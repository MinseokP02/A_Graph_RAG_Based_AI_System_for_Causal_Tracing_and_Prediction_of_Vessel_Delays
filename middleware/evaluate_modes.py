"""Compare naive, hybrid and Neo4j+mix through the same middleware API."""

import argparse
import asyncio
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODES = ("naive", "hybrid", "neo4j_mix")
DEFAULT_DIR = Path(__file__).with_name("evaluation")
JUDGE_RUBRICS = {
    "context_relevance": """Context Relevance를 0~1 실수로 평가한다.
검색 근거의 전체 정보 중 질문에 답하는 데 필요하거나 잠재적으로 도움이 되는 정보의 비율이다.
질문의 대상 선박, 항만, 날짜, 운항 사건과 무관하거나 중복된 정보는 관련 없는 정보로 본다.
근거가 비어 있거나 질문에 답할 정보가 전혀 없으면 0이다.""",
    "faithfulness": """Faithfulness를 0~1 실수로 평가한다.
답변을 검증 가능한 원자적 claim으로 나눈 뒤, 검색 근거에서 직접 확인하거나 합리적으로 추론할 수 있는 claim의 비율을 점수로 준다.
근거 없는 숫자, 사건, 인과관계 단정과 결측값을 0으로 채운 표현은 미지원 claim이다.
추측이라고 명시한 해석은 사실 주장과 구분되어 있고 제시된 근거와 논리적으로 연결될 때만 지원된 것으로 본다.""",
    "answer_relevance": """Answer Relevance를 0~1 실수로 평가한다.
답변이 질문을 직접적이고 적절하게 다루는 정도다.
질문의 대상 선박, 항만, 날짜, 운항 또는 지연 사건에 집중했는지 평가한다.
불완전한 답변, 질문과 무관한 내용, 불필요한 반복이 많을수록 감점한다. 사실성은 이 지표에서 평가하지 않는다.""",
    "comprehensiveness": """Comprehensiveness를 1~5 정수로 평가한다.
1: 질문이 요구한 핵심 정보를 거의 다루지 않음.
2: 일부 정보만 언급하며 중요한 요소가 다수 누락됨.
3: 주요 정보를 다루지만 일부 관련 요인 또는 관계가 누락됨.
4: 검색된 근거에서 확인 가능한 대부분의 관련 정보와 관계를 충분히 설명함.
5: 검색된 근거에서 확인 가능한 주요 정보, 관계, 상호작용을 빠짐없이 포괄적으로 설명함.
검색 근거에 존재하지 않는 내용을 추가하지 않았다는 이유로 감점하지 않는다.""",
}

JUDGE_METRICS = (
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "comprehensiveness",
)


def _evidence_text(evidence: list[dict], limit: int = 16000) -> str:
    text = json.dumps(evidence, ensure_ascii=False, default=str)
    return text[:limit]


async def judge_metric(
    client, judge_url, judge_model, metric, question, evidence, answer
):
    """Call one isolated judge with only the inputs allowed for that metric."""
    if metric == "context_relevance":
        inputs = {"question": question, "retrieved_evidence": _evidence_text(evidence)}
        score_schema = {"type": "number", "minimum": 0, "maximum": 1}
    elif metric == "faithfulness":
        inputs = {"retrieved_evidence": _evidence_text(evidence), "answer": answer}
        score_schema = {"type": "number", "minimum": 0, "maximum": 1}
    elif metric == "answer_relevance":
        inputs = {"question": question, "answer": answer}
        score_schema = {"type": "number", "minimum": 0, "maximum": 1}
    elif metric == "comprehensiveness":
        inputs = {
            "question": question,
            "retrieved_evidence": _evidence_text(evidence),
            "answer": answer,
        }
        score_schema = {"type": "integer", "minimum": 1, "maximum": 5}
    else:
        raise ValueError(f"Unknown judge metric: {metric}")
    schema = {
        "type": "object",
        "properties": {
            "score": score_schema,
            "rationale": {"type": "string"},
        },
        "required": ["score", "rationale"],
        "additionalProperties": False,
    }
    response = await client.post(
        judge_url.rstrip("/") + "/api/chat",
        json={
            "model": judge_model,
            "stream": False,
            "format": schema,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 RAG 평가자입니다. "
                    + JUDGE_RUBRICS[metric]
                    + " 반드시 제공된 입력만 사용하고 rationale은 한국어로 "
                    "작성하여 JSON으로 답하세요.",
                },
                {"role": "user", "content": json.dumps(inputs, ensure_ascii=False)},
            ],
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 512},
        },
    )
    response.raise_for_status()
    result = json.loads(response.json()["message"]["content"])
    score = result["score"]
    if metric != "comprehensiveness":
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            raise ValueError(f"Invalid {metric} score")
        score = float(score)
    elif isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"Invalid {metric} score")
    if not isinstance(result.get("rationale"), str) or not result["rationale"].strip():
        raise ValueError("Judge rationale is missing")
    return {"score": score, "rationale": result["rationale"].strip()}


async def judge_answer(client, args, question, evidence, answer):
    if args.skip_judge:
        return {"judge": None, "judge_latency_ms": None, "judge_error": None}
    started = time.perf_counter()
    try:
        judged = {}
        for metric in JUDGE_METRICS:
            judged[metric] = await judge_metric(
                client,
                args.judge_url,
                args.judge_model,
                metric,
                question,
                evidence,
                answer,
            )
        return {
            "judge": judged,
            "judge_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "judge_error": None,
        }
    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "judge": None,
            "judge_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "judge_error": type(exc).__name__,
        }


def load_suite(questions_path: Path, answers_path: Path) -> list[dict]:
    """Validate one-to-one IDs before making any API request."""
    questions_doc = json.loads(questions_path.read_text(encoding="utf-8"))
    answers_doc = json.loads(answers_path.read_text(encoding="utf-8"))
    if questions_doc.get("dataset") != answers_doc.get("dataset"):
        raise ValueError("Question and answer dataset versions differ")
    questions = questions_doc.get("questions")
    answers = answers_doc.get("answers")
    if not isinstance(questions, list) or not isinstance(answers, list):
        raise TypeError("Questions and answers must be lists")
    q_by_id = {item.get("id"): item for item in questions}
    a_by_id = {item.get("id"): item for item in answers}
    if None in q_by_id or None in a_by_id or len(q_by_id) != len(questions):
        raise ValueError("Question and answer IDs must be present and unique")
    if len(a_by_id) != len(answers) or set(q_by_id) != set(a_by_id):
        raise ValueError("Question and answer IDs do not match")
    cases = []
    for item in questions:
        answer = a_by_id[item["id"]]
        terms = answer.get("required_terms")
        if (
            not isinstance(item.get("question"), str)
            or not isinstance(terms, list)
            or not terms
        ):
            raise ValueError(f"Invalid evaluation case: {item['id']}")
        cases.append({**item, **answer})
    return cases


CAUSAL_UNCERTAINTY_PATTERNS = (
    r"단정.{0,16}(?:없|어렵|곤란|부족|불충분)",
    r"확정.{0,16}(?:없|어렵|곤란|부족|불충분)",
    r"판단.{0,16}(?:없|어렵|곤란|부족|불충분)",
    r"(?:직접|공식|명확한?)\s*(?:원인|인과관계).{0,16}(?:없|알 수 없|확인할 수 없|어렵)",
    r"원인.{0,16}(?:추가|더 많은)\s*(?:정보|근거).{0,12}(?:필요)",
)


def _term_match(answer: str, term: str) -> bool:
    """Match literal requirements plus the declared causal-uncertainty concept."""
    if term.casefold() in answer.casefold():
        return True
    if term == "단정할 수 없다":
        return any(
            re.search(pattern, answer) for pattern in CAUSAL_UNCERTAINTY_PATTERNS
        )
    return False


def score_answer(answer: str, required_terms: list[str]) -> dict:
    """Return transparent deterministic term recall; never invoke a judge LLM."""
    matches = {term: _term_match(answer, term) for term in required_terms}
    return {
        "required_term_recall": sum(matches.values()) / len(matches),
        "all_required_terms": all(matches.values()),
        "term_matches": matches,
    }


async def run_case(client, args, case, mode):
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    body = {
        "question": case["question"],
        "plan": case.get("plan"),
        "mode": mode,
        "use_documents": True,
        "max_records": args.max_records,
    }
    started = time.perf_counter()
    try:
        response = await client.post(
            args.endpoint.rstrip("/") + "/ask", headers=headers, json=body
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("answer", "")
        scores = score_answer(answer, case["required_terms"])
        evidence = payload.get("evidence", [])
        judged = await judge_answer(client, args, case["question"], evidence, answer)
        return {
            "case_id": case["id"],
            "category": case.get("category"),
            "mode": mode,
            "status": payload.get("status"),
            "latency_ms": latency_ms,
            **scores,
            "answer": answer,
            "ground_truth": case["ground_truth"],
            "graph_evidence": sum(e.get("kind") == "graph" for e in evidence),
            "document_evidence": sum(e.get("kind") == "document" for e in evidence),
            "warnings": payload.get("warnings", []),
            "error": None,
            **judged,
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return {
            "case_id": case["id"],
            "category": case.get("category"),
            "mode": mode,
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "required_term_recall": 0.0,
            "all_required_terms": False,
            "term_matches": {},
            "answer": "",
            "ground_truth": case["ground_truth"],
            "graph_evidence": 0,
            "document_evidence": 0,
            "warnings": [],
            "error": type(exc).__name__,
            "judge": None,
            "judge_latency_ms": None,
            "judge_error": None,
        }


def summarize(results):
    summary = {}
    for mode in MODES:
        rows = [row for row in results if row["mode"] == mode]
        successful = [row for row in rows if row["status"] == "ok"]
        judged = [row for row in successful if row.get("judge")]
        summary[mode] = {
            "cases": len(rows),
            "successful": len(successful),
            "all_required_terms_rate": (
                sum(row["all_required_terms"] for row in rows) / len(rows)
                if rows
                else 0
            ),
            "mean_required_term_recall": (
                sum(row["required_term_recall"] for row in rows) / len(rows)
                if rows
                else 0
            ),
            "mean_latency_ms": (
                sum(row["latency_ms"] for row in successful) / len(successful)
                if successful
                else None
            ),
            "judged": len(judged),
            **{
                f"mean_{metric}": (
                    sum(row["judge"][metric]["score"] for row in judged) / len(judged)
                    if judged
                    else None
                )
                for metric in JUDGE_METRICS
            },
        }
    return summary


def write_results(output_dir: Path, report: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"mode_comparison_{stamp}.json"
    csv_path = output_dir / f"mode_comparison_{stamp}.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "case_id",
        "category",
        "mode",
        "status",
        "required_term_recall",
        "all_required_terms",
        "latency_ms",
        "graph_evidence",
        "document_evidence",
        "context_relevance",
        "faithfulness",
        "answer_relevance",
        "comprehensiveness",
        "judge_latency_ms",
        "judge_error",
        "answer",
        "ground_truth",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in report["results"]:
            flat = dict(row)
            judge = row.get("judge") or {}
            for metric in JUDGE_METRICS:
                flat[metric] = judge.get(metric, {}).get("score")
            writer.writerow(flat)
    return json_path, csv_path


async def evaluate(args, cases):
    timeout = httpx.Timeout(args.timeout, connect=10)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        results = []
        for case in cases:
            for mode in MODES:
                results.append(await run_case(client, args, case, mode))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:9622")
    parser.add_argument("--api-key", default=os.getenv("MIDDLEWARE_API_KEY", ""))
    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_DIR / "questions.json"
    )
    parser.add_argument("--answers", type=Path, default=DEFAULT_DIR / "answers.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "results")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-records", type=int, default=10, choices=range(1, 51))
    parser.add_argument(
        "--judge-url",
        default=os.getenv(
            "JUDGE_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434")
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "JUDGE_MODEL", os.getenv("QWEN_MODEL", "qwen2.5:7b-instruct")
        ),
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        cases = load_suite(args.questions, args.answers)
        if args.dry_run:
            print(f"Validated {len(cases)} cases for {len(MODES)} modes")
            return 0
        results = asyncio.run(evaluate(args, cases))
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": args.endpoint,
            "modes": list(MODES),
            "judge": {
                "url": args.judge_url,
                "model": args.judge_model,
                "metrics": JUDGE_RUBRICS,
            },
            "summary": summarize(results),
            "results": results,
        }
        json_path, csv_path = write_results(args.output, report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"JSON: {json_path}\nCSV: {csv_path}")
        return 1 if any(row["status"] == "error" for row in results) else 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

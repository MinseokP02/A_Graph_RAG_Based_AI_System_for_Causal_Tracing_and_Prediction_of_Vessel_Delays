"""Offline validation for the deterministic three-mode comparison."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from middleware.evaluate_modes import (
    DEFAULT_DIR,
    judge_metric,
    load_suite,
    score_answer,
    summarize,
)


def test_bundled_suite_is_paired_and_versioned():
    cases = load_suite(DEFAULT_DIR / "questions.json", DEFAULT_DIR / "answers.json")
    assert len(cases) == 8
    assert len({case["id"] for case in cases}) == 8
    assert sum(case["category"] == "delay_analysis" for case in cases) == 4
    assert all(case["plan"]["date_basis"] == "departure" for case in cases)


def test_suite_rejects_mismatched_answer_ids(tmp_path: Path):
    questions = tmp_path / "questions.json"
    answers = tmp_path / "answers.json"
    questions.write_text(
        json.dumps({"dataset": "v1", "questions": [{"id": "q", "question": "abc"}]}),
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps(
            {"dataset": "v1", "answers": [{"id": "other", "required_terms": ["x"]}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="IDs do not match"):
        load_suite(questions, answers)


def test_scoring_and_summary_are_deterministic():
    score = score_answer("D7DZ는 12:20 출항", ["d7dz", "12:20", "부산"])
    assert score["required_term_recall"] == pytest.approx(2 / 3)
    rows = [
        {
            "mode": mode,
            "status": "ok",
            "all_required_terms": mode == "neo4j_mix",
            "required_term_recall": 1 if mode == "neo4j_mix" else 0.5,
            "latency_ms": 10,
        }
        for mode in ("naive", "hybrid", "neo4j_mix")
    ]
    summary = summarize(rows)
    assert summary["neo4j_mix"]["all_required_terms_rate"] == 1
    assert summary["naive"]["mean_required_term_recall"] == 0.5


@pytest.mark.parametrize(
    "wording",
    [
        "인과관계로 단정할 수 없다.",
        "지연 원인으로 단정하기에는 근거가 부족합니다.",
        "정확한 지연 원인은 확정하기 어렵습니다.",
        "현재 자료만으로 원인을 판단할 수 없습니다.",
        "직접 원인인지 확인할 수 없습니다.",
        "원인을 밝히려면 추가 정보가 필요합니다.",
    ],
)
def test_causal_uncertainty_equivalent_wordings_are_accepted(wording):
    result = score_answer(wording, ["단정할 수 없다"])
    assert result["all_required_terms"] is True


def test_positive_causal_claim_does_not_match_uncertainty():
    result = score_answer("강우가 지연의 직접 원인이라고 단정했다.", ["단정할 수 없다"])
    assert result["all_required_terms"] is False


def test_metric_judges_receive_only_their_declared_inputs():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        inputs = json.loads(body["messages"][1]["content"])
        seen.append(inputs)
        score = (
            4 if set(inputs) == {"question", "retrieved_evidence", "answer"} else 0.75
        )
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps({"score": score, "rationale": "근거"})
                }
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for metric in (
                "context_relevance",
                "faithfulness",
                "answer_relevance",
                "comprehensiveness",
            ):
                await judge_metric(
                    client,
                    "http://judge",
                    "model",
                    metric,
                    "질문",
                    [{"content": "근거"}],
                    "답변",
                )

    asyncio.run(run())
    assert set(seen[0]) == {"question", "retrieved_evidence"}
    assert set(seen[1]) == {"retrieved_evidence", "answer"}
    assert set(seen[2]) == {"question", "answer"}
    assert set(seen[3]) == {"question", "retrieved_evidence", "answer"}

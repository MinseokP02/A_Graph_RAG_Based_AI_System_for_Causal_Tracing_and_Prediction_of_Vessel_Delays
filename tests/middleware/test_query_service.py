"""Offline query API tests, with injected models and HTTP transports."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from middleware.graph_retriever import (
    CALL_SIGN_MATCH,
    DETAILS,
    SUMMARY,
    parameters,
    retrieval_queries,
)
from middleware.query_api import create_app
from middleware.query_config import QueryConfig
from middleware.query_models import AskRequest, QueryPlan
from middleware.rag_service import (
    DocumentRetriever,
    QwenClient,
    RagService,
    UpstreamError,
)


def dependencies():
    """Return an injectable service with a real graph-shaped response."""
    graph, qwen, docs = AsyncMock(), AsyncMock(), AsyncMock()
    graph.retrieve.return_value = {
        "summary": {
            "matchingRecords": 3,
            "vessels": 1,
            "knownDelayRecords": 0,
            "knownStayRecords": 0,
            "delayedRecords": 0,
        },
        "records": [
            {"recordId": "r1", "vesselName": "TEST", "departureDelayMinutes": None}
        ],
    }
    qwen.plan.return_value = QueryPlan(vessel_name="TEST")
    qwen.chat.return_value = "근거에 따른 답변 [G1] [D1]"
    docs.retrieve.return_value = [
        {"content": "port congestion context", "source": "a.txt", "kind": "document"}
    ]
    config = QueryConfig()
    return RagService(graph, qwen, docs, config), graph, qwen, docs


def test_graph_only_skips_documents_and_planning():
    service, _graph, qwen, docs = dependencies()
    response = asyncio.run(
        service.ask(
            AskRequest(question="선박 조회", plan=QueryPlan(), use_documents=False)
        )
    )
    docs.retrieve.assert_not_awaited()
    qwen.plan.assert_not_awaited()
    assert response["evidence"][0]["id"] == "G1"
    assert any("unknown" in w for w in response["warnings"])


def test_combined_context_and_sampling():
    service, _graph, qwen, _docs = dependencies()
    response = asyncio.run(service.ask(AskRequest(question="선박 조회")))
    assert [e["id"] for e in response["evidence"]] == ["G1", "D1"]
    assert any("sample" in w for w in response["warnings"])
    payload = qwen.chat.await_args.args[1]
    assert payload["evidence"][1]["source"] == "a.txt"
    assert payload["summary"]["matchingRecords"] == 3


def test_count_answer_uses_full_graph_summary_without_llm_sample_confusion():
    service, graph, qwen, docs = dependencies()
    graph.retrieve.return_value["summary"]["matchingRecords"] = 44
    response = asyncio.run(
        service.ask(
            AskRequest(
                question="D7DZ 출항 기록은 몇 건이야?",
                plan=QueryPlan(intent="count", call_sign="D7DZ"),
                use_documents=False,
            )
        )
    )
    qwen.chat.assert_not_awaited()
    docs.retrieve.assert_not_awaited()
    assert "44건" in response["answer"]
    assert "D7DZ" in response["answer"]
    assert response["graph"]["summary"]["matchingRecords"] == 44


def test_answer_prompt_requires_weather_and_causality_limits_for_delay_analysis():
    service, _graph, qwen, _docs = dependencies()
    asyncio.run(
        service.ask(
            AskRequest(
                question="호출부호 A8XY6 선박의 지연 원인을 분석해줘.",
                plan=QueryPlan(intent="delays", call_sign="A8XY6"),
            )
        )
    )
    system = qwen.chat.await_args.args[0]
    payload = qwen.chat.await_args.args[1]
    assert "delayPairingMethod가 기록된 PortCall" in system
    assert "기온, 풍속, 강우량을 모두 빠짐없이" in system
    assert "인과관계로 단정할 수 없다" in system
    assert "구분 | 내용 | 근거 | 확실성" in system
    assert "구분을 `추측`으로 표시" in system
    assert len(payload["mandatory_answer_requirements"]) == 7
    assert "departureDelayMinutes" in payload["mandatory_answer_requirements"][0]


@pytest.mark.parametrize("mode", ["naive", "hybrid"])
def test_document_only_modes_skip_neo4j_and_select_lightrag_mode(mode):
    service, graph, qwen, docs = dependencies()
    response = asyncio.run(service.ask(AskRequest(question="선박 조회", mode=mode)))
    graph.retrieve.assert_not_awaited()
    qwen.plan.assert_not_awaited()
    docs.retrieve.assert_awaited_once_with("선박 조회", mode)
    assert response["mode"] == mode
    assert response["graph"] == {"summary": {}, "returned_records": 0}
    assert [row["kind"] for row in response["evidence"]] == ["document"]


def test_neo4j_mix_selects_mix_document_retrieval():
    service, graph, _qwen, docs = dependencies()
    response = asyncio.run(service.ask(AskRequest(question="선박 조회")))
    graph.retrieve.assert_awaited_once()
    docs.retrieve.assert_awaited_once_with("선박 조회", "mix")
    assert response["mode"] == "neo4j_mix"


def test_document_outage_is_explicit():
    service, _graph, _qwen, docs = dependencies()
    docs.retrieve.side_effect = UpstreamError("unavailable")
    response = asyncio.run(service.ask(AskRequest(question="선박 조회")))
    assert response["status"] == "ok"
    assert any("unavailable" in w for w in response["warnings"])


def test_document_only_mode_fails_when_its_only_retriever_is_unavailable():
    service, graph, qwen, docs = dependencies()
    docs.retrieve.side_effect = UpstreamError("unavailable")
    with pytest.raises(UpstreamError, match="unavailable"):
        asyncio.run(service.ask(AskRequest(question="선박 조회", mode="naive")))
    graph.retrieve.assert_not_awaited()
    qwen.chat.assert_not_awaited()


def test_graph_outage_never_generates_an_answer():
    service, graph, qwen, _docs = dependencies()
    graph.retrieve.side_effect = RuntimeError("password=secret")
    with pytest.raises(UpstreamError, match="Neo4j retrieval unavailable"):
        asyncio.run(service.ask(AskRequest(question="선박 조회")))
    qwen.chat.assert_not_awaited()


def test_clarification_does_not_query():
    service, graph, qwen, _docs = dependencies()
    qwen.plan.return_value = QueryPlan(clarification="조회 연도를 알려주세요.")
    response = asyncio.run(service.ask(AskRequest(question="8월 선박 조회")))
    assert response["status"] == "clarification_required"
    graph.retrieve.assert_not_awaited()


def test_context_budget_is_applied():
    service, _graph, _qwen, docs = dependencies()
    service.config = QueryConfig(document_chars=1000)
    docs.retrieve.return_value = [
        {"kind": "document", "content": "a" * 3000, "source": "x"}
    ]
    response = asyncio.run(service.ask(AskRequest(question="선박 조회")))
    assert len(response["evidence"][1]["content"]) == 1000
    assert response["evidence"][1]["truncated"]


def test_dates_and_parameter_injection():
    plan = QueryPlan(
        start_date="2026-08-01", end_date="2026-08-31", vessel_name="'; DELETE n"
    )
    params = parameters(plan, 10, "Asia/Seoul")
    assert params["end"] == "2026-09-01T00:00:00+09:00"
    assert params["start"] == "2026-08-01T00:00:00+09:00"
    assert "DELETE" not in SUMMARY + DETAILS
    assert params["vessel_name"] == "'; DELETE N"
    with pytest.raises(ValidationError):
        QueryPlan(start_date="2026-09-01", end_date="2026-08-01")
    with pytest.raises(ValidationError):
        QueryPlan(cypher="MATCH (n) RETURN n")


def test_call_sign_query_uses_index_friendly_exact_match_template():
    summary, details = retrieval_queries(QueryPlan(call_sign="D7DZ"))
    assert summary.startswith(CALL_SIGN_MATCH)
    assert details.startswith(CALL_SIGN_MATCH)
    assert "v.callSign = $call_sign" in summary
    assert "vesselKey: $vessel_key" in summary
    assert "$call_sign IS NULL OR" not in summary
    generic_summary, _ = retrieval_queries(QueryPlan())
    assert generic_summary == SUMMARY
    assert (
        parameters(QueryPlan(call_sign="d7dz"), 10, "Asia/Seoul")["vessel_key"]
        == 'callsign:["D7DZ"]'
    )


def test_http_document_contract():
    def handler(request):
        """Assert retrieval-only API fields and return one chunk."""
        body = json.loads(request.content)
        assert request.url.path == "/query/data"
        assert body["mode"] == "naive"
        assert request.headers["X-API-Key"] == "test-key"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "chunks": [
                        {
                            "content": "evidence",
                            "file_path": "a.txt",
                            "chunk_id": "chunk1",
                        }
                    ]
                },
            },
        )

    async def run():
        """Use a mocked HTTP transport, never a live service."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await DocumentRetriever(
                http, QueryConfig(lightrag_key="test-key")
            ).retrieve("query")

    assert asyncio.run(run())[0]["chunk_id"] == "chunk1"


def test_http_document_contract_accepts_hybrid():
    def handler(request):
        assert json.loads(request.content)["mode"] == "hybrid"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "chunks": [],
                    "entities": [
                        {
                            "entity_name": "Vessel:A8XY6",
                            "entity_type": "Vessel",
                            "description": "callSign=A8XY6",
                            "source_id": "entity-chunk",
                            "file_path": "custom_kg",
                        }
                    ],
                    "relationships": [
                        {
                            "src_id": "Vessel:A8XY6",
                            "tgt_id": "PortCall:1",
                            "keywords": "MADE_PORT_CALL",
                            "description": "A8XY6 made port call 1",
                            "source_id": "relation-chunk",
                            "file_path": "custom_kg",
                        }
                    ],
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await DocumentRetriever(http, QueryConfig()).retrieve(
                "query", "hybrid"
            )

    result = asyncio.run(run())
    assert [row["evidence_type"] for row in result] == ["entity", "relationship"]
    assert result[0]["chunk_id"] == "entity-chunk"
    assert "MADE_PORT_CALL" in result[1]["content"]


def test_empty_document_result_does_not_generate_zero_placeholders():
    service, graph, qwen, docs = dependencies()
    docs.retrieve.return_value = []
    response = asyncio.run(
        service.ask(
            AskRequest(
                question="A8XY6 지연 원인을 분석해줘.",
                mode="hybrid",
                plan=QueryPlan(intent="delays", call_sign="A8XY6"),
            )
        )
    )
    graph.retrieve.assert_not_awaited()
    qwen.chat.assert_not_awaited()
    assert response["evidence"] == []
    assert "A8XY6" in response["answer"]
    assert "근거가 없어" in response["answer"]
    assert "결측값은 0으로 간주하지 않습니다" in response["answer"]


def test_qwen_schema_and_invalid_plan():
    def handler(request):
        """Return deliberately invalid structured output."""
        body = json.loads(request.content)
        assert body["stream"] is False
        assert "format" in body
        return httpx.Response(200, json={"message": {"content": '{"cypher":"DELETE"}'}})

    async def run():
        """Reject model-produced fields outside the plan schema."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await QwenClient(http, QueryConfig()).plan("query")

    with pytest.raises(UpstreamError, match="invalid query plan"):
        asyncio.run(run())


def test_api_auth_validation_and_upstream_error():
    service, graph, _qwen, _docs = dependencies()
    with TestClient(create_app(service, QueryConfig(api_key="test"))) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/ask", json={"question": "선박 조회"}).status_code == 401
        headers = {"X-API-Key": "test"}
        assert (
            client.post("/ask", headers=headers, json={"question": "x"}).status_code
            == 422
        )
        assert (
            client.post(
                "/ask",
                headers=headers,
                json={"question": "선박 조회", "max_records": 1000},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/ask", headers=headers, json={"question": "선박 조회"}
            ).status_code
            == 200
        )
        graph.retrieve.side_effect = RuntimeError("secret")
        response = client.post("/ask", headers=headers, json={"question": "선박 조회"})
        assert response.status_code == 502
        assert "secret" not in response.text


def test_extension_point_adds_evidence():
    service, _graph, _qwen, _docs = dependencies()
    extra = AsyncMock()
    extra.retrieve.return_value = [
        {"kind": "graph_description", "content": "description", "source": "node-id"}
    ]
    service.additional_retrievers = (extra,)
    response = asyncio.run(
        service.ask(AskRequest(question="선박 조회", use_documents=False))
    )
    assert response["evidence"][1]["kind"] == "graph_description"


def test_common_korean_question_uses_exact_parser():
    """Unambiguous common filters do not depend on Qwen's extraction accuracy."""
    qwen = QwenClient(None, QueryConfig())
    plan = asyncio.run(
        qwen.plan("호출부호 202809 선박의 2026년 8월 입출항과 사용 계선시설을 알려줘.")
    )
    assert plan.call_sign == "202809"
    assert plan.end_date.isoformat() == "2026-08-31"
    assert plan.intent == "berths"
    assert plan.date_basis == "event"

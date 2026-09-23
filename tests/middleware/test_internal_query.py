"""Internal graph/document composition without network services."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from lightrag import QueryParam, portmis


def make_rag():
    return SimpleNamespace(
        _build_global_config=lambda: {
            "role_llm_funcs": {"query": AsyncMock()},
            "tokenizer": SimpleNamespace(encode=lambda s: list(s)),
        },
        aquery_data=AsyncMock(
            return_value={
                "data": {
                    "chunks": [{"content": "document evidence", "reference_id": "1"}]
                }
            }
        ),
        aquery_llm=AsyncMock(
            return_value={"status": "success", "llm_response": {"content": "answer"}}
        ),
        _query_done=AsyncMock(),
    )


def test_combines_evidence_without_http(monkeypatch):
    monkeypatch.setattr(
        portmis,
        "graph_context",
        AsyncMock(
            return_value={
                "summary": {"matchingRecords": 4},
                "records": [{"evidence_id": "G1", "recordId": "a"}],
            }
        ),
    )
    rag = make_rag()
    param = QueryParam(mode="mix", stream=True)
    result = asyncio.run(portmis.query_with_portmis(rag, "question", param))
    assert result["metadata"]["portmis"]["records"][0]["recordId"] == "a"
    args = rag.aquery_llm.call_args.args
    assert args[1].mode == "bypass" and args[1].stream
    assert "document evidence" in args[2] and "G1" in args[2]
    assert param.mode == "mix"


def test_graph_without_documents(monkeypatch):
    monkeypatch.setattr(
        portmis,
        "graph_context",
        AsyncMock(return_value={"summary": {}, "records": [{"recordId": "a"}]}),
    )
    rag = make_rag()
    rag.aquery_data.return_value = {"status": "failure", "data": {}}
    result = asyncio.run(portmis.query_with_portmis(rag, "question", QueryParam()))
    assert result["status"] == "success"
    assert result["metadata"]["portmis"]["records"]


def test_failure_is_explicit_and_context_only_skips_generation(monkeypatch):
    monkeypatch.setattr(
        portmis, "graph_context", AsyncMock(side_effect=RuntimeError("secret"))
    )
    rag = make_rag()
    result = asyncio.run(
        portmis.query_with_portmis(rag, "question", QueryParam(only_need_context=True))
    )
    assert result["metadata"]["warnings"]
    assert "secret" not in str(result)
    rag.aquery_llm.assert_not_called()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PORTMIS_GRAPH_ENABLED", raising=False)
    assert not portmis.internal_query_enabled()

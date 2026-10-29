"""Offline forecasting contracts; synthetic outputs are confined to tests."""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from middleware.forecast import (
    ForecastRequest,
    ForecastResult,
    TFTClient,
    extract_forecast_date,
    is_forecast_question,
)
from middleware.query_api import create_app
from middleware.query_config import QueryConfig
from middleware.query_models import AskRequest, QueryPlan
from middleware.rag_service import QwenClient, UpstreamError
from middleware.tft_api import (
    FeatureBundle,
    FeatureWindow,
    TFTBackend,
    estimate_delay_probability,
)
from middleware.tft_api import create_app as create_tft_app
from tests.middleware.test_query_service import dependencies


def prediction():
    """Return explicitly synthetic, typed predictions for transport tests."""
    return ForecastResult(
        status="ok",
        model_version="test-only",
        series_id="TEST",
        as_of="2026-09-27T10:00:00+09:00",
        predictions=[
            {
                "timestamp": "2026-09-27T11:00:00+09:00",
                "p10": 5,
                "p50": 20,
                "p90": 40,
                "delay_probability_percent": 10,
            }
        ],
        delay_threshold_minutes=60,
        probability_method="quantile_linear_interpolation_capped",
    )


@pytest.mark.parametrize(
    "question",
    [
        "2026-09-27 지연 예측",
        "TEST 2026년 9월 27일 지연예측 알려줘",
        "2026.09.27 지연  예측 부탁해",
    ],
)
def test_trigger_overrides_lookup_intent_and_includes_forecast(question):
    service, graph, qwen, _ = dependencies()
    service.forecaster = AsyncMock()
    service.forecaster.predict.return_value = prediction()
    with TestClient(create_app(service, QueryConfig())) as client:
        response = client.post(
            "/ask",
            json={
                "question": question,
                "plan": {"intent": "visits", "vessel_name": "TEST"},
                "use_documents": False,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["intent"] == "delay_forecast"
    assert body["forecast"]["predictions"][0]["p50"] == 20
    assert body["evidence"][-1]["id"] == "F1"
    assert "P50 (분)" in body["answer"] and "[F1]" in body["answer"]
    assert qwen.chat.await_args.args[1]["forecast"]["status"] == "ok"
    assert service.forecaster.predict.await_args.args[0].plan.vessel_name == "TEST"
    graph.retrieve.assert_awaited_once()
    qwen.plan.assert_not_awaited()


def test_no_trigger_does_not_call_tft():
    service, _, _, _ = dependencies()
    service.forecaster = AsyncMock()
    body = asyncio.run(
        service.ask(
            AskRequest(question="지연 기록 조회", plan=QueryPlan(intent="delays"))
        )
    )
    service.forecaster.predict.assert_not_awaited()
    assert "forecast" not in body


def test_bare_trigger_needs_no_llm_planner():
    assert (
        asyncio.run(QwenClient(None, QueryConfig()).plan("지연 예측")).intent
        == "delay_forecast"
    )


def test_any_prediction_keyword_routes_and_extracts_one_date():
    assert is_forecast_question("2026-11-03 도착시간 예측")
    assert (
        extract_forecast_date("2026년 11월 3일 지연을 예측해줘").isoformat()
        == "2026-11-03"
    )


def test_prediction_without_date_requests_clarification():
    service, graph, _, _ = dependencies()
    service.forecaster = AsyncMock()

    body = asyncio.run(
        service.ask(AskRequest(question="TEST 선박의 지연 예측", plan=QueryPlan()))
    )

    assert body["status"] == "clarification_required"
    assert "날짜" in body["answer"]
    service.forecaster.predict.assert_not_awaited()
    graph.retrieve.assert_not_awaited()


@pytest.mark.parametrize(
    "status", ["model_not_ready", "insufficient_data", "unavailable"]
)
def test_unready_status_is_explicit_in_answer(status):
    service, _, _, _ = dependencies()
    service.forecaster = AsyncMock()
    service.forecaster.predict.return_value = ForecastResult(status=status)
    body = asyncio.run(
        service.ask(AskRequest(question="2026-09-27 지연 예측", plan=QueryPlan()))
    )
    assert body["forecast"]["status"] == status
    assert body["forecast"]["predictions"] == []
    assert all(e["kind"] != "forecast" for e in body["evidence"])
    assert "TFT" in body["answer"]


def test_tft_survives_graph_and_narrative_outage():
    service, graph, qwen, _ = dependencies()
    graph.retrieve.side_effect = RuntimeError("secret")
    qwen.chat.side_effect = UpstreamError("Qwen/Ollama response unavailable")
    service.forecaster = AsyncMock()
    service.forecaster.predict.return_value = prediction()
    body = asyncio.run(
        service.ask(AskRequest(question="2026-09-27 지연 예측", plan=QueryPlan()))
    )
    assert body["forecast"]["status"] == "ok"
    assert "[F1]" in body["answer"]
    assert "secret" not in json.dumps(body)


@pytest.mark.parametrize(
    "broken", ["timeout", "http", "invalid", "horizon", "crossed", "nan"]
)
def test_bad_tft_responses_are_sanitized(broken):
    def handler(request):
        """Emulate transport errors and untrustworthy model responses."""
        assert request.headers["X-API-Key"] == "test"
        request_body = json.loads(request.content)
        assert request_body["steps"] == 1
        assert request_body["delay_threshold_minutes"] == 60
        if broken == "timeout":
            raise httpx.ReadTimeout("secret", request=request)
        if broken == "http":
            return httpx.Response(500, text="secret")
        data = prediction().model_dump(mode="json")
        if broken == "invalid":
            data = {"status": "ok"}
        elif broken == "horizon":
            data["predictions"].append(
                {**data["predictions"][0], "timestamp": "2026-09-27T12:00:00+09:00"}
            )
        elif broken == "crossed":
            data["predictions"][0]["p10"] = 100
        elif broken == "nan":
            data["predictions"][0]["p50"] = "NaN"
        return httpx.Response(200, json=data)

    async def run():
        """Exercise the real HTTP adapter with no external service."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await TFTClient(http, QueryConfig(tft_api_key="test")).predict(
                ForecastRequest(plan=QueryPlan())
            )

    assert asyncio.run(run()).status == "unavailable"


def test_sidecar_auth_validation_and_missing_checkpoint():
    with TestClient(create_tft_app(TFTBackend(), api_key="test")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"plan": {}}).status_code == 401
        response = client.post(
            "/predict", json={"plan": {}}, headers={"X-API-Key": "test"}
        )
        assert response.json()["status"] == "model_not_ready"
        assert response.json()["predictions"] == []
        assert (
            client.post(
                "/predict",
                json={"plan": {}, "steps": 25},
                headers={"X-API-Key": "test"},
            ).status_code
            == 422
        )


def test_probability_uses_capped_quantile_interpolation():
    assert estimate_delay_probability(10, 30, 90, 0) == 90
    assert estimate_delay_probability(10, 30, 90, 60) == 30
    assert estimate_delay_probability(10, 30, 90, 100) == 10


def window():
    """Make a fresh, single-series input fixture, not production features."""
    now = datetime.now(timezone.utc) - timedelta(seconds=1)
    return FeatureWindow(
        series_id="TEST",
        selectors=QueryPlan(vessel_name="TEST"),
        as_of=now,
        forecast_times=[now + timedelta(hours=1)],
        rows=[{"time_idx": i, "vessel": "TEST", "delay": 5.0} for i in range(4)],
    )


def test_feature_selection_rejects_ambiguity_staleness_and_wrong_scope():
    backend = TFTBackend(model_version="test")
    feature = window()
    bundle = FeatureBundle(model_version="test", windows=[feature])
    request = ForecastRequest(plan=QueryPlan(vessel_name="test"))
    assert backend.select_window(bundle, request) is feature
    assert (
        backend.select_window(bundle, ForecastRequest(plan=QueryPlan(port_code="020")))
        is None
    )
    assert (
        backend.select_window(bundle, ForecastRequest(plan=QueryPlan(), steps=2))
        is None
    )
    bundle.windows.append(feature)
    assert backend.select_window(bundle, request) is None
    bundle.windows = [
        feature.model_copy(update={"as_of": feature.as_of - timedelta(days=1)})
    ]
    assert backend.select_window(bundle, request) is None
    bundle.windows = [feature]
    bundle.model_version = "wrong"
    assert backend.select_window(bundle, request) is None


def test_real_http_boundary_between_apis():
    backend = Mock()
    backend.predict.return_value = prediction()
    tft_app = create_tft_app(backend, api_key="test")

    async def run():
        """Use the service endpoint and HTTP client together, without a socket."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=tft_app)
        ) as http:
            service, _, _, _ = dependencies()
            service.forecaster = TFTClient(http, QueryConfig(tft_api_key="test"))
            return await service.ask(
                AskRequest(
                    question="2026-09-27 지연 예측",
                    plan=QueryPlan(intent="delay_forecast"),
                )
            )

    assert asyncio.run(run())["forecast"]["status"] == "ok"


def test_checkpoint_adapter_preserves_training_parameters_and_quantile_order(
    monkeypatch,
):
    """Exercise preprocessing and mapping with lightweight stand-ins for torch."""
    pf, metrics = (
        ModuleType("pytorch_forecasting"),
        ModuleType("pytorch_forecasting.metrics"),
    )
    metrics.QuantileLoss = type("QuantileLoss", (), {})
    loss = metrics.QuantileLoss()
    loss.quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    tensor = Mock()
    tensor.shape = (1, 1, 7)
    tensor.detach.return_value.cpu.return_value.tolist.return_value = np.array(
        [[[1, 5, 10, 20, 30, 40, 50]]]
    ).tolist()
    params = {
        "max_prediction_length": 1,
        "min_encoder_length": 2,
        "target": "delay",
        "time_idx": "time_idx",
        "group_ids": ["vessel"],
    }
    model = Mock(dataset_parameters=params, loss=loss)
    model.predict.return_value = tensor
    pf.TemporalFusionTransformer = SimpleNamespace(
        load_from_checkpoint=Mock(return_value=model)
    )
    pf.TimeSeriesDataSet = SimpleNamespace(from_parameters=Mock(return_value="dataset"))
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", pf)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting.metrics", metrics)
    result = TFTBackend(checkpoint="trusted.ckpt").infer(
        window(), ForecastRequest(plan=QueryPlan())
    )
    assert result.predictions[0].p50 == 20
    assert result.predictions[0].p90 == 40
    assert result.predictions[0].delay_probability_percent == 10
    assert result.delay_threshold_minutes == 60
    assert result.probability_method == "quantile_linear_interpolation_capped"
    assert result.basis is not None
    args = pf.TimeSeriesDataSet.from_parameters.call_args
    assert args.args[0] is params
    assert args.args[1]["delay"].tolist() == [5, 5, 5, 0]
    assert args.kwargs == {"predict": True, "stop_randomization": True}
    assert model.predict.call_args.kwargs["mode"] == "quantiles"


def test_forecast_schema_rejects_numbers_when_not_ready():
    with pytest.raises(ValidationError):
        ForecastResult.model_validate(
            {**prediction().model_dump(), "status": "model_not_ready"}
        )


def test_untrained_sidecar_returns_readiness_through_query_service():
    """The real unconfigured backend works without torch or a running LLM."""

    async def run():
        """Wire both service boundaries and preserve the no-model status."""
        app = create_tft_app(TFTBackend(), api_key="")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as http:
            service, _, qwen, _ = dependencies()
            qwen.chat.side_effect = UpstreamError("unavailable")
            service.forecaster = TFTClient(http, QueryConfig())
            return await service.ask(
                AskRequest(
                    question="2026-09-27 출항 지연을 예측해줘",
                    use_documents=False,
                    plan=QueryPlan(intent="delay_forecast"),
                )
            )

    body = asyncio.run(run())
    assert body["forecast"]["status"] == "model_not_ready"
    assert body["forecast"]["predictions"] == []
    assert "학습 모델이 아직 준비되지 않아" in body["answer"]


def test_clarification_does_not_invoke_prediction():
    """The trigger does not discard unresolved scope clarification."""
    service, graph, _, _ = dependencies()
    service.forecaster = AsyncMock()
    body = asyncio.run(
        service.ask(
            AskRequest(
                question="지연 예측",
                plan=QueryPlan(clarification="어느 선박인가요?"),
            )
        )
    )
    assert body["status"] == "clarification_required"
    service.forecaster.predict.assert_not_awaited()
    graph.retrieve.assert_not_awaited()

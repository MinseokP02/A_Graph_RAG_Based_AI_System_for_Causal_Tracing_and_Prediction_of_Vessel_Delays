"""Independent TFT inference API; start with python -m middleware.tft_api."""

import hmac
import json
import math
import os
import threading
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .forecast import ForecastBasis, ForecastPoint, ForecastRequest, ForecastResult
from .query_models import QueryPlan


class FeatureWindow(BaseModel):
    """One prepared series: historical encoder rows followed by decoder rows."""

    model_config = ConfigDict(extra="forbid")
    series_id: str = Field(min_length=1, max_length=200)
    selectors: QueryPlan
    as_of: AwareDatetime
    forecast_times: list[AwareDatetime] = Field(min_length=1, max_length=24)
    rows: list[dict] = Field(min_length=2, max_length=10000)

    @model_validator(mode="after")
    def chronological(self) -> "FeatureWindow":
        """Require future timestamps ordered relative to the snapshot cutoff."""
        times = [self.as_of, *self.forecast_times]
        if any(a >= b for a, b in pairwise(times)):
            raise ValueError("Feature-window forecast times must increase after as_of")
        return self


class FeatureBundle(BaseModel):
    """Locally managed features produced with the checkpoint's training schema."""

    model_config = ConfigDict(extra="forbid")
    model_version: str = Field(min_length=1, max_length=100)
    windows: list[FeatureWindow] = Field(default_factory=list, max_length=1000)


def estimate_delay_probability(
    p10: float, p50: float, p90: float, threshold: float
) -> float:
    """Estimate threshold exceedance from three quantiles, capped at 10-90%."""

    points = ((p10, 0.1), (p50, 0.5), (p90, 0.9))
    if threshold <= p10:
        cdf = 0.1
    elif threshold >= p90:
        cdf = 0.9
    else:
        lower, upper = (
            (points[0], points[1]) if threshold <= p50 else (points[1], points[2])
        )
        if upper[0] == lower[0]:
            cdf = upper[1]
        else:
            ratio = (threshold - lower[0]) / (upper[0] - lower[0])
            cdf = lower[1] + ratio * (upper[1] - lower[1])
    return round((1 - cdf) * 100, 1)


def _optional_finite(row: dict, key: str) -> float | None:
    try:
        value = float(row[key])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def forecast_basis(row: dict) -> ForecastBasis:
    """Expose a small, typed subset of the decoder inputs as provenance."""

    return ForecastBasis.model_validate(
        {
            "planned_departure": row.get("planned_departure_utc"),
            "feature_cutoff": row.get("feature_cutoff_utc"),
            "ais_observation_count": _optional_finite(row, "ais_observation_count"),
            "latest_sog_knots": _optional_finite(row, "ais_sog"),
            "latest_latitude": _optional_finite(row, "ais_latitude"),
            "latest_longitude": _optional_finite(row, "ais_longitude"),
            "gross_tonnage": _optional_finite(row, "gross_tonnage"),
            "cargo_ton": _optional_finite(row, "cargo_ton"),
            "hours_to_arrival": _optional_finite(row, "hours_to_arrival"),
            "weather_age_hours": _optional_finite(row, "weather_age_hours"),
            "temperature_2m_c": _optional_finite(row, "temperature_2m_c"),
            "wind_speed_10m_ms": _optional_finite(row, "wind_speed_10m_ms"),
            "rain_mm": _optional_finite(row, "rain_mm"),
        }
    )


class TFTBackend:
    """Lazy CPU inference with optional heavy dependencies and a serialized model."""

    def __init__(
        self,
        checkpoint: str = "",
        features: str = "",
        model_version: str = "local",
        max_age_seconds: int = 3600,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("TFT_MAX_FEATURE_AGE_SECONDS must be positive")
        self.checkpoint, self.features = checkpoint, features
        self.model_version, self.max_age_seconds = model_version, max_age_seconds
        self.model = None
        self.lock = threading.Lock()

    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Do not load torch until a checkpoint and feature source are available."""
        if not self.checkpoint or not Path(self.checkpoint).is_file():
            return ForecastResult(status="model_not_ready")
        if not self.features or not Path(self.features).is_file():
            return ForecastResult(status="insufficient_data")
        with self.lock:
            try:
                bundle = FeatureBundle.model_validate(
                    json.loads(Path(self.features).read_text(encoding="utf-8"))
                )
                window = self.select_window(bundle, request)
                if window is None:
                    return ForecastResult(status="insufficient_data")
                return self.infer(window, request)
            except Exception:  # noqa: BLE001 -- isolate optional model/runtime failures
                # The API must not expose local paths, features, or library errors.
                return ForecastResult(status="unavailable")

    def select_window(
        self,
        bundle: FeatureBundle,
        request: ForecastRequest,
    ) -> FeatureWindow | None:
        """Match all supplied entity filters; never guess among multiple series."""
        if bundle.model_version != self.model_version or request.plan.clarification:
            return None
        matches = []
        for window in bundle.windows:
            if any(
                value is not None
                and value.casefold()
                != (getattr(window.selectors, key) or "").casefold()
                for key in (
                    "call_sign",
                    "vessel_name",
                    "port_code",
                    "port_name",
                    "berth_name",
                )
                if (value := getattr(request.plan, key)) is not None
            ):
                continue
            age = (datetime.now(timezone.utc) - window.as_of).total_seconds()
            if not 0 <= age <= self.max_age_seconds or request.steps > len(
                window.forecast_times
            ):
                continue
            dates = [t.date() for t in window.forecast_times[: request.steps]]
            if request.plan.start_date and min(dates) < request.plan.start_date:
                continue
            if request.plan.end_date and max(dates) > request.plan.end_date:
                continue
            matches.append(window)
        return matches[0] if len(matches) == 1 else None

    def infer(self, window: FeatureWindow, request: ForecastRequest) -> ForecastResult:
        """Use training encoders/scalers from a trusted TFT checkpoint unchanged."""
        import pandas as pd
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import QuantileLoss

        if self.model is None:
            self.model = TemporalFusionTransformer.load_from_checkpoint(
                self.checkpoint, map_location="cpu"
            )
            self.model.eval()
        model = self.model
        params = model.dataset_parameters
        horizon = params["max_prediction_length"]
        if (
            not isinstance(model.loss, QuantileLoss)
            or not isinstance(params["target"], str)
            or len(window.forecast_times) != horizon
            or len(window.rows) < params["min_encoder_length"] + horizon
        ):
            return ForecastResult(status="insufficient_data")
        quantiles = list(model.loss.quantiles)
        if not all(q in quantiles for q in (0.1, 0.5, 0.9)):
            return ForecastResult(status="insufficient_data")
        frame = pd.DataFrame(window.rows)
        time_idx = params["time_idx"]
        frame = frame.sort_values(time_idx).reset_index(drop=True)
        if (
            frame[params["group_ids"]].drop_duplicates().shape[0] != 1
            or not pd.api.types.is_integer_dtype(frame[time_idx])
            or not frame[time_idx].diff().dropna().eq(1).all()
        ):
            return ForecastResult(status="insufficient_data")
        # Future target values are unknown; placeholders are never model evidence.
        frame.loc[len(frame) - horizon :, params["target"]] = 0.0
        dataset = TimeSeriesDataSet.from_parameters(
            params, frame, predict=True, stop_randomization=True
        )
        output = model.predict(
            dataset,
            mode="quantiles",
            trainer_kwargs={"accelerator": "cpu", "devices": 1, "logger": False},
        )
        if tuple(output.shape) != (1, horizon, len(quantiles)):
            raise ValueError("Unexpected TFT output shape")
        values = output.detach().cpu().tolist()[0]
        indices = [quantiles.index(q) for q in (0.1, 0.5, 0.9)]
        predictions = []
        for timestamp, row in zip(
            window.forecast_times[: request.steps], values[: request.steps]
        ):
            p10, p50, p90 = (float(row[index]) for index in indices)
            predictions.append(
                ForecastPoint(
                    timestamp=timestamp,
                    p10=p10,
                    p50=p50,
                    p90=p90,
                    delay_probability_percent=estimate_delay_probability(
                        p10, p50, p90, request.delay_threshold_minutes
                    ),
                )
            )
        return ForecastResult(
            status="ok",
            model_version=self.model_version,
            as_of=window.as_of,
            series_id=window.series_id,
            predictions=predictions,
            delay_threshold_minutes=request.delay_threshold_minutes,
            probability_method="quantile_linear_interpolation_capped",
            basis=forecast_basis(window.rows[-1]),
        )


def create_app(
    backend: TFTBackend | None = None, api_key: str | None = None
) -> FastAPI:
    """Run inference in FastAPI's thread pool, outside the query event loop."""
    from dotenv import load_dotenv

    load_dotenv(
        os.getenv("MIDDLEWARE_ENV_FILE", Path(__file__).with_name(".env")),
        override=False,
    )
    backend = (
        backend
        if backend is not None
        else TFTBackend(
            checkpoint=os.getenv("TFT_CHECKPOINT", ""),
            features=os.getenv("TFT_FEATURES_PATH", ""),
            model_version=os.getenv("TFT_MODEL_VERSION", "local"),
            max_age_seconds=int(os.getenv("TFT_MAX_FEATURE_AGE_SECONDS", "3600")),
        )
    )
    api_key = os.getenv("TFT_API_KEY", "") if api_key is None else api_key
    app = FastAPI(title="TFT departure delay forecast API")

    @app.get("/health")
    def health() -> dict:
        """Report only process liveness; predictions validate model readiness."""
        return {"status": "alive", "model_verified": False}

    @app.post("/predict", response_model=ForecastResult)
    def predict(
        request: ForecastRequest, x_api_key: str | None = Header(default=None)
    ) -> ForecastResult:
        """Return a typed forecast or a non-numeric readiness status."""
        if api_key and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(status_code=401, detail="Invalid TFT API key")
        return backend.predict(request)

    return app


def main() -> None:
    """Bind the standalone model service to loopback by default."""
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=os.getenv("TFT_HOST", "127.0.0.1"),
        port=int(os.getenv("TFT_PORT", "9623")),
    )


if __name__ == "__main__":
    main()

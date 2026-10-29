"""Run a one-step holdout backtest from a trained local TFT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

from .tft_api import estimate_delay_probability


def backtest(
    checkpoint: Path,
    input_path: Path,
    *,
    delay_threshold_minutes: float = 60.0,
) -> dict[str, object]:
    """Predict the final event while reusing checkpoint encoders and scalers."""

    import pandas as pd
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    model.eval()
    params = model.dataset_parameters
    if not isinstance(model.loss, QuantileLoss):
        raise TypeError("Checkpoint does not use QuantileLoss")
    quantiles = list(model.loss.quantiles)
    if not all(value in quantiles for value in (0.1, 0.5, 0.9)):
        raise ValueError("Checkpoint is missing P10/P50/P90 quantiles")

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    time_idx = params["time_idx"]
    target = params["target"]
    frame[time_idx] = frame[time_idx].astype(int)
    for name in (
        set(params["group_ids"])
        | set(params.get("static_categoricals", []))
        | set(params.get("time_varying_known_categoricals", []))
    ):
        frame[name] = frame[name].fillna("UNKNOWN").astype(str)
    actual = float(frame.iloc[-1][target])
    holdout = frame.iloc[-1]
    frame.loc[frame.index[-params["max_prediction_length"] :], target] = 0.0
    dataset = TimeSeriesDataSet.from_parameters(
        params, frame, predict=True, stop_randomization=True
    )
    output = model.predict(
        dataset,
        mode="quantiles",
        trainer_kwargs={"accelerator": "cpu", "devices": 1, "logger": False},
    )
    values = output.detach().cpu().tolist()[0][0]
    p10, p50, p90 = (float(values[quantiles.index(value)]) for value in (0.1, 0.5, 0.9))
    return {
        "status": "experimental",
        "scope": "single_last_event_holdout",
        "checkpoint": str(checkpoint.resolve()),
        "input": str(input_path.resolve()),
        "training_rows": len(frame),
        "encoder_length": params["max_encoder_length"],
        "vessel_name": str(holdout.get("vessel_name", "")),
        "voyage_id": str(holdout.get("voyage_id", "")),
        "forecast_time": str(holdout.get("planned_departure_utc", "")),
        "actual_delay_minutes": actual,
        "prediction": {
            "p10_minutes": p10,
            "p50_minutes": p50,
            "p90_minutes": p90,
            "p50_absolute_error_minutes": abs(p50 - actual),
            "delay_threshold_minutes": delay_threshold_minutes,
            "delay_probability_percent": estimate_delay_probability(
                p10, p50, p90, delay_threshold_minutes
            ),
        },
        "warning": (
            "Five matched voyages are insufficient for accuracy claims; this result only "
            "verifies the training and inference path."
        ),
    }


def build_holdout_bundle(
    checkpoint: Path,
    input_path: Path,
    *,
    model_version: str,
) -> dict[str, object]:
    """Create one historical feature window for exercising the HTTP pipeline."""

    import pandas as pd
    from pytorch_forecasting import TemporalFusionTransformer

    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint, map_location="cpu"
    )
    params = model.dataset_parameters
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    horizon = int(params["max_prediction_length"])
    required = int(params["max_encoder_length"]) + horizon
    if len(frame) < required:
        raise ValueError(f"Feature window needs at least {required} rows")
    window = frame.tail(required).copy().reset_index(drop=True)
    categorical_names = (
        set(params["group_ids"])
        | set(params.get("static_categoricals", []))
        | set(params.get("time_varying_known_categoricals", []))
    )
    for name in categorical_names:
        window[name] = window[name].fillna("UNKNOWN").astype(str)
    window[params["time_idx"]] = range(required)
    window.loc[window.index[-horizon:], params["target"]] = 0.0
    forecast_time = pd.Timestamp(
        window.iloc[-1]["planned_departure_utc"]
    ).to_pydatetime()
    if forecast_time.tzinfo is None:
        raise ValueError("planned_departure_utc must be timezone-aware")
    rows = json.loads(window.to_json(orient="records", date_format="iso"))
    last = rows[-1]
    return {
        "model_version": model_version,
        "windows": [
            {
                "series_id": f"experimental-{last['voyage_id']}",
                "selectors": {
                    "call_sign": last["call_sign"],
                    "vessel_name": last["vessel_name"],
                    "port_code": str(last["port_code"]),
                },
                "as_of": (forecast_time - timedelta(seconds=1)).isoformat(),
                "forecast_times": [forecast_time.isoformat()],
                "rows": rows,
            }
        ],
    }


def _write_json(path: Path, value: dict[str, object], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"Output exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=path.parent
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary:
            temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-threshold-minutes", type=float, default=60.0)
    parser.add_argument("--features-output", type=Path)
    parser.add_argument(
        "--model-version", default="incheon-probabilistic-v0-experimental"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.delay_threshold_minutes < 0:
            raise ValueError("delay-threshold-minutes must be non-negative")
        result = backtest(
            args.checkpoint,
            args.input,
            delay_threshold_minutes=args.delay_threshold_minutes,
        )
        _write_json(args.output, result, args.overwrite)
        if args.features_output:
            bundle = build_holdout_bundle(
                args.checkpoint,
                args.input,
                model_version=args.model_version,
            )
            _write_json(args.features_output, bundle, args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

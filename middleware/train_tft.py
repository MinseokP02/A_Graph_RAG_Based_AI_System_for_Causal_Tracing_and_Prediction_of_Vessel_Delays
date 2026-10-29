"""Train the repository's departure-delay TFT from a canonical event CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .tft_features import TRAINING_COLUMNS, TFTDataError

KNOWN_REALS = [
    "time_idx",
    "planned_hour_sin",
    "planned_hour_cos",
    "planned_dow_sin",
    "planned_dow_cos",
    "gross_tonnage",
    "tug_used",
    "pilotage_used",
    "cargo_ton",
    "has_ais",
    "ais_observation_count",
    "ais_latitude",
    "ais_longitude",
    "ais_sog",
    "ais_sog_mean",
    "ais_cog",
    "ais_heading",
    "hours_to_arrival",
    "has_weather",
    "weather_age_hours",
    "temperature_2m_c",
    "wind_speed_10m_ms",
    "rain_mm",
]


def validate_training_csv(path: Path, encoder_length: int) -> int:
    """Fail before importing ML libraries when the dataset is not trainable."""

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing = set(TRAINING_COLUMNS) - set(header)
        if missing:
            raise TFTDataError(
                f"Training columns missing: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if len(rows) < encoder_length + 2:
        raise TFTDataError(
            f"Need at least {encoder_length + 2} rows, found {len(rows)}"
        )
    if any(not row["delay_minutes"].strip() for row in rows):
        raise TFTDataError("Training CSV contains an unlabeled delay_minutes row")
    if not any(row["has_ais"].strip() == "1.0" for row in rows):
        raise TFTDataError("Training CSV contains no AIS-backed rows")
    return len(rows)


def train(
    input_path: Path,
    output_path: Path,
    *,
    encoder_length: int = 48,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.03,
    hidden_size: int = 32,
    attention_heads: int = 4,
) -> dict[str, object]:
    """Train one-step P10/P50/P90 departure-delay quantiles on CPU."""

    row_count = validate_training_csv(input_path, encoder_length)
    if output_path.suffix != ".ckpt":
        raise TFTDataError("output checkpoint must use the .ckpt suffix")
    try:
        import pandas as pd
        import torch
        from lightning.pytorch import Trainer, seed_everything
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
        from pytorch_forecasting.metrics import QuantileLoss
    except ImportError as exc:
        raise TFTDataError(
            "TFT training dependencies are missing; install middleware/requirements-tft.txt"
        ) from exc

    if encoder_length < 2 or max_epochs < 1 or batch_size < 1:
        raise TFTDataError("encoder_length, max_epochs and batch_size must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise TFTDataError("learning_rate must be positive and finite")
    seed_everything(42, workers=True)
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    frame["time_idx"] = frame["time_idx"].astype(int)
    frame["port_code"] = frame["port_code"].astype(str)
    frame["call_sign"] = frame["call_sign"].fillna("UNKNOWN").astype(str)
    frame["vessel_name"] = frame["vessel_name"].fillna("UNKNOWN").astype(str)
    numeric = ["delay_minutes", *[name for name in KNOWN_REALS if name != "time_idx"]]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not frame[numeric].apply(lambda column: column.map(math.isfinite).all()).all():
        raise TFTDataError("Training data contains non-finite numeric values")
    counts = frame.groupby("port_code")["time_idx"].count()
    if counts.max() < encoder_length + 2:
        raise TFTDataError(
            "No port series is long enough for the encoder and validation"
        )
    training_cutoff = int(frame["time_idx"].max()) - 1
    encoders = {
        "port_code": NaNLabelEncoder(add_nan=True),
        "call_sign": NaNLabelEncoder(add_nan=True),
        "vessel_name": NaNLabelEncoder(add_nan=True),
    }
    training = TimeSeriesDataSet(
        frame[lambda value: value.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="delay_minutes",
        group_ids=["port_code"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["port_code"],
        time_varying_known_categoricals=["call_sign", "vessel_name"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=["delay_minutes"],
        categorical_encoders=encoders,
        target_normalizer=GroupNormalizer(
            groups=["port_code"], transformation="softplus"
        ),
        allow_missing_timesteps=False,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, frame, predict=True, stop_randomization=True
    )
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    validation_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_heads,
        hidden_continuous_size=max(8, hidden_size // 2),
        dropout=0.1,
        output_size=3,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=-1,
        reduce_on_plateau_patience=3,
    )
    checkpoint = ModelCheckpoint(
        dirpath=output_path.parent,
        filename=output_path.stem,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            checkpoint,
        ],
        logger=False,
        enable_checkpointing=True,
        deterministic=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )
    if Path(checkpoint.best_model_path).resolve() != output_path.resolve():
        raise TFTDataError(
            f"Best checkpoint was written to an unexpected path: {checkpoint.best_model_path}"
        )
    return {
        "checkpoint": str(output_path.resolve()),
        "rows": row_count,
        "encoder_length": encoder_length,
        "prediction_length": 1,
        "quantiles": [0.1, 0.5, 0.9],
        "torch_version": torch.__version__,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/tft/training.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/departure-delay.ckpt")
    )
    parser.add_argument("--encoder-length", type=int, default=48)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = train(
            args.input,
            args.output,
            encoder_length=args.encoder_length,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            attention_heads=args.attention_heads,
        )
    except (OSError, TFTDataError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

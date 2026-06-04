"""
TFT Training Job — ANALYTICA Sprint 5 D3
Training job for climate_transformer_tft (TFTClimateModel).

Data source: FeatureStoreClient.get_historical_features()
Splits: 70% train / 15% val / 15% test (chronological)
Epochs: 50, early stopping patience 5 on val_loss
Best checkpoint: workspace/models/climate_transformer/best.ckpt (overwrite)

MLflow experiment: climate_transformer_tft
  Logs: loss curves (train_loss, val_loss), T2M RMSE, RH RMSE per epoch
  Registers: climate_transformer_tft@Candidate on completion
  Tags: feature_store_version (via FeatureStoreClient.tag_mlflow_run)

Prometheus: tropi_training_job_duration_seconds{model_name}  ← Gauge
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from prometheus_client import Gauge
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ANALYTICA module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.feature_store_client import FeatureStoreClient  # noqa: E402
from models.climate_transformer import TFTClimateModel    # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
_JOB_DURATION = Gauge(
    "tropi_training_job_duration_seconds",
    "Total wall-clock duration of a training job run",
    ["model_name"],
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME          = "climate_transformer_tft"
MLFLOW_URI          = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.analytica.svc:5000")
EXPERIMENT_NAME     = "climate_transformer_tft"
REGISTRY_NAME       = "climate_transformer_tft"
REGISTRY_STAGE      = "Candidate"

CHECKPOINT_DIR      = Path("workspace/models/climate_transformer")
CHECKPOINT_PATH     = CHECKPOINT_DIR / "best.ckpt"

MAX_EPOCHS          = 50
EARLY_STOP_PATIENCE = 5      # on val_loss
BATCH_SIZE          = 64
ENCODER_LENGTH      = 30
PRED_HORIZON        = 7

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = remaining 15%

# Features consumed from the Feast feature store
FEATURE_REFS = [
    "weather_stats:t2m_mean_24h",
    "weather_stats:t2m_min_24h",
    "weather_stats:t2m_max_24h",
    "weather_stats:rh_mean_24h",
    "weather_stats:rh_min_24h",
    "weather_stats:precip_sum_24h",
    "ocean_teleconnection:oni_index",
    "ocean_teleconnection:iod_dmi",
    "satellite_derived:lst_day_mean",
    "satellite_derived:ndvi_16day",
]

# Entity IDs to pull — all 38 provinces, represented by their BMKG anchor station
ENTITY_DF_PATH = Path("workspace/data/training/tft_entity_df.parquet")
OUTPUT_LOG     = Path("workspace/output/tft_training_log.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_entity_df() -> pd.DataFrame:
    if ENTITY_DF_PATH.exists():
        return pd.read_parquet(ENTITY_DF_PATH)
    # Minimal fallback: build from last 3 years of province-level timestamps
    logger.warning(
        "Entity DataFrame not found at %s — building synthetic entity_df fallback.", ENTITY_DF_PATH
    )
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D", tz="UTC")
    provinces = [f"province_{i:02d}" for i in range(1, 39)]
    rows = [
        {"event_timestamp": d, "station_id": p}
        for d in dates
        for p in provinces
    ]
    return pd.DataFrame(rows)


def _split_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/15/15 split on time_idx."""
    df = df.sort_values("time_idx").reset_index(drop=True)
    n = df["time_idx"].nunique()
    train_cut = int(n * TRAIN_FRAC)
    val_cut   = int(n * (TRAIN_FRAC + VAL_FRAC))
    sorted_idx = sorted(df["time_idx"].unique())
    train_df = df[df["time_idx"] <= sorted_idx[train_cut - 1]]
    val_df   = df[(df["time_idx"] > sorted_idx[train_cut - 1]) & (df["time_idx"] <= sorted_idx[val_cut - 1])]
    test_df  = df[df["time_idx"] > sorted_idx[val_cut - 1]]
    logger.info(
        "Split → train: %d rows (time_idx ≤ %d) | val: %d rows | test: %d rows",
        len(train_df), sorted_idx[train_cut - 1], len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def _rmse(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def _write_log(payload: dict) -> None:
    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Training log written to %s", OUTPUT_LOG)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training() -> dict:
    job_start = time.perf_counter()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"tft_train_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    ) as run:
        run_id = run.info.run_id
        logger.info("MLflow run: %s", run_id)

        # --- Feature retrieval ---
        client = FeatureStoreClient()
        FeatureStoreClient.tag_mlflow_run(run)

        entity_df = _load_entity_df()
        logger.info("Fetching historical features for %d entity rows …", len(entity_df))
        df = client.get_historical_features(
            entity_df=entity_df,
            feature_refs=FEATURE_REFS,
            start_dt=datetime(2023, 1, 1),
            end_dt=datetime(2025, 12, 31),
        )

        # Ensure time_idx exists (integer day offset)
        if "time_idx" not in df.columns:
            df = df.sort_values("event_timestamp")
            min_date = df["event_timestamp"].min()
            df["time_idx"] = (
                (df["event_timestamp"] - min_date).dt.days.astype(int)
            )
        if "group_id" not in df.columns:
            df["group_id"] = df.get("station_id", "province_01").astype(str)

        df["t2m"] = df.get("weather_stats__t2m_mean_24h", df.get("t2m_mean_24h", 0.0))
        df["rh"]  = df.get("weather_stats__rh_mean_24h",  df.get("rh_mean_24h",  50.0))
        df = df.dropna(subset=["t2m", "rh", "time_idx", "group_id"])

        mlflow.log_params({
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "batch_size": BATCH_SIZE,
            "encoder_length": ENCODER_LENGTH,
            "pred_horizon": PRED_HORIZON,
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "feature_refs": FEATURE_REFS,
        })

        # --- Data splits ---
        train_df, val_df, test_df = _split_time_series(df)

        # --- Datasets ---
        time_varying_unknown = [
            c for c in df.columns
            if c not in ("time_idx", "group_id", "event_timestamp", "station_id")
        ]

        training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=["t2m", "rh"],
            group_ids=["group_id"],
            min_encoder_length=ENCODER_LENGTH // 2,
            max_encoder_length=ENCODER_LENGTH,
            min_prediction_length=1,
            max_prediction_length=PRED_HORIZON,
            time_varying_unknown_reals=time_varying_unknown,
            static_categoricals=["group_id"],
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        val_dataset  = TimeSeriesDataSet.from_dataset(training_dataset, val_df,  predict=False, stop_randomization=True)
        test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, test_df, predict=True,  stop_randomization=True)

        train_loader = training_dataset.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
        val_loader   = val_dataset.to_dataloader(     train=False, batch_size=BATCH_SIZE, num_workers=0)
        test_loader  = test_dataset.to_dataloader(    train=False, batch_size=BATCH_SIZE, num_workers=0)

        # --- Model ---
        model = TFTClimateModel.from_dataset(
            training_dataset,
            learning_rate=3e-3,
            hidden_size=256,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=64,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        logger.info("TFTClimateModel — parameters: %d", sum(p.numel() for p in model.parameters()))

        # --- Callbacks ---
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, verbose=True, mode="min"
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(CHECKPOINT_DIR),
            filename="best",       # → best.ckpt (overwrite each run)
            save_top_k=1,
            mode="min",
        )

        # --- Trainer ---
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            gradient_clip_val=0.1,
            enable_progress_bar=True,
            callbacks=[early_stop, checkpoint],
        )

        logger.info("trainer.fit() starting — max_epochs=%d, patience=%d", MAX_EPOCHS, EARLY_STOP_PATIENCE)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logger.info("Training done — stopped at epoch %d", trainer.current_epoch)

        # Log loss curves
        if hasattr(trainer.logger, "history"):
            for step, metrics in enumerate(trainer.logger.history or []):
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if k in ("train_loss", "val_loss")},
                    step=step,
                )

        # --- Load best checkpoint & evaluate on test set ---
        best_model = TFTClimateModel.load_from_checkpoint(str(CHECKPOINT_PATH))
        preds = best_model.predict(test_loader, return_y=True,
                                   trainer_kwargs={"accelerator": "auto"})

        pred_t2m = preds.output[:, :, 0].numpy().flatten()
        pred_rh  = preds.output[:, :, 1].numpy().flatten()
        true_t2m = preds.y[0][:, :, 0].numpy().flatten()
        true_rh  = preds.y[0][:, :, 1].numpy().flatten()

        t2m_rmse = _rmse(pred_t2m, true_t2m)
        rh_rmse  = _rmse(pred_rh,  true_rh)

        mlflow.log_metrics({"t2m_rmse": t2m_rmse, "rh_rmse": rh_rmse})
        logger.info("Test metrics — T2M RMSE: %.4f°C | RH RMSE: %.2f%%", t2m_rmse, rh_rmse)

        # --- Register to Model Registry as Candidate ---
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="tft_climate_model",
            registered_model_name=REGISTRY_NAME,
        )
        reg_client = mlflow.tracking.MlflowClient()
        latest_ver = reg_client.get_latest_versions(REGISTRY_NAME, stages=["None"])[0].version
        reg_client.transition_model_version_stage(
            name=REGISTRY_NAME, version=latest_ver, stage=REGISTRY_STAGE
        )
        logger.info("Registered %s v%s → %s", REGISTRY_NAME, latest_ver, REGISTRY_STAGE)

        # --- Prometheus duration ---
        duration = time.perf_counter() - job_start
        _JOB_DURATION.labels(model_name=MODEL_NAME).set(duration)
        logger.info("Total job duration: %.1f s", duration)

        log_payload = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": MODEL_NAME,
            "epochs_trained": trainer.current_epoch,
            "checkpoint": str(CHECKPOINT_PATH),
            "metrics": {"t2m_rmse": round(t2m_rmse, 4), "rh_rmse": round(rh_rmse, 4)},
            "registry": {"name": REGISTRY_NAME, "version": latest_ver, "stage": REGISTRY_STAGE},
            "duration_seconds": round(duration, 2),
        }
        _write_log(log_payload)
        return log_payload


if __name__ == "__main__":
    result = run_training()
    print(json.dumps(result, indent=2, default=str))

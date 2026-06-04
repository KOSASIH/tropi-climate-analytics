"""
TFT Training Job — ANALYTICA Sprint 5 D5
One-shot training execution for TFTClimateModel (climate_transformer.py).
Designed to be triggered by model_serving_health_dag or run manually.

Promotion thresholds:
  T2M RMSE ≤ 1.2°C  AND  RH RMSE ≤ 8%
    → MLflow Production registry as tropi_climate_tft v1
  Otherwise
    → MLflow Staging registry + failure report

Emits: tropi_tft_training_complete gauge (1=production, 0=staging)
Output: workspace/output/tft_training_log.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from prometheus_client import Gauge
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Import project model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.climate_transformer import TFTClimateModel  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH    = Path("workspace/data/training/tft_dataset.parquet")
OUTPUT_PATH     = Path("workspace/output/tft_training_log.json")
MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.analytica.svc:5000")
EXPERIMENT_NAME = "climate_transformer"
REGISTRY_NAME   = "tropi_climate_tft"
VALIDATION_DAYS = 90
MAX_EPOCHS      = 50
EARLY_STOP_PAT  = 10

T2M_RMSE_THRESHOLD = 1.2   # °C
RH_RMSE_THRESHOLD  = 8.0   # %

# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
_TRAINING_COMPLETE = Gauge(
    "tropi_tft_training_complete",
    "TFT training result: 1=promoted_to_production, 0=staging_only",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"TFT training dataset not found at {DATASET_PATH}. "
            "Coordinate with DATA-FLOW to confirm feature export is complete."
        )
    logger.info("Loading TFT training dataset from %s", DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH)
    logger.info("Dataset shape: %s | columns: %s", df.shape, list(df.columns))
    return df


def _split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split last VALIDATION_DAYS as validation; remainder as train."""
    df = df.sort_values("time_idx")
    max_time_idx = df["time_idx"].max()
    val_cutoff   = max_time_idx - VALIDATION_DAYS
    train_df = df[df["time_idx"] <= val_cutoff].copy()
    val_df   = df[df["time_idx"] >  val_cutoff].copy()
    logger.info(
        "Split: train=%d rows (time_idx ≤ %d), val=%d rows (time_idx > %d)",
        len(train_df), val_cutoff, len(val_df), val_cutoff,
    )
    return train_df, val_df


def _compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def _write_log(payload: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Training log written to %s", OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"tft_training_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}") as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        # --- Load data ---
        df = _load_dataset()
        train_df, val_df = _split_train_val(df)

        # --- Build TimeSeriesDataSet ---
        # Expects columns: time_idx (int), group_id (str), t2m (float), rh (float),
        # plus any known/unknown regressors DATA-FLOW provides.
        time_varying_known_reals   = [c for c in df.columns if c.startswith("known_")]
        time_varying_unknown_reals = [c for c in df.columns if c.startswith("unknown_")]
        static_categoricals        = [c for c in df.columns if c.startswith("static_cat_")]
        static_reals               = [c for c in df.columns if c.startswith("static_real_")]

        training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=["t2m", "rh"],
            group_ids=["group_id"],
            min_encoder_length=15,
            max_encoder_length=30,
            min_prediction_length=1,
            max_prediction_length=7,
            time_varying_known_reals=time_varying_known_reals or ["time_idx"],
            time_varying_unknown_reals=time_varying_unknown_reals + ["t2m", "rh"],
            static_categoricals=static_categoricals or ["group_id"],
            static_reals=static_reals,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        val_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, val_df, predict=False, stop_randomization=True
        )

        train_loader = training_dataset.to_dataloader(train=True,  batch_size=64, num_workers=0)
        val_loader   = val_dataset.to_dataloader(  train=False, batch_size=64, num_workers=0)

        # --- Instantiate model ---
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
        logger.info("TFTClimateModel instantiated — parameters: %d", sum(p.numel() for p in model.parameters()))

        # --- Callbacks ---
        early_stop = EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PAT, verbose=True, mode="min"
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath="workspace/output/tft_checkpoints",
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
        )

        # --- Trainer ---
        trainer = Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="auto",
            enable_progress_bar=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop, checkpoint],
        )

        mlflow.log_params({
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PAT,
            "hidden_size": 256,
            "attention_heads": 4,
            "validation_days": VALIDATION_DAYS,
            "t2m_threshold": T2M_RMSE_THRESHOLD,
            "rh_threshold": RH_RMSE_THRESHOLD,
        })

        logger.info("Starting trainer.fit() — max_epochs=%d, early_stop_patience=%d", MAX_EPOCHS, EARLY_STOP_PAT)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        logger.info("Training complete — stopped at epoch %d", trainer.current_epoch)

        # --- Evaluate on validation set ---
        predictions = model.predict(val_loader, return_y=True, trainer_kwargs={"accelerator": "auto"})
        pred_t2m = predictions.output[:, :, 0].numpy().flatten()
        pred_rh  = predictions.output[:, :, 1].numpy().flatten()
        true_t2m = predictions.y[0][:, :, 0].numpy().flatten()
        true_rh  = predictions.y[0][:, :, 1].numpy().flatten()

        t2m_rmse = _compute_rmse(pred_t2m, true_t2m)
        rh_rmse  = _compute_rmse(pred_rh,  true_rh)

        mlflow.log_metrics({"t2m_rmse": t2m_rmse, "rh_rmse": rh_rmse})
        logger.info("Validation metrics — T2M RMSE: %.4f°C | RH RMSE: %.2f%%", t2m_rmse, rh_rmse)

        # --- Promotion decision ---
        promoted = t2m_rmse <= T2M_RMSE_THRESHOLD and rh_rmse <= RH_RMSE_THRESHOLD
        registry_stage = "Production" if promoted else "Staging"

        mlflow.pytorch.log_model(
            model,
            artifact_path="tft_climate_model",
            registered_model_name=REGISTRY_NAME,
        )

        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(REGISTRY_NAME, stages=["None"])[0].version
        client.transition_model_version_stage(
            name=REGISTRY_NAME,
            version=latest_version,
            stage=registry_stage,
            archive_existing_versions=(registry_stage == "Production"),
        )
        logger.info(
            "Model registered as %s v%s — stage: %s",
            REGISTRY_NAME, latest_version, registry_stage,
        )

        if promoted:
            logger.info("✅ Thresholds met — promoted to Production.")
        else:
            logger.warning(
                "⚠️  Thresholds NOT met (T2M RMSE %.4f > %.1f°C or RH RMSE %.2f > %.0f%%) "
                "— model logged to Staging only.",
                t2m_rmse, T2M_RMSE_THRESHOLD, rh_rmse, RH_RMSE_THRESHOLD,
            )

        _TRAINING_COMPLETE.set(1 if promoted else 0)

        log_payload = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epochs_trained": trainer.current_epoch,
            "metrics": {
                "t2m_rmse": round(t2m_rmse, 4),
                "rh_rmse": round(rh_rmse, 4),
            },
            "thresholds": {
                "t2m_rmse": T2M_RMSE_THRESHOLD,
                "rh_rmse": RH_RMSE_THRESHOLD,
            },
            "registry": {
                "name": REGISTRY_NAME,
                "version": latest_version,
                "stage": registry_stage,
            },
            "promoted_to_production": promoted,
            **(
                {}
                if promoted
                else {
                    "failure_report": {
                        "t2m_rmse_actual": round(t2m_rmse, 4),
                        "t2m_rmse_target": T2M_RMSE_THRESHOLD,
                        "rh_rmse_actual": round(rh_rmse, 4),
                        "rh_rmse_target": RH_RMSE_THRESHOLD,
                        "t2m_delta": round(t2m_rmse - T2M_RMSE_THRESHOLD, 4),
                        "rh_delta": round(rh_rmse - RH_RMSE_THRESHOLD, 4),
                        "recommended_actions": [
                            "Increase training data window (coordinate with DATA-FLOW)",
                            "Tune hidden_size / attention_head_size hyperparameters",
                            "Review feature quality for ENSO transition months",
                        ],
                    }
                }
            ),
        }
        _write_log(log_payload)
        return log_payload


if __name__ == "__main__":
    result = run_training()
    print(json.dumps(result, indent=2, default=str))

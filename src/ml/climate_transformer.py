"""
ANALYTICA Sprint 3 — Temporal Fusion Transformer for multi-variate climate forecasting
src/ml/climate_transformer.py

Inputs : GPM precip + MODIS LST + SMAP soil moisture + BMKG obs (0.1° grid, 6-hourly)
Targets: 24 h / 48 h / 72 h / 120 h / 168 h precipitation anomaly per grid cell
Arch   : PyTorch Forecasting TFT — encoder 30 days, prediction 7 days, hidden 256, heads 4
MLflow : experiment 'analytica_tft', hyperparams + RMSE/MAE/CRPS
Ckpt   : workspace/models/tft_climate_v1.pt
Registry: 'tropi_climate_tft'
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("analytica.climate_transformer")

MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT   = "analytica_tft"
REGISTRY_NAME       = "tropi_climate_tft"
CHECKPOINT_DIR      = Path(
    os.getenv(
        "TFT_CHECKPOINT_DIR",
        "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/models",
    )
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACT_PATH  = CHECKPOINT_DIR / "tft_climate_v1.pt"
FEATURE_DATA_DIR     = Path(
    "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/output/features"
)

# Input grid: 0.1° resolution, 6-hourly
GRID_RESOLUTION_DEG  = 0.1
TIME_STEP_HOURS      = 6
ENCODER_STEPS        = 30 * (24 // TIME_STEP_HOURS)  # 30 days × 4 steps/day = 120
PREDICTION_STEPS     = 7  * (24 // TIME_STEP_HOURS)  # 7 days  × 4 steps/day = 28
HORIZON_HOURS        = [24, 48, 72, 120, 168]

INPUT_VARS = ["gpm_precip_mm", "modis_lst_k", "smap_sm_m3m3", "bmkg_t2m_c", "bmkg_rh_pct", "bmkg_wind_ms"]
TARGET_VAR  = "precip_anomaly_mm"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFTConfig:
    hidden_size:             int   = 256
    attention_head_size:     int   = 4
    dropout:                 float = 0.1
    hidden_continuous_size:  int   = 32
    lstm_layers:             int   = 2
    learning_rate:           float = 3e-4
    max_epochs:              int   = 60
    batch_size:              int   = 128
    gradient_clip_val:       float = 0.1
    early_stopping_patience: int   = 7
    encoder_length:          int   = ENCODER_STEPS
    prediction_length:       int   = PREDICTION_STEPS
    input_vars:              List[str] = field(default_factory=lambda: INPUT_VARS)
    target_var:              str   = TARGET_VAR
    horizon_hours:           List[int] = field(default_factory=lambda: HORIZON_HOURS)
    mlflow_experiment:       str   = MLFLOW_EXPERIMENT
    registry_name:           str   = REGISTRY_NAME


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel

class TFTForecast(BaseModel):
    """Forecast output for a single grid cell and horizon."""
    grid_cell_id:    str
    variable:        str
    horizon_hours:   int
    forecast_value:  float
    quantile_10:     float
    quantile_90:     float
    issued_at:       datetime


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_timeseries_dataset(df: pd.DataFrame, config: TFTConfig, predict: bool = False):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    return TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target=config.target_var,
        group_ids=["grid_cell_id"],
        max_encoder_length=config.encoder_length,
        max_prediction_length=config.prediction_length,
        static_categoricals=["grid_cell_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[v for v in config.input_vars if v != config.target_var],
        target_normalizer=GroupNormalizer(groups=["grid_cell_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        predict_mode=predict,
    )


def build_tft_model(config: TFTConfig, training_ds):
    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    return TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=7,     # 7-quantile output
        loss=QuantileLoss(),
        reduce_on_plateau_patience=5,
        lstm_layers=config.lstm_layers,
        log_interval=20,
        log_val_interval=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics: RMSE, MAE, CRPS
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _crps(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> float:
    """Simplified CRPS proxy: mean interval score for PI coverage."""
    width    = q90 - q10
    below    = np.maximum(q10 - y_true, 0)
    above    = np.maximum(y_true - q90, 0)
    alpha    = 0.1
    return float(np.mean(width + (2 / alpha) * below + (2 / alpha) * above))


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, q10: np.ndarray, q90: np.ndarray
) -> Dict[str, float]:
    return {
        "rmse": round(_rmse(y_true, y_pred), 4),
        "mae":  round(_mae(y_true,  y_pred), 4),
        "crps": round(_crps(y_true, q10, q90), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(config: Optional[TFTConfig] = None, df: Optional[pd.DataFrame] = None) -> str:
    """
    Train TFT → save workspace/models/tft_climate_v1.pt
               → register as 'tropi_climate_tft' in MLflow registry.
    Returns artifact path.
    """
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    config = config or TFTConfig()
    _init_mlflow(config)

    if df is None:
        df = _load_feature_data()
    train_df, val_df = _train_val_split(df)

    training_ds  = build_timeseries_dataset(train_df, config)
    val_ds       = build_timeseries_dataset(val_df,   config)
    train_loader = DataLoader(training_ds, batch_size=config.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,      batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = build_tft_model(config, training_ds)

    ckpt_cb  = pl.callbacks.ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="tft_climate-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=3,
    )
    early_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.early_stopping_patience, mode="min"
    )
    lr_cb    = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[ckpt_cb, early_cb, lr_cb],
        logger=pl.loggers.MLFlowLogger(
            experiment_name=config.mlflow_experiment,
            tracking_uri=MLFLOW_TRACKING_URI,
        ),
        enable_progress_bar=True,
    )

    log.info(
        "Training TFT: encoder=%d pred=%d hidden=%d heads=%d epochs=%d",
        config.encoder_length, config.prediction_length,
        config.hidden_size, config.attention_head_size, config.max_epochs,
    )
    trainer.fit(model, train_loader, val_loader)

    # Save to canonical .pt artifact path
    import shutil
    best_path = ckpt_cb.best_model_path
    if best_path and Path(best_path).exists():
        shutil.copy2(best_path, MODEL_ARTIFACT_PATH)
        log.info("Saved → %s", MODEL_ARTIFACT_PATH)

    # Evaluate on validation set
    metrics = _evaluate(model, val_loader)

    # Log hyperparams + metrics to MLflow and register model
    _register(model, config, metrics)

    return str(MODEL_ARTIFACT_PATH)


def _evaluate(model, val_loader) -> Dict[str, float]:
    import torch
    preds, q10s, q90s, actuals = [], [], [], []
    for batch in val_loader:
        x, y = batch
        with torch.no_grad():
            out = model(x)
        preds.append(out["prediction"][:, :, 3].cpu().numpy())   # p50
        q10s.append(out["prediction"][:, :, 1].cpu().numpy())    # p10
        q90s.append(out["prediction"][:, :, 5].cpu().numpy())    # p90
        actuals.append(y[0].cpu().numpy())
    y_true = np.concatenate(actuals)
    y_pred = np.concatenate(preds)
    q10    = np.concatenate(q10s)
    q90    = np.concatenate(q90s)
    m = compute_metrics(y_true[:, 0], y_pred[:, 0], q10[:, 0], q90[:, 0])
    log.info("Validation — RMSE=%.4f MAE=%.4f CRPS=%.4f", m["rmse"], m["mae"], m["crps"])
    return m


def _register(model, config: TFTConfig, metrics: Dict[str, float]) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name="tft_registration") as run:
            mlflow.log_params({
                "model_type":       "TemporalFusionTransformer",
                "encoder_length":   config.encoder_length,
                "prediction_length": config.prediction_length,
                "hidden_size":      config.hidden_size,
                "attention_heads":  config.attention_head_size,
                "dropout":          config.dropout,
                "input_vars":       ",".join(config.input_vars),
                "target_var":       config.target_var,
                "time_step_hours":  TIME_STEP_HOURS,
                "grid_resolution":  GRID_RESOLUTION_DEG,
            })
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(MODEL_ARTIFACT_PATH), artifact_path="model")
            mlflow.pytorch.log_model(
                model,
                artifact_path="tft_pytorch",
                registered_model_name=config.registry_name,
            )
        log.info("Registered '%s' to MLflow registry", config.registry_name)
    except Exception as exc:
        log.warning("MLflow registration failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, config: Optional[TFTConfig] = None) -> List[TFTForecast]:
    """Run inference, return TFTForecast list for each grid cell × horizon."""
    import torch
    from pytorch_forecasting.models import TemporalFusionTransformer
    from torch.utils.data import DataLoader

    config = config or TFTConfig()
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {MODEL_ARTIFACT_PATH}")
    model = TemporalFusionTransformer.load_from_checkpoint(str(MODEL_ARTIFACT_PATH))
    model.eval()

    pred_ds     = build_timeseries_dataset(df, config, predict=True)
    pred_loader = DataLoader(pred_ds, batch_size=256, shuffle=False)
    results: List[TFTForecast] = []
    issued_at = datetime.now(timezone.utc)

    with torch.no_grad():
        for i, (batch, _) in enumerate(pred_loader):
            out = model(batch)
            for j, (p50, p10, p90) in enumerate(zip(
                out["prediction"][:, :, 3].cpu().numpy(),
                out["prediction"][:, :, 1].cpu().numpy(),
                out["prediction"][:, :, 5].cpu().numpy(),
            )):
                grid_cell_id = str(i * 256 + j)
                # Emit one TFTForecast per configured horizon
                for step_idx, horizon_h in enumerate(config.horizon_hours):
                    arr_idx = min(step_idx * (TIME_STEP_HOURS), len(p50) - 1)
                    results.append(TFTForecast(
                        grid_cell_id=grid_cell_id,
                        variable=config.target_var,
                        horizon_hours=horizon_h,
                        forecast_value=round(float(p50[arr_idx]), 4),
                        quantile_10=round(float(p10[arr_idx]), 4),
                        quantile_90=round(float(p90[arr_idx]), 4),
                        issued_at=issued_at,
                    ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_mlflow(config: TFTConfig) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
    except Exception as exc:
        log.warning("MLflow init failed: %s", exc)


def _load_feature_data() -> pd.DataFrame:
    files = sorted(FEATURE_DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {FEATURE_DATA_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "time_idx" not in df.columns:
        df = df.sort_values(["grid_cell_id", "timestamp"])
        df["time_idx"] = df.groupby("grid_cell_id").cumcount()
    return df


def _train_val_split(df: pd.DataFrame, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trains, vals = [], []
    for _, g in df.groupby("grid_cell_id"):
        cut = int(len(g) * (1 - val_frac))
        trains.append(g.iloc[:cut])
        vals.append(g.iloc[cut:])
    return pd.concat(trains), pd.concat(vals)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        print("Checkpoint:", train())
    else:
        print("Usage: train")

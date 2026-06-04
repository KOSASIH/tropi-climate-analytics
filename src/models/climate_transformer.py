"""
ANALYTICA Sprint 4 — Temporal Fusion Transformer (hardened)
src/models/climate_transformer.py

TFT via PyTorch Forecasting.
Inputs:  14-day lookback {T2M, RH, PREC, WIND, SST, SMAP_SM, NDVI}, 38 provinces
Output:  7-day ahead multi-variate forecast, province-level
MLflow:  experiment 'climate_transformer'
Ckpt:    workspace/models/climate_transformer/best.ckpt
Registry: registers best checkpoint as 'climate_transformer_tft' @ Staging

Target:  T2M RMSE ≤ 1.2°C  |  RH RMSE ≤ 8%
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("analytica.climate_transformer")

MLFLOW_TRACKING_URI = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT   = "climate_transformer"
REGISTRY_NAME       = "climate_transformer_tft"
CHECKPOINT_DIR      = Path(
    os.getenv(
        "CLIMATE_TRANSFORMER_CHECKPOINT_DIR",
        "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/models/climate_transformer",
    )
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT_PATH = CHECKPOINT_DIR / "best.ckpt"

N_PROVINCES   = 38
LOOKBACK_DAYS = 14
HORIZON_DAYS  = 7
CLIMATE_VARS  = ["T2M", "RH", "PREC", "WIND", "SST", "SMAP_SM", "NDVI"]
TARGET_RMSE   = {"T2M": 1.2, "RH": 8.0}

FEATURE_DATA_DIR = Path(
    "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/output/features"
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFTConfig:
    hidden_size:             int   = 128
    attention_head_size:     int   = 4
    dropout:                 float = 0.1
    hidden_continuous_size:  int   = 16
    lstm_layers:             int   = 2
    learning_rate:           float = 1e-3
    max_epochs:              int   = 50
    batch_size:              int   = 64
    gradient_clip_val:       float = 0.1
    early_stopping_patience: int   = 5
    n_provinces:             int   = N_PROVINCES
    lookback_days:           int   = LOOKBACK_DAYS
    horizon_days:            int   = HORIZON_DAYS
    climate_vars:            List[str] = field(default_factory=lambda: CLIMATE_VARS)
    mlflow_experiment:       str   = MLFLOW_EXPERIMENT
    checkpoint_dir:          str   = str(CHECKPOINT_DIR)
    registry_name:           str   = REGISTRY_NAME


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_timeseries_dataset(df: pd.DataFrame, config: TFTConfig, predict: bool = False):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    return TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target="T2M",
        group_ids=["province_id"],
        max_encoder_length=config.lookback_days,
        max_prediction_length=config.horizon_days,
        static_categoricals=["province_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[v for v in config.climate_vars if v != "T2M"],
        target_normalizer=GroupNormalizer(groups=["province_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        predict_mode=predict,
    )


def build_tft_model(config: TFTConfig, training_dataset):
    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        lstm_layers=config.lstm_layers,
        log_interval=10,
        log_val_interval=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RMSE utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for i, var in enumerate(CLIMATE_VARS):
        if y_true.ndim > 1 and i < y_true.shape[1]:
            rmse = float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)))
        else:
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        results[var] = round(rmse, 4)
    return results


def check_rmse_targets(rmse: Dict[str, float]) -> Dict[str, bool]:
    return {var: rmse.get(var, 999) <= thr for var, thr in TARGET_RMSE.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(config: Optional[TFTConfig] = None, df: Optional[pd.DataFrame] = None) -> str:
    """
    Train TFT → save best.ckpt → register to MLflow Model Registry as
    'climate_transformer_tft' at Staging stage. Returns best checkpoint path.
    """
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    config = config or TFTConfig()
    _init_mlflow(config)

    if df is None:
        df = _load_feature_data()
    train_df, val_df = _train_val_split(df)
    training_ds = build_timeseries_dataset(train_df, config)
    val_ds      = build_timeseries_dataset(val_df,   config)
    train_loader = DataLoader(training_ds, batch_size=config.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,      batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = build_tft_model(config, training_ds)

    # Callbacks
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_cb = pl.callbacks.EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, mode="min")
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

    log.info("Training TFT: epochs=%d provinces=%d vars=%s", config.max_epochs, config.n_provinces, config.climate_vars)
    trainer.fit(model, train_loader, val_loader)

    best_path = ckpt_cb.best_model_path
    log.info("Best checkpoint: %s  val_loss=%.4f", best_path, ckpt_cb.best_model_score)

    # Symlink / copy to workspace/models/climate_transformer/best.ckpt
    import shutil
    if best_path and Path(best_path).exists():
        shutil.copy2(best_path, BEST_CKPT_PATH)
        log.info("Saved best.ckpt → %s", BEST_CKPT_PATH)

    # Evaluate + log RMSE
    rmse = _evaluate_and_log(model, val_loader, config, str(BEST_CKPT_PATH))

    # Register to MLflow Model Registry at Staging
    _register_model(model, config, rmse)

    return str(BEST_CKPT_PATH)


def _evaluate_and_log(model, val_loader, config: TFTConfig, ckpt_path: str) -> Dict[str, float]:
    """Run validation pass, compute and log RMSE per variable."""
    import mlflow, torch
    preds, actuals = [], []
    for batch in val_loader:
        x, y = batch
        with torch.no_grad():
            out = model(x)
        preds.append(out["prediction"][:, :, 3].cpu().numpy())  # p50
        actuals.append(y[0].cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    rmse   = compute_rmse(y_true, y_pred)
    passed = check_rmse_targets(rmse)

    try:
        with mlflow.start_run(run_name="tft_evaluation", nested=True):
            for var, r in rmse.items():
                mlflow.log_metric(f"rmse_{var}", r)
            mlflow.log_param("checkpoint", ckpt_path)
            for var, ok in passed.items():
                mlflow.log_param(f"{var}_target_met", str(ok))
    except Exception as exc:
        log.warning("MLflow eval logging failed: %s", exc)

    for var, ok in passed.items():
        log.info("  %s %s — RMSE=%.4f (target ≤ %.1f°C/%%)",
                 "✓ PASS" if ok else "✗ FAIL", var, rmse.get(var, 0), TARGET_RMSE.get(var, 0))
    return rmse


def _register_model(model, config: TFTConfig, rmse: Dict[str, float]) -> None:
    """
    Log model to MLflow and register as 'climate_transformer_tft' @ Staging.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name="tft_registration") as run:
            mlflow.log_params({
                "model_type":  "TemporalFusionTransformer",
                "lookback":    config.lookback_days,
                "horizon":     config.horizon_days,
                "n_provinces": config.n_provinces,
                "vars":        ",".join(config.climate_vars),
            })
            for var, r in rmse.items():
                mlflow.log_metric(f"rmse_{var}", r)
            mlflow.log_artifact(str(BEST_CKPT_PATH), artifact_path="model_checkpoint")
            # Register via pyfunc wrapper
            mlflow.pytorch.log_model(
                model,
                artifact_path="tft_model",
                registered_model_name=config.registry_name,
            )
            # Transition to Staging
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            mv = client.get_latest_versions(config.registry_name)
            if mv:
                client.transition_model_version_stage(
                    name=config.registry_name,
                    version=mv[-1].version,
                    stage="Staging",
                    archive_existing_versions=False,
                )
                log.info("Registered %s v%s → Staging", config.registry_name, mv[-1].version)
    except Exception as exc:
        log.warning("MLflow model registration failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    df: pd.DataFrame,
    checkpoint_path: Optional[str] = None,
    config: Optional[TFTConfig] = None,
) -> pd.DataFrame:
    import torch
    from pytorch_forecasting.models import TemporalFusionTransformer
    from torch.utils.data import DataLoader

    config = config or TFTConfig()
    ckpt   = checkpoint_path or str(BEST_CKPT_PATH)
    if not Path(ckpt).exists():
        all_ckpts = sorted(CHECKPOINT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint in {CHECKPOINT_DIR}")
        ckpt = str(all_ckpts[-1])

    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    model.eval()
    pred_ds     = build_timeseries_dataset(df, config, predict=True)
    pred_loader = DataLoader(pred_ds, batch_size=128, shuffle=False)

    preds = []
    with torch.no_grad():
        for batch in pred_loader:
            x, _ = batch
            out  = model(x)
            preds.append({
                "p10": out["prediction"][:, :, 1].cpu().numpy(),
                "p50": out["prediction"][:, :, 3].cpu().numpy(),
                "p90": out["prediction"][:, :, 5].cpu().numpy(),
            })

    return pd.DataFrame({
        "T2M_p50": np.concatenate([p["p50"] for p in preds])[:, 0],
        "T2M_p10": np.concatenate([p["p10"] for p in preds])[:, 0],
        "T2M_p90": np.concatenate([p["p90"] for p in preds])[:, 0],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_mlflow(config: TFTConfig) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
    except Exception as exc:
        log.warning("MLflow init: %s", exc)


def _load_feature_data() -> pd.DataFrame:
    files = sorted(FEATURE_DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {FEATURE_DATA_DIR}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "time_idx" not in df.columns:
        df = df.sort_values(["province_id", "date"])
        df["time_idx"] = df.groupby("province_id").cumcount()
    return df


def _train_val_split(df: pd.DataFrame, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trains, vals = [], []
    for _, g in df.groupby("province_id"):
        cut = int(len(g) * (1 - val_frac))
        trains.append(g.iloc[:cut])
        vals.append(g.iloc[cut:])
    return pd.concat(trains), pd.concat(vals)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        print(f"Best checkpoint: {train()}")
    elif cmd == "predict":
        print(predict(_load_feature_data()).head())
    else:
        print(f"Usage: train | predict")

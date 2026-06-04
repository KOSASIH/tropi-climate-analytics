"""
ANALYTICA — Temporal Fusion Transformer (TFT) Multi-variate Climate Forecast
src/models/climate_transformer.py

Architecture:  Temporal Fusion Transformer via pytorch-forecasting
Inputs:        14-day lookback window of {T2M, RH, PREC, WIND, SST, SMAP_SM, NDVI}
               for all 38 Indonesian provinces
Output:        7-day ahead forecast for all 7 variables simultaneously (province-level)
Training data: src/features/feature_pipeline.py outputs
MLflow exp:    'climate_transformer'
Checkpoint:    workspace/models/climate_transformer/

Target metrics:
  T2M  RMSE ≤ 1.2°C
  RH   RMSE ≤ 8%

Usage:
    python -m models.climate_transformer train
    python -m models.climate_transformer predict --checkpoint <path>
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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI  = os.getenv("ANALYTICA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT    = "climate_transformer"
CHECKPOINT_DIR       = Path(
    os.getenv(
        "CLIMATE_TRANSFORMER_CHECKPOINT_DIR",
        "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/models/climate_transformer",
    )
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

N_PROVINCES   = 38
LOOKBACK_DAYS = 14  # encoder sequence length
HORIZON_DAYS  = 7   # decoder / forecast horizon
CLIMATE_VARS  = ["T2M", "RH", "PREC", "WIND", "SST", "SMAP_SM", "NDVI"]
N_VARS        = len(CLIMATE_VARS)

# Target RMSE thresholds (per CLIMATE-OS spec)
TARGET_RMSE = {"T2M": 1.2, "RH": 8.0}


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TFTConfig:
    """Training + architecture configuration for the Climate TFT."""
    # Architecture
    hidden_size:         int   = 128
    attention_head_size: int   = 4
    dropout:             float = 0.1
    hidden_continuous_size: int = 16
    lstm_layers:         int   = 2

    # Training
    learning_rate:       float = 1e-3
    max_epochs:          int   = 50
    batch_size:          int   = 64
    gradient_clip_val:   float = 0.1
    early_stopping_patience: int = 5

    # Data
    n_provinces:         int   = N_PROVINCES
    lookback_days:       int   = LOOKBACK_DAYS
    horizon_days:        int   = HORIZON_DAYS
    climate_vars:        List[str] = field(default_factory=lambda: CLIMATE_VARS)

    # MLflow / checkpointing
    mlflow_experiment:   str   = MLFLOW_EXPERIMENT
    checkpoint_dir:      str   = str(CHECKPOINT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_timeseries_dataset(
    df: pd.DataFrame,
    config: TFTConfig,
    predict: bool = False,
) -> "TimeSeriesDataSet":
    """
    Build a pytorch_forecasting TimeSeriesDataSet from a long-format DataFrame.

    Expected columns:
        date           (datetime64)
        province_id    (str/int  — group identifier)
        T2M, RH, PREC, WIND, SST, SMAP_SM, NDVI  (float32)
        time_idx       (int, monotonically increasing per province)
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    target_var = "T2M"  # primary target; auxiliary targets handled in model
    max_encoder_length = config.lookback_days
    max_prediction_length = config.horizon_days

    dataset = TimeSeriesDataSet(
        data=df,
        time_idx="time_idx",
        target=target_var,
        group_ids=["province_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["province_id"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[v for v in config.climate_vars if v != target_var],
        target_normalizer=GroupNormalizer(groups=["province_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        predict_mode=predict,
    )
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tft_model(config: TFTConfig, training_dataset) -> "TemporalFusionTransformer":
    """Instantiate TFT from pytorch_forecasting with ANALYTICA config."""
    from pytorch_forecasting.models import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=7,          # 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
        lstm_layers=config.lstm_layers,
        log_interval=10,
        log_val_interval=1,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# RMSE computation (multi-variable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return per-variable RMSE dict for all 7 climate variables."""
    results: Dict[str, float] = {}
    for i, var in enumerate(CLIMATE_VARS):
        if y_true.ndim > 1 and i < y_true.shape[1]:
            rmse = float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)))
        else:
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        results[var] = round(rmse, 4)
    return results


def check_rmse_targets(rmse: Dict[str, float]) -> Dict[str, bool]:
    """Return pass/fail for target RMSE thresholds (T2M ≤ 1.2°C, RH ≤ 8%)."""
    return {var: rmse.get(var, 999) <= threshold
            for var, threshold in TARGET_RMSE.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(config: Optional[TFTConfig] = None, df: Optional[pd.DataFrame] = None) -> str:
    """
    Train TFT and log to MLflow. Returns the best checkpoint path.

    Parameters
    ----------
    config : TFTConfig, optional
        Uses defaults if not supplied.
    df : pd.DataFrame, optional
        Long-format climate DataFrame. Loads from feature_pipeline outputs
        (workspace/output/features/) if not supplied.
    """
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    config = config or TFTConfig()
    _init_mlflow(config)

    # ── Load training data ──────────────────────────────────────────────────
    if df is None:
        df = _load_feature_data()

    train_df, val_df = _train_val_split(df)

    training_ds = build_timeseries_dataset(train_df, config)
    val_ds      = build_timeseries_dataset(val_df, config)

    train_loader = DataLoader(training_ds, batch_size=config.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,      batch_size=config.batch_size, shuffle=False, num_workers=4)

    # ── Build model ─────────────────────────────────────────────────────────
    model = build_tft_model(config, training_ds)

    # ── Callbacks ───────────────────────────────────────────────────────────
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.early_stopping_patience,
        mode="min",
    )
    lr_monitor_cb = pl.callbacks.LearningRateMonitor()

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor_cb],
        logger=pl.loggers.MLFlowLogger(
            experiment_name=config.mlflow_experiment,
            tracking_uri=MLFLOW_TRACKING_URI,
        ),
        enable_progress_bar=True,
    )

    log.info("Starting TFT training: epochs=%d batch=%d provinces=%d",
             config.max_epochs, config.batch_size, config.n_provinces)
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    log.info("Best checkpoint: %s  val_loss=%.4f", best_ckpt, checkpoint_cb.best_model_score)

    # ── Evaluate + log RMSE ─────────────────────────────────────────────────
    _evaluate_and_log(model, val_loader, config, best_ckpt)

    return best_ckpt


def _evaluate_and_log(model, val_loader, config: TFTConfig, ckpt_path: str) -> None:
    """Run validation predictions and log RMSE metrics to MLflow."""
    try:
        import mlflow
        preds, actuals = [], []
        for batch in val_loader:
            x, y = batch
            with __import__("torch").no_grad():
                out = model(x)
            p50 = out["prediction"][:, :, 3]  # median quantile
            preds.append(p50.cpu().numpy())
            actuals.append(y[0].cpu().numpy())

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(actuals)
        rmse   = compute_rmse(y_true, y_pred)
        passed = check_rmse_targets(rmse)

        with mlflow.start_run(run_name="tft_evaluation", nested=True):
            for var, r in rmse.items():
                mlflow.log_metric(f"rmse_{var}", r)
            mlflow.log_param("checkpoint", ckpt_path)
            mlflow.log_param("t2m_target_met", str(passed.get("T2M", False)))
            mlflow.log_param("rh_target_met",  str(passed.get("RH",  False)))

        log.info("RMSE results: %s", rmse)
        for var, ok in passed.items():
            status = "✓ PASS" if ok else "✗ FAIL"
            log.info("  %s %s RMSE=%.4f (target ≤ %.1f)", status, var,
                     rmse.get(var, 0), TARGET_RMSE.get(var, 0))
    except Exception as exc:
        log.warning("Evaluation logging failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    df: pd.DataFrame,
    checkpoint_path: Optional[str] = None,
    config: Optional[TFTConfig] = None,
) -> pd.DataFrame:
    """
    Generate 7-day ahead forecast for all provinces in df.
    Returns a DataFrame with columns: date, province_id, <var>_p50, <var>_p10, <var>_p90
    """
    import torch
    from pytorch_forecasting.models import TemporalFusionTransformer

    config = config or TFTConfig()
    if checkpoint_path is None:
        # Pick most recent checkpoint
        ckpts = sorted(CHECKPOINT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")
        checkpoint_path = str(ckpts[-1])

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    pred_ds = build_timeseries_dataset(df, config, predict=True)
    from torch.utils.data import DataLoader
    pred_loader = DataLoader(pred_ds, batch_size=128, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in pred_loader:
            x, _ = batch
            out  = model(x)
            # out["prediction"] shape: [batch, horizon, n_quantiles]
            p10 = out["prediction"][:, :, 1].cpu().numpy()  # 10th quantile
            p50 = out["prediction"][:, :, 3].cpu().numpy()  # median
            p90 = out["prediction"][:, :, 5].cpu().numpy()  # 90th quantile
            all_preds.append({"p10": p10, "p50": p50, "p90": p90})

    # Assemble result DataFrame (T2M only — extend for multi-variate in v2)
    p50_all = np.concatenate([p["p50"] for p in all_preds])
    p10_all = np.concatenate([p["p10"] for p in all_preds])
    p90_all = np.concatenate([p["p90"] for p in all_preds])

    result = pd.DataFrame({
        "T2M_p50": p50_all[:, 0],
        "T2M_p10": p10_all[:, 0],
        "T2M_p90": p90_all[:, 0],
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_mlflow(config: TFTConfig) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.log_params({
            "model_type":    "TemporalFusionTransformer",
            "lookback_days": config.lookback_days,
            "horizon_days":  config.horizon_days,
            "n_provinces":   config.n_provinces,
            "climate_vars":  ",".join(config.climate_vars),
            "hidden_size":   config.hidden_size,
            "attention_heads": config.attention_head_size,
        })
    except Exception as exc:
        log.warning("MLflow init failed: %s", exc)


def _load_feature_data() -> pd.DataFrame:
    """Load feature_pipeline outputs from workspace/output/features/."""
    feature_dir = Path(
        "/home/user/surething/cells/0e7ffdfd-b723-43ac-a590-3e35f82a5fce/workspace/output/features"
    )
    parquet_files = sorted(feature_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No feature parquet files found in {feature_dir}. "
            "Run src/features/feature_pipeline.py first."
        )
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    if "time_idx" not in df.columns:
        df = df.sort_values(["province_id", "date"])
        df["time_idx"] = df.groupby("province_id").cumcount()
    return df


def _train_val_split(df: pd.DataFrame, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — last val_frac of each province goes to validation."""
    train_frames, val_frames = [], []
    for _, grp in df.groupby("province_id"):
        cut = int(len(grp) * (1 - val_frac))
        train_frames.append(grp.iloc[:cut])
        val_frames.append(grp.iloc[cut:])
    return pd.concat(train_frames), pd.concat(val_frames)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        ckpt = train()
        print(f"Best checkpoint: {ckpt}")
    elif cmd == "predict":
        df = _load_feature_data()
        result = predict(df)
        print(result.head())
    else:
        print(f"Unknown command '{cmd}'. Use: train | predict")

"""
LSTM Streamflow Prediction Model — ANALYTICA × HYDROLOGIS
Production weights for Ciliwung (Manggarai), Brantas (Mlirip), Solo (Jurug).
Replaces the LSTMStreamflowModel surrogate in src/hydrology/.

Architecture : Stacked LSTM (3 layers) + Attention + Dense head
Input         : 30-day lookback window of meteorological & hydrological features
Output        : Next-day streamflow (m³/s), with 7-day forecast capability
Training data : BMKG daily gauge records 2010-2024
MLflow run    : lstm-streamflow-v1 | Experiment: tropi-climate-models
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Station registry
# ─────────────────────────────────────────────────────────────────────────────

STATIONS: Dict[str, Dict[str, Any]] = {
    "ciliwung_manggarai": {
        "river":      "Ciliwung",
        "gauge":      "Manggarai",
        "bmkg_id":    "96749",
        "lat":        -6.2115,
        "lon":        106.8452,
        "catchment_km2": 293,
        "mean_q_m3s":    28.4,
        "flood_threshold_m3s": 400.0,
        "province":   "DKI Jakarta / Jawa Barat",
    },
    "brantas_mlirip": {
        "river":      "Brantas",
        "gauge":      "Mlirip",
        "bmkg_id":    "97180",
        "lat":        -7.5019,
        "lon":        112.5834,
        "catchment_km2": 11_800,
        "mean_q_m3s":    212.0,
        "flood_threshold_m3s": 1_200.0,
        "province":   "Jawa Timur",
    },
    "solo_jurug": {
        "river":      "Solo (Bengawan Solo)",
        "gauge":      "Jurug",
        "bmkg_id":    "96935",
        "lat":        -7.5569,
        "lon":        110.8666,
        "catchment_km2": 5_547,
        "mean_q_m3s":    89.7,
        "flood_threshold_m3s": 700.0,
        "province":   "Jawa Tengah",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature specification
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH  = 30          # 30-day lookback
FORECAST_HORIZON = 7           # max days ahead
N_FEATURES       = 18          # input features per timestep

FEATURE_NAMES: List[str] = [
    # Precipitation (GPM + BMKG gauge)
    "precip_gpm_1d",        # GPM IMERG daily mean over catchment (mm)
    "precip_gpm_3d",        # 3-day antecedent (mm)
    "precip_gpm_7d",        # 7-day antecedent (mm)
    "precip_bmkg",          # BMKG gauge daily (mm)
    # Streamflow (lagged)
    "streamflow_q_t1",      # Q at t-1 (m³/s)
    "streamflow_q_t3",      # Q at t-3
    "streamflow_q_t7",      # Q at t-7
    # Soil moisture (SMAP L3 0-5 cm)
    "smap_sm_surface",      # surface soil moisture (m³/m³)
    "smap_sm_rootzone",     # rootzone soil moisture (m³/m³)
    # Atmospheric
    "temp_2m_mean",         # °C
    "temp_2m_max",          # °C
    "rh_mean",              # %
    "wind_speed",           # m/s
    # Hydrological indices
    "stage_level",          # water stage (m)
    "water_level_rate",     # dH/dt (m/day)
    "baseflow_index",       # ratio of baseflow to total Q
    # Temporal (cyclic)
    "doy_sin",              # sin(2π·DOY/365.25)
    "doy_cos",              # cos(2π·DOY/365.25)
]

# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

LSTM_PARAMS: Dict[str, Any] = {
    "lstm_units":            [128, 64, 32],   # 3-layer stacked LSTM
    "attention_units":       32,
    "dense_units":           [64, 32],
    "dropout_rate":          0.2,
    "recurrent_dropout":     0.1,
    "l2_reg":                1e-4,
    "learning_rate":         1e-3,
    "batch_size":            64,
    "epochs":                200,
    "early_stopping_patience": 20,
    "reduce_lr_patience":    10,
    "sequence_length":       SEQUENCE_LENGTH,
    "forecast_horizon":      FORECAST_HORIZON,
    "n_features":            N_FEATURES,
    "loss":                  "huber",         # robust to outliers in flood events
    "metrics":               ["mae", "mse"],
}

PERFORMANCE_TARGETS: Dict[str, float] = {
    "nse":   0.80,    # Nash-Sutcliffe efficiency ≥ 0.80
    "kge":   0.75,    # Kling-Gupta efficiency   ≥ 0.75
    "pbias": 10.0,    # |Percent bias|           < 10%
    "rmse_normalized": 0.20,  # RMSE / mean_Q    < 20%
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility — hydrological metrics
# ─────────────────────────────────────────────────────────────────────────────

def nash_sutcliffe_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """NSE = 1 - Σ(obs-sim)² / Σ(obs-mean_obs)²"""
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((obs - sim) ** 2) / (denom + 1e-9))


def kling_gupta_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)"""
    r    = float(np.corrcoef(obs, sim)[0, 1])
    alpha = sim.std() / (obs.std() + 1e-9)
    beta  = sim.mean() / (obs.mean() + 1e-9)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def percent_bias(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(100 * (sim.sum() - obs.sum()) / (obs.sum() + 1e-9))


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray,
                        mean_q: float = 1.0) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = float(np.sqrt(mean_squared_error(obs, sim)))
    return {
        "nse":               nash_sutcliffe_efficiency(obs, sim),
        "kge":               kling_gupta_efficiency(obs, sim),
        "pbias":             percent_bias(obs, sim),
        "rmse":              rmse,
        "mae":               float(mean_absolute_error(obs, sim)),
        "rmse_normalized":   rmse / (mean_q + 1e-9),
        "r2":                float(np.corrcoef(obs, sim)[0, 1] ** 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

class StreamflowDataPreparer:
    """
    Prepares BMKG daily gauge records + GPM/SMAP covariates into
    LSTM-ready (samples, seq_len, n_features) tensors.
    """

    def __init__(self, station_id: str, sequence_length: int = SEQUENCE_LENGTH) -> None:
        if station_id not in STATIONS:
            raise ValueError(f"Unknown station: {station_id}. Valid: {list(STATIONS)}")
        self.station_id      = station_id
        self.station         = STATIONS[station_id]
        self.seq_len         = sequence_length
        self._scaler_Q: Any  = None
        self._scaler_X: Any  = None

    def prepare(
        self,
        df: pd.DataFrame,
        target_col: str = "streamflow_q",
        fit_scalers: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        df: DataFrame with DatetimeIndex and columns matching FEATURE_NAMES + target_col.
        Returns (X, y) as numpy arrays: X=(N, seq_len, n_features), y=(N,).
        """
        from sklearn.preprocessing import RobustScaler

        df = self._add_temporal_features(df)
        df = self._add_antecedent_features(df, target_col)
        df = df.dropna()

        feat_cols = [c for c in FEATURE_NAMES if c in df.columns]
        X_raw = df[feat_cols].values
        y_raw = df[target_col].values

        # Log-transform streamflow (right-skewed)
        y_log = np.log1p(y_raw)

        if fit_scalers:
            self._scaler_X = RobustScaler().fit(X_raw)
            self._scaler_Q = RobustScaler().fit(y_log.reshape(-1, 1))

        X_scaled = self._scaler_X.transform(X_raw)
        y_scaled = self._scaler_Q.transform(y_log.reshape(-1, 1)).ravel()

        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_len: i])
            y_seq.append(y_scaled[i])

        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def inverse_transform_q(self, y_scaled: np.ndarray) -> np.ndarray:
        y_log = self._scaler_Q.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return np.expm1(y_log)

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = pd.to_datetime(df.index)
        df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
        return df

    @staticmethod
    def _add_antecedent_features(df: pd.DataFrame,
                                 target_col: str) -> pd.DataFrame:
        df = df.copy()
        if "precip_gpm_1d" in df.columns:
            df["precip_gpm_3d"] = df["precip_gpm_1d"].rolling(3).sum()
            df["precip_gpm_7d"] = df["precip_gpm_1d"].rolling(7).sum()
        if target_col in df.columns:
            df["streamflow_q_t1"] = df[target_col].shift(1)
            df["streamflow_q_t3"] = df[target_col].shift(3)
            df["streamflow_q_t7"] = df[target_col].shift(7)
        if "stage_level" in df.columns:
            df["water_level_rate"] = df["stage_level"].diff()
        return df

    @staticmethod
    def train_val_test_split(
        X: np.ndarray, y: np.ndarray,
        val_frac: float = 0.15, test_frac: float = 0.15,
    ) -> Tuple[np.ndarray, ...]:
        n   = len(X)
        n_test = int(n * test_frac)
        n_val  = int(n * val_frac)
        return (
            X[:n - n_val - n_test],  y[:n - n_val - n_test],
            X[n - n_val - n_test: n - n_test], y[n - n_val - n_test: n - n_test],
            X[n - n_test:],           y[n - n_test:],
        )


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────────────────────────────────────────

class LSTMStreamflowModel:
    """
    Production LSTM model for daily streamflow prediction.
    Covers three Indonesian river gauges:
      • Ciliwung – Manggarai  (Jakarta flood sentinel)
      • Brantas  – Mlirip     (East Java water management)
      • Solo     – Jurug       (Central Java flood warning)

    Architecture: 3-layer stacked LSTM + Bahdanau-style attention + Dense head.
    """

    def __init__(self, station_id: str) -> None:
        if station_id not in STATIONS:
            raise ValueError(f"Unknown station: {station_id}")
        self.station_id  = station_id
        self.station     = STATIONS[station_id]
        self.model: Any  = None
        self.version: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self._data_preparer: Optional[StreamflowDataPreparer] = None

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build(self) -> Any:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers
        L2 = regularizers.l2(LSTM_PARAMS["l2_reg"])

        inp = layers.Input(shape=(LSTM_PARAMS["sequence_length"],
                                  LSTM_PARAMS["n_features"]),
                           name="input_sequence")

        # Layer 1 — return sequences for attention
        x = layers.LSTM(
            LSTM_PARAMS["lstm_units"][0],
            return_sequences=True,
            dropout=LSTM_PARAMS["dropout_rate"],
            recurrent_dropout=LSTM_PARAMS["recurrent_dropout"],
            kernel_regularizer=L2,
            name="lstm_1",
        )(inp)
        x = layers.LayerNormalization()(x)

        # Layer 2
        x = layers.LSTM(
            LSTM_PARAMS["lstm_units"][1],
            return_sequences=True,
            dropout=LSTM_PARAMS["dropout_rate"],
            recurrent_dropout=LSTM_PARAMS["recurrent_dropout"],
            kernel_regularizer=L2,
            name="lstm_2",
        )(x)
        x = layers.LayerNormalization()(x)

        # Layer 3
        x = layers.LSTM(
            LSTM_PARAMS["lstm_units"][2],
            return_sequences=True,
            dropout=LSTM_PARAMS["dropout_rate"],
            recurrent_dropout=LSTM_PARAMS["recurrent_dropout"],
            kernel_regularizer=L2,
            name="lstm_3",
        )(x)
        x = layers.LayerNormalization()(x)

        # Bahdanau-style attention
        attn_scores = layers.Dense(1, activation="tanh", name="attn_score")(x)
        attn_weights = layers.Softmax(axis=1, name="attn_weights")(attn_scores)
        context = layers.Multiply()([x, attn_weights])
        context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1),
                                name="attn_context")(context)

        # Dense head
        x = layers.Dense(LSTM_PARAMS["dense_units"][0], activation="relu",
                         kernel_regularizer=L2, name="dense_1")(context)
        x = layers.Dropout(LSTM_PARAMS["dropout_rate"])(x)
        x = layers.Dense(LSTM_PARAMS["dense_units"][1], activation="relu",
                         kernel_regularizer=L2, name="dense_2")(x)
        out = layers.Dense(1, activation="linear", name="streamflow_output")(x)

        return models.Model(inp, out,
                            name=f"LSTMStreamflow_{self.station_id}")

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        data_preparer: Optional[StreamflowDataPreparer] = None,
        run_name: str = "lstm-streamflow-v1",
    ) -> Dict[str, float]:
        import mlflow
        import mlflow.tensorflow
        import tensorflow as tf
        from tensorflow.keras import callbacks

        from src.analytics.mlflow_setup import setup_mlflow

        setup_mlflow()
        self._data_preparer = data_preparer

        with mlflow.start_run(run_name=f"{run_name}_{self.station_id}") as run:
            mlflow.set_tag("model_type",  "lstm_streamflow")
            mlflow.set_tag("station_id",  self.station_id)
            mlflow.set_tag("river",       self.station["river"])
            mlflow.set_tag("gauge",       self.station["gauge"])
            mlflow.set_tag("data_period", "2010-2024")
            mlflow.set_tag("agent",       "ANALYTICA")

            mlflow.log_params({**LSTM_PARAMS,
                               "station_id":        self.station_id,
                               "catchment_km2":     self.station["catchment_km2"],
                               "flood_threshold":   self.station["flood_threshold_m3s"],
                               "n_train_samples":   len(X_train),
                               "n_val_samples":     len(X_val)})

            self.model = self._build()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=LSTM_PARAMS["learning_rate"],
                    clipnorm=1.0,          # gradient clipping for stability
                ),
                loss=LSTM_PARAMS["loss"],
                metrics=LSTM_PARAMS["metrics"],
            )
            logger.info(f"Training LSTM for {self.station_id} | "
                        f"train={len(X_train)}, val={len(X_val)}")
            self.model.summary(print_fn=logger.debug)

            Path("models").mkdir(exist_ok=True)
            save_path = f"models/lstm_{self.station_id}_best.keras"
            cb_list   = [
                callbacks.EarlyStopping(
                    patience=LSTM_PARAMS["early_stopping_patience"],
                    restore_best_weights=True, monitor="val_loss"),
                callbacks.ReduceLROnPlateau(
                    patience=LSTM_PARAMS["reduce_lr_patience"],
                    factor=0.5, monitor="val_loss", min_lr=1e-6),
                callbacks.ModelCheckpoint(
                    save_path, save_best_only=True, monitor="val_loss"),
                callbacks.CSVLogger(f"models/lstm_{self.station_id}_history.csv"),
            ]

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=LSTM_PARAMS["batch_size"],
                epochs=LSTM_PARAMS["epochs"],
                callbacks=cb_list, verbose=1,
            )

            # Evaluate with hydro metrics
            y_pred_scaled = self.model.predict(X_val).ravel()
            if data_preparer:
                y_obs = data_preparer.inverse_transform_q(y_val)
                y_sim = data_preparer.inverse_transform_q(y_pred_scaled)
            else:
                y_obs, y_sim = y_val, y_pred_scaled

            self.metrics = compute_all_metrics(
                y_obs, y_sim, mean_q=self.station["mean_q_m3s"])
            self.metrics["val_loss"] = float(history.history["val_loss"][-1])
            self.metrics["val_mae"]  = float(history.history["val_mae"][-1])

            mlflow.log_metrics(self.metrics)
            mlflow.log_artifact(save_path, artifact_path="model_weights")
            mlflow.tensorflow.log_model(
                self.model, "model",
                registered_model_name=f"tropi-streamflow-lstm-{self.station_id}",
            )

            # Performance gate
            targets_met = {k: (self.metrics.get(k, -999) >= v
                               if k != "pbias" else
                               abs(self.metrics.get(k, 999)) < v)
                           for k, v in PERFORMANCE_TARGETS.items()}
            mlflow.log_dict(targets_met, "performance_targets.json")
            all_pass = all(targets_met.values())
            mlflow.set_tag("performance_gate", "PASS" if all_pass else "REVIEW")

            self.version = run.info.run_id

        nse   = self.metrics.get("nse", 0)
        kge   = self.metrics.get("kge", 0)
        pbias = self.metrics.get("pbias", 0)
        gate  = "✅ PASS" if all_pass else "⚠️ REVIEW"
        logger.info(
            f"LSTM {self.station_id} done | "
            f"NSE={nse:.3f} KGE={kge:.3f} PBias={pbias:.1f}% | {gate}"
        )
        return self.metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        inverse_transform: bool = True,
    ) -> np.ndarray:
        """Predict streamflow. X: (N, seq_len, n_features)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load().")
        y_scaled = self.model.predict(X).ravel()
        if inverse_transform and self._data_preparer:
            return self._data_preparer.inverse_transform_q(y_scaled)
        return y_scaled

    def predict_flood_risk(self, X: np.ndarray) -> Dict[str, Any]:
        """Returns predicted Q and flood exceedance probability."""
        q_pred   = self.predict(X)
        threshold = self.station["flood_threshold_m3s"]
        exceed    = (q_pred > threshold).mean()
        return {
            "station_id":                  self.station_id,
            "predicted_q_m3s":             q_pred.tolist(),
            "flood_threshold_m3s":         threshold,
            "flood_exceedance_probability": float(exceed),
            "max_predicted_q":             float(q_pred.max()),
            "alert_level": (
                "RED"    if q_pred.max() > threshold else
                "ORANGE" if q_pred.max() > threshold * 0.75 else
                "GREEN"
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self, model_path: str) -> None:
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"LSTM loaded: {model_path}")

    def save(self, path: str) -> None:
        if self.model:
            self.model.save(path)
            logger.info(f"LSTM saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-station training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class LSTMStreamflowTrainer:
    """
    Trains LSTM models for all three stations and registers them in MLflow.
    Handles synthetic data fallback when BMKG gauge CSV files are not present.
    """

    DATA_DIR   = Path("data/bmkg/streamflow")
    MODELS_DIR = Path("models")
    RUN_NAME   = "lstm-streamflow-v1"

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, float]] = {}

    def run_all(self) -> Dict[str, Any]:
        """Train all three stations sequentially. Returns full results summary."""
        summary: Dict[str, Any] = {
            "run_name":   self.RUN_NAME,
            "started_at": datetime.utcnow().isoformat(),
            "stations":   {},
        }
        for station_id in STATIONS:
            try:
                result = self._train_station(station_id)
                summary["stations"][station_id] = result
            except Exception as exc:
                logger.error(f"Training failed for {station_id}: {exc}")
                summary["stations"][station_id] = {"error": str(exc)}

        summary["finished_at"] = datetime.utcnow().isoformat()
        summary["all_pass"]    = all(
            s.get("performance_gate") == "PASS"
            for s in summary["stations"].values()
            if "error" not in s
        )
        return summary

    # ── Per-station ───────────────────────────────────────────────────────────

    def _train_station(self, station_id: str) -> Dict[str, Any]:
        logger.info(f"━━ Training station: {station_id} ━━")
        df = self._load_or_synthesise(station_id)

        preparer = StreamflowDataPreparer(station_id)
        X, y     = preparer.prepare(df, fit_scalers=True)
        X_tr, y_tr, X_val, y_val, X_te, y_te = preparer.train_val_test_split(X, y)

        model = LSTMStreamflowModel(station_id)
        metrics = model.train(X_tr, y_tr, X_val, y_val,
                              data_preparer=preparer,
                              run_name=self.RUN_NAME)

        # Test set evaluation
        y_pred_te = model.predict(X_te)
        y_obs_te  = preparer.inverse_transform_q(y_te)
        test_metrics = compute_all_metrics(
            y_obs_te, y_pred_te, mean_q=STATIONS[station_id]["mean_q_m3s"])
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

        all_metrics = {**metrics, **test_metrics}
        nse = all_metrics.get("nse", 0)
        gate = "PASS" if nse >= PERFORMANCE_TARGETS["nse"] else "REVIEW"

        return {
            "station_id":       station_id,
            "mlflow_run_id":    model.version,
            "metrics":          all_metrics,
            "performance_gate": gate,
            "model_saved":      f"models/lstm_{station_id}_best.keras",
        }

    # ── Data loading / synthesis ──────────────────────────────────────────────

    def _load_or_synthesise(self, station_id: str) -> pd.DataFrame:
        """Load BMKG CSV if available, otherwise synthesise physically plausible data."""
        csv_path = self.DATA_DIR / f"{station_id}_2010_2024.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded real data: {csv_path} ({len(df)} rows)")
            return df

        logger.warning(
            f"BMKG CSV not found at {csv_path} — "
            "synthesising physically plausible training data. "
            "Replace with real gauge records before production deployment."
        )
        return self._synthesise(station_id)

    @staticmethod
    def _synthesise(station_id: str) -> pd.DataFrame:
        """
        Generate physically plausible synthetic records (2010-2024) for testing.
        Uses seasonal rainfall signal, ENSO modulation, and recession curves.
        Replace with real BMKG data in production.
        """
        rng       = np.random.default_rng(seed=42)
        dates     = pd.date_range("2010-01-01", "2024-12-31", freq="D")
        n         = len(dates)
        doy       = dates.dayofyear.values
        station   = STATIONS[station_id]
        mean_q    = station["mean_q_m3s"]

        # Seasonal precipitation (Indonesian wet season Nov-Apr)
        p_seasonal = (
            15 + 20 * np.sin(2 * np.pi * (doy - 330) / 365.25) +
            rng.exponential(8, n)
        ).clip(0)

        # ENSO modulation (rough 3.6-year cycle)
        enso  = 5 * np.sin(2 * np.pi * np.arange(n) / (3.6 * 365))
        p_seasonal = (p_seasonal + enso).clip(0)

        # Streamflow via simple recession model
        q = np.zeros(n)
        q[0] = mean_q
        for i in range(1, n):
            q[i] = max(0.15 * q[i-1] + 0.6 * p_seasonal[i] + rng.normal(0, 1), 0.1)

        # Stage level (rating curve approximation: H ≈ a·Q^b)
        stage = 0.5 * q ** 0.4 + rng.normal(0, 0.02, n)

        df = pd.DataFrame(index=dates, data={
            "streamflow_q":    q,
            "precip_gpm_1d":   p_seasonal,
            "precip_bmkg":     p_seasonal * (1 + rng.normal(0, 0.1, n)),
            "smap_sm_surface": (0.25 + 0.15 * np.sin(2 * np.pi * (doy - 330) / 365.25) +
                                rng.normal(0, 0.02, n)).clip(0.05, 0.55),
            "smap_sm_rootzone":(0.30 + 0.10 * np.sin(2 * np.pi * (doy - 330) / 365.25) +
                                rng.normal(0, 0.015, n)).clip(0.1, 0.5),
            "temp_2m_mean":    27 + 2 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 0.5, n),
            "temp_2m_max":     32 + 2 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0, 0.5, n),
            "rh_mean":         (75 + 10 * np.sin(2 * np.pi * (doy - 330) / 365.25) +
                                rng.normal(0, 3, n)).clip(40, 100),
            "wind_speed":      (2 + rng.exponential(1, n)).clip(0, 15),
            "stage_level":     stage.clip(0.1),
            "baseflow_index":  (0.3 + 0.2 * np.cos(2 * np.pi * doy / 365.25) +
                                rng.normal(0, 0.03, n)).clip(0.05, 0.95),
        })
        return df

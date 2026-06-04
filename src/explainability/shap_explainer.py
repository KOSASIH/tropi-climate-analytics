"""
ANALYTICA SHAP Explainability — Sprint 6 H2
Class: SHAPExplainer

Methods:
  explain_xgb(model, X: pd.DataFrame)  -> list[dict]  — TreeExplainer (XGBoost/sklearn trees)
  explain_deep(model, X: pd.DataFrame) -> list[dict]  — DeepExplainer (LSTM, TFT, PyTorch/TF)
  explain_cnn(model, X: np.ndarray)    -> list[dict]  — GradientExplainer (CNN)

Output shape per method: [{feature: str, shap_value: float, rank: int}]
  - Sorted by abs(shap_value) DESC
  - Top-10 returned; caller slices to top-5 for API response

Sidecar write: workspace/output/explainability/{model_id}_{grid_cell_id}_{date}.json
Called by inference_api.py for feature_importance_top5 field.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("workspace/output/explainability")


class SHAPExplainer:
    """
    Unified SHAP explainability interface for all ANALYTICA model types.

    Usage:
        explainer = SHAPExplainer()

        # XGBoost precipitation nowcast
        shap_vals = explainer.explain_xgb(xgb_model, features_df)

        # LSTM / TFT deep models
        shap_vals = explainer.explain_deep(lstm_model, features_df)

        # CNN land cover
        shap_vals = explainer.explain_cnn(cnn_model, image_array)

        # Auto-dispatch by model_id prefix
        shap_vals = explainer.explain_for_model(model_id, model, X)
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public explain methods
    # ------------------------------------------------------------------

    def explain_xgb(self, model: Any, X: pd.DataFrame) -> List[dict]:
        """
        TreeExplainer for XGBoost / sklearn gradient-boosted trees.
        Returns top-10 features sorted by mean absolute SHAP value across samples.

        Args:
            model: Fitted XGBoost or sklearn tree model (or mlflow.pyfunc wrapper).
            X:     Feature DataFrame (n_samples x n_features).

        Returns:
            List of {feature, shap_value, rank} dicts, sorted by abs(shap_value) DESC.
        """
        import shap as shap_lib

        raw_model = _unwrap_pyfunc(model)
        explainer  = shap_lib.TreeExplainer(raw_model)
        shap_vals  = explainer.shap_values(X)

        # For multi-class output, take mean over classes
        if isinstance(shap_vals, list):
            arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            arr = np.abs(shap_vals)

        mean_abs = np.mean(arr, axis=0)
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f_{i}" for i in range(len(mean_abs))]
        return self._rank_and_trim(feature_names, mean_abs, self.top_k)

    def explain_deep(self, model: Any, X: pd.DataFrame) -> List[dict]:
        """
        DeepExplainer for LSTM / TFT (PyTorch or TensorFlow).
        Uses a background sample (first min(100, n) rows) as the reference distribution.

        Args:
            model: Fitted deep learning model (or mlflow.pyfunc wrapper).
            X:     Feature DataFrame (n_samples x n_features).

        Returns:
            List of {feature, shap_value, rank} dicts, sorted by abs(shap_value) DESC.
        """
        import shap as shap_lib

        raw_model  = _unwrap_pyfunc(model)
        X_arr      = X.values.astype(np.float32)
        n_bg       = min(100, len(X_arr))
        background = X_arr[:n_bg]

        try:
            explainer  = shap_lib.DeepExplainer(raw_model, background)
            shap_vals  = explainer.shap_values(X_arr)
        except Exception as deep_exc:
            logger.warning("DeepExplainer failed (%s) — falling back to KernelExplainer", deep_exc)
            def _predict_fn(data):
                if hasattr(raw_model, "predict"):
                    out = raw_model.predict(pd.DataFrame(data, columns=X.columns))
                    return np.array(out) if not isinstance(out, np.ndarray) else out
                return np.zeros(len(data))
            explainer  = shap_lib.KernelExplainer(_predict_fn, background)
            shap_vals  = explainer.shap_values(X_arr, nsamples=50)

        if isinstance(shap_vals, list):
            arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            arr = np.abs(shap_vals)

        mean_abs     = np.mean(arr, axis=0).flatten()
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"f_{i}" for i in range(len(mean_abs))]
        return self._rank_and_trim(feature_names, mean_abs, self.top_k)

    def explain_cnn(self, model: Any, X: np.ndarray) -> List[dict]:
        """
        GradientExplainer for CNN land cover model.
        Computes per-channel spatial mean of absolute SHAP values.

        Args:
            model: Fitted CNN (PyTorch or TensorFlow).
            X:     Input array shape (n_samples, H, W, C) or (n_samples, C, H, W).

        Returns:
            List of {feature, shap_value, rank} dicts (one entry per channel), sorted DESC.
        """
        import shap as shap_lib

        raw_model = _unwrap_pyfunc(model)
        n_bg      = min(50, len(X))
        background = X[:n_bg]

        try:
            explainer  = shap_lib.GradientExplainer(raw_model, background)
            shap_vals  = explainer.shap_values(X)
        except Exception as grad_exc:
            logger.warning("GradientExplainer failed (%s) — returning zero-SHAP stub", grad_exc)
            n_channels = X.shape[-1] if X.ndim == 4 else X.shape[1]
            return [
                {"feature": f"channel_{c}", "shap_value": 0.0, "rank": c + 1}
                for c in range(min(n_channels, self.top_k))
            ]

        if isinstance(shap_vals, list):
            sv_arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            sv_arr = np.abs(shap_vals)

        # Per-channel spatial mean: collapse spatial dims, keep channel axis
        # Supports (N, H, W, C) and (N, C, H, W)
        if sv_arr.ndim == 4:
            channel_axis = -1 if sv_arr.shape[-1] <= sv_arr.shape[1] else 1
            if channel_axis == -1:
                mean_abs = sv_arr.mean(axis=(0, 1, 2))  # -> (C,)
            else:
                mean_abs = sv_arr.mean(axis=(0, 2, 3))  # -> (C,)
        else:
            mean_abs = sv_arr.mean(axis=0).flatten()

        feature_names = [f"channel_{i}" for i in range(len(mean_abs))]
        return self._rank_and_trim(feature_names, mean_abs, self.top_k)

    def explain_for_model(
        self, model_id: str, model: Any, X: Any, is_cnn_input: bool = False
    ) -> List[dict]:
        """
        Auto-dispatch to the correct explain method based on model_id prefix.

        model_id prefixes:
            xgb_*       -> explain_xgb
            cnn_*       -> explain_cnn (X must be np.ndarray)
            prophet_*   -> explain_deep (fallback)
            lstm_*      -> explain_deep
            tft_*       -> explain_deep
        """
        model_id_lower = model_id.lower()
        if model_id_lower.startswith("xgb"):
            return self.explain_xgb(model, X)
        elif model_id_lower.startswith("cnn"):
            X_arr = X if isinstance(X, np.ndarray) else X.values
            return self.explain_cnn(model, X_arr)
        else:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            return self.explain_deep(model, X)

    def write_sidecar(
        self,
        shap_entries:  List[dict],
        model_id:      str,
        grid_cell_id:  str,
        issued_date:   date,
    ) -> Path:
        """
        Persist SHAP output to workspace/output/explainability/.
        Returns the path written.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname    = f"{model_id}_{grid_cell_id}_{issued_date.isoformat()}.json"
        out_path = OUTPUT_DIR / fname
        payload  = {
            "model_id":     model_id,
            "grid_cell_id": grid_cell_id,
            "issued_date":  issued_date.isoformat(),
            "shap_values":  shap_entries,
            "written_at":   datetime.now(timezone.utc).isoformat(),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.debug("SHAP sidecar written: %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_and_trim(
        feature_names: List[str], mean_abs: np.ndarray, top_k: int
    ) -> List[dict]:
        """Sort features by mean abs SHAP value, trim to top_k, assign ranks."""
        paired   = sorted(
            zip(feature_names, mean_abs.tolist()),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:top_k]
        return [
            {"feature": name, "shap_value": round(float(val), 6), "rank": rank + 1}
            for rank, (name, val) in enumerate(paired)
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_pyfunc(model: Any) -> Any:
    """
    If model is an mlflow.pyfunc wrapper, attempt to extract the underlying
    Python model object for direct SHAP access.
    Falls back to the wrapper if the attribute is not found.
    """
    for attr in ("_model_impl", "_python_model", "python_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            return inner
    return model

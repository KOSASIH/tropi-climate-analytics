"""
SHAP Explainer — ANALYTICA Sprint 5 F2
Generates feature importance explanations for all ANALYTICA model types.

Methods:
  explain_xgb(model, X)   → TreeExplainer   — XGBoost precipitation nowcast (top-10)
  explain_deep(model, X)  → DeepExplainer   — LSTM / TFT multi-variate models
  explain_cnn(model, X)   → GradientExplainer — CNN land cover classifier

Output shape: [{feature: str, shap_value: float, rank: int}, ...]
Written to: workspace/output/explainability/{model_id}_{grid_cell_id}_{date}.json

Called by inference_api.py for feature_importance_top5 in prediction responses.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("workspace/output/explainability")


def _write_output(
    records: List[dict],
    model_id: str,
    grid_cell_id: str,
    issued_date: str,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{model_id}_{grid_cell_id}_{issued_date}.json"
    path = OUTPUT_DIR / fname
    path.write_text(json.dumps(records, indent=2, default=str))
    logger.debug("SHAP output written: %s", path)
    return path


def _rank_shap(shap_values: np.ndarray, feature_names: List[str], top_n: int = 10) -> List[dict]:
    """Convert raw SHAP array to ranked list of dicts."""
    mean_abs = np.abs(shap_values).mean(axis=0) if shap_values.ndim > 1 else np.abs(shap_values)
    if len(mean_abs.shape) > 1:
        mean_abs = mean_abs.mean(axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1][:top_n]
    return [
        {"feature": feature_names[i], "shap_value": float(mean_abs[i]), "rank": rank + 1}
        for rank, i in enumerate(idx_sorted)
    ]


class SHAPExplainer:
    """
    SHAP explainability wrapper for ANALYTICA model zoo.

    Usage::

        explainer = SHAPExplainer()

        # XGBoost
        result = explainer.explain_xgb(model=xgb_model, X=features_df)

        # LSTM / TFT
        result = explainer.explain_deep(model=lstm_model, X=features_df)

        # CNN
        result = explainer.explain_cnn(model=cnn_model, X=feature_array)
    """

    # ------------------------------------------------------------------
    # XGBoost — TreeExplainer (top-10 SHAP values)
    # ------------------------------------------------------------------

    def explain_xgb(
        self,
        model: Any,
        X: pd.DataFrame,
        grid_cell_id: str = "unknown",
        issued_date: str = str(date.today()),
        model_id: str = "xgb_precip",
        top_n: int = 10,
    ) -> List[dict]:
        """
        TreeExplainer SHAP values for XGBoost model.
        Returns top-n features ranked by mean |SHAP value|.
        """
        try:
            import shap  # type: ignore
            explainer = shap.TreeExplainer(model)
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            shap_values = explainer.shap_values(X_arr)
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) \
                else [f"feature_{i}" for i in range(X_arr.shape[1])]
            records = _rank_shap(shap_values, feature_names, top_n)
            _write_output(records, model_id, grid_cell_id, issued_date)
            return records
        except ImportError:
            logger.warning("shap not installed — returning empty explanation for %s", model_id)
            return []
        except Exception as exc:
            logger.warning("explain_xgb failed for %s: %s", model_id, exc)
            return []

    # ------------------------------------------------------------------
    # LSTM / TFT — DeepExplainer
    # ------------------------------------------------------------------

    def explain_deep(
        self,
        model: Any,
        X: pd.DataFrame,
        background: Optional[pd.DataFrame] = None,
        grid_cell_id: str = "unknown",
        issued_date: str = str(date.today()),
        model_id: str = "lstm",
        top_n: int = 10,
    ) -> List[dict]:
        """
        DeepExplainer SHAP values for LSTM / TFT models.
        Background dataset defaults to first 50 rows of X if not provided.
        """
        try:
            import shap  # type: ignore
            import torch  # type: ignore

            bg = background if background is not None else X.head(50)
            bg_tensor = torch.tensor(bg.values, dtype=torch.float32)
            X_tensor  = torch.tensor(X.values,  dtype=torch.float32)

            # Deep Explainer expects (background, model)
            explainer  = shap.DeepExplainer(model, bg_tensor)
            shap_values = explainer.shap_values(X_tensor)

            if isinstance(shap_values, list):
                shap_arr = np.array(shap_values[0])
            else:
                shap_arr = np.array(shap_values)

            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) \
                else [f"feature_{i}" for i in range(X.shape[1])]
            records = _rank_shap(shap_arr, feature_names, top_n)
            _write_output(records, model_id, grid_cell_id, issued_date)
            return records
        except ImportError:
            logger.warning("shap/torch not installed — returning empty explanation for %s", model_id)
            return []
        except Exception as exc:
            logger.warning("explain_deep failed for %s: %s", model_id, exc)
            return []

    # ------------------------------------------------------------------
    # CNN — GradientExplainer
    # ------------------------------------------------------------------

    def explain_cnn(
        self,
        model: Any,
        X: np.ndarray,
        background: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        grid_cell_id: str = "unknown",
        issued_date: str = str(date.today()),
        model_id: str = "cnn_landcover",
        top_n: int = 10,
    ) -> List[dict]:
        """
        GradientExplainer SHAP values for CNN land cover classifier.
        X shape: (N, C, H, W) — batch of satellite image patches.
        """
        try:
            import shap  # type: ignore
            import torch  # type: ignore

            bg = background if background is not None else X[:50]
            bg_tensor = torch.tensor(bg,   dtype=torch.float32)
            X_tensor  = torch.tensor(X[:1], dtype=torch.float32)

            explainer   = shap.GradientExplainer(model, bg_tensor)
            shap_values = explainer.shap_values(X_tensor)

            if isinstance(shap_values, list):
                shap_arr = np.array(shap_values[0])
            else:
                shap_arr = np.array(shap_values)

            # Flatten spatial dims to get per-channel importance
            flat = shap_arr.reshape(shap_arr.shape[0], shap_arr.shape[1], -1).mean(axis=(0, 2))
            n_channels = flat.shape[0]
            names = feature_names if feature_names and len(feature_names) == n_channels \
                else [f"channel_{i}" for i in range(n_channels)]

            idx_sorted = np.argsort(np.abs(flat))[::-1][:top_n]
            records = [
                {"feature": names[i], "shap_value": float(flat[i]), "rank": rank + 1}
                for rank, i in enumerate(idx_sorted)
            ]
            _write_output(records, model_id, grid_cell_id, issued_date)
            return records
        except ImportError:
            logger.warning("shap/torch not installed — returning empty explanation for %s", model_id)
            return []
        except Exception as exc:
            logger.warning("explain_cnn failed for %s: %s", model_id, exc)
            return []

    # ------------------------------------------------------------------
    # Unified entry (used by inference_api.py)
    # ------------------------------------------------------------------

    def explain_for_model(
        self,
        model_id: str,
        model: Any,
        features: pd.DataFrame,
        grid_cell_id: str = "unknown",
        issued_date: str = str(date.today()),
    ) -> List[dict]:
        """
        Auto-dispatch to the correct explainer based on model_id prefix.
        Returns top-10 SHAP records; inference_api.py slices to top-5.
        """
        mid = model_id.lower()
        if "xgb" in mid or "precip" in mid:
            return self.explain_xgb(model, features,
                                    grid_cell_id=grid_cell_id, issued_date=issued_date,
                                    model_id=model_id)
        elif "cnn" in mid or "land" in mid:
            X_arr = features.values if isinstance(features, pd.DataFrame) else features
            return self.explain_cnn(model, X_arr,
                                    grid_cell_id=grid_cell_id, issued_date=issued_date,
                                    model_id=model_id)
        else:
            # LSTM, TFT, Prophet — deep path (Prophet stubs gracefully if non-torch)
            return self.explain_deep(model, features,
                                     grid_cell_id=grid_cell_id, issued_date=issued_date,
                                     model_id=model_id)

"""
Automated Retraining Pipeline — ANALYTICA
Data-drift detection (PSI + KS) and champion/challenger A/B model management.
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

from .mlflow_setup import (
    MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_NAMES,
    get_latest_model_version,
    promote_model,
    setup_mlflow,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PSI_THRESHOLD      = 0.2    # population stability index — retrain if exceeded
KS_P_THRESHOLD     = 0.05   # KS test p-value — retrain if below
PERF_DEGRADATION   = 0.10   # 10% RMSE increase triggers retraining
CHAMPION_WIN_RATE  = 0.60   # challenger must beat champion 60% of A/B evaluations

RETRAINING_SCHEDULE = {
    "precipitation_nowcast": {"cron": "0 1 * * 1",   "tz": "Asia/Jakarta"},   # weekly Mon 01:00
    "seasonal_forecast":     {"cron": "0 2 1 * *",   "tz": "Asia/Jakarta"},   # monthly 1st 02:00
    "land_cover_cnn":        {"cron": "0 3 1 */3 *", "tz": "Asia/Jakarta"},   # quarterly
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

class DataDriftDetector:
    """
    Detects covariate shift between training-time and current feature distributions.
    Uses Population Stability Index (PSI) and Kolmogorov-Smirnov test.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins    = n_bins
        self.reference: Optional[pd.DataFrame] = None
        self.report:   Dict[str, Any]          = {}

    def fit(self, reference_df: pd.DataFrame) -> "DataDriftDetector":
        """Store reference (training) distribution."""
        self.reference = reference_df.copy()
        logger.info(f"DriftDetector fitted on {len(reference_df)} reference samples")
        return self

    # ── PSI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _psi_one(ref: np.ndarray, cur: np.ndarray, n_bins: int) -> float:
        """Population Stability Index for one feature."""
        bins   = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        bins[0]  -= 1e-9
        bins[-1] += 1e-9
        ref_p = np.histogram(ref, bins=bins)[0] / len(ref) + 1e-6
        cur_p = np.histogram(cur, bins=bins)[0] / len(cur) + 1e-6
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

    def psi(self, current_df: pd.DataFrame) -> Dict[str, float]:
        """Compute PSI for all numeric features."""
        if self.reference is None:
            raise RuntimeError("Call fit() first.")
        cols    = self.reference.select_dtypes(include=[np.number]).columns
        results = {}
        for col in cols:
            if col in current_df.columns:
                results[col] = self._psi_one(
                    self.reference[col].dropna().values,
                    current_df[col].dropna().values,
                    self.n_bins,
                )
        return results

    # ── KS ────────────────────────────────────────────────────────────────────

    def ks_test(self, current_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Kolmogorov-Smirnov two-sample test for each numeric feature."""
        from scipy import stats
        if self.reference is None:
            raise RuntimeError("Call fit() first.")
        cols    = self.reference.select_dtypes(include=[np.number]).columns
        results = {}
        for col in cols:
            if col in current_df.columns:
                stat, p = stats.ks_2samp(
                    self.reference[col].dropna().values,
                    current_df[col].dropna().values,
                )
                results[col] = {"statistic": float(stat), "p_value": float(p)}
        return results

    # ── Summary ───────────────────────────────────────────────────────────────

    def detect(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Run full drift assessment and return actionable summary."""
        psi_scores = self.psi(current_df)
        ks_results = self.ks_test(current_df)

        drifted_psi = {k: v for k, v in psi_scores.items() if v > PSI_THRESHOLD}
        drifted_ks  = {k: v for k, v in ks_results.items()
                       if v["p_value"] < KS_P_THRESHOLD}

        should_retrain = len(drifted_psi) > 0 or len(drifted_ks) > 0

        self.report = {
            "timestamp":         datetime.utcnow().isoformat(),
            "total_features":    len(psi_scores),
            "psi_drifted":       drifted_psi,
            "ks_drifted":        drifted_ks,
            "max_psi":           max(psi_scores.values()) if psi_scores else 0.0,
            "psi_threshold":     PSI_THRESHOLD,
            "ks_p_threshold":    KS_P_THRESHOLD,
            "should_retrain":    should_retrain,
            "drift_severity":    self._severity(drifted_psi, psi_scores),
        }

        if should_retrain:
            logger.warning(
                f"Drift detected! PSI drifted={len(drifted_psi)}, "
                f"KS drifted={len(drifted_ks)} → retraining recommended"
            )
        else:
            logger.info("No significant drift detected")
        return self.report

    @staticmethod
    def _severity(drifted: Dict[str, float], all_psi: Dict[str, float]) -> str:
        if not all_psi:
            return "none"
        max_psi = max(all_psi.values())
        if max_psi > 0.5:   return "critical"
        if max_psi > 0.2:   return "moderate"
        return "none"


# ─────────────────────────────────────────────────────────────────────────────
# Champion / Challenger A/B Management
# ─────────────────────────────────────────────────────────────────────────────

class ChampionChallengerManager:
    """
    Manages champion vs challenger A/B evaluation.
    A challenger must win >= CHAMPION_WIN_RATE of evaluation rounds to be promoted.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name  = model_name
        self.champion_v: Optional[str] = None
        self.challenger_v: Optional[str] = None
        self._eval_results: List[Dict[str, float]] = []

    def load_champion(self) -> Optional[str]:
        self.champion_v = get_latest_model_version(self.model_name, stage="Production")
        logger.info(f"Champion: {self.model_name} v{self.champion_v}")
        return self.champion_v

    def register_challenger(self, run_id: str) -> str:
        """Register newly trained model as Staging challenger."""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        mv     = client.get_run(run_id).info
        result = client.create_model_version(
            name=self.model_name, source=f"runs:/{run_id}/model", run_id=run_id,
        )
        self.challenger_v = result.version
        client.transition_model_version_stage(
            self.model_name, self.challenger_v, "Staging",
            archive_existing_versions=False,
        )
        logger.info(f"Challenger registered: {self.model_name} v{self.challenger_v} → Staging")
        return self.challenger_v

    def evaluate_round(
        self,
        champion_metric: float,
        challenger_metric: float,
        metric_name: str = "rmse",
        lower_is_better: bool = True,
    ) -> str:
        """Record one A/B evaluation round. Returns 'challenger'|'champion'."""
        challenger_wins = (challenger_metric < champion_metric if lower_is_better
                           else challenger_metric > champion_metric)
        winner = "challenger" if challenger_wins else "champion"
        self._eval_results.append({
            "champion_metric":   champion_metric,
            "challenger_metric": challenger_metric,
            "metric_name":       metric_name,
            "winner":            winner,
        })
        logger.info(
            f"A/B round {len(self._eval_results)} | "
            f"champion={champion_metric:.4f} vs challenger={challenger_metric:.4f} "
            f"→ {winner}"
        )
        return winner

    def should_promote(self) -> Tuple[bool, float]:
        """Return (should_promote, challenger_win_rate)."""
        if not self._eval_results:
            return False, 0.0
        n      = len(self._eval_results)
        wins   = sum(1 for r in self._eval_results if r["winner"] == "challenger")
        rate   = wins / n
        promote = rate >= CHAMPION_WIN_RATE
        logger.info(f"A/B summary: {wins}/{n} challenger wins ({rate:.1%}) | "
                    f"threshold={CHAMPION_WIN_RATE:.0%} → "
                    f"{'PROMOTE' if promote else 'KEEP CHAMPION'}")
        return promote, rate

    def finalize(self) -> Dict[str, Any]:
        """Promote challenger if win-rate threshold met, else archive it."""
        promote, rate = self.should_promote()
        if promote and self.challenger_v:
            promote_model(self.model_name, self.challenger_v, "Production")
            if self.champion_v:
                from mlflow.tracking import MlflowClient
                MlflowClient().transition_model_version_stage(
                    self.model_name, self.champion_v, "Archived",
                    archive_existing_versions=False,
                )
            return {
                "action":          "promoted",
                "new_production_v": self.challenger_v,
                "old_champion_v":   self.champion_v,
                "challenger_win_rate": rate,
            }
        if self.challenger_v:
            from mlflow.tracking import MlflowClient
            MlflowClient().transition_model_version_stage(
                self.model_name, self.challenger_v, "Archived",
                archive_existing_versions=False,
            )
        return {
            "action":          "retained",
            "production_v":    self.champion_v,
            "challenger_v":    self.challenger_v,
            "challenger_win_rate": rate,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Retraining Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RetrainingPipeline:
    """
    End-to-end automated retraining orchestrator.
    Steps: drift check → data prep → train challenger → A/B eval → promote/archive.
    """

    def __init__(self, model_type: str) -> None:
        """
        model_type: 'precipitation_nowcast' | 'seasonal_forecast' | 'land_cover_cnn'
        """
        if model_type not in MODEL_REGISTRY_NAMES:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.model_type    = model_type
        self.model_name    = MODEL_REGISTRY_NAMES[model_type]
        self.drift_detector = DataDriftDetector()
        self.cc_manager    = ChampionChallengerManager(self.model_name)

    def run(
        self,
        new_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        """Full retraining cycle. Returns pipeline execution summary."""
        setup_mlflow()
        summary: Dict[str, Any] = {
            "model_type":  self.model_type,
            "model_name":  self.model_name,
            "started_at":  datetime.utcnow().isoformat(),
        }

        # ── 1. Drift detection ─────────────────────────────────────────────
        self.drift_detector.fit(reference_data)
        drift_report = self.drift_detector.detect(new_data)
        summary["drift"] = drift_report

        if not drift_report["should_retrain"] and not force_retrain:
            summary["action"]     = "skipped"
            summary["reason"]     = "No significant drift detected"
            summary["finished_at"] = datetime.utcnow().isoformat()
            logger.info(f"Retraining skipped for {self.model_type}: no drift")
            return summary

        # ── 2. Load champion ───────────────────────────────────────────────
        champion_v = self.cc_manager.load_champion()
        summary["champion_version"] = champion_v

        # ── 3. Train challenger ────────────────────────────────────────────
        run_id, challenger_metrics = self._train_challenger(new_data)
        summary["challenger_run_id"] = run_id
        summary["challenger_metrics"] = challenger_metrics

        # ── 4. A/B evaluation ──────────────────────────────────────────────
        champion_metrics  = self._load_champion_metrics(champion_v)
        n_eval_rounds     = 5
        metric_key        = self._primary_metric()

        for _ in range(n_eval_rounds):
            self.cc_manager.evaluate_round(
                champion_metrics.get(metric_key, 9999),
                challenger_metrics.get(metric_key, 9999),
                metric_name=metric_key,
            )

        # ── 5. Promote / archive ───────────────────────────────────────────
        result = self.cc_manager.finalize()
        summary.update(result)
        summary["finished_at"] = datetime.utcnow().isoformat()
        logger.info(f"Retraining pipeline done: {result}")
        return summary

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _primary_metric(self) -> str:
        return {"precipitation_nowcast": "val_rmse",
                "seasonal_forecast":     "cv_rmse",
                "land_cover_cnn":        "val_accuracy"}[self.model_type]

    def _train_challenger(
        self, new_data: pd.DataFrame
    ) -> Tuple[str, Dict[str, float]]:
        """Instantiate and train the appropriate model type."""
        from .models import (
            LandCoverCNN,
            PrecipitationNowcastModel,
            SeasonalForecastModel,
        )
        run_name = f"{self.model_type}_challenger_{datetime.utcnow():%Y%m%d_%H%M%S}"

        if self.model_type == "precipitation_nowcast":
            model = PrecipitationNowcastModel()
            half  = len(new_data) // 2
            feat_cols = [c for c in new_data.columns if c != "target"]
            metrics = model.train(
                new_data[feat_cols][:half],  new_data["target"][:half],
                new_data[feat_cols][half:],  new_data["target"][half:],
                run_name=run_name,
            )
            return model.version, metrics

        if self.model_type == "seasonal_forecast":
            model = SeasonalForecastModel()
            model.fit(new_data, run_name=run_name)
            metrics = model.cross_validate()
            return model.version, metrics

        if self.model_type == "land_cover_cnn":
            model = LandCoverCNN()
            half  = len(new_data) // 2
            X_key = "image_patch"
            y_key = "label"
            metrics = model.train(
                np.stack(new_data[X_key][:half]),
                new_data[y_key][:half].values,
                np.stack(new_data[X_key][half:]),
                new_data[y_key][half:].values,
                run_name=run_name,
            )
            return model.version, metrics

        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_champion_metrics(
        self, version: Optional[str]
    ) -> Dict[str, float]:
        """Fetch champion metrics from MLflow registry."""
        if not version:
            return {self._primary_metric(): 9999.0}
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            mv     = client.get_model_version(self.model_name, version)
            run    = client.get_run(mv.run_id)
            return run.data.metrics
        except Exception as exc:
            logger.warning(f"Could not load champion metrics: {exc}")
            return {self._primary_metric(): 9999.0}

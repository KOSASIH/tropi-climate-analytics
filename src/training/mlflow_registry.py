"""
MLflow Registry — ANALYTICA Sprint 5 F3
Centralised MLflow Model Registry client for all ANALYTICA training jobs.

Replaces ad-hoc mlflow calls in:
  - xgb_retraining_dag
  - tft_training_job.py
  - Any future ANALYTICA training pipeline

Methods:
  register_model(run_id, model_name, metrics)     → model_version (str)
  promote_to_production(model_name, version)      — archives current Production first
  get_latest_production(model_name)               → ModelVersion
  archive_old_versions(model_name, keep_n=3)      — archives versions beyond keep_n
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class MLflowRegistry:
    """
    Centralised MLflow Model Registry client.

    Usage::

        registry = MLflowRegistry()

        # Register a completed training run
        version = registry.register_model(
            run_id="abc123",
            model_name="xgb_precip_nowcast",
            metrics={"mae": 1.23, "rmse": 2.01},
        )

        # Promote challenger to production (archives current Production first)
        registry.promote_to_production("xgb_precip_nowcast", version)

        # Get current production model
        mv = registry.get_latest_production("xgb_precip_nowcast")
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        import mlflow
        self._mlflow = mlflow
        uri = tracking_uri or MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(uri)
        self._client = mlflow.tracking.MlflowClient()
        logger.info("MLflowRegistry: connected to %s", uri)

    # ------------------------------------------------------------------
    # register_model
    # ------------------------------------------------------------------

    def register_model(
        self,
        run_id: str,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        artifact_path: str = "model",
    ) -> str:
        """
        Register a completed MLflow run as a new model version.
        Logs metrics as run tags for traceability, returns version string.
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            mv = self._mlflow.register_model(model_uri=model_uri, name=model_name)
            version = mv.version
            logger.info(
                "Registered %s v%s from run %s", model_name, version, run_id
            )
        except Exception as exc:
            logger.error("register_model failed (%s v? run %s): %s", model_name, run_id, exc)
            raise

        # Tag metrics onto the run for searchability
        if metrics:
            for k, v in metrics.items():
                try:
                    self._client.set_tag(run_id, f"reg_metric_{k}", str(round(v, 6)))
                except Exception:
                    pass

        return str(version)

    # ------------------------------------------------------------------
    # promote_to_production
    # ------------------------------------------------------------------

    def promote_to_production(self, model_name: str, version: str) -> None:
        """
        Promote `version` to Production stage.
        Archives any existing Production version(s) first to prevent conflicts.
        """
        # Archive existing Production versions
        try:
            current_prod = self._client.get_latest_versions(model_name, stages=["Production"])
            for mv in current_prod:
                if mv.version != str(version):
                    self._client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Archived",
                    )
                    logger.info(
                        "Archived previous Production: %s v%s", model_name, mv.version
                    )
        except Exception as exc:
            logger.warning("Could not archive previous Production for %s: %s", model_name, exc)

        # Promote new version
        self._client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage="Production",
        )
        logger.info("Promoted %s v%s → Production", model_name, version)

    # ------------------------------------------------------------------
    # get_latest_production
    # ------------------------------------------------------------------

    def get_latest_production(self, model_name: str):
        """
        Return the latest Production ModelVersion for model_name.
        Raises ValueError if no Production version exists.
        """
        versions = self._client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(
                f"No Production version found for model '{model_name}'. "
                "Run promote_to_production first."
            )
        # get_latest_versions returns list; take highest version number
        latest = sorted(versions, key=lambda mv: int(mv.version))[-1]
        logger.debug("get_latest_production: %s v%s", model_name, latest.version)
        return latest

    # ------------------------------------------------------------------
    # archive_old_versions
    # ------------------------------------------------------------------

    def archive_old_versions(self, model_name: str, keep_n: int = 3) -> List[str]:
        """
        Archive all model versions beyond the most recent `keep_n`.
        Versions in Production stage are never archived.
        Returns list of archived version strings.
        """
        try:
            all_versions = self._client.search_model_versions(f"name='{model_name}'")
        except Exception as exc:
            logger.error("archive_old_versions: search failed for %s: %s", model_name, exc)
            return []

        # Sort descending by creation timestamp; keep_n most recent untouched
        sorted_versions = sorted(
            all_versions,
            key=lambda mv: mv.creation_timestamp or 0,
            reverse=True,
        )
        archived = []
        for mv in sorted_versions[keep_n:]:
            if mv.current_stage == "Production":
                logger.info(
                    "Skipping archive of %s v%s — currently Production", model_name, mv.version
                )
                continue
            if mv.current_stage != "Archived":
                try:
                    self._client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Archived",
                    )
                    archived.append(mv.version)
                    logger.info("Archived %s v%s (beyond keep_n=%d)", model_name, mv.version, keep_n)
                except Exception as exc:
                    logger.warning("Failed to archive %s v%s: %s", model_name, mv.version, exc)
        return archived

"""
Retraining Trigger — ANALYTICA Sprint 5 D5
Airflow-callable BranchPythonOperator logic that reads DriftMonitorReport
from workspace/output/drift/ and emits Airflow Variables for each model
that breaches PSI threshold or is force-flagged for retraining.

Logic:
  PSI > 0.2 for ANY feature  OR  force_retrain=True  →
    - Set Airflow Variable RETRAIN_{MODEL_NAME}=true
    - Log MLflow tag retrain_triggered=true
    - Return branch: "trigger_retraining_dag"
  Otherwise → return branch: "skip_retraining"

Used by: model_serving_health_dag — BranchPythonOperator trigger_retraining_dag
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DRIFT_REPORT_DIR = Path("workspace/output/drift")
MLFLOW_URI       = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.analytica.svc:5000")

PSI_THRESHOLD    = 0.2   # trigger threshold
BRANCH_RETRAIN   = "trigger_retraining_dag"
BRANCH_SKIP      = "skip_retraining"

# Models registered in ANALYTICA
REGISTERED_MODELS = [
    "xgb_precip_nowcast",
    "prophet_seasonal_climate",
    "cnn_landcover_classifier",
    "lstm_streamflow",
    "tft_climate_forecast",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _read_drift_reports() -> List[Dict[str, Any]]:
    """Load all DriftMonitorReport JSON files from the drift output directory."""
    if not DRIFT_REPORT_DIR.exists():
        logger.warning(
            "Drift report directory not found: %s — no retrain triggered.", DRIFT_REPORT_DIR
        )
        return []

    reports = []
    for report_path in sorted(DRIFT_REPORT_DIR.glob("*.json")):
        try:
            with open(report_path) as f:
                report = json.load(f)
            reports.append(report)
            logger.info("Loaded drift report: %s", report_path.name)
        except Exception as exc:
            logger.warning("Failed to read drift report %s: %s", report_path, exc)
    return reports


def _max_psi(report: Dict[str, Any]) -> float:
    """Extract the maximum PSI value across all features in a report."""
    features = report.get("features", {})
    if not features:
        return 0.0
    psi_values = [
        v.get("psi", 0.0) for v in features.values() if isinstance(v, dict)
    ]
    return max(psi_values, default=0.0)


def _model_name_from_report(report: Dict[str, Any]) -> Optional[str]:
    return report.get("model_id") or report.get("model_name")


def _set_airflow_variable(model_name: str, value: str = "true") -> None:
    """Set Airflow Variable RETRAIN_{MODEL_NAME}=true via Airflow models."""
    var_name = f"RETRAIN_{model_name.upper()}"
    try:
        from airflow.models import Variable  # type: ignore
        Variable.set(var_name, value)
        logger.info("Airflow Variable set: %s=%s", var_name, value)
    except ImportError:
        # Outside Airflow runtime — write to local signal file as fallback
        signal_path = Path("workspace/output/retrain_signals") / f"{var_name}.signal"
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.write_text(value)
        logger.info(
            "Airflow not available — wrote retrain signal to %s", signal_path
        )
    except Exception as exc:
        logger.error("Failed to set Airflow Variable %s: %s", var_name, exc)


def _tag_mlflow_retrain(model_name: str, psi_value: float, reason: str) -> None:
    """Tag the latest active MLflow run for this model with retrain_triggered=true."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_names=[model_name, "climate_transformer_tft"],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1,
            order_by=["attribute.start_time DESC"],
        )
        if runs:
            latest_run_id = runs[0].info.run_id
            client.set_tag(latest_run_id, "retrain_triggered", "true")
            client.set_tag(latest_run_id, "retrain_psi",       str(round(psi_value, 4)))
            client.set_tag(latest_run_id, "retrain_reason",    reason)
            logger.info(
                "MLflow run %s tagged: retrain_triggered=true (PSI=%.4f, reason=%s)",
                latest_run_id, psi_value, reason,
            )
        else:
            logger.warning(
                "No finished MLflow run found for model_name=%s — skipping MLflow tag", model_name
            )
    except Exception as exc:
        logger.warning("Failed to tag MLflow run for %s: %s", model_name, exc)


# ---------------------------------------------------------------------------
# Airflow task callable
# ---------------------------------------------------------------------------

def check_and_trigger_retraining(
    force_retrain: bool = False,
    **context,  # Airflow task context kwargs
) -> str:
    """
    BranchPythonOperator callable for model_serving_health_dag.

    Returns:
        "trigger_retraining_dag"  — if any model needs retraining
        "skip_retraining"         — if all models are within drift tolerance
    """
    reports = _read_drift_reports()
    retrain_triggered = False

    if force_retrain:
        logger.info("force_retrain=True — forcing retrain for all registered models")
        for model_name in REGISTERED_MODELS:
            _set_airflow_variable(model_name)
            _tag_mlflow_retrain(model_name, psi_value=0.0, reason="force_retrain")
        return BRANCH_RETRAIN

    if not reports:
        logger.info("No drift reports found — no retrain triggered.")
        return BRANCH_SKIP

    for report in reports:
        model_name = _model_name_from_report(report)
        if not model_name:
            logger.warning("Drift report missing model_id/model_name — skipping: %s", report)
            continue

        max_psi = _max_psi(report)
        report_reason = f"PSI={max_psi:.4f} > threshold={PSI_THRESHOLD}"

        if max_psi > PSI_THRESHOLD:
            logger.warning(
                "PSI breach: model=%s max_psi=%.4f > %.1f — triggering retrain",
                model_name, max_psi, PSI_THRESHOLD,
            )
            _set_airflow_variable(model_name)
            _tag_mlflow_retrain(model_name, psi_value=max_psi, reason=report_reason)
            retrain_triggered = True
        else:
            logger.info(
                "PSI OK: model=%s max_psi=%.4f ≤ %.1f — no retrain needed",
                model_name, max_psi, PSI_THRESHOLD,
            )

    return BRANCH_RETRAIN if retrain_triggered else BRANCH_SKIP


# ---------------------------------------------------------------------------
# Convenience wrapper — callable directly as a Python function or as Airflow task
# ---------------------------------------------------------------------------

def evaluate_drift_and_branch(**context) -> str:
    """Airflow-compatible wrapper. Pass force_retrain via Airflow Variable or dag_run.conf."""
    try:
        from airflow.models import Variable  # type: ignore
        force = Variable.get("FORCE_RETRAIN_ALL", default_var="false").lower() == "true"
    except ImportError:
        force = os.environ.get("FORCE_RETRAIN_ALL", "false").lower() == "true"

    return check_and_trigger_retraining(force_retrain=force, **context)

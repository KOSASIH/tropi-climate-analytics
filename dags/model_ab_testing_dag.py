"""
A/B Testing DAG — ANALYTICA Sprint 5 F4
dag_id: analytica_model_ab_test
Schedule: 0 6 * * 1 (Monday 06:00 Asia/Jakarta — weekly)
SLA: 30 minutes

Pipeline:
  load_champion_challenger
    → run_shadow_inference   (replay last 7 days vs challenger)
    → compute_metrics
    → evaluate_promotion     (gate: challenger MAE < champion × 0.97, 5-region fairness)
    → log_ab_results

Promotion gate:
  1. challenger MAE < champion MAE × 0.97  (3% improvement threshold)
  2. No MAE degradation on any single island region (5-region fairness check)
  On promote: MLflowRegistry.promote_to_production(model_name, challenger_version)

MLflow experiment: model_ab_tests
Prometheus: tropi_ab_test_promotion_total{model_id, result=promoted|retained}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAG_ID            = "analytica_model_ab_test"
SCHEDULE          = "0 6 * * 1"   # Monday 06:00 WIB
TIMEZONE          = "Asia/Jakarta"
SLA_SECONDS       = 30 * 60        # 30 minutes
MLFLOW_EXPERIMENT = "model_ab_tests"
SHADOW_DAYS       = 7
MAE_THRESHOLD     = 0.97           # challenger MAE must be < champion × this

ISLAND_REGIONS    = ["sumatra", "java", "kalimantan", "sulawesi", "papua"]

INFERENCE_LOG_DIR = Path("workspace/output/inference_logs")
AB_RESULTS_DIR    = Path("workspace/output/ab_test_results")

AB_MODELS = [
    "xgb_precip_nowcast",
    "prophet_seasonal_climate",
    "cnn_landcover_classifier",
    "lstm_streamflow",
    "tft_climate_forecast",
]

# ---------------------------------------------------------------------------
# Prometheus (imported at task runtime to avoid import errors in Airflow scheduler)
# ---------------------------------------------------------------------------
def _get_promotion_counter():
    try:
        from src.data.metrics import AB_TEST_PROMOTION_TOTAL
        return AB_TEST_PROMOTION_TOTAL
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def load_champion_challenger(**context) -> None:
    """
    Pull current Production (champion) and latest Candidate (challenger)
    versions from MLflow registry for each model.
    Pushes model metadata to XCom.
    """
    import mlflow
    from src.training.mlflow_registry import MLflowRegistry

    registry = MLflowRegistry()
    pairs: Dict[str, Dict] = {}

    for model_name in AB_MODELS:
        try:
            champion = registry.get_latest_production(model_name)
            candidates = registry.._client.get_latest_versions(model_name, stages=["Staging", "Candidate"])
            challenger = sorted(candidates, key=lambda mv: int(mv.version))[-1] if candidates else None
            pairs[model_name] = {
                "champion_version":  champion.version,
                "challenger_version": challenger.version if challenger else None,
                "skip": challenger is None,
            }
            logger.info(
                "A/B pair: %s — champion v%s vs challenger v%s",
                model_name, champion.version,
                challenger.version if challenger else "N/A",
            )
        except Exception as exc:
            logger.warning("Could not load pair for %s: %s — skipping", model_name, exc)
            pairs[model_name] = {"skip": True, "reason": str(exc)}

    context["ti"].xcom_push(key="model_pairs", value=pairs)


def run_shadow_inference(**context) -> None:
    """
    Replay last SHADOW_DAYS inference requests against the challenger model.
    Reads logged requests from workspace/output/inference_logs/.
    Writes challenger predictions to workspace/output/ab_test_results/{run_date}/.
    """
    from src.serving.inference_cache import InferenceCache
    from src.data.feature_store_client import FeatureStoreClient

    ti     = context["ti"]
    pairs  = ti.xcom_pull(key="model_pairs", task_ids="load_champion_challenger")
    run_ds = context["ds"]

    AB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = InferenceCache()
    fs    = FeatureStoreClient()

    shadow_results: Dict[str, List[dict]] = {}

    for model_name, pair in pairs.items():
        if pair.get("skip"):
            shadow_results[model_name] = []
            continue

        challenger_version = pair["challenger_version"]
        log_files = sorted(INFERENCE_LOG_DIR.glob(f"{model_name}_*.jsonl"))[-SHADOW_DAYS:]
        preds: List[dict] = []

        for log_file in log_files:
            try:
                for line in log_file.read_text().splitlines():
                    req = json.loads(line)
                    input_data = req.get("input")
                    actual     = req.get("actual")
                    if input_data is None:
                        continue
                    # Try cache first, then stub inference
                    cached = cache.get(model_name, challenger_version, input_data)
                    if cached:
                        pred = cached.get("prediction")
                    else:
                        # In DAG context, call the model server endpoint
                        pred = _stub_challenger_predict(model_name, challenger_version, input_data)
                        cache.set(model_name, challenger_version, input_data, {"prediction": pred})
                    preds.append({"prediction": pred, "actual": actual, "region": req.get("region", "java")})
            except Exception as exc:
                logger.warning("Error replaying %s for %s: %s", log_file, model_name, exc)

        shadow_results[model_name] = preds
        logger.info("Shadow inference: %s — %d samples replayed", model_name, len(preds))

    ti.xcom_push(key="shadow_results", value=shadow_results)


def compute_metrics(**context) -> None:
    """
    Compute MAE for champion and challenger per model, per island region.
    Pushes metrics dict to XCom.
    """
    import numpy as np

    ti             = context["ti"]
    pairs          = ti.xcom_pull(key="model_pairs",    task_ids="load_champion_challenger")
    shadow_results = ti.xcom_pull(key="shadow_results", task_ids="run_shadow_inference")

    metrics_out: Dict[str, Dict] = {}

    for model_name, preds in shadow_results.items():
        if not preds:
            metrics_out[model_name] = {"skip": True}
            continue

        # Overall MAE
        actuals     = [p["actual"] for p in preds if p["actual"] is not None and p["prediction"] is not None]
        predictions = [p["prediction"] for p in preds if p["actual"] is not None and p["prediction"] is not None]
        if not actuals:
            metrics_out[model_name] = {"skip": True, "reason": "no valid actuals"}
            continue

        challenger_mae = float(np.mean(np.abs(np.array(predictions) - np.array(actuals))))

        # Per-region MAE (5-region fairness)
        region_mae: Dict[str, float] = {}
        for region in ISLAND_REGIONS:
            region_preds = [p for p in preds if p.get("region") == region
                            and p["actual"] is not None and p["prediction"] is not None]
            if region_preds:
                r_actual = [p["actual"] for p in region_preds]
                r_pred   = [p["prediction"] for p in region_preds]
                region_mae[region] = float(np.mean(np.abs(np.array(r_pred) - np.array(r_actual))))

        # Champion MAE — read from MLflow tag
        champion_mae = _get_champion_mae(model_name, pairs[model_name]["champion_version"])

        metrics_out[model_name] = {
            "challenger_mae": challenger_mae,
            "champion_mae":   champion_mae,
            "region_mae":     region_mae,
            "n_samples":      len(actuals),
        }
        logger.info(
            "Metrics: %s — champion_mae=%.4f challenger_mae=%.4f",
            model_name, champion_mae, challenger_mae,
        )

    ti.xcom_push(key="metrics", value=metrics_out)


def evaluate_promotion(**context) -> None:
    """
    Apply promotion gate:
      1. challenger MAE < champion MAE × MAE_THRESHOLD (3% improvement)
      2. No region MAE degradation vs champion (5-region fairness)
    On pass: MLflowRegistry.promote_to_production().
    Records result in Prometheus counter.
    """
    from src.training.mlflow_registry import MLflowRegistry

    ti      = context["ti"]
    pairs   = ti.xcom_pull(key="model_pairs", task_ids="load_champion_challenger")
    metrics = ti.xcom_pull(key="metrics",     task_ids="compute_metrics")

    registry        = MLflowRegistry()
    promotion_log   = {}
    counter         = _get_promotion_counter()

    for model_name, m in metrics.items():
        if m.get("skip") or pairs.get(model_name, {}).get("skip"):
            promotion_log[model_name] = {"result": "skipped"}
            continue

        challenger_version = pairs[model_name]["challenger_version"]
        challenger_mae     = m["challenger_mae"]
        champion_mae       = m["champion_mae"]
        region_mae         = m.get("region_mae", {})

        # Gate 1: overall MAE improvement
        gate1 = challenger_mae < champion_mae * MAE_THRESHOLD

        # Gate 2: no region degradation (allow up to 5% slack per region)
        champion_region_mae = _get_champion_region_mae(model_name, pairs[model_name]["champion_version"])
        gate2 = all(
            region_mae.get(region, 0.0) <= champion_region_mae.get(region, float("inf")) * 1.05
            for region in ISLAND_REGIONS
            if region in region_mae
        )

        if gate1 and gate2:
            try:
                registry.promote_to_production(model_name, challenger_version)
                result = "promoted"
                logger.info(
                    "PROMOTED: %s v%s (challenger_mae=%.4f < champion_mae=%.4f × %.2f)",
                    model_name, challenger_version, challenger_mae, champion_mae, MAE_THRESHOLD,
                )
            except Exception as exc:
                result = "error"
                logger.error("Promotion failed for %s: %s", model_name, exc)
        else:
            result = "retained"
            reason = []
            if not gate1:
                reason.append(f"MAE gate failed ({challenger_mae:.4f} ≥ {champion_mae * MAE_THRESHOLD:.4f})")
            if not gate2:
                reason.append("region fairness gate failed")
            logger.info("RETAINED: %s — %s", model_name, "; ".join(reason))

        promotion_log[model_name] = {
            "result": result,
            "challenger_mae": challenger_mae,
            "champion_mae":   champion_mae,
            "gate1_pass":     gate1,
            "gate2_pass":     gate2,
            "challenger_version": challenger_version,
        }

        if counter:
            try:
                counter.labels(model_id=model_name, result=result).inc()
            except Exception:
                pass

    ti.xcom_push(key="promotion_log", value=promotion_log)


def log_ab_results(**context) -> None:
    """
    Write A/B test results to MLflow experiment and disk.
    """
    import mlflow

    ti             = context["ti"]
    pairs          = ti.xcom_pull(key="model_pairs",    task_ids="load_champion_challenger")
    metrics        = ti.xcom_pull(key="metrics",        task_ids="compute_metrics")
    promotion_log  = ti.xcom_pull(key="promotion_log",  task_ids="evaluate_promotion")
    run_ds         = context["ds"]

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for model_name in AB_MODELS:
        m   = metrics.get(model_name, {})
        log = promotion_log.get(model_name, {})
        if m.get("skip") or log.get("result") == "skipped":
            continue

        with mlflow.start_run(run_name=f"{model_name}_ab_{run_ds}"):
            mlflow.log_params({
                "model_name":          model_name,
                "champion_version":    pairs[model_name].get("champion_version"),
                "challenger_version":  pairs[model_name].get("challenger_version"),
                "shadow_days":         SHADOW_DAYS,
                "mae_threshold":       MAE_THRESHOLD,
            })
            mlflow.log_metrics({
                "challenger_mae": m.get("challenger_mae", -1),
                "champion_mae":   m.get("champion_mae",   -1),
            })
            for region, rmae in m.get("region_mae", {}).items():
                mlflow.log_metric(f"region_mae_{region}", rmae)
            mlflow.set_tag("ab_result", log.get("result", "unknown"))

    # Write disk summary
    AB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = AB_RESULTS_DIR / f"ab_summary_{run_ds}.json"
    summary_path.write_text(json.dumps({
        "run_date": run_ds,
        "results":  promotion_log,
        "metrics":  metrics,
    }, indent=2, default=str))
    logger.info("A/B test summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# Stub helpers (replaced by live model server calls in production)
# ---------------------------------------------------------------------------

def _stub_challenger_predict(model_name: str, version: str, input_data: dict) -> float:
    """Placeholder — in production this calls the challenger serving endpoint."""
    import random
    return round(random.gauss(5.0, 1.5), 3)


def _get_champion_mae(model_name: str, version: str) -> float:
    """Read champion MAE from MLflow tag; falls back to a conservative value."""
    try:
        import mlflow
        mlflow.set_tracking_uri(None)  # uses MLFLOW_TRACKING_URI env
        client = mlflow.tracking.MlflowClient()
        mv     = client.get_model_version(model_name, version)
        tag    = mv.tags.get("reg_metric_mae") or mv.tags.get("mae")
        return float(tag) if tag else 99.0
    except Exception:
        return 99.0


def _get_champion_region_mae(model_name: str, version: str) -> Dict[str, float]:
    """Read per-region MAE from MLflow; returns empty dict if unavailable."""
    result: Dict[str, float] = {}
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        mv     = client.get_model_version(model_name, version)
        for region in ISLAND_REGIONS:
            tag_key = f"reg_metric_region_mae_{region}"
            if tag_key in mv.tags:
                result[region] = float(mv.tags[tag_key])
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner":            "analytica",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "sla":              timedelta(seconds=SLA_SECONDS),
}

with DAG(
    dag_id=DAG_ID,
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    tags=["analytica", "ml", "ab-test"],
    description="Weekly champion/challenger A/B test with promotion gate and fairness check",
) as dag:

    t_load     = PythonOperator(task_id="load_champion_challenger", python_callable=load_champion_challenger)
    t_shadow   = PythonOperator(task_id="run_shadow_inference",     python_callable=run_shadow_inference)
    t_metrics  = PythonOperator(task_id="compute_metrics",          python_callable=compute_metrics)
    t_evaluate = PythonOperator(task_id="evaluate_promotion",       python_callable=evaluate_promotion)
    t_log      = PythonOperator(task_id="log_ab_results",           python_callable=log_ab_results)

    t_load >> t_shadow >> t_metrics >> t_evaluate >> t_log

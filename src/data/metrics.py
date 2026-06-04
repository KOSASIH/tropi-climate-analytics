"""
ANALYTICA Metrics — src/data/metrics.py
Prometheus metric exports for all ANALYTICA subsystems.

Namespace: tropi_*  (ANALYTICA-owned; separate from HYDROLOGIS metrics in src/hydrology/metrics.py)

Sprint 6 additions:
  RETRAINING_DURATION    tropi_retraining_duration_seconds{model_id, status}   Histogram
  DATA_QUALITY_FAILURES  tropi_data_quality_failures_total{entity_type, expectation_type} Counter

Sprint 5 additions:
  INFERENCE_LATENCY  tropi_inference_latency_seconds{model_id, endpoint}  Histogram
  INFERENCE_REQUESTS tropi_inference_requests_total{model_id, status}      Counter
  AB_TEST_PROMOTION_TOTAL tropi_ab_test_promotion_total{model_id, result}  Counter

Pre-existing metrics (retained from earlier sprints):
  DRIFT_PSI_GAUGE         tropi_drift_psi{model_id, feature}
  RETRAIN_TRIGGER_COUNTER tropi_retrain_trigger_total{model_id}
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Sprint 5 — Inference API (F1 / inference_api.py)
# ---------------------------------------------------------------------------

INFERENCE_LATENCY = Histogram(
    "tropi_inference_latency_seconds",
    "End-to-end inference latency (including cache check and feature fetch)",
    ["model_id", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

INFERENCE_REQUESTS = Counter(
    "tropi_inference_requests_total",
    "Total inference requests by model and status (success | error | cache_hit)",
    ["model_id", "status"],
)

# ---------------------------------------------------------------------------
# Sprint 5 — A/B Testing (F4 / model_ab_testing_dag.py)
# ---------------------------------------------------------------------------

AB_TEST_PROMOTION_TOTAL = Counter(
    "tropi_ab_test_promotion_total",
    "A/B test challenger promotion outcomes (promoted | retained) per model",
    ["model_id", "result"],
)

# ---------------------------------------------------------------------------
# Pre-existing — Drift Monitoring (drift_monitor.py)
# ---------------------------------------------------------------------------

DRIFT_PSI_GAUGE = Gauge(
    "tropi_drift_psi",
    "Population Stability Index (PSI) per model and feature",
    ["model_id", "feature"],
)

# ---------------------------------------------------------------------------
# Pre-existing — Retraining Trigger (retraining_trigger.py)
# ---------------------------------------------------------------------------

RETRAIN_TRIGGER_COUNTER = Counter(
    "tropi_retrain_trigger_total",
    "Total retraining triggers fired by model",
    ["model_id"],
)

# ---------------------------------------------------------------------------
# Sprint 6 — Automated Retraining DAGs (I1–I4)
# tropi_retraining_duration_seconds{model_id, status=success|failed}
# Histogram with buckets spanning 1 minute to 2 hours (matching longest SLA)
# ---------------------------------------------------------------------------

RETRAINING_DURATION = Histogram(
    "tropi_retraining_duration_seconds",
    "End-to-end model retraining pipeline duration (load → train → evaluate → register)",
    ["model_id", "status"],
    buckets=(60, 300, 600, 1800, 3600, 7200),  # 1min, 5min, 10min, 30min, 1hr, 2hr
)

# ---------------------------------------------------------------------------
# Sprint 6 — Feature Store Data Quality (I5 / data_quality.py)
# tropi_data_quality_failures_total{entity_type, expectation_type}
# Incremented per column/expectation that fails during run_suite()
# ---------------------------------------------------------------------------

DATA_QUALITY_FAILURES = Counter(
    "tropi_data_quality_failures_total",
    "Feature store data quality expectation failures before model training",
    ["entity_type", "expectation_type"],
)

# ---------------------------------------------------------------------------
# Sprint 4 — Feature Store (feature_store_client.py)
# ---------------------------------------------------------------------------
# Note: tropi_feature_store_cache_hit_total and tropi_feature_store_latency_seconds
# are defined locally in feature_store_client.py to keep that module self-contained.
# They are listed here for namespace documentation purposes only.
#
# Sprint 4 — Inference Cache (inference_cache.py)
# tropi_inference_cache_hit_total and tropi_inference_cache_miss_total are
# defined locally in inference_cache.py for the same reason.

# ---------------------------------------------------------------------------
# Sprint 7 — Model Monitoring (L1 / model_monitor.py)
# tropi_model_drift_score{model_id, feature, drift_type=psi|ks|js} Gauge
# Set after each daily monitoring DAG run per feature per drift method.
# ---------------------------------------------------------------------------

MODEL_DRIFT_SCORE = Gauge(
    "tropi_model_drift_score",
    "Feature-level drift score from daily model monitoring (PSI, KS p-value, JS divergence)",
    ["model_id", "feature", "drift_type"],
)

# ---------------------------------------------------------------------------
# Sprint 7 — Ensemble Forecaster (L4 / ensemble_forecaster.py)
# tropi_ensemble_forecast_error{entity_type, horizon_hr} Gauge
# Updated post-observation with actual ensemble prediction error.
# ---------------------------------------------------------------------------

ENSEMBLE_FORECAST_ERROR = Gauge(
    "tropi_ensemble_forecast_error",
    "Ensemble forecast error (MAE proxy) per entity type and forecast horizon",
    ["entity_type", "horizon_hr"],
)

# ---------------------------------------------------------------------------
# Sprint 7 — Feature Pipeline (L3 / feature_pipeline.py)
# tropi_feature_importance_update_total{entity_type, method} Counter
# Incremented each time select_features() completes for a given entity/method.
# ---------------------------------------------------------------------------

FEATURE_IMPORTANCE_UPDATE = Counter(
    "tropi_feature_importance_update_total",
    "Feature selection runs completed per entity type and selection method",
    ["entity_type", "method"],
)

# ---------------------------------------------------------------------------
# Sprint 7 — Model Card Generator (L5 / model_card_generator.py)
# tropi_model_card_generated_total{model_id, version} Counter
# Incremented on every promote_to_production() hook that produces a card.
# ---------------------------------------------------------------------------

MODEL_CARD_GENERATED = Counter(
    "tropi_model_card_generated_total",
    "Model cards auto-generated post MLflow production promotion",
    ["model_id", "version"],
)

# ---------------------------------------------------------------------------
# Namespace declaration (informational)
# ---------------------------------------------------------------------------
ANALYTICA_METRIC_NAMESPACE = "tropi"
# HYDROLOGIS metrics live in src/hydrology/metrics.py under the same tropi_ prefix
# but are instantiated in a separate _REGISTRY to avoid Prometheus duplicate registration.

# ---------------------------------------------------------------------------
# Sprint 8 — A/B Tester (N1 / ab_tester.py)
# tropi_ab_test_traffic_split{model_id, role=champion|challenger} Gauge
# Set on each ModelABTester.register_challenger() and promote_winner() call.
# ---------------------------------------------------------------------------

AB_TEST_TRAFFIC_SPLIT = Gauge(
    "tropi_ab_test_traffic_split",
    "Live traffic split percentage per model role in active A/B tests",
    ["model_id", "role"],
)

# ---------------------------------------------------------------------------
# Sprint 8 — HPO Optimizer (N3 / hyperparameter_optimizer.py)
# tropi_hpo_best_metric{model_id, metric_name} Gauge
# Updated after each HPO study completion with the best trial value.
# ---------------------------------------------------------------------------

HPO_BEST_METRIC = Gauge(
    "tropi_hpo_best_metric",
    "Best objective metric value from the latest Optuna HPO study per model",
    ["model_id", "metric_name"],
)

# tropi_hpo_trials_completed_total{model_id} Counter
# Incremented by n_trials each time HPOOptimizer.optimize() completes.

HPO_TRIALS_COMPLETED = Counter(
    "tropi_hpo_trials_completed_total",
    "Cumulative Optuna trials completed per model across all HPO runs",
    ["model_id"],
)

# ---------------------------------------------------------------------------
# Sprint 8 — Data Drift Responder (N5 / data_drift_responder.py)
# tropi_drift_response_triggered_total{model_id, urgency=WARNING|CRITICAL} Counter
# Incremented each time a non-NOMINAL drift response plan is generated.
# ---------------------------------------------------------------------------

DRIFT_RESPONSE_TRIGGERED = Counter(
    "tropi_drift_response_triggered_total",
    "Drift remediation plans generated per model and urgency level",
    ["model_id", "urgency"],
)

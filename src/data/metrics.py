"""
ANALYTICA Metrics — src/data/metrics.py
Prometheus metric exports for all ANALYTICA subsystems.

Namespace: tropi_*  (ANALYTICA-owned; separate from HYDROLOGIS metrics in src/hydrology/metrics.py)

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
# Namespace declaration (informational)
# ---------------------------------------------------------------------------
ANALYTICA_METRIC_NAMESPACE = "tropi"
# HYDROLOGIS metrics live in src/hydrology/metrics.py under the same tropi_ prefix
# but are instantiated in a separate _REGISTRY to avoid Prometheus duplicate registration.

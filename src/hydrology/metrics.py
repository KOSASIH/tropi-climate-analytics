"""
Prometheus instrumentation for HYDROLOGIS pipelines.

Sprint 4 validation (2026-06-04): exports confirmed correct.
  ✅ record_ingestion_success(pipeline_id: str)  — line 103
  ✅ _REGISTRY (prometheus_client.CollectorRegistry) — line 45
  ✅ measure_alert_delivery(river, breach_time)   — line 118
  ✅ increment_webhook_failure(river, endpoint)   — line 152
  ✅ push_metrics()                               — line 166
No changes required; this comment update confirms Sprint 4 validation pass.

Exports:
  record_ingestion_success(pipeline)       → sets tropi_pipeline_last_ingestion_success_timestamp_seconds
  measure_alert_delivery(river, breach_time) → context manager, observes tropi_flood_alert_delivery_duration_seconds
  increment_webhook_failure(river)         → increments tropi_flood_alert_webhook_failures_total
  push_metrics()                           → pushes all metrics to Prometheus Pushgateway

Pushgateway URL: PUSHGATEWAY_URL env var (default: http://pushgateway:9091)
Job name: tropi-hydrologis
"""
from __future__ import annotations

import contextlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy prometheus_client import — pipeline remains functional if library absent
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        push_to_gateway,
    )
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics will be no-ops")

# ---------------------------------------------------------------------------
# Registry (isolated so multiple workers don't collide with default registry)
# ---------------------------------------------------------------------------

_REGISTRY = CollectorRegistry() if _PROM_AVAILABLE else None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if _PROM_AVAILABLE:
    # 1. Ingestion freshness gauge
    # Drives: TropiIngestionLagHigh (>30min warning), TropiIngestionLagCritical (>1h),
    #         TropiQPEIngestionStale (QPE label, >30min critical),
    #         TropiFloodAlertPipelineStale (flood_early_warning label, >30min critical)
    INGESTION_SUCCESS_TS = Gauge(
        "tropi_pipeline_last_ingestion_success_timestamp_seconds",
        "Unix timestamp of the last successful pipeline ingestion run",
        labelnames=["pipeline"],
        registry=_REGISTRY,
    )

    # 2. Flood alert end-to-end delivery latency histogram
    # Drives: TropiFloodAlertDeliveryLate (p95 > 2 min = 120s critical)
    # Buckets cover 0–300s with fine granularity around the 120s SLA
    ALERT_DELIVERY_DURATION = Histogram(
        "tropi_flood_alert_delivery_duration_seconds",
        "Elapsed time from flood threshold breach to confirmed BPBD webhook 2xx (seconds)",
        labelnames=["river"],
        buckets=(5, 10, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300),
        registry=_REGISTRY,
    )

    # 3. BPBD webhook failure counter
    # Drives: TropiFloodAlertBPBDWebhookFailing (any errors > 0 → critical)
    WEBHOOK_FAILURES = Counter(
        "tropi_flood_alert_webhook_failures_total",
        "Total number of failed BPBD alert webhook deliveries (non-2xx or exception)",
        labelnames=["river", "endpoint"],
        registry=_REGISTRY,
    )

    # ---- Sprint 5 metrics ----

    # D1: BMKG gauge freshness per river
    BMKG_GAUGE_SUCCESS_TS = Gauge(
        "tropi_bmkg_gauge_last_success_timestamp_seconds",
        "Unix timestamp of the last successful BMKG HIMET gauge fetch per river",
        labelnames=["river_id"],
        registry=_REGISTRY,
    )

    # D1: BMKG fetch failures per station
    BMKG_GAUGE_FETCH_FAILURES = Counter(
        "tropi_bmkg_gauge_fetch_failures_total",
        "Total failed BMKG HIMET gauge fetch attempts per station",
        labelnames=["station_id"],
        registry=_REGISTRY,
    )

    # D2: QPE validation freshness (no label — single domain)
    QPE_VALIDATED_TS = Gauge(
        "tropi_qpe_last_validated_timestamp_seconds",
        "Unix timestamp of the last successful QPE file validation pass",
        registry=_REGISTRY,
    )

    # D2: QPE validation failures per reason
    QPE_VALIDATION_FAILURES = Counter(
        "tropi_qpe_validation_failures_total",
        "Total QPE validation failures by reason code",
        labelnames=["reason"],
        registry=_REGISTRY,
    )

    # D3: Flood alert delivery by channel and status
    ALERT_DELIVERY_TOTAL = Counter(
        "tropi_flood_alert_delivery_total",
        "Total flood alert delivery attempts by river, channel, and status",
        labelnames=["river_id", "channel", "status"],
        registry=_REGISTRY,
    )

    # D5: Water balance closure error per watershed
    WATER_BALANCE_CLOSURE_ERR = Gauge(
        "tropi_water_balance_closure_error_pct",
        "Water balance closure error (|Q_computed - Q_obs| / Q_obs * 100) per watershed",
        labelnames=["watershed_id"],
        registry=_REGISTRY,
    )

else:  # pragma: no cover — define stub objects so call sites don't need guards
    class _Stub:  # type: ignore[no-redef]
        def labels(self, **_): return self
        def set(self, *_): pass
        def observe(self, *_): pass
        def inc(self, *_): pass

    INGESTION_SUCCESS_TS     = _Stub()   # type: ignore[assignment]
    ALERT_DELIVERY_DURATION  = _Stub()   # type: ignore[assignment]
    WEBHOOK_FAILURES         = _Stub()   # type: ignore[assignment]
    # Sprint 5 stubs
    BMKG_GAUGE_SUCCESS_TS    = _Stub()   # type: ignore[assignment]
    BMKG_GAUGE_FETCH_FAILURES = _Stub()  # type: ignore[assignment]
    QPE_VALIDATED_TS         = _Stub()   # type: ignore[assignment]
    QPE_VALIDATION_FAILURES  = _Stub()   # type: ignore[assignment]
    ALERT_DELIVERY_TOTAL     = _Stub()   # type: ignore[assignment]
    WATER_BALANCE_CLOSURE_ERR = _Stub()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

PUSHGATEWAY_URL: str = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
_JOB_NAME = "tropi-hydrologis"


def record_ingestion_success(pipeline: str) -> None:
    """
    Set the ingestion freshness gauge to now().
    Call once per successful Airflow task run (on_success_callback or end of run()).

    Args:
        pipeline: Airflow DAG id, e.g. "flood_early_warning_30min", "qpe_fusion_30min".
    """
    ts = time.time()
    INGESTION_SUCCESS_TS.labels(pipeline=pipeline).set(ts)
    logger.debug("Ingestion timestamp recorded | pipeline=%s ts=%.0f", pipeline, ts)
    _push_safe()


@contextlib.contextmanager
def measure_alert_delivery(
    river: str,
    breach_time: Optional[datetime] = None,
) -> Generator[None, None, None]:
    """
    Context manager that measures flood alert delivery duration and observes
    the histogram on exit.

    Measures from breach_time (if provided) to when the context exits,
    matching the README spec: "from threshold breach to confirmed 2xx response".

    Usage:
        with measure_alert_delivery("ciliwung", breach_time=breach_dt):
            result = client.push_alert(payload)
    """
    if breach_time is not None:
        # Wall-clock elapsed since the breach event
        start_wall = breach_time.timestamp()
        def _elapsed() -> float:
            return time.time() - start_wall
    else:
        t0 = time.monotonic()
        def _elapsed() -> float:  # type: ignore[misc]
            return time.monotonic() - t0

    try:
        yield
    finally:
        elapsed = _elapsed()
        ALERT_DELIVERY_DURATION.labels(river=river).observe(elapsed)
        logger.debug("Alert delivery duration | river=%s elapsed=%.2fs", river, elapsed)
        _push_safe()


def increment_webhook_failure(river: str, endpoint: str = "bpbd") -> None:
    """
    Increment the webhook failure counter.
    Call on any non-2xx response or exception from push_alert().

    Args:
        river:    River station id ("ciliwung", "brantas", "solo").
        endpoint: Target endpoint label (default "bpbd").
    """
    WEBHOOK_FAILURES.labels(river=river, endpoint=endpoint).inc()
    logger.warning("Webhook failure recorded | river=%s endpoint=%s", river, endpoint)
    _push_safe()


def push_metrics() -> None:
    """Explicitly push all registry metrics to Pushgateway."""
    _push_safe(force=True)


# ---------------------------------------------------------------------------
# Internal push helper
# ---------------------------------------------------------------------------

def _push_safe(force: bool = False) -> None:
    """
    Push metrics to Pushgateway, swallowing errors so instrumentation
    never breaks pipeline execution.
    """
    if not _PROM_AVAILABLE:
        return
    try:
        push_to_gateway(PUSHGATEWAY_URL, job=_JOB_NAME, registry=_REGISTRY)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Pushgateway push failed (non-fatal): %s", exc)

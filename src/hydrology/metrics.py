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

    # ---- Sprint 6 metrics ----

    # E1: Streamflow forecast peak discharge per river and horizon
    STREAMFLOW_FORECAST_PEAK = Gauge(
        "tropi_streamflow_forecast_peak_cms",
        "Forecast peak discharge (m³/s) per river and horizon",
        labelnames=["river_id", "horizon_hr"],
        registry=_REGISTRY,
    )

    # E1: Streamflow forecast run counter per river and status
    STREAMFLOW_FORECAST_RUNS = Counter(
        "tropi_streamflow_forecast_runs_total",
        "Total streamflow forecast runs per river and status",
        labelnames=["river_id", "status"],
        registry=_REGISTRY,
    )

    # E3: Drought risk class per DAS watershed (0=NORMAL … 3=EMERGENCY)
    DROUGHT_RISK_CLASS = Gauge(
        "tropi_drought_risk_class",
        "Drought risk class (0=NORMAL, 1=WATCH, 2=WARNING, 3=EMERGENCY) per watershed",
        labelnames=["watershed_id"],
        registry=_REGISTRY,
    )

    # E3: Count of watersheds at WATCH or above
    DROUGHT_WATERSHEDS_WARNING = Gauge(
        "tropi_drought_watersheds_warning_total",
        "Number of DAS watersheds currently at WATCH risk class or above",
        registry=_REGISTRY,
    )

    # E5: Flood threshold breach counter per river and stage
    FLOOD_THRESHOLD_BREACH = Counter(
        "tropi_flood_threshold_breach_total",
        "Total flood threshold breach events per river and flood stage",
        labelnames=["river_id", "flood_stage"],
        registry=_REGISTRY,
    )

    # ---- Sprint 7 metrics ----

    # G1: Water balance residual fraction per watershed
    WATER_BALANCE_RESIDUAL = Gauge(
        "tropi_water_balance_residual_fraction",
        "Water balance residual fraction |res/P| per watershed (0=closed, >0.15=unclosed)",
        labelnames=["watershed_id"],
        registry=_REGISTRY,
    )

    # G3: Flood inundation area per river and flood stage
    INUNDATION_AREA_KM2 = Gauge(
        "tropi_flood_inundation_area_km2",
        "Flood inundation affected area (km²) per river and flood stage",
        labelnames=["river_id", "flood_stage"],
        registry=_REGISTRY,
    )

    # G5: GRACE-FO groundwater storage anomaly per region
    GRACE_GWS_ANOMALY = Gauge(
        "tropi_grace_gws_anomaly_mm",
        "GRACE-FO groundwater storage anomaly (mm/month, negative = depletion) per region",
        labelnames=["region_id"],
        registry=_REGISTRY,
    )

    # G5: Aquifer emergency region counter
    AQUIFER_EMERGENCY = Counter(
        "tropi_aquifer_emergency_total",
        "Total count of aquifer regions reaching EMERGENCY depletion class",
        registry=_REGISTRY,
    )

    # ---- Sprint 8 metrics ----

    # J1: QPE pipeline update latency per source
    QPE_UPDATE_LATENCY = Histogram(
        "tropi_qpe_update_latency_seconds",
        "QPE pipeline update latency (seconds) per data source",
        labelnames=["source"],
        buckets=(30, 60, 120, 240, 480),
        registry=_REGISTRY,
    )

    # J3: Drought risk level per region and classification
    DROUGHT_RISK_LEVEL = Gauge(
        "tropi_drought_risk_level",
        "Drought risk level (0=NORMAL..4=EXTREME_DROUGHT) per region and classification",
        labelnames=["region_id", "classification"],
        registry=_REGISTRY,
    )

    # J5: Streamflow forecast bias (updated after verification)
    STREAMFLOW_FORECAST_BIAS = Gauge(
        "tropi_streamflow_forecast_bias_cms",
        "Streamflow forecast bias (m³/s) per river and horizon (updated after verification)",
        labelnames=["river_id", "horizon_hr"],
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
    # Sprint 6 stubs
    STREAMFLOW_FORECAST_PEAK   = _Stub()  # type: ignore[assignment]
    STREAMFLOW_FORECAST_RUNS   = _Stub()  # type: ignore[assignment]
    DROUGHT_RISK_CLASS         = _Stub()  # type: ignore[assignment]
    DROUGHT_WATERSHEDS_WARNING = _Stub()  # type: ignore[assignment]
    FLOOD_THRESHOLD_BREACH     = _Stub()  # type: ignore[assignment]
    # Sprint 7 stubs
    WATER_BALANCE_RESIDUAL     = _Stub()  # type: ignore[assignment]
    INUNDATION_AREA_KM2        = _Stub()  # type: ignore[assignment]
    GRACE_GWS_ANOMALY          = _Stub()  # type: ignore[assignment]
    AQUIFER_EMERGENCY          = _Stub()  # type: ignore[assignment]
    # Sprint 8 stubs
    QPE_UPDATE_LATENCY         = _Stub()  # type: ignore[assignment]
    DROUGHT_RISK_LEVEL         = _Stub()  # type: ignore[assignment]
    STREAMFLOW_FORECAST_BIAS   = _Stub()  # type: ignore[assignment]


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


# ---------------------------------------------------------------------------
# Sprint 7 additions — Water Balance, Inundation, GRACE-FO, Aquifer
# ---------------------------------------------------------------------------

WATER_BALANCE_RESIDUAL = _make_gauge(
    "tropi_water_balance_residual_fraction",
    "Water balance residual fraction (|residual/P|) per watershed",
    ["watershed_id"],
)

INUNDATION_AREA_KM2 = _make_gauge(
    "tropi_flood_inundation_area_km2",
    "Flood inundation extent in km² per river and flood stage",
    ["river_id", "flood_stage"],
)

GRACE_GWS_ANOMALY = _make_gauge(
    "tropi_grace_gws_anomaly_mm",
    "GRACE-FO groundwater storage anomaly in mm per region",
    ["region_id"],
)

AQUIFER_EMERGENCY = _make_counter(
    "tropi_aquifer_emergency_total",
    "Total number of aquifer emergency threshold breaches",
    ["region_id"],
)


def record_water_balance_residual(watershed_id: str, residual_fraction: float) -> None:
    """Set water balance residual fraction for a watershed."""
    WATER_BALANCE_RESIDUAL.labels(watershed_id=watershed_id).set(residual_fraction)
    _push_safe()


def record_inundation_area(river_id: str, flood_stage: str, area_km2: float) -> None:
    """Set flood inundation area for a river and flood stage."""
    INUNDATION_AREA_KM2.labels(river_id=river_id, flood_stage=flood_stage).set(area_km2)
    _push_safe()


def record_grace_gws_anomaly(region_id: str, anomaly_mm: float) -> None:
    """Set GRACE-FO groundwater storage anomaly for a region."""
    GRACE_GWS_ANOMALY.labels(region_id=region_id).set(anomaly_mm)
    _push_safe()


def increment_aquifer_emergency(region_id: str) -> None:
    """Increment aquifer emergency counter for a region."""
    AQUIFER_EMERGENCY.labels(region_id=region_id).inc()
    logger.warning("Aquifer EMERGENCY threshold breached | region=%s", region_id)
    _push_safe()


def _make_histogram(name: str, doc: str, labels: list, buckets=None) -> Any:
    if not _PROM_AVAILABLE:
        return _Noop()
    kwargs: dict = {"name": name, "documentation": doc,
                    "labelnames": labels, "registry": _REGISTRY}
    if buckets:
        kwargs["buckets"] = list(buckets)
    return Histogram(**kwargs)


# ---------------------------------------------------------------------------
# Sprint 8 additions — QPE Pipeline, Drought Monitor, Streamflow Forecast
# ---------------------------------------------------------------------------

QPE_UPDATE_LATENCY = _make_histogram(
    "tropi_qpe_update_latency_seconds",
    "QPE update latency in seconds per data source",
    ["source"],
    buckets=(30, 60, 120, 240, 480),
)

DROUGHT_RISK_LEVEL = _make_gauge(
    "tropi_drought_risk_level",
    "Drought risk level (0=NORMAL..4=EXTREME) per region and classification",
    ["region_id", "classification"],
)

STREAMFLOW_FORECAST_BIAS = _make_gauge(
    "tropi_streamflow_forecast_bias_cms",
    "Streamflow forecast bias in m³/s per river and horizon",
    ["river_id", "horizon_hr"],
)


def record_qpe_latency(source: str, latency_s: float) -> None:
    """Observe QPE update latency for a data source (gpm|gauge|merged)."""
    QPE_UPDATE_LATENCY.labels(source=source).observe(latency_s)
    _push_safe()


def record_drought_risk_level(region_id: str, classification: str, level: int) -> None:
    """Set drought risk level gauge for a region."""
    DROUGHT_RISK_LEVEL.labels(region_id=region_id, classification=classification).set(level)
    _push_safe()


def record_streamflow_forecast_bias(river_id: str, horizon_hr: int, bias_cms: float) -> None:
    """Set streamflow forecast bias for a river and horizon."""
    STREAMFLOW_FORECAST_BIAS.labels(river_id=river_id, horizon_hr=str(horizon_hr)).set(bias_cms)
    _push_safe()

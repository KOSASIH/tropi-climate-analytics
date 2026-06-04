"""
Flood Alert Notifier — HYDROLOGIS Sprint 5 | Deliverable 3

Delivers emergency flood alerts via two channels:
  1. HTTP POST to BMKG webhook (env BMKG_WEBHOOK_URL, timeout 8s, retry 3×)
  2. JSON artifact to workspace/output/alerts/ (consumed by VISUALIA)

Prometheus (all on shared _REGISTRY):
  tropi_flood_alert_delivery_total{river_id, channel, status}   Counter
  tropi_flood_alert_delivery_duration_seconds{river}             Histogram (from metrics.py)
  tropi_flood_alert_webhook_failures_total{river, endpoint}      Counter (from metrics.py)

Dependencies (all from metrics._REGISTRY):
  measure_alert_delivery(river, breach_time)
  increment_webhook_failure(river, endpoint)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("WORKSPACE_ROOT", "workspace")
BMKG_WEBHOOK_URL = os.getenv("BMKG_WEBHOOK_URL", "https://webhook.bmkg.go.id/api/flood-alert")
BMKG_WEBHOOK_TIMEOUT_S = 8
BMKG_WEBHOOK_RETRIES   = 3

# ---------------------------------------------------------------------------
# Prometheus — additional Sprint 5 counter (delivery total by channel/status)
# ---------------------------------------------------------------------------

try:
    from src.hydrology.metrics import (
        ALERT_DELIVERY_TOTAL,
        _REGISTRY,
        increment_webhook_failure,
        measure_alert_delivery,
        push_metrics,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    logger.warning("metrics module unavailable — Prometheus instrumentation disabled")

    # Stub callables so class remains functional
    def measure_alert_delivery(river, breach_time=None):  # type: ignore[misc]
        import contextlib
        return contextlib.nullcontext()

    def increment_webhook_failure(river, endpoint="bpbd"):  # type: ignore[misc]
        pass

    def push_metrics():  # type: ignore[misc]
        pass

    class _AlertDeliveryTotalStub:
        def labels(self, **_): return self
        def inc(self): pass

    ALERT_DELIVERY_TOTAL = _AlertDeliveryTotalStub()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class FloodAlertPayload:
    """Immutable alert payload passed to send_flood_alert."""

    def __init__(
        self,
        river_id: str,
        flood_stage: str,
        peak_q_m3s: float,
        affected_villages: list[str],
        forecast_horizon_hours: int,
        breach_time: Optional[datetime] = None,
    ) -> None:
        self.river_id              = river_id
        self.flood_stage           = flood_stage
        self.peak_q_m3s            = peak_q_m3s
        self.affected_villages     = affected_villages
        self.forecast_horizon_hours = forecast_horizon_hours
        self.breach_time           = breach_time or datetime.now(timezone.utc)
        self.alert_id              = f"FLOOD-{river_id.upper()}-{int(self.breach_time.timestamp())}"

    def to_dict(self) -> dict:
        return {
            "alert_id":               self.alert_id,
            "river_id":               self.river_id,
            "flood_stage":            self.flood_stage,
            "peak_discharge_m3s":     self.peak_q_m3s,
            "affected_villages":      self.affected_villages,
            "forecast_horizon_hours": self.forecast_horizon_hours,
            "breach_time_utc":        self.breach_time.isoformat(),
            "issued_at_utc":          datetime.now(timezone.utc).isoformat(),
            "source":                 "HYDROLOGIS/flood_alert_notifier",
        }


class FloodAlertDeliveryResult:
    def __init__(
        self,
        alert_id: str,
        bmkg_webhook_ok: bool,
        visualia_path: Optional[str],
        error: Optional[str] = None,
    ) -> None:
        self.alert_id        = alert_id
        self.bmkg_webhook_ok = bmkg_webhook_ok
        self.visualia_path   = visualia_path
        self.error           = error
        self.success         = bmkg_webhook_ok and visualia_path is not None


# ---------------------------------------------------------------------------
# FloodAlertNotifier
# ---------------------------------------------------------------------------

class FloodAlertNotifier:
    """
    Delivers flood alert via BMKG webhook + VISUALIA JSON artifact.

    Usage:
        notifier = FloodAlertNotifier()
        result = notifier.send_flood_alert(
            river_id="ciliwung",
            flood_stage="EMERGENCY",
            peak_q_m3s=1420.0,
            affected_villages=["Kampung Melayu", "Bukit Duri", "Pengadegan"],
            forecast_horizon_hours=6,
        )
    """

    def __init__(self, workspace: Optional[str] = None) -> None:
        self.workspace = workspace or WORKSPACE
        self.alerts_dir = os.path.join(self.workspace, "output", "alerts")
        os.makedirs(self.alerts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_flood_alert(
        self,
        river_id: str,
        flood_stage: str,
        peak_q_m3s: float,
        affected_villages: list[str],
        forecast_horizon_hours: int,
        breach_time: Optional[datetime] = None,
    ) -> FloodAlertDeliveryResult:
        """
        Deliver flood alert to both channels.

        Args:
            river_id:               "ciliwung" | "brantas" | "solo"
            flood_stage:            "WARNING" | "EMERGENCY"
            peak_q_m3s:             Peak discharge in m³/s
            affected_villages:      List of affected village names
            forecast_horizon_hours: Alert horizon (6 / 12 / 24)
            breach_time:            Time of threshold breach (default: now)

        Returns:
            FloodAlertDeliveryResult with per-channel status.
        """
        payload = FloodAlertPayload(
            river_id=river_id,
            flood_stage=flood_stage,
            peak_q_m3s=peak_q_m3s,
            affected_villages=affected_villages,
            forecast_horizon_hours=forecast_horizon_hours,
            breach_time=breach_time,
        )

        logger.warning(
            "Flood alert | id=%s river=%s stage=%s peak_q=%.1f m³/s villages=%d horizon=%dh",
            payload.alert_id, river_id, flood_stage, peak_q_m3s,
            len(affected_villages), forecast_horizon_hours,
        )

        # Channel 1: BMKG webhook (timed, retried)
        with measure_alert_delivery(river_id, breach_time=payload.breach_time):
            bmkg_ok = self._send_bmkg_webhook(payload)

        # Channel 2: VISUALIA JSON artifact
        visualia_path = self._write_visualia_payload(payload)

        result = FloodAlertDeliveryResult(
            alert_id=payload.alert_id,
            bmkg_webhook_ok=bmkg_ok,
            visualia_path=visualia_path,
        )

        push_metrics()
        return result

    # ------------------------------------------------------------------
    # Channel 1 — BMKG webhook
    # ------------------------------------------------------------------

    def _send_bmkg_webhook(self, payload: FloodAlertPayload) -> bool:
        """
        POST alert to BMKG_WEBHOOK_URL with up to BMKG_WEBHOOK_RETRIES retries.
        On any failure: increment_webhook_failure and continue to next retry.
        Returns True if any attempt succeeded (2xx), False otherwise.
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library not available — BMKG webhook skipped")
            increment_webhook_failure(payload.river_id, endpoint="bmkg_webhook")
            ALERT_DELIVERY_TOTAL.labels(
                river_id=payload.river_id, channel="bmkg_webhook", status="error"
            ).inc()
            return False

        body = payload.to_dict()

        for attempt in range(1, BMKG_WEBHOOK_RETRIES + 1):
            try:
                resp = requests.post(
                    BMKG_WEBHOOK_URL,
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=BMKG_WEBHOOK_TIMEOUT_S,
                )
                if resp.status_code < 300:
                    logger.info(
                        "BMKG webhook delivered | alert=%s status=%d attempt=%d",
                        payload.alert_id, resp.status_code, attempt,
                    )
                    ALERT_DELIVERY_TOTAL.labels(
                        river_id=payload.river_id, channel="bmkg_webhook", status="success"
                    ).inc()
                    return True

                logger.warning(
                    "BMKG webhook non-2xx | alert=%s status=%d attempt=%d",
                    payload.alert_id, resp.status_code, attempt,
                )
                increment_webhook_failure(payload.river_id, endpoint="bmkg_webhook")

            except requests.Timeout:
                logger.warning(
                    "BMKG webhook timeout | alert=%s timeout=%ds attempt=%d",
                    payload.alert_id, BMKG_WEBHOOK_TIMEOUT_S, attempt,
                )
                increment_webhook_failure(payload.river_id, endpoint="bmkg_webhook")

            except Exception as exc:
                logger.error(
                    "BMKG webhook exception | alert=%s attempt=%d error=%s",
                    payload.alert_id, attempt, exc,
                )
                increment_webhook_failure(payload.river_id, endpoint="bmkg_webhook")

            if attempt < BMKG_WEBHOOK_RETRIES:
                time.sleep(2 ** attempt)  # Exponential back-off: 2s, 4s

        ALERT_DELIVERY_TOTAL.labels(
            river_id=payload.river_id, channel="bmkg_webhook", status="failed"
        ).inc()
        logger.error(
            "BMKG webhook delivery failed after %d attempts | alert=%s",
            BMKG_WEBHOOK_RETRIES, payload.alert_id,
        )
        return False

    # ------------------------------------------------------------------
    # Channel 2 — VISUALIA JSON artifact
    # ------------------------------------------------------------------

    def _write_visualia_payload(self, payload: FloodAlertPayload) -> Optional[str]:
        """
        Write alert JSON to workspace/output/alerts/flood_{river_id}_{timestamp}.json.
        Returns file path on success, None on failure.
        """
        ts_str    = payload.breach_time.strftime("%Y%m%d_%H%M%S")
        fname     = f"flood_{payload.river_id}_{ts_str}.json"
        out_path  = os.path.join(self.alerts_dir, fname)

        # Also write a 'latest' symlink-equivalent for VISUALIA polling
        latest_path = os.path.join(self.alerts_dir, f"flood_{payload.river_id}_latest.json")

        visualia_body = {
            **payload.to_dict(),
            # VISUALIA-specific extensions
            "display": {
                "color":       "#FF0000" if payload.flood_stage == "EMERGENCY" else "#FF8C00",
                "severity":    3 if payload.flood_stage == "EMERGENCY" else 2,
                "label":       f"🚨 BANJIR {payload.flood_stage} — {payload.river_id.upper()}",
                "description": (
                    f"Debit puncak {payload.peak_q_m3s:.0f} m³/s diprakirakan dalam "
                    f"{payload.forecast_horizon_hours} jam ke depan. "
                    f"Desa terdampak: {', '.join(payload.affected_villages[:5])}"
                    + (f" (+{len(payload.affected_villages)-5} lainnya)" if len(payload.affected_villages) > 5 else "")
                ),
            },
        }

        try:
            with open(out_path, "w") as fh:
                json.dump(visualia_body, fh, indent=2, ensure_ascii=False)

            # Overwrite latest
            with open(latest_path, "w") as fh:
                json.dump(visualia_body, fh, indent=2, ensure_ascii=False)

            logger.info(
                "VISUALIA alert payload written | alert=%s path=%s",
                payload.alert_id, out_path,
            )
            ALERT_DELIVERY_TOTAL.labels(
                river_id=payload.river_id, channel="visualia_json", status="success"
            ).inc()
            return out_path

        except Exception as exc:
            logger.error(
                "VISUALIA payload write failed | alert=%s error=%s",
                payload.alert_id, exc,
            )
            ALERT_DELIVERY_TOTAL.labels(
                river_id=payload.river_id, channel="visualia_json", status="error"
            ).inc()
            return None

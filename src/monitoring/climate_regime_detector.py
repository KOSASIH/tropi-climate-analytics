"""
Climate Regime Detector — ANALYTICA Sprint 9 Q4
Module: src/monitoring/climate_regime_detector.py

Class: ClimateRegimeDetector
  fetch_oni(lookback_months=3) → ONIRecord
  fetch_dmi(lookback_months=3) → DMIRecord
  classify_regime(oni, dmi) → ClimateRegime
  get_bias_correction_factors(regime, model_id) → BiasFactors
  update_regime_context() → RegimeContext

Data sources:
  ONI: NOAA CPC https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
  DMI: JAMSTEC (monthly)
  Fallback: synthetic PDO proxy

Regimes: El Niño, La Niña, Neutral, Positive/Negative IOD, SEVERE_DRY, SEVERE_WET
Bias factors applied by EnsembleForecaster as multiplicative post-processing.
HYDROLOGIS integration: regime_context.json read by seasonal_water_forecast.py.

Output:
  workspace/output/climate_indices/regime_context.json (rolling sidecar)
  workspace/output/climate_indices/oni_dmi_{YYYYMM}.json
Prometheus: ONI_VALUE Gauge, DMI_VALUE Gauge, CLIMATE_REGIME_ACTIVE{regime} Gauge
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

CLIMATE_INDEX_DIR = Path("workspace/data/climate_indices")
OUTPUT_DIR        = Path("workspace/output/climate_indices")
REGIME_CONTEXT_F  = OUTPUT_DIR / "regime_context.json"

ONI_URL   = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
TIMEOUT_S = 15

# Thresholds
ONI_ELNINO   =  0.5
ONI_LANINA   = -0.5
DMI_POS      =  0.4
DMI_NEG      = -0.4
CONSEC_MONTHS = 3

# Bias correction table
BIAS_TABLE = {
    "EL_NINO":        {"precipitation_multiplier": 0.72, "temp_offset":  0.4, "flood_alert": False, "drought_alert": False},
    "LA_NINA":        {"precipitation_multiplier": 1.35, "temp_offset": -0.2, "flood_alert": False, "drought_alert": False},
    "NEUTRAL":        {"precipitation_multiplier": 1.00, "temp_offset":  0.0, "flood_alert": False, "drought_alert": False},
    "POSITIVE_IOD":   {"precipitation_multiplier": 0.85, "temp_offset":  0.1, "flood_alert": False, "drought_alert": False},
    "NEGATIVE_IOD":   {"precipitation_multiplier": 1.20, "temp_offset": -0.1, "flood_alert": False, "drought_alert": False},
    "SEVERE_DRY":     {"precipitation_multiplier": 0.55, "temp_offset":  0.6, "flood_alert": False, "drought_alert": True},
    "SEVERE_WET":     {"precipitation_multiplier": 1.60, "temp_offset": -0.3, "flood_alert": True,  "drought_alert": False},
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ONIRecord:
    values:        List[float]     # most recent N monthly ONI values (°C)
    months:        List[str]       # YYYY-MM labels
    latest_value:  float
    source:        str             # "noaa_cpc" | "synthetic"
    fetched_at:    str

    def to_dict(self) -> dict:
        return {
            "values":       [round(v, 2) for v in self.values],
            "months":       self.months,
            "latest_value": round(self.latest_value, 2),
            "source":       self.source,
            "fetched_at":   self.fetched_at,
        }


@dataclass
class DMIRecord:
    values:        List[float]
    months:        List[str]
    latest_value:  float
    source:        str             # "jamstec" | "synthetic"
    fetched_at:    str

    def to_dict(self) -> dict:
        return {
            "values":       [round(v, 2) for v in self.values],
            "months":       self.months,
            "latest_value": round(self.latest_value, 2),
            "source":       self.source,
            "fetched_at":   self.fetched_at,
        }


@dataclass
class BiasFactors:
    model_id:                 str
    regime:                   str
    precipitation_multiplier: float
    temp_offset:              float
    flood_alert:              bool
    drought_alert:            bool

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class RegimeContext:
    regime:           str
    oni_value:        float
    dmi_value:        float
    regime_start_date: str
    months_active:    int
    bias_factors:     Dict[str, dict]     # keyed by model_id
    flood_alert:      bool
    drought_alert:    bool
    next_update_date: str
    data_sources:     List[str]
    updated_at:       str = ""

    def to_dict(self) -> dict:
        return {
            "regime":             self.regime,
            "oni_value":          round(self.oni_value, 2),
            "dmi_value":          round(self.dmi_value, 2),
            "regime_start_date":  self.regime_start_date,
            "months_active":      self.months_active,
            "bias_factors":       self.bias_factors,
            "flood_alert":        self.flood_alert,
            "drought_alert":      self.drought_alert,
            "next_update_date":   self.next_update_date,
            "data_sources":       self.data_sources,
            "updated_at":         self.updated_at,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClimateRegimeDetector:
    """
    ENSO / IOD climate regime detection and bias correction for ANALYTICA models.
    regime_context.json is the shared sidecar read by EnsembleForecaster and HYDROLOGIS.
    """

    def fetch_oni(self, lookback_months: int = 3) -> ONIRecord:
        """
        Fetch the Oceanic Niño Index from NOAA CPC.
        Falls back to synthetic PDO proxy on network failure.

        Returns:
            ONIRecord with the latest N monthly values.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        try:
            values, months = self._fetch_noaa_oni(lookback_months)
            src = "noaa_cpc"
        except Exception as exc:
            logger.warning("NOAA ONI fetch failed (%s) — using synthetic fallback", exc)
            values, months = self._synthetic_oni(lookback_months)
            src = "synthetic"

        record = ONIRecord(
            values=values, months=months, latest_value=values[-1],
            source=src, fetched_at=now_str,
        )
        self._cache_index("oni", record.to_dict())
        return record

    def fetch_dmi(self, lookback_months: int = 3) -> DMIRecord:
        """
        Fetch the Dipole Mode Index (Indian Ocean Dipole) from JAMSTEC.
        Falls back to synthetic if unavailable.

        Returns:
            DMIRecord with the latest N monthly values.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        try:
            values, months = self._fetch_jamstec_dmi(lookback_months)
            src = "jamstec"
        except Exception as exc:
            logger.warning("JAMSTEC DMI fetch failed (%s) — using synthetic fallback", exc)
            values, months = self._synthetic_dmi(lookback_months)
            src = "synthetic"

        record = DMIRecord(
            values=values, months=months, latest_value=values[-1],
            source=src, fetched_at=now_str,
        )
        self._cache_index("dmi", record.to_dict())
        return record

    def classify_regime(self, oni: ONIRecord, dmi: DMIRecord) -> str:
        """
        Classify the current ENSO/IOD regime from ONI and DMI records.

        Rules:
          El Niño:  ONI ≥ +0.5 for ≥ 3 consecutive months
          La Niña:  ONI ≤ -0.5 for ≥ 3 consecutive months
          Positive IOD: DMI ≥ +0.4
          Negative IOD: DMI ≤ -0.4
          Neutral:  otherwise
          Compounds take priority: El Niño + Pos IOD → SEVERE_DRY; La Niña + Neg IOD → SEVERE_WET
        """
        oni_vals = oni.values[-CONSEC_MONTHS:]
        dmi_latest = dmi.latest_value

        el_nino = len(oni_vals) >= CONSEC_MONTHS and all(v >= ONI_ELNINO for v in oni_vals)
        la_nina = len(oni_vals) >= CONSEC_MONTHS and all(v <= ONI_LANINA for v in oni_vals)
        pos_iod = dmi_latest >= DMI_POS
        neg_iod = dmi_latest <= DMI_NEG

        # Compound regimes first (highest priority)
        if el_nino and pos_iod:
            return "SEVERE_DRY"
        if la_nina and neg_iod:
            return "SEVERE_WET"

        # Primary ENSO signal
        if el_nino:
            return "EL_NINO"
        if la_nina:
            return "LA_NINA"

        # IOD only
        if pos_iod:
            return "POSITIVE_IOD"
        if neg_iod:
            return "NEGATIVE_IOD"

        return "NEUTRAL"

    def get_bias_correction_factors(self, regime: str, model_id: str) -> BiasFactors:
        """
        Return the regime-specific bias correction factors for a given model.

        Args:
            regime:   Climate regime string (e.g., 'EL_NINO', 'SEVERE_DRY').
            model_id: ANALYTICA model identifier.

        Returns:
            BiasFactors for multiplicative post-processing in EnsembleForecaster.
        """
        table = BIAS_TABLE.get(regime, BIAS_TABLE["NEUTRAL"])
        return BiasFactors(
            model_id=model_id,
            regime=regime,
            precipitation_multiplier=table["precipitation_multiplier"],
            temp_offset=table["temp_offset"],
            flood_alert=table["flood_alert"],
            drought_alert=table["drought_alert"],
        )

    def update_regime_context(self, run_date: Optional[date] = None) -> RegimeContext:
        """
        Full update cycle: fetch → classify → compute bias factors → write sidecar.
        Called monthly by model_monitoring_dag or on-demand.

        Returns:
            RegimeContext dict written to workspace/output/climate_indices/regime_context.json.
        """
        run_date = run_date or date.today()
        yyyymm   = run_date.strftime("%Y%m")

        oni = self.fetch_oni(lookback_months=CONSEC_MONTHS)
        dmi = self.fetch_dmi(lookback_months=CONSEC_MONTHS)

        regime = self.classify_regime(oni, dmi)
        table  = BIAS_TABLE.get(regime, BIAS_TABLE["NEUTRAL"])

        # Compute bias factors for all 4 ANALYTICA models
        model_ids = ["xgb_precipitation", "prophet_seasonal", "lstm_streamflow", "cnn_landcover"]
        bias_dict = {mid: self.get_bias_correction_factors(regime, mid).to_dict() for mid in model_ids}

        # Estimate months_active: count consecutive months matching current classification
        months_active = self._count_consecutive_months(oni.values, dmi.latest_value, regime)

        next_update = (run_date.replace(day=1) + timedelta(days=32)).replace(day=1)

        ctx = RegimeContext(
            regime=regime,
            oni_value=oni.latest_value,
            dmi_value=dmi.latest_value,
            regime_start_date=(run_date - timedelta(days=30 * months_active)).isoformat(),
            months_active=months_active,
            bias_factors=bias_dict,
            flood_alert=table["flood_alert"],
            drought_alert=table["drought_alert"],
            next_update_date=next_update.isoformat(),
            data_sources=[oni.source, dmi.source],
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REGIME_CONTEXT_F.write_text(json.dumps(ctx.to_dict(), indent=2, default=str))
        logger.info("Regime context updated: %s (ONI=%.2f DMI=%.2f flood=%s drought=%s)",
                    regime, oni.latest_value, dmi.latest_value, ctx.flood_alert, ctx.drought_alert)

        # Write monthly oni_dmi archive
        archive = OUTPUT_DIR / f"oni_dmi_{yyyymm}.json"
        archive.write_text(json.dumps({
            "run_date": run_date.isoformat(),
            "oni":      oni.to_dict(),
            "dmi":      dmi.to_dict(),
            "regime":   regime,
        }, indent=2, default=str))

        # Emit Prometheus
        self._emit_metrics(oni.latest_value, dmi.latest_value, regime)

        return ctx

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_noaa_oni(n_months: int) -> tuple[list[float], list[str]]:
        """Parse NOAA CPC ONI ASCII table. Returns last n_months values."""
        req = urllib.request.Request(ONI_URL, headers={"User-Agent": "ANALYTICA/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            text = resp.read().decode("utf-8", errors="ignore")

        lines  = [l for l in text.splitlines() if l.strip() and not l.startswith("SEAS")]
        records = []
        for line in lines[-36:]:   # last 3 years
            parts = line.split()
            if len(parts) >= 3:
                try:
                    year  = int(parts[0])
                    month = int(parts[1]) if parts[1].isdigit() else records[-1][1] + 1
                    value = float(parts[-1])   # ANOM column
                    records.append((year, month, value))
                except ValueError:
                    continue

        records  = records[-n_months:]
        values   = [r[2] for r in records]
        months   = [f"{r[0]}-{r[1]:02d}" for r in records]
        return values, months

    @staticmethod
    def _fetch_jamstec_dmi(n_months: int) -> tuple[list[float], list[str]]:
        """Fetch JAMSTEC DMI monthly data."""
        # JAMSTEC DMI endpoint (public, monthly updates)
        dmi_url = "https://www.jamstec.go.jp/aplinfo/sintexf/iod/dmi.monthly.txt"
        req     = urllib.request.Request(dmi_url, headers={"User-Agent": "ANALYTICA/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            text = resp.read().decode("utf-8", errors="ignore")

        records = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    year   = int(parts[0])
                    month  = int(parts[1])
                    value  = float(parts[2])
                    records.append((year, month, value))
                except ValueError:
                    continue

        records = records[-n_months:]
        values  = [r[2] for r in records]
        months  = [f"{r[0]}-{r[1]:02d}" for r in records]
        return values, months

    @staticmethod
    def _synthetic_oni(n_months: int) -> tuple[list[float], list[str]]:
        """Generate synthetic ONI values from PDO proxy (fallback)."""
        rng    = np.random.default_rng(seed=int(date.today().strftime("%Y%m")))
        base   = rng.normal(0, 0.4, n_months)
        today  = date.today()
        months = [(today.replace(day=1) - timedelta(days=30 * i)) for i in range(n_months)]
        months = sorted([m.strftime("%Y-%m") for m in months])
        return list(np.round(base, 2)), months

    @staticmethod
    def _synthetic_dmi(n_months: int) -> tuple[list[float], list[str]]:
        rng    = np.random.default_rng(seed=int(date.today().strftime("%Y%m")) + 1)
        base   = rng.normal(0, 0.3, n_months)
        today  = date.today()
        months = [(today.replace(day=1) - timedelta(days=30 * i)) for i in range(n_months)]
        months = sorted([m.strftime("%Y-%m") for m in months])
        return list(np.round(base, 2)), months

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_consecutive_months(oni_values: List[float], dmi_latest: float, regime: str) -> int:
        """Count consecutive months matching the classified regime."""
        count = 0
        for v in reversed(oni_values):
            if regime in ("EL_NINO", "SEVERE_DRY") and v >= ONI_ELNINO:
                count += 1
            elif regime in ("LA_NINA", "SEVERE_WET") and v <= ONI_LANINA:
                count += 1
            elif regime == "NEUTRAL" and ONI_LANINA < v < ONI_ELNINO:
                count += 1
            else:
                break
        return max(count, 1)

    @staticmethod
    def _cache_index(index_type: str, data: dict) -> None:
        CLIMATE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        (CLIMATE_INDEX_DIR / f"{index_type}_latest.json").write_text(
            json.dumps(data, indent=2, default=str)
        )

    @staticmethod
    def _emit_metrics(oni: float, dmi: float, regime: str) -> None:
        try:
            from src.data.metrics import ONI_VALUE, DMI_VALUE, CLIMATE_REGIME_ACTIVE
            ONI_VALUE.set(oni)
            DMI_VALUE.set(dmi)
            for r in BIAS_TABLE:
                CLIMATE_REGIME_ACTIVE.labels(regime=r).set(1.0 if r == regime else 0.0)
        except ImportError:
            pass

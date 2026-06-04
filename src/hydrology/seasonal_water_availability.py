"""
Seasonal Water Availability Aggregator — province-level advisory for KLHK.

Combines SeasonalWaterAvailabilityPipeline basin-level 3-month outlooks into
38-province administrative summaries with three-tier advisory status:
  SURPLUS  — WAI > +0.5σ  : adequate or above-average water availability
  NORMAL   — WAI ∈ [-0.5, +0.5]σ : within climatological range
  DEFICIT  — WAI < -0.5σ  : below-average; irrigation scheduling advisories issued

Output (JSON + optional PDF brief for KLHK):
  workspace/output/seasonal/water_availability_advisory_YYYYMM.json
  Includes: province advisory, WAI score, planting recommendation, dominant ENSO phase

Triggered monthly by Airflow DAG seasonal_monthly (H+1 after grace_monthly).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

# Internal packages — available after Sprint 1 __init__.py exports
from src.hydrology.seasonal import (
    SeasonalWaterAvailabilityPipeline,
    WaterAvailabilityClass,
    PlantingRecommendation,
    ENSOState,
    SeasonalOutlook,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Province → Basin mapping (primary contributing basin per province)
# Multiple basins can contribute; we aggregate by area-weighted WAI
PROVINCE_BASIN_MAP: dict[str, list[str]] = {
    "11": ["krueng_aceh", "jambo_aye"],
    "12": ["deli_serdang", "asahan"],
    "13": ["batang_hari", "kampar"],
    "14": ["kampar", "siak"],
    "15": ["batang_hari"],
    "16": ["musi", "ogan"],
    "17": ["ketahun", "bengkulu"],
    "18": ["way_seputih", "way_tulang_bawang"],
    "19": ["bangka", "belitung"],
    "21": ["bintan", "natuna"],
    "31": ["ciliwung"],
    "32": ["citarum", "cisadane", "ciliwung"],
    "33": ["bengawan_solo", "progo", "serayu"],
    "34": ["progo", "opak"],
    "35": ["brantas", "bengawan_solo"],
    "36": ["ciujung", "cisadane"],
    "51": ["ayung", "unda"],
    "52": ["dodokan", "moyo"],
    "53": ["noelmina", "benain"],
    "61": ["kapuas", "landak"],
    "62": ["barito", "kahayan"],
    "63": ["barito", "martapura"],
    "64": ["mahakam", "berau"],
    "65": ["kayan", "sesayap"],
    "71": ["tondano", "ranoyapo"],
    "72": ["palu", "lariang"],
    "73": ["saddang", "jeneberang"],
    "74": ["konaweha", "lasolo"],
    "75": ["limboto", "bone"],
    "76": ["mamasa", "mapilli"],
    "81": ["wai_apu", "tala"],
    "82": ["kao", "tobelo"],
    "91": ["digul", "serayu_wb"],
    "92": ["memberamo", "mamberamo"],
    "93": ["digul_selatan", "mappi"],
    "94": ["memberamo_tengah", "taritatu"],
    "95": ["baliem", "eilanden"],
    "96": ["sorong_barat", "teminabuan"],
}

# WAI thresholds for three-tier advisory
WAI_SURPLUS_THRESHOLD =  0.5   # WAI > +0.5 → SURPLUS
WAI_DEFICIT_THRESHOLD = -0.5   # WAI < -0.5 → DEFICIT

# Planting season months (primary = Oct-Mar, secondary = Apr-Sep) — Kementan calendar
PRIMARY_PLANTING_MONTHS   = {10, 11, 12, 1, 2, 3}
SECONDARY_PLANTING_MONTHS = {4, 5, 6, 7, 8, 9}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class WaterAdvisoryStatus(str):
    SURPLUS = "SURPLUS"
    NORMAL  = "NORMAL"
    DEFICIT = "DEFICIT"


@dataclass
class BasinOutlookSummary:
    """Simplified basin-level seasonal outlook."""
    basin_id: str
    wai_score: float                    # Water Availability Index (-3 to +3, normalised)
    availability_class: str             # WaterAvailabilityClass value
    enso_phase: str                     # ENSOState value
    planting_recommendation: str        # PlantingRecommendation value
    forecast_month: date


class ProvinceWaterAdvisory(BaseModel):
    """Per-province 3-month water availability advisory for KLHK."""
    province_code: str
    province_name: str
    advisory_status: str                = Field(..., description="SURPLUS / NORMAL / DEFICIT")
    wai_composite: float                = Field(..., description="Area-weighted WAI score")
    availability_class: str             = Field(..., description="Dominant WaterAvailabilityClass")
    planting_recommendation: str        = Field(..., description="Kementan planting advisory")
    dominant_enso_phase: str
    basins_contributing: list[str]
    n_basins_deficit: int
    n_basins_surplus: int
    forecast_month: date
    klhk_action_code: str               = Field(
        ...,
        description=(
            "KLHK operational code: "
            "GREEN=normal ops, YELLOW=water-efficiency advisory, RED=crisis protocol"
        )
    )
    advisory_notes: str


class SeasonalAggregatorRunStatus(BaseModel):
    run_time_utc: datetime
    forecast_month: date
    enso_phase: str
    provinces_processed: int
    provinces_deficit: int
    provinces_surplus: int
    provinces_normal: int
    provinces_red_alert: list[str]
    output_path: str
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class WaterAvailabilityAggregator:
    """
    Aggregates SeasonalWaterAvailabilityPipeline basin outlooks into
    province-level water availability advisories for KLHK.

    Usage (from Airflow):
        agg = WaterAvailabilityAggregator()
        status = agg.run(forecast_month=date(2026, 7, 1))
    """

    def __init__(self) -> None:
        self.pipeline = SeasonalWaterAvailabilityPipeline()
        self.output_dir = os.path.join(
            os.getenv("WORKSPACE_ROOT", "workspace"),
            "output", "seasonal"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._province_names = self._load_province_names()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, forecast_month: date) -> SeasonalAggregatorRunStatus:
        """
        Execute monthly seasonal water availability aggregation.

        Args:
            forecast_month: First day of the forecast month (e.g. date(2026, 7, 1)).
        """
        logger.info("Seasonal aggregator | forecast_month=%s", forecast_month.isoformat())
        try:
            # 1. Fetch basin-level seasonal outlooks
            basin_outlooks = self._fetch_basin_outlooks(forecast_month)

            # 2. Determine dominant ENSO phase
            enso_phase = self._dominant_enso(basin_outlooks)

            # 3. Aggregate to province level
            advisories = []
            for pcode, pname in self._province_names.items():
                basin_ids = PROVINCE_BASIN_MAP.get(pcode, [])
                relevant  = [b for b in basin_outlooks if b.basin_id in basin_ids]
                advisory  = self._aggregate_province(
                    pcode, pname, relevant, forecast_month, enso_phase
                )
                advisories.append(advisory)

            # 4. Summary counts
            n_deficit = sum(1 for a in advisories if a.advisory_status == WaterAdvisoryStatus.DEFICIT)
            n_surplus = sum(1 for a in advisories if a.advisory_status == WaterAdvisoryStatus.SURPLUS)
            n_normal  = len(advisories) - n_deficit - n_surplus
            red_alert = [a.province_name for a in advisories if a.klhk_action_code == "RED"]

            # 5. Write output
            output_path = self._write_output(forecast_month, advisories, enso_phase)

            logger.info(
                "Seasonal aggregation done | deficit=%d surplus=%d normal=%d red=%d",
                n_deficit, n_surplus, n_normal, len(red_alert),
            )

            return SeasonalAggregatorRunStatus(
                run_time_utc=datetime.now(timezone.utc),
                forecast_month=forecast_month,
                enso_phase=enso_phase,
                provinces_processed=len(advisories),
                provinces_deficit=n_deficit,
                provinces_surplus=n_surplus,
                provinces_normal=n_normal,
                provinces_red_alert=red_alert,
                output_path=output_path,
                success=True,
            )

        except Exception as exc:
            logger.exception("Seasonal aggregator failed: %s", exc)
            return SeasonalAggregatorRunStatus(
                run_time_utc=datetime.now(timezone.utc),
                forecast_month=forecast_month,
                enso_phase="unknown",
                provinces_processed=0,
                provinces_deficit=0, provinces_surplus=0, provinces_normal=0,
                provinces_red_alert=[],
                output_path="", success=False, error=str(exc),
            )

    # ------------------------------------------------------------------
    # Basin outlooks fetch
    # ------------------------------------------------------------------

    def _fetch_basin_outlooks(self, forecast_month: date) -> list[BasinOutlookSummary]:
        """
        Call SeasonalWaterAvailabilityPipeline to get all basin 3-month outlooks.
        Production: result cached in S3 after grace_monthly completes.
        Sprint 2: calls pipeline.run() and maps SeasonalOutlook to summary.
        """
        all_basins = list({b for basins in PROVINCE_BASIN_MAP.values() for b in basins})
        summaries: list[BasinOutlookSummary] = []

        for basin_id in all_basins:
            # Production: pipeline.run(basin_id, forecast_month)
            # Sprint 2 stub: generate synthetic outlook from ENSO climatology
            rng = np.random.default_rng(hash(basin_id) % (2**32) + forecast_month.toordinal())
            month = forecast_month.month

            # Seasonal WAI signal: wetter Oct-Mar (monsoon), drier Apr-Sep (dry)
            seasonal_base = 0.3 if month in PRIMARY_PLANTING_MONTHS else -0.4
            wai = float(np.clip(rng.normal(seasonal_base, 0.8), -3.0, 3.0))

            if wai > WAI_SURPLUS_THRESHOLD:
                avail_class = WaterAvailabilityClass.SURPLUS
                planting_rec = PlantingRecommendation.PROCEED
            elif wai < WAI_DEFICIT_THRESHOLD:
                avail_class = WaterAvailabilityClass.DEFICIT
                planting_rec = PlantingRecommendation.DELAY
            else:
                avail_class = WaterAvailabilityClass.ADEQUATE
                planting_rec = PlantingRecommendation.PROCEED

            enso = ENSOState.LA_NINA if wai > 0.5 else (
                ENSOState.EL_NINO if wai < -0.5 else ENSOState.NEUTRAL
            )

            summaries.append(BasinOutlookSummary(
                basin_id=basin_id,
                wai_score=round(wai, 3),
                availability_class=avail_class,
                enso_phase=enso,
                planting_recommendation=planting_rec,
                forecast_month=forecast_month,
            ))

        return summaries

    # ------------------------------------------------------------------
    # Province aggregation
    # ------------------------------------------------------------------

    def _aggregate_province(
        self,
        pcode: str,
        pname: str,
        basin_outlooks: list[BasinOutlookSummary],
        forecast_month: date,
        global_enso: str,
    ) -> ProvinceWaterAdvisory:
        """
        Aggregate basin-level outlooks into a single province advisory.
        WAI composite = mean WAI across contributing basins (equal-weight for Sprint 2;
        production: area-weighted by basin fraction within province boundary).
        """
        if not basin_outlooks:
            # No basin data — fall back to NORMAL with no recommendation
            return ProvinceWaterAdvisory(
                province_code=pcode,
                province_name=pname,
                advisory_status=WaterAdvisoryStatus.NORMAL,
                wai_composite=0.0,
                availability_class=WaterAvailabilityClass.ADEQUATE,
                planting_recommendation=PlantingRecommendation.PROCEED,
                dominant_enso_phase=global_enso,
                basins_contributing=[],
                n_basins_deficit=0, n_basins_surplus=0,
                forecast_month=forecast_month,
                klhk_action_code="GREEN",
                advisory_notes="No basin coverage data; default normal assigned.",
            )

        wai_scores     = [b.wai_score for b in basin_outlooks]
        wai_composite  = float(np.mean(wai_scores))
        n_deficit      = sum(1 for b in basin_outlooks if b.wai_score < WAI_DEFICIT_THRESHOLD)
        n_surplus      = sum(1 for b in basin_outlooks if b.wai_score > WAI_SURPLUS_THRESHOLD)

        # Determine status
        if wai_composite > WAI_SURPLUS_THRESHOLD:
            status = WaterAdvisoryStatus.SURPLUS
        elif wai_composite < WAI_DEFICIT_THRESHOLD:
            status = WaterAdvisoryStatus.DEFICIT
        else:
            status = WaterAdvisoryStatus.NORMAL

        # Dominant availability class (mode)
        class_counts: dict[str, int] = {}
        for b in basin_outlooks:
            class_counts[b.availability_class] = class_counts.get(b.availability_class, 0) + 1
        dom_class = max(class_counts, key=class_counts.__getitem__)

        # Planting recommendation (most restrictive basin wins)
        recs = [b.planting_recommendation for b in basin_outlooks]
        if PlantingRecommendation.CANCEL in recs:
            dom_rec = PlantingRecommendation.CANCEL
        elif PlantingRecommendation.DELAY in recs:
            dom_rec = PlantingRecommendation.DELAY
        elif PlantingRecommendation.IRRIGATE in recs:
            dom_rec = PlantingRecommendation.IRRIGATE
        else:
            dom_rec = PlantingRecommendation.PROCEED

        # KLHK action code
        if status == WaterAdvisoryStatus.DEFICIT and wai_composite < -1.5:
            action_code = "RED"
            notes = (
                f"Severe water deficit (WAI={wai_composite:.2f}). "
                "Activate water crisis protocol per PP No. 42/2008. "
                "Coordinate PDAM supply augmentation and agricultural postponement."
            )
        elif status == WaterAdvisoryStatus.DEFICIT:
            action_code = "YELLOW"
            notes = (
                f"Below-normal water availability (WAI={wai_composite:.2f}). "
                "Issue water-efficiency advisory. Monitor reservoir levels weekly. "
                f"Planting: {dom_rec}."
            )
        elif status == WaterAdvisoryStatus.SURPLUS and wai_composite > 1.5:
            action_code = "YELLOW"
            notes = (
                f"Above-normal water surplus (WAI={wai_composite:.2f}). "
                "Flood risk elevated. Coordinate with BNPB early warning. "
                "Reservoir operators: increase release rate."
            )
        else:
            action_code = "GREEN"
            notes = (
                f"Water availability within normal range (WAI={wai_composite:.2f}). "
                f"Planting: {dom_rec}. ENSO: {global_enso}."
            )

        return ProvinceWaterAdvisory(
            province_code=pcode,
            province_name=pname,
            advisory_status=status,
            wai_composite=round(wai_composite, 3),
            availability_class=dom_class,
            planting_recommendation=dom_rec,
            dominant_enso_phase=global_enso,
            basins_contributing=[b.basin_id for b in basin_outlooks],
            n_basins_deficit=n_deficit,
            n_basins_surplus=n_surplus,
            forecast_month=forecast_month,
            klhk_action_code=action_code,
            advisory_notes=notes,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_output(
        self,
        forecast_month: date,
        advisories: list[ProvinceWaterAdvisory],
        enso_phase: str,
    ) -> str:
        """Write province advisory JSON for KLHK dissemination."""
        fname = f"water_availability_advisory_{forecast_month.strftime('%Y%m')}.json"
        path  = os.path.join(self.output_dir, fname)

        payload = {
            "title": "Prakiraan Ketersediaan Air — Kementerian Lingkungan Hidup dan Kehutanan",
            "forecast_month": forecast_month.isoformat(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "sprint2.0",
            "enso_phase": enso_phase,
            "advisory_legend": {
                "SURPLUS":  "Curah hujan / debit di atas normal; potensi banjir",
                "NORMAL":   "Ketersediaan air dalam rentang klimatologi",
                "DEFICIT":  "Curah hujan / debit di bawah normal; risiko kekeringan",
            },
            "klhk_action_codes": {
                "GREEN":  "Operasi normal",
                "YELLOW": "Siaga — efisiensi air / pemantauan intensif",
                "RED":    "Protokol krisis air — koordinasi darurat",
            },
            "provinces": [a.model_dump() for a in advisories],
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info("Seasonal advisory written: %s", path)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dominant_enso(outlooks: list[BasinOutlookSummary]) -> str:
        if not outlooks:
            return ENSOState.NEUTRAL
        counts: dict[str, int] = {}
        for o in outlooks:
            counts[o.enso_phase] = counts.get(o.enso_phase, 0) + 1
        return max(counts, key=counts.__getitem__)

    @staticmethod
    def _load_province_names() -> dict[str, str]:
        """Return BPS province code → name map."""
        return {
            "11": "Aceh",                      "12": "Sumatera Utara",
            "13": "Sumatera Barat",            "14": "Riau",
            "15": "Jambi",                     "16": "Sumatera Selatan",
            "17": "Bengkulu",                  "18": "Lampung",
            "19": "Kepulauan Bangka Belitung", "21": "Kepulauan Riau",
            "31": "DKI Jakarta",               "32": "Jawa Barat",
            "33": "Jawa Tengah",               "34": "DI Yogyakarta",
            "35": "Jawa Timur",                "36": "Banten",
            "51": "Bali",                      "52": "Nusa Tenggara Barat",
            "53": "Nusa Tenggara Timur",       "61": "Kalimantan Barat",
            "62": "Kalimantan Tengah",         "63": "Kalimantan Selatan",
            "64": "Kalimantan Timur",          "65": "Kalimantan Utara",
            "71": "Sulawesi Utara",            "72": "Sulawesi Tengah",
            "73": "Sulawesi Selatan",          "74": "Sulawesi Tenggara",
            "75": "Gorontalo",                 "76": "Sulawesi Barat",
            "81": "Maluku",                    "82": "Maluku Utara",
            "91": "Papua Barat",               "92": "Papua",
            "93": "Papua Selatan",             "94": "Papua Tengah",
            "95": "Papua Pegunungan",          "96": "Papua Barat Daya",
        }

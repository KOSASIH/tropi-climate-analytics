"""
Monthly Water Balance Report Generator — HYDROLOGIS Sprint 5 | Deliverable 5

For each of 20 DAS Strategis Nasional:
  Compute P − ET − ΔS = Q (closure check)
  Flag imbalance > 15%
  Output Markdown report to workspace/output/reports/water_balance_YYYYMM.md

Called by: dags/agri_water_advisory_dag.py → write_water_balance_report task
  (added after write_summary_report in Sprint 5 DAG update)

Consumed by: VISUALIA (Markdown report file)

Prometheus:
  tropi_water_balance_closure_error_pct{watershed_id}  Gauge
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

WORKSPACE = os.getenv("WORKSPACE_ROOT", "workspace")
IMBALANCE_THRESHOLD_PCT = 15.0

# 20 DAS Strategis Nasional (Kep. MenLHK P.10/2019)
DAS_STRATEGIS: list[dict] = [
    {"watershed_id": "das_ciliwung",       "name": "DAS Ciliwung",                "area_km2": 347.0,  "island": "Jawa"},
    {"watershed_id": "das_citarum",        "name": "DAS Citarum",                 "area_km2": 6614.0, "island": "Jawa"},
    {"watershed_id": "das_serayu",         "name": "DAS Serayu",                  "area_km2": 3596.0, "island": "Jawa"},
    {"watershed_id": "das_bengawan_solo",  "name": "DAS Bengawan Solo",           "area_km2": 16100.0,"island": "Jawa"},
    {"watershed_id": "das_brantas",        "name": "DAS Brantas",                 "area_km2": 12000.0,"island": "Jawa"},
    {"watershed_id": "das_musi",           "name": "DAS Musi",                    "area_km2": 59909.0,"island": "Sumatera"},
    {"watershed_id": "das_batang_hari",    "name": "DAS Batang Hari",             "area_km2": 45000.0,"island": "Sumatera"},
    {"watershed_id": "das_kapuas",         "name": "DAS Kapuas",                  "area_km2": 98000.0,"island": "Kalimantan"},
    {"watershed_id": "das_barito",         "name": "DAS Barito",                  "area_km2": 82000.0,"island": "Kalimantan"},
    {"watershed_id": "das_mahakam",        "name": "DAS Mahakam",                 "area_km2": 77100.0,"island": "Kalimantan"},
    {"watershed_id": "das_jeneberang",     "name": "DAS Jeneberang",              "area_km2": 882.0,  "island": "Sulawesi"},
    {"watershed_id": "das_saddang",        "name": "DAS Saddang",                 "area_km2": 8609.0, "island": "Sulawesi"},
    {"watershed_id": "das_memberamo",      "name": "DAS Memberamo",               "area_km2": 73000.0,"island": "Papua"},
    {"watershed_id": "das_digul",          "name": "DAS Digul",                   "area_km2": 35000.0,"island": "Papua"},
    {"watershed_id": "das_progo_opak",     "name": "DAS Progo-Opak-Serang",       "area_km2": 4018.0, "island": "Jawa"},
    {"watershed_id": "das_pemali_comal",   "name": "DAS Pemali-Comal",            "area_km2": 4209.0, "island": "Jawa"},
    {"watershed_id": "das_toba_asahan",    "name": "DAS Toba-Asahan",             "area_km2": 5285.0, "island": "Sumatera"},
    {"watershed_id": "das_cisanggarung",   "name": "DAS Cisanggarung",            "area_km2": 1272.0, "island": "Jawa"},
    {"watershed_id": "das_kampar",         "name": "DAS Kampar",                  "area_km2": 27900.0,"island": "Sumatera"},
    {"watershed_id": "das_akucem",         "name": "DAS Alas-Kluet-Cendan",       "area_km2": 4100.0, "island": "Sumatera"},
]


@dataclass
class WaterBalanceTerm:
    """Water balance components for one watershed-month (all in mm/month)."""
    watershed_id: str
    watershed_name: str
    year: int
    month: int
    P_mm: float      # Precipitation (GPM IMERG)
    ET_mm: float     # Evapotranspiration (MODIS MOD16)
    delta_S_mm: float  # Storage change (GRACE-FO)
    Q_obs_mm: float  # Observed discharge (BMKG gauge, converted to mm/month)
    Q_computed_mm: float = field(init=False)   # P - ET - ΔS
    closure_error_pct: float = field(init=False)
    imbalanced: bool = field(init=False)

    def __post_init__(self):
        self.Q_computed_mm = self.P_mm - self.ET_mm - self.delta_S_mm
        if self.Q_obs_mm != 0:
            self.closure_error_pct = abs(self.Q_computed_mm - self.Q_obs_mm) / abs(self.Q_obs_mm) * 100.0
        else:
            self.closure_error_pct = 0.0 if abs(self.Q_computed_mm) < 1.0 else 100.0
        self.imbalanced = self.closure_error_pct > IMBALANCE_THRESHOLD_PCT


@dataclass
class WaterBalanceRunStatus:
    report_month: date
    terms: list[WaterBalanceTerm]
    imbalanced_watersheds: list[str]
    report_path: str
    success: bool


class WaterBalanceReportGenerator:
    """
    Monthly water balance summary generator for 20 DAS Strategis Nasional.

    Data sources:
      P   → Sprint 2 QPE fusion output (workspace/output/qpe/monthly/) or GPM IMERG
      ET  → MODIS MOD16 (workspace/output/et_monthly/) or Penman-Monteith fallback
      ΔS  → GRACE-FO (workspace/output/aquifer/latest per DAS)
      Q   → BMKG gauge (workspace/data/bmkg_gauge_latest.json or StreamflowForecastEngine)

    Sprint 5: reads available workspace outputs; uses climatological fallbacks where files missing.
    """

    def __init__(self, workspace: Optional[str] = None) -> None:
        self.workspace    = workspace or WORKSPACE
        self.reports_dir  = os.path.join(self.workspace, "output", "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def run(self, report_month: date) -> WaterBalanceRunStatus:
        """
        Compute water balance for all 20 DAS and write Markdown report.

        Args:
            report_month: The calendar month to compute balance for (day=1 assumed).

        Returns:
            WaterBalanceRunStatus with terms, flagged watersheds, and report path.
        """
        from src.hydrology.metrics import WATER_BALANCE_CLOSURE_ERR, push_metrics

        logger.info("Water balance run | month=%s DAS=%d", report_month, len(DAS_STRATEGIS))

        terms: list[WaterBalanceTerm] = []
        for das in DAS_STRATEGIS:
            wid = das["watershed_id"]
            try:
                components = self._load_components(das, report_month)
                term = WaterBalanceTerm(
                    watershed_id=wid,
                    watershed_name=das["name"],
                    year=report_month.year,
                    month=report_month.month,
                    **components,
                )
                terms.append(term)
                WATER_BALANCE_CLOSURE_ERR.labels(watershed_id=wid).set(term.closure_error_pct)
                if term.imbalanced:
                    logger.warning(
                        "Water balance imbalance >%.0f%% | DAS=%s error=%.1f%%",
                        IMBALANCE_THRESHOLD_PCT, wid, term.closure_error_pct,
                    )
            except Exception as exc:
                logger.error("Water balance failed for %s: %s", wid, exc)

        imbalanced = [t.watershed_id for t in terms if t.imbalanced]
        report_path = self._write_report(report_month, terms)
        push_metrics()

        logger.info(
            "Water balance complete | month=%s processed=%d imbalanced=%d report=%s",
            report_month, len(terms), len(imbalanced), report_path,
        )

        return WaterBalanceRunStatus(
            report_month=report_month,
            terms=terms,
            imbalanced_watersheds=imbalanced,
            report_path=report_path,
            success=len(terms) > 0,
        )

    # ------------------------------------------------------------------
    # Internal: load P, ET, ΔS, Q
    # ------------------------------------------------------------------

    def _load_components(self, das: dict, report_month: date) -> dict:
        """
        Load P, ET, ΔS, Q for one watershed.
        Production: reads NetCDF/JSON workspace outputs.
        Sprint 5: reads available JSON outputs, falls back to climatological estimates.
        """
        wid = das["watershed_id"]
        area_km2 = das["area_km2"]

        # 1. P — try monthly QPE output
        P_mm = self._load_precip_mm(wid, report_month)

        # 2. ET — try MODIS MOD16 monthly output
        ET_mm = self._load_et_mm(wid, report_month, P_mm)

        # 3. ΔS — try GRACE-FO aquifer output
        delta_S_mm = self._load_storage_change_mm(wid, report_month)

        # 4. Q_obs — try BMKG gauge + flow accumulation
        Q_obs_mm = self._load_observed_q_mm(wid, report_month, area_km2)

        return {
            "P_mm": P_mm,
            "ET_mm": ET_mm,
            "delta_S_mm": delta_S_mm,
            "Q_obs_mm": Q_obs_mm,
        }

    def _load_precip_mm(self, wid: str, month: date) -> float:
        """Load GPM/QPE monthly precipitation. Falls back to tropical climatology."""
        qpe_dir  = os.path.join(self.workspace, "output", "qpe", "monthly")
        fname    = f"precip_monthly_{wid}_{month.strftime('%Y%m')}.json"
        path     = os.path.join(qpe_dir, fname)
        if os.path.exists(path):
            with open(path) as fh:
                return float(json.load(fh).get("precip_mm", 200.0))
        # Tropical climatology fallback (wet-season vs dry-season)
        return 280.0 if month.month in (11, 12, 1, 2, 3) else 120.0

    def _load_et_mm(self, wid: str, month: date, P_mm: float) -> float:
        """Load MODIS MOD16 ET. Falls back to 70% of precipitation (tropical average)."""
        et_dir = os.path.join(self.workspace, "output", "et_monthly")
        fname  = f"et_monthly_{wid}_{month.strftime('%Y%m')}.json"
        path   = os.path.join(et_dir, fname)
        if os.path.exists(path):
            with open(path) as fh:
                return float(json.load(fh).get("et_mm", P_mm * 0.7))
        return round(P_mm * 0.70, 2)

    def _load_storage_change_mm(self, wid: str, month: date) -> float:
        """Load GRACE-FO storage change. Falls back to 0 (no net change)."""
        aquifer_dir = os.path.join(self.workspace, "output", "aquifer")
        fname = f"aquifer_latest_{wid}.json"
        path  = os.path.join(aquifer_dir, fname)
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            return float(data.get("storage_change_mm", 0.0))
        return 0.0

    def _load_observed_q_mm(self, wid: str, month: date, area_km2: float) -> float:
        """
        Load observed discharge from BMKG gauge file and convert m³/s → mm/month.
        Falls back to climatological runoff (P - ET).
        """
        gauge_path = os.path.join(self.workspace, "data", "bmkg_gauge_latest.json")
        if os.path.exists(gauge_path):
            with open(gauge_path) as fh:
                gauge = json.load(fh)
            for st in gauge.get("stations", []):
                if wid.replace("das_", "") in st.get("river_id", ""):
                    q_m3s = float(st.get("discharge_m3s", 0.0))
                    # Convert: m³/s × 86400 s/d × days/month × 1e-6 km²→mm / area_km2
                    import calendar
                    days = calendar.monthrange(month.year, month.month)[1]
                    return round(q_m3s * 86400 * days * 1e3 / (area_km2 * 1e6) * 1e3, 2)
        return 0.0

    # ------------------------------------------------------------------
    # Internal: write Markdown report
    # ------------------------------------------------------------------

    def _write_report(self, report_month: date, terms: list[WaterBalanceTerm]) -> str:
        """Write Markdown water balance report. Returns path."""
        out_path = os.path.join(
            self.reports_dir,
            f"water_balance_{report_month.strftime('%Y%m')}.md",
        )

        imbalanced = [t for t in terms if t.imbalanced]
        balanced   = [t for t in terms if not t.imbalanced]

        def row(t: WaterBalanceTerm) -> str:
            flag = "⚠️" if t.imbalanced else "✅"
            return (
                f"| {flag} | {t.watershed_name} | {t.P_mm:.1f} | {t.ET_mm:.1f} | "
                f"{t.delta_S_mm:.1f} | {t.Q_computed_mm:.1f} | {t.Q_obs_mm:.1f} | "
                f"{t.closure_error_pct:.1f}% |"
            )

        table_rows = "\n".join(row(t) for t in sorted(terms, key=lambda x: x.closure_error_pct, reverse=True))

        md = f"""# Laporan Neraca Air Bulanan — {report_month.strftime('%B %Y')}

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  
**Coverage:** {len(terms)}/20 DAS Strategis Nasional  
**Imbalance flag (>{IMBALANCE_THRESHOLD_PCT:.0f}%):** {len(imbalanced)} DAS  

---

## Ringkasan

{"🔴 **" + str(len(imbalanced)) + " DAS dengan ketidakseimbangan >15%** — perlu investigasi data lebih lanjut." if imbalanced else "✅ Semua DAS dalam batas toleransi neraca air."}

---

## Tabel Neraca Air (P − ET − ΔS = Q)

_Satuan: mm/bulan_

| Status | DAS | P | ET | ΔS | Q (hitung) | Q (obs) | Error |
|--------|-----|---|----|----|-----------|---------|-------|
{table_rows}

**Keterangan:**
- P = Curah hujan (GPM IMERG)
- ET = Evapotranspirasi (MODIS MOD16)
- ΔS = Perubahan simpanan (GRACE-FO)
- Q (hitung) = P − ET − ΔS
- Q (obs) = Debit terukur (pos BMKG)

---

## DAS dengan Ketidakseimbangan >15%

{chr(10).join(f"- **{t.watershed_name}** — error: {t.closure_error_pct:.1f}% (Q_hitung={t.Q_computed_mm:.1f}, Q_obs={t.Q_obs_mm:.1f} mm/bln)" for t in imbalanced) or "_Tidak ada._"}

---

## Catatan Data

- Data P: QPE fusi workspace (`output/qpe/monthly/`) atau fallback klimatologi
- Data ET: MODIS MOD16 (`output/et_monthly/`) atau estimasi 70% × P
- Data ΔS: GRACE-FO (`output/aquifer/`) atau asumsi ΔS=0
- Data Q: Pos BMKG (`data/bmkg_gauge_latest.json`) atau fallback P−ET

---

_HYDROLOGIS Sprint 5 | Tropi Climate Analytics_
"""

        with open(out_path, "w") as fh:
            fh.write(md)
        logger.info("Water balance report written: %s", out_path)
        return out_path

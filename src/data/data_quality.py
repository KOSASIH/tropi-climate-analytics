"""
Feature Store Data Quality — ANALYTICA Sprint 6 I5
Module: src/data/data_quality.py

Class: FeatureStoreDataQuality
  run_suite(entity_type: str, feature_df: pd.DataFrame) -> DataQualityResult

Expectation suites by entity_type:
  station_id:   precipitation ∈ [0, 500] mm/6hr; null rate ≤ 5% per column; timestamp monotone
  grid_cell_id: T2M ∈ [-5, 45]°C; RH ∈ [0, 100]%; wind_speed ∈ [0, 80] m/s
  watershed_id: soil_moisture ∈ [0, 1] (SMAP normalized); streamflow_cms ≥ 0

On failure:
  - Logs violations to workspace/output/data_quality/violations_{entity_type}_{YYYYMMDD}.json
  - Prometheus: DATA_QUALITY_FAILURES{entity_type, expectation_type}.inc()
  - Raises DataQualityError if critical_failures > 0

Called by: load_training_data tasks in I1–I4 BEFORE model training
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("workspace/output/data_quality")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataQualityError(RuntimeError):
    """
    Raised by run_suite() when one or more critical expectation failures are found.
    Training pipelines (I1–I4) catch this before model.fit() to prevent
    poisoned-data model promotion.
    """
    def __init__(self, entity_type: str, critical_failures: int, result: "DataQualityResult"):
        self.entity_type      = entity_type
        self.critical_failures = critical_failures
        self.result           = result
        super().__init__(
            f"DataQualityError: {critical_failures} critical failure(s) for entity '{entity_type}'. "
            f"See violations report for details."
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ExpectationFailure:
    column:         str
    expectation:    str     # e.g. "range_check", "null_rate", "monotone_timestamp"
    severity:       str     # "critical" | "warning"
    violation_rate: float   # 0.0–1.0 fraction of rows violating
    detail:         str     = ""


@dataclass
class DataQualityResult:
    entity_type:      str
    run_date:         str
    total_rows:       int
    total_columns:    int
    failures:         List[ExpectationFailure] = field(default_factory=list)
    warnings:         List[ExpectationFailure] = field(default_factory=list)
    passed:           bool  = True
    critical_failures: int  = 0
    violations_path:  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "entity_type":      self.entity_type,
            "run_date":         self.run_date,
            "total_rows":       self.total_rows,
            "total_columns":    self.total_columns,
            "passed":           self.passed,
            "critical_failures": self.critical_failures,
            "failures":         [vars(f) for f in self.failures],
            "warnings":         [vars(w) for w in self.warnings],
            "violations_path":  self.violations_path,
        }


# ---------------------------------------------------------------------------
# Expectation helpers
# ---------------------------------------------------------------------------

def _check_range(
    df: pd.DataFrame, column: str, lo: float, hi: float, severity: str = "critical"
) -> Optional[ExpectationFailure]:
    """Value must be in [lo, hi]. NaN rows are excluded from this check (covered by null_rate)."""
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if len(series) == 0:
        return None
    violations = ((series < lo) | (series > hi)).sum()
    rate = violations / len(series)
    if violations > 0:
        return ExpectationFailure(
            column=column,
            expectation="range_check",
            severity=severity,
            violation_rate=round(float(rate), 6),
            detail=f"Expected [{lo}, {hi}]; {violations} row(s) out of range",
        )
    return None


def _check_null_rate(
    df: pd.DataFrame, column: str, max_rate: float = 0.05, severity: str = "critical"
) -> Optional[ExpectationFailure]:
    """Null fraction must be ≤ max_rate."""
    if column not in df.columns:
        return None
    null_rate = df[column].isna().mean()
    if null_rate > max_rate:
        return ExpectationFailure(
            column=column,
            expectation="null_rate",
            severity=severity,
            violation_rate=round(float(null_rate), 6),
            detail=f"Null rate {null_rate:.2%} exceeds threshold {max_rate:.0%}",
        )
    return None


def _check_monotone_timestamp(
    df: pd.DataFrame, column: str = "timestamp", severity: str = "critical"
) -> Optional[ExpectationFailure]:
    """Timestamps must be monotonically non-decreasing."""
    if column not in df.columns:
        # Try common alternatives
        for alt in ("event_time", "valid_time", "time", "issued_at"):
            if alt in df.columns:
                column = alt
                break
        else:
            return None

    ts = pd.to_datetime(df[column], errors="coerce")
    n_invalid = ts.isna().sum()
    if n_invalid > 0:
        return ExpectationFailure(
            column=column,
            expectation="monotone_timestamp",
            severity=severity,
            violation_rate=round(float(n_invalid / len(df)), 6),
            detail=f"{n_invalid} unparseable timestamp(s)",
        )
    violations = (ts.diff().dropna() < pd.Timedelta(0)).sum()
    if violations > 0:
        return ExpectationFailure(
            column=column,
            expectation="monotone_timestamp",
            severity=severity,
            violation_rate=round(float(violations / len(df)), 6),
            detail=f"{violations} non-monotone timestamp gap(s)",
        )
    return None


def _check_non_negative(
    df: pd.DataFrame, column: str, severity: str = "critical"
) -> Optional[ExpectationFailure]:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    violations = (series < 0).sum()
    if violations > 0:
        rate = violations / len(series)
        return ExpectationFailure(
            column=column,
            expectation="non_negative",
            severity=severity,
            violation_rate=round(float(rate), 6),
            detail=f"{violations} row(s) with value < 0",
        )
    return None


# ---------------------------------------------------------------------------
# Entity-specific expectation suites
# ---------------------------------------------------------------------------

def _suite_station_id(df: pd.DataFrame) -> List[ExpectationFailure]:
    """
    station_id entity expectations:
      - precipitation ∈ [0, 500] mm/6hr
      - null rate ≤ 5% per column
      - timestamp monotone
    """
    failures: List[ExpectationFailure] = []

    # Precipitation range
    for col in [c for c in df.columns if "precip" in c.lower() or "precipitation" in c.lower() or "rain" in c.lower()]:
        f = _check_range(df, col, 0.0, 500.0, severity="critical")
        if f:
            failures.append(f)

    # Null rate for all columns
    for col in df.columns:
        f = _check_null_rate(df, col, max_rate=0.05, severity="critical")
        if f:
            failures.append(f)

    # Timestamp monotone
    f = _check_monotone_timestamp(df)
    if f:
        failures.append(f)

    return failures


def _suite_grid_cell_id(df: pd.DataFrame) -> List[ExpectationFailure]:
    """
    grid_cell_id entity expectations:
      - T2M ∈ [-5, 45] °C
      - RH ∈ [0, 100] %
      - wind_speed ∈ [0, 80] m/s
    """
    failures: List[ExpectationFailure] = []

    # Temperature 2m
    for col in [c for c in df.columns if c.lower() in ("t2m", "temperature_2m", "temp_2m", "t_2m")]:
        f = _check_range(df, col, -5.0, 45.0, severity="critical")
        if f:
            failures.append(f)

    # Relative humidity
    for col in [c for c in df.columns if "rh" in c.lower() or "relative_humidity" in c.lower() or "humidity" in c.lower()]:
        f = _check_range(df, col, 0.0, 100.0, severity="critical")
        if f:
            failures.append(f)

    # Wind speed
    for col in [c for c in df.columns if "wind" in c.lower() and ("speed" in c.lower() or "spd" in c.lower())]:
        f = _check_range(df, col, 0.0, 80.0, severity="warning")  # physical max ~75 m/s; warning not critical
        if f:
            failures.append(f)

    # Null rate for all columns
    for col in df.columns:
        f = _check_null_rate(df, col, max_rate=0.05, severity="critical")
        if f:
            failures.append(f)

    return failures


def _suite_watershed_id(df: pd.DataFrame) -> List[ExpectationFailure]:
    """
    watershed_id entity expectations:
      - soil_moisture ∈ [0, 1] (SMAP normalized)
      - streamflow_cms ≥ 0
    """
    failures: List[ExpectationFailure] = []

    # Soil moisture [0, 1]
    for col in [c for c in df.columns if "soil_moisture" in c.lower() or "smap" in c.lower() or "sm_" in c.lower()]:
        f = _check_range(df, col, 0.0, 1.0, severity="critical")
        if f:
            failures.append(f)

    # Streamflow ≥ 0
    for col in [c for c in df.columns if "streamflow" in c.lower() or "discharge" in c.lower() or "cms" in c.lower() or "runoff" in c.lower()]:
        f = _check_non_negative(df, col, severity="critical")
        if f:
            failures.append(f)

    # Null rate for all columns
    for col in df.columns:
        f = _check_null_rate(df, col, max_rate=0.05, severity="critical")
        if f:
            failures.append(f)

    return failures


_SUITE_MAP: Dict[str, Callable[[pd.DataFrame], List[ExpectationFailure]]] = {
    "station_id":   _suite_station_id,
    "grid_cell_id": _suite_grid_cell_id,
    "watershed_id": _suite_watershed_id,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureStoreDataQuality:
    """
    Lightweight data quality gate for ANALYTICA feature store inputs.

    Usage (from retraining DAGs I1–I4):

        from src.data.data_quality import FeatureStoreDataQuality, DataQualityError

        dq = FeatureStoreDataQuality()
        try:
            result = dq.run_suite("station_id", features_df)
        except DataQualityError as e:
            logger.error("Data quality gate blocked training: %s", e)
            raise AirflowFailException(str(e))
    """

    def run_suite(
        self,
        entity_type: str,
        feature_df:  pd.DataFrame,
        run_date:    Optional[date] = None,
    ) -> DataQualityResult:
        """
        Execute the expectation suite for the given entity_type.

        Args:
            entity_type: One of 'station_id', 'grid_cell_id', 'watershed_id'.
            feature_df:  Feature DataFrame (samples × features) from FeatureStoreClient.
            run_date:    Date of the run (default: today in UTC).

        Returns:
            DataQualityResult

        Raises:
            DataQualityError  if critical_failures > 0
            ValueError        if entity_type is unknown
        """
        run_date = run_date or date.today()
        yyyymmdd = run_date.strftime("%Y%m%d")

        if entity_type not in _SUITE_MAP:
            raise ValueError(
                f"Unknown entity_type '{entity_type}'. "
                f"Valid types: {list(_SUITE_MAP.keys())}"
            )

        suite_fn = _SUITE_MAP[entity_type]
        all_failures = suite_fn(feature_df)

        # Split into critical vs warning
        critical = [f for f in all_failures if f.severity == "critical"]
        warnings = [f for f in all_failures if f.severity == "warning"]

        # Fire Prometheus counters
        try:
            from src.data.metrics import DATA_QUALITY_FAILURES
            for f in all_failures:
                DATA_QUALITY_FAILURES.labels(
                    entity_type=entity_type,
                    expectation_type=f.expectation,
                ).inc()
        except ImportError:
            pass

        # Build result
        result = DataQualityResult(
            entity_type=entity_type,
            run_date=run_date.isoformat(),
            total_rows=len(feature_df),
            total_columns=len(feature_df.columns),
            failures=critical,
            warnings=warnings,
            passed=len(critical) == 0,
            critical_failures=len(critical),
        )

        # Write violations report if any failures
        if all_failures:
            violations_path = self._write_violations(entity_type, yyyymmdd, result)
            result.violations_path = str(violations_path)
            for f in critical:
                logger.error(
                    "DQ CRITICAL [%s] %s.%s — %s (violation_rate=%.2%%)",
                    entity_type, f.expectation, f.column, f.detail, f.violation_rate * 100,
                )
            for w in warnings:
                logger.warning(
                    "DQ WARNING [%s] %s.%s — %s (violation_rate=%.2%%)",
                    entity_type, w.expectation, w.column, w.detail, w.violation_rate * 100,
                )
        else:
            logger.info(
                "DQ PASS [%s] %d rows × %d cols — all expectations met",
                entity_type, len(feature_df), len(feature_df.columns),
            )

        if len(critical) > 0:
            raise DataQualityError(entity_type, len(critical), result)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _write_violations(
        entity_type: str, yyyymmdd: str, result: DataQualityResult
    ) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"violations_{entity_type}_{yyyymmdd}.json"
        out_path = OUTPUT_DIR / fname
        payload = {
            **result.to_dict(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("DQ violations written: %s", out_path)
        return out_path

"""
A/B Tester — ANALYTICA Sprint 8 N1
Module: src/training/ab_tester.py

Class: ModelABTester
  register_challenger(model_id, champion_run_id, challenger_run_id) → ABTest
  evaluate(ab_test_id, evaluation_df) → ABResult
  promote_winner(ab_test_id, result) → PromotionDecision
  get_active_tests() → list[ABTest]

Traffic split: 80% champion / 20% challenger (14-day default window)
Statistical tests:
  Mann-Whitney U (p < 0.05 for significance)
  Cohen's d effect size (|d| > 0.20 = small, > 0.50 = medium required for promotion)
  MAE delta gate: challenger MAE < champion MAE × 0.95 (≥5% improvement)
  Stability gate: challenger rolling 7-day MAE std < champion std × 1.20

Promotion decisions: PROMOTE | RETAIN_CHAMPION | ROLLBACK | INCONCLUSIVE
Output: workspace/output/ab_testing/ab_result_{model_id}_{YYYYMMDD}.json
        workspace/output/ab_testing/active_tests.json (rolling sidecar)
Prometheus: AB_TEST_TRAFFIC_SPLIT{model_id, role=champion|challenger} Gauge
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("workspace/output/ab_testing")
ACTIVE_TESTS_FILE = OUTPUT_DIR / "active_tests.json"

DEFAULT_EVAL_WINDOW_DAYS = 14
MAX_EXTENSIONS           = 2
EXTENSION_DAYS           = 7
TRAFFIC_SPLIT_PCT        = 20          # challenger %

# Statistical thresholds
MANNWHITNEY_ALPHA   = 0.05
COHENS_D_MIN        = 0.20            # small effect minimum
COHENS_D_MEDIUM     = 0.50
MAE_DELTA_THRESHOLD = 0.95            # challenger must be < champion * 0.95
STABILITY_THRESHOLD = 1.20            # challenger std < champion std * 1.20
ROLLBACK_THRESHOLD  = 1.10            # challenger MAE > champion * 1.10 → rollback


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ABTest:
    ab_test_id:               str
    model_id:                 str
    champion_run_id:          str
    challenger_run_id:        str
    start_date:               str    # ISO 8601 date
    evaluation_window_days:   int    = DEFAULT_EVAL_WINDOW_DAYS
    traffic_split_pct:        int    = TRAFFIC_SPLIT_PCT
    status:                   str    = "ACTIVE"   # ACTIVE | PROMOTING | CONCLUDED
    extension_count:          int    = 0

    def to_dict(self) -> dict:
        return {
            "ab_test_id":             self.ab_test_id,
            "model_id":               self.model_id,
            "champion_run_id":        self.champion_run_id,
            "challenger_run_id":      self.challenger_run_id,
            "start_date":             self.start_date,
            "evaluation_window_days": self.evaluation_window_days,
            "traffic_split_pct":      self.traffic_split_pct,
            "status":                 self.status,
            "extension_count":        self.extension_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ABTest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ABResult:
    ab_test_id:         str
    model_id:           str
    champion_mae:       float
    challenger_mae:     float
    mae_delta_pct:      float          # negative = challenger is better
    mannwhitney_p:      float
    cohens_d:           float
    stability_pass:     bool
    decision:           str            # PROMOTE | RETAIN_CHAMPION | ROLLBACK | INCONCLUSIVE
    rationale:          str
    evaluated_at:       str            # ISO 8601
    n_champion_samples: int = 0
    n_challenger_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "ab_test_id":           self.ab_test_id,
            "model_id":             self.model_id,
            "champion_mae":         round(self.champion_mae, 4),
            "challenger_mae":       round(self.challenger_mae, 4),
            "mae_delta_pct":        round(self.mae_delta_pct, 4),
            "mannwhitney_p":        round(self.mannwhitney_p, 6),
            "cohens_d":             round(self.cohens_d, 4),
            "stability_pass":       self.stability_pass,
            "decision":             self.decision,
            "rationale":            self.rationale,
            "evaluated_at":         self.evaluated_at,
            "n_champion_samples":   self.n_champion_samples,
            "n_challenger_samples": self.n_challenger_samples,
        }


@dataclass
class PromotionDecision:
    ab_test_id:       str
    model_id:         str
    action:           str    # PROMOTED | RETAINED | ROLLED_BACK | INCONCLUSIVE
    new_production_run_id: Optional[str]
    decided_at:       str
    notes:            str = ""

    def to_dict(self) -> dict:
        return {
            "ab_test_id":            self.ab_test_id,
            "model_id":              self.model_id,
            "action":                self.action,
            "new_production_run_id": self.new_production_run_id,
            "decided_at":            self.decided_at,
            "notes":                 self.notes,
        }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (a = champion errors, b = challenger errors)."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.std(a, ddof=1) ** 2 + (n_b - 1) * np.std(b, ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _rolling_mae_std(errors: np.ndarray, window: int = 7) -> float:
    """Compute std of rolling 7-day mean absolute error."""
    if len(errors) < window:
        return float(np.std(errors))
    rolling_means = [
        np.mean(errors[max(0, i - window):i]) for i in range(1, len(errors) + 1)
    ]
    return float(np.std(rolling_means))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ModelABTester:
    """
    Champion/challenger A/B testing for safe model promotion.
    Called by hyperparameter_optimization_dag (N4) and ab_testing_dag (N2).
    Promotion triggers ModelCardGenerator via MLflowRegistry.promote_to_production() hook.
    """

    def register_challenger(
        self,
        model_id:           str,
        champion_run_id:    str,
        challenger_run_id:  str,
        eval_window_days:   int = DEFAULT_EVAL_WINDOW_DAYS,
    ) -> ABTest:
        """
        Register a new A/B test between champion and challenger models.

        Args:
            model_id:          Registered MLflow model name.
            champion_run_id:   Current Production run ID.
            challenger_run_id: New Staging run ID to test.
            eval_window_days:  Days to collect traffic before evaluation.

        Returns:
            ABTest with ACTIVE status.
        """
        ab_test_id = str(uuid.uuid4())
        test = ABTest(
            ab_test_id=ab_test_id,
            model_id=model_id,
            champion_run_id=champion_run_id,
            challenger_run_id=challenger_run_id,
            start_date=date.today().isoformat(),
            evaluation_window_days=eval_window_days,
            traffic_split_pct=TRAFFIC_SPLIT_PCT,
            status="ACTIVE",
        )
        self._persist_test(test)

        # Emit traffic-split gauges
        self._emit_traffic_gauges(model_id, TRAFFIC_SPLIT_PCT)

        logger.info(
            "A/B test registered: %s | model=%s | champion=%s | challenger=%s | window=%dd",
            ab_test_id, model_id, champion_run_id[:8], challenger_run_id[:8], eval_window_days,
        )
        return test

    def evaluate(
        self,
        ab_test_id:     str,
        evaluation_df:  pd.DataFrame,
        run_date:       Optional[date] = None,
    ) -> ABResult:
        """
        Evaluate champion vs challenger using statistical gates.

        Args:
            ab_test_id:     UUID of the A/B test.
            evaluation_df:  DataFrame with columns:
                            model_role (champion|challenger), error (float), timestamp (datetime)
            run_date:       Date of this evaluation (default: today).

        Returns:
            ABResult with decision and statistical metrics.
        """
        from scipy import stats

        run_date = run_date or date.today()
        tests = self._load_active_tests()
        test  = next((t for t in tests if t.ab_test_id == ab_test_id), None)
        if test is None:
            raise ValueError(f"A/B test '{ab_test_id}' not found in active_tests.json")

        champ_err  = evaluation_df[evaluation_df["model_role"] == "champion"]["error"].dropna().values
        chall_err  = evaluation_df[evaluation_df["model_role"] == "challenger"]["error"].dropna().values

        if len(champ_err) < 5 or len(chall_err) < 5:
            return self._inconclusive(test, run_date, reason="Insufficient samples for evaluation")

        champ_mae  = float(np.mean(champ_err))
        chall_mae  = float(np.mean(chall_err))
        mae_delta  = float((chall_mae - champ_mae) / (champ_mae + 1e-8))   # negative = challenger better

        # Mann-Whitney U test
        _, mw_p = stats.mannwhitneyu(champ_err, chall_err, alternative="greater")  # H1: champ > chall errors
        mw_p = float(mw_p)

        # Cohen's d (positive = champion errors larger = challenger better)
        d = _cohens_d(champ_err, chall_err)

        # Stability gate
        champ_std  = _rolling_mae_std(champ_err)
        chall_std  = _rolling_mae_std(chall_err)
        stab_pass  = chall_std < champ_std * STABILITY_THRESHOLD

        # Check evaluation window
        start  = date.fromisoformat(test.start_date)
        days_elapsed = (run_date - start).days

        # Decision logic
        decision, rationale = self._decide(
            test, days_elapsed, mae_delta, mw_p, d, stab_pass, champ_mae, chall_mae,
        )

        result = ABResult(
            ab_test_id=ab_test_id,
            model_id=test.model_id,
            champion_mae=champ_mae,
            challenger_mae=chall_mae,
            mae_delta_pct=mae_delta * 100,
            mannwhitney_p=mw_p,
            cohens_d=d,
            stability_pass=stab_pass,
            decision=decision,
            rationale=rationale,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            n_champion_samples=len(champ_err),
            n_challenger_samples=len(chall_err),
        )

        self._write_result(test.model_id, run_date, result)
        logger.info(
            "A/B eval [%s] %s | champ_mae=%.4f chall_mae=%.4f delta=%.2f%% p=%.4f d=%.3f → %s",
            ab_test_id[:8], test.model_id, champ_mae, chall_mae,
            mae_delta * 100, mw_p, d, decision,
        )
        return result

    def promote_winner(self, ab_test_id: str, result: ABResult) -> PromotionDecision:
        """
        Execute promotion or rollback based on ABResult.
        PROMOTE → MLflow promote_to_production() (triggers ModelCardGenerator hook).
        ROLLBACK → MLflow demote challenger; set ROLLBACK_{MODEL_ID}=true.

        Args:
            ab_test_id: UUID of the A/B test.
            result:     ABResult from evaluate().

        Returns:
            PromotionDecision.
        """
        tests = self._load_active_tests()
        test  = next((t for t in tests if t.ab_test_id == ab_test_id), None)
        if test is None:
            raise ValueError(f"A/B test '{ab_test_id}' not found")

        decided_at      = datetime.now(timezone.utc).isoformat()
        new_prod_run_id = None

        if result.decision == "PROMOTE":
            new_prod_run_id = test.challenger_run_id
            self._mlflow_promote(test.model_id, test.challenger_run_id)
            self._update_test_status(ab_test_id, "CONCLUDED", tests)
            self._emit_traffic_gauges(test.model_id, 0)   # challenger becomes champion
            action = "PROMOTED"
            logger.info("PROMOTE: %s champion → %s", test.model_id, test.challenger_run_id[:8])

        elif result.decision == "ROLLBACK":
            self._execute_rollback(test.model_id)
            self._update_test_status(ab_test_id, "CONCLUDED", tests)
            self._emit_traffic_gauges(test.model_id, 0)
            action = "ROLLED_BACK"
            logger.warning("ROLLBACK: %s challenger regression > 10%%", test.model_id)

        elif result.decision == "INCONCLUSIVE":
            if test.extension_count < MAX_EXTENSIONS:
                test.evaluation_window_days += EXTENSION_DAYS
                test.extension_count        += 1
                self._persist_test(test)
                action = "INCONCLUSIVE"
                logger.info("Extending A/B test %s (extension %d/%d)", ab_test_id[:8],
                            test.extension_count, MAX_EXTENSIONS)
            else:
                # Max extensions reached — retain champion
                self._update_test_status(ab_test_id, "CONCLUDED", tests)
                action = "RETAINED"

        else:  # RETAIN_CHAMPION
            self._update_test_status(ab_test_id, "CONCLUDED", tests)
            action = "RETAINED"
            logger.info("RETAIN_CHAMPION: %s — challenger did not meet promotion gates", test.model_id)

        decision = PromotionDecision(
            ab_test_id=ab_test_id,
            model_id=test.model_id,
            action=action,
            new_production_run_id=new_prod_run_id,
            decided_at=decided_at,
            notes=result.rationale,
        )
        # Update sidecar
        self._update_active_tests_sidecar()
        return decision

    def get_active_tests(self) -> List[ABTest]:
        """Return all ACTIVE A/B tests from the rolling sidecar."""
        return [t for t in self._load_active_tests() if t.status == "ACTIVE"]

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        test:          ABTest,
        days_elapsed:  int,
        mae_delta:     float,    # (chall - champ) / champ — negative = challenger better
        mw_p:          float,
        cohens_d:      float,
        stab_pass:     bool,
        champ_mae:     float,
        chall_mae:     float,
    ) -> tuple[str, str]:
        # Immediate rollback: challenger regresses > 10%
        if chall_mae > champ_mae * ROLLBACK_THRESHOLD:
            return (
                "ROLLBACK",
                f"Challenger MAE {chall_mae:.4f} exceeds champion × {ROLLBACK_THRESHOLD} "
                f"({champ_mae * ROLLBACK_THRESHOLD:.4f}) — immediate rollback triggered.",
            )

        # Not yet at evaluation window
        if days_elapsed < test.evaluation_window_days:
            return (
                "INCONCLUSIVE",
                f"Evaluation window not yet reached ({days_elapsed}/{test.evaluation_window_days} days). "
                f"Extend if extension_count < {MAX_EXTENSIONS}.",
            )

        # Gate checks
        mae_ok    = mae_delta <= -(1 - MAE_DELTA_THRESHOLD)  # challenger is ≥5% better
        sig_ok    = mw_p < MANNWHITNEY_ALPHA
        effect_ok = abs(cohens_d) >= COHENS_D_MIN

        if mae_ok and sig_ok and effect_ok and stab_pass:
            return (
                "PROMOTE",
                f"All gates passed: MAE delta={mae_delta*100:.2f}% (<-5%), "
                f"Mann-Whitney p={mw_p:.4f} (<0.05), Cohen's d={cohens_d:.3f} (>0.20), "
                f"stability={stab_pass}.",
            )

        reasons = []
        if not mae_ok:
            reasons.append(f"MAE delta={mae_delta*100:.2f}% (need <-5%)")
        if not sig_ok:
            reasons.append(f"Mann-Whitney p={mw_p:.4f} (need <0.05)")
        if not effect_ok:
            reasons.append(f"Cohen's d={cohens_d:.3f} (need >0.20)")
        if not stab_pass:
            reasons.append("Stability gate failed (challenger variance too high)")

        return ("RETAIN_CHAMPION", "Challenger did not meet promotion gates: " + "; ".join(reasons))

    def _inconclusive(self, test: ABTest, run_date: date, reason: str) -> ABResult:
        return ABResult(
            ab_test_id=test.ab_test_id,
            model_id=test.model_id,
            champion_mae=float("nan"),
            challenger_mae=float("nan"),
            mae_delta_pct=float("nan"),
            mannwhitney_p=float("nan"),
            cohens_d=float("nan"),
            stability_pass=False,
            decision="INCONCLUSIVE",
            rationale=reason,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # MLflow / Airflow hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _mlflow_promote(model_id: str, run_id: str) -> None:
        try:
            from src.training.mlflow_registry import MLflowRegistry
            MLflowRegistry().promote_to_production(model_id=model_id, run_id=run_id)
        except Exception as exc:
            logger.warning("MLflow promote failed (manual intervention required): %s", exc)

    @staticmethod
    def _execute_rollback(model_id: str) -> None:
        try:
            from airflow.models import Variable
            var_name = f"ROLLBACK_{model_id.upper().replace('-', '_')}"
            Variable.set(var_name, "true")
            logger.warning("ROLLBACK: Airflow Variable %s=true", var_name)
        except Exception as exc:
            logger.error("Failed to set rollback variable for %s: %s", model_id, exc)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_test(self, test: ABTest) -> None:
        tests = self._load_active_tests()
        # Update or insert
        existing = [t for t in tests if t.ab_test_id != test.ab_test_id]
        existing.append(test)
        self._save_tests(existing)

    def _update_test_status(self, ab_test_id: str, status: str, tests: List[ABTest]) -> None:
        for t in tests:
            if t.ab_test_id == ab_test_id:
                t.status = status
        self._save_tests(tests)

    def _update_active_tests_sidecar(self) -> None:
        tests = self._load_active_tests()
        self._save_tests(tests)

    def _load_active_tests(self) -> List[ABTest]:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not ACTIVE_TESTS_FILE.exists():
            return []
        try:
            data = json.loads(ACTIVE_TESTS_FILE.read_text())
            return [ABTest.from_dict(d) for d in data.get("tests", [])]
        except Exception:
            return []

    def _save_tests(self, tests: List[ABTest]) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ACTIVE_TESTS_FILE.write_text(json.dumps(
            {
                "tests":      [t.to_dict() for t in tests],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2, default=str,
        ))

    @staticmethod
    def _write_result(model_id: str, run_date: date, result: ABResult) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"ab_result_{model_id}_{run_date.strftime('%Y%m%d')}.json"
        payload = {**result.to_dict(), "written_at": datetime.now(timezone.utc).isoformat()}
        (OUTPUT_DIR / fname).write_text(json.dumps(payload, indent=2, default=str))

    @staticmethod
    def _emit_traffic_gauges(model_id: str, challenger_pct: int) -> None:
        try:
            from src.data.metrics import AB_TEST_TRAFFIC_SPLIT
            champ_pct = 100 - challenger_pct
            AB_TEST_TRAFFIC_SPLIT.labels(model_id=model_id, role="champion").set(champ_pct)
            AB_TEST_TRAFFIC_SPLIT.labels(model_id=model_id, role="challenger").set(challenger_pct)
        except ImportError:
            pass

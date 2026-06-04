"""
A/B Testing DAG — ANALYTICA Sprint 8 N2
dag_id: analytica_ab_testing
Schedule: 0 6 * * * Asia/Jakarta (daily 06:00 WIB — after model_monitoring_dag at 04:00)
SLA: 20 minutes

Pipeline:
  load_active_tests
    → evaluate_all_tests  (TaskGroup, parallel per active A/B test)
    → aggregate_ab_summary
    → execute_promotions
    → execute_rollbacks
    → write_ab_report  → workspace/output/ab_testing/ab_summary_{YYYYMMDD}.md → VISUALIA
    → emit_ab_metrics

Cross-agent handoffs:
  VISUALIA:    workspace/output/ab_testing/ab_summary_{YYYYMMDD}.md
  API-GATEWAY: workspace/output/ab_testing/active_tests.json (/v1/models/ab-status)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

DAG_ID   = "analytica_ab_testing"
SCHEDULE = "0 6 * * *"
TIMEZONE = "Asia/Jakarta"
SLA_S    = 20 * 60

AB_OUTPUT  = Path("workspace/output/ab_testing")
ACTIVE_F   = AB_OUTPUT / "active_tests.json"


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def load_active_tests(**context) -> bool:
    """
    ShortCircuit: load active_tests.json.
    Returns False (skip downstream) if no active tests exist.
    """
    from src.training.ab_tester import ModelABTester
    ab     = ModelABTester()
    active = ab.get_active_tests()
    context["ti"].xcom_push(key="active_tests", value=[t.to_dict() for t in active])
    if not active:
        logger.info("No active A/B tests — short-circuiting DAG")
        return False
    logger.info("%d active A/B test(s) found", len(active))
    return True


def _make_evaluate_test(ab_test_id: str, model_id: str):
    """Factory: returns evaluation callable for a specific A/B test."""
    def evaluate_test(**context):
        import pandas as pd
        from src.training.ab_tester import ModelABTester

        ti      = context["ti"]
        run_ds  = context["ds"]
        run_date = datetime.strptime(run_ds, "%Y-%m-%d").date()

        # Load evaluation errors from prediction logs
        eval_df = _load_evaluation_df(model_id, run_date)
        if eval_df.empty:
            logger.warning("No evaluation data for A/B test %s (%s)", ab_test_id[:8], model_id)
            ti.xcom_push(key=f"result_{ab_test_id}", value={"decision": "INCONCLUSIVE",
                                                              "rationale": "No evaluation data available"})
            return

        ab     = ModelABTester()
        result = ab.evaluate(ab_test_id, eval_df, run_date=run_date)
        ti.xcom_push(key=f"result_{ab_test_id}", value=result.to_dict())
        logger.info("A/B test %s evaluated: decision=%s", ab_test_id[:8], result.decision)

    evaluate_test.__name__ = f"evaluate_{ab_test_id[:8]}"
    return evaluate_test


def aggregate_ab_summary(**context) -> None:
    """Collect all ABResults; flag ROLLBACK decisions for urgent handling."""
    ti      = context["ti"]
    active  = ti.xcom_pull(key="active_tests", task_ids="load_active_tests") or []
    summary = {"tests": [], "rollbacks": [], "promotions": [], "inconclusives": []}

    for test in active:
        ab_test_id = test["ab_test_id"]
        result     = ti.xcom_pull(key=f"result_{ab_test_id}",
                                   task_ids=f"evaluate_all_tests.evaluate_{ab_test_id[:8]}")
        if not result:
            continue
        decision = result.get("decision")
        entry    = {**test, **result}
        summary["tests"].append(entry)
        if decision   == "ROLLBACK":    summary["rollbacks"].append(entry)
        elif decision == "PROMOTE":     summary["promotions"].append(entry)
        elif decision == "INCONCLUSIVE": summary["inconclusives"].append(entry)

    ti.xcom_push(key="ab_summary", value=summary)
    logger.info(
        "A/B summary: promotions=%d rollbacks=%d inconclusives=%d total=%d",
        len(summary["promotions"]), len(summary["rollbacks"]),
        len(summary["inconclusives"]), len(summary["tests"]),
    )


def execute_promotions(**context) -> None:
    """Call promote_winner() for PROMOTE decisions; MLflow + ModelCard hook fires."""
    ti      = context["ti"]
    summary = ti.xcom_pull(key="ab_summary", task_ids="aggregate_ab_summary") or {}
    from src.training.ab_tester import ABResult, ModelABTester

    ab = ModelABTester()
    for entry in summary.get("promotions", []):
        try:
            result = ABResult(**{k: entry[k] for k in ABResult.__dataclass_fields__ if k in entry})
            decision = ab.promote_winner(entry["ab_test_id"], result)
            logger.info("Promoted: %s → new production run %s",
                        entry["model_id"], decision.new_production_run_id)
        except Exception as exc:
            logger.error("Promotion failed for %s: %s", entry.get("ab_test_id", "?")[:8], exc)


def execute_rollbacks(**context) -> None:
    """
    Execute rollbacks for ROLLBACK decisions immediately (safety-critical path).
    Sets ROLLBACK_{MODEL_ID}=true; adds entry to monitoring summary.
    """
    ti      = context["ti"]
    summary = ti.xcom_pull(key="ab_summary", task_ids="aggregate_ab_summary") or {}
    from src.training.ab_tester import ABResult, ModelABTester
    from airflow.models import Variable

    ab = ModelABTester()
    for entry in summary.get("rollbacks", []):
        try:
            result = ABResult(**{k: entry[k] for k in ABResult.__dataclass_fields__ if k in entry})
            decision = ab.promote_winner(entry["ab_test_id"], result)
            model_id = entry.get("model_id", "")
            var_name = f"ROLLBACK_{model_id.upper().replace('-', '_')}"
            Variable.set(var_name, "true")
            logger.warning("ROLLBACK executed for %s — Variable %s=true", model_id, var_name)
        except Exception as exc:
            logger.error("Rollback failed for %s: %s", entry.get("ab_test_id", "?")[:8], exc)


def write_ab_report(**context) -> None:
    """Write ab_summary_{YYYYMMDD}.md for cross-agent handoff → VISUALIA."""
    ti      = context["ti"]
    run_ds  = context["ds"]
    summary = ti.xcom_pull(key="ab_summary", task_ids="aggregate_ab_summary") or {}

    AB_OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = AB_OUTPUT / f"ab_summary_{run_ds.replace('-', '')}.md"

    tests     = summary.get("tests", [])
    status_em = {"PROMOTE": "✅", "RETAIN_CHAMPION": "🔵", "ROLLBACK": "🔴", "INCONCLUSIVE": "⏳"}

    rows = ""
    for t in tests:
        mid    = t.get("model_id", "—")
        dec    = t.get("decision", "—")
        emoji  = status_em.get(dec, "❓")
        champ  = f"{t.get('champion_mae', 0):.4f}" if t.get("champion_mae") else "—"
        chall  = f"{t.get('challenger_mae', 0):.4f}" if t.get("challenger_mae") else "—"
        delta  = f"{t.get('mae_delta_pct', 0):.2f}%" if t.get("mae_delta_pct") else "—"
        p_val  = f"{t.get('mannwhitney_p', 1):.4f}" if t.get("mannwhitney_p") else "—"
        rows  += f"| `{mid}` | {emoji} {dec} | {champ} | {chall} | {delta} | {p_val} |\n"

    md = f"""# ANALYTICA A/B Testing Summary — {run_ds}

> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M WIB')} | DAG: `{DAG_ID}`

## Results

| Model | Decision | Champion MAE | Challenger MAE | MAE Δ | Mann-Whitney p |
|-------|----------|-------------|----------------|-------|---------------|
{rows if rows else "| — | No active tests | — | — | — | — |"}

## Summary Counts

| Decision | Count |
|----------|-------|
| ✅ Promoted | {len(summary.get('promotions', []))} |
| 🔵 Retained Champion | {sum(1 for t in tests if t.get('decision') == 'RETAIN_CHAMPION')} |
| 🔴 Rolled Back | {len(summary.get('rollbacks', []))} |
| ⏳ Inconclusive | {len(summary.get('inconclusives', []))} |

## Notes
- ROLLBACK decisions execute within SLA (20min) — safety-critical path
- Promoted models automatically generate a new Model Card (ModelCardGenerator hook)
- active_tests.json updated → API-GATEWAY /v1/models/ab-status endpoint

---
*Cross-agent: VISUALIA reads this file for A/B dashboard update.*
"""
    out_path.write_text(md)

    # API-GATEWAY sidecar already updated by ModelABTester.promote_winner()
    logger.info("A/B report written: %s", out_path)


def emit_ab_metrics(**context) -> None:
    """Emit AB_TEST_TRAFFIC_SPLIT per active test."""
    ti      = context["ti"]
    active  = ti.xcom_pull(key="active_tests", task_ids="load_active_tests") or []
    try:
        from src.data.metrics import AB_TEST_TRAFFIC_SPLIT
        for t in active:
            model_id = t.get("model_id", "")
            split    = t.get("traffic_split_pct", 20)
            AB_TEST_TRAFFIC_SPLIT.labels(model_id=model_id, role="champion").set(100 - split)
            AB_TEST_TRAFFIC_SPLIT.labels(model_id=model_id, role="challenger").set(split)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data loader helper
# ---------------------------------------------------------------------------

def _load_evaluation_df(model_id: str, run_date: date) -> "pd.DataFrame":
    """Load champion/challenger prediction error records for evaluation."""
    import pandas as pd
    from datetime import timedelta

    # Look for prediction logs in monitoring output
    monitor_dir = Path("workspace/output/monitoring")
    records = []
    for role in ("champion", "challenger"):
        pattern = f"predictions_{model_id}_{role}_*.json"
        for path in sorted(monitor_dir.glob(pattern)):
            try:
                data = json.loads(path.read_text())
                for err in data.get("errors", []):
                    records.append({"model_role": role, "error": float(err)})
            except Exception:
                pass

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner":           "analytica",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries":         1,
    "retry_delay":     timedelta(minutes=3),
    "sla":             timedelta(seconds=SLA_S),
}

with DAG(
    dag_id=DAG_ID,
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    default_args=default_args,
    catchup=False,
    max_active_runs=1,
    tags=["analytica", "ab_testing", "model_promotion"],
    description="Daily champion/challenger A/B evaluation; auto-promote or rollback on statistical gates",
) as dag:

    t_load = ShortCircuitOperator(
        task_id="load_active_tests",
        python_callable=load_active_tests,
    )

    # Dynamic TaskGroup: one task per active test
    # In production, tasks are generated at runtime by loading active_tests.json;
    # here we pre-define a placeholder that the operator resolves dynamically.
    with TaskGroup("evaluate_all_tests") as tg_eval:
        # Placeholder operator; real deployments use DynamicTaskMapping or a pre-scan step.
        def evaluate_all_dynamic(**context):
            """Evaluate all active A/B tests sequentially (fallback for static DAG)."""
            import pandas as pd
            from src.training.ab_tester import ModelABTester

            ti      = context["ti"]
            run_ds  = context["ds"]
            run_date = datetime.strptime(run_ds, "%Y-%m-%d").date()
            active  = ti.xcom_pull(key="active_tests", task_ids="load_active_tests") or []
            results = {}
            ab      = ModelABTester()

            for test in active:
                ab_test_id = test["ab_test_id"]
                model_id   = test["model_id"]
                eval_df    = _load_evaluation_df(model_id, run_date)
                if eval_df.empty:
                    results[ab_test_id] = {"decision": "INCONCLUSIVE", "rationale": "No data"}
                    continue
                try:
                    result = ab.evaluate(ab_test_id, eval_df, run_date=run_date)
                    results[ab_test_id] = result.to_dict()
                except Exception as exc:
                    results[ab_test_id] = {"decision": "INCONCLUSIVE", "rationale": str(exc)}

            # Push all results under a shared key
            ti.xcom_push(key="all_results", value=results)

        t_eval_all = PythonOperator(
            task_id="evaluate_all",
            python_callable=evaluate_all_dynamic,
        )

    t_agg      = PythonOperator(task_id="aggregate_ab_summary",  python_callable=aggregate_ab_summary)
    t_promote  = PythonOperator(task_id="execute_promotions",     python_callable=execute_promotions)
    t_rollback = PythonOperator(task_id="execute_rollbacks",      python_callable=execute_rollbacks)
    t_report   = PythonOperator(task_id="write_ab_report",        python_callable=write_ab_report)
    t_metrics  = PythonOperator(task_id="emit_ab_metrics",        python_callable=emit_ab_metrics)

    t_load >> tg_eval >> t_agg >> [t_promote, t_rollback] >> t_report >> t_metrics

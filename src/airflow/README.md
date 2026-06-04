# ANALYTICA Airflow Retraining DAGs
## `src/airflow/retraining_dag.py` — Sprint 2 Documentation

> **Agent**: ANALYTICA | **Sprint**: 2 | **Repo path**: `src/airflow/retraining_dag.py`

---

## Overview

`retraining_dag.py` defines **four Airflow DAGs** that automate ML model retraining for the Tropi-Climate-Analytics platform. Each DAG wires `retraining_pipeline.run_scheduled_retrain()` to an Airflow `PythonOperator` and routes outcomes to the `#mlops-alerts` Slack channel via a downstream notification stub.

---

## DAG Summary

| DAG ID | Model | Schedule | UTC Time | Cron |
|---|---|---|---|---|
| `analytica_retrain_xgb_precip` | XGBoost precipitation nowcast | Weekly | Mon 18:00 | `0 18 * * 1` |
| `analytica_retrain_prophet_seasonal` | Prophet seasonal forecast | Monthly | 1st 19:00 | `0 19 1 * *` |
| `analytica_retrain_cnn_landcover` | ResNet+Attention CNN land-cover | Quarterly | 1st 20:00 | `0 20 1 */3 *` |
| `analytica_retrain_lstm_streamflow` | LSTM streamflow forecast | Monthly | 1st 19:00 | `0 19 1 * *` |

---

## DAG Topology

Each DAG follows an identical two-task pattern:

```
[retrain_{model}]  ──►  [mlops_slack_alert_{model}]
  PythonOperator           PythonOperator (Slack stub)
  trigger_rule: default    trigger_rule: all_done
```

### Task 1 — `retrain_{model}` (PythonOperator)

| Property | Value |
|---|---|
| `python_callable` | `run_scheduled_retrain(model_type)` |
| `depends_on_past` | `False` |
| `retries` | 2 |
| `retry_delay` | 10 minutes |
| `execution_timeout` | 6 hours |
| `on_failure_callback` | `sns_failure_callback` (SNS alert) |

**What `run_scheduled_retrain` does:**
1. Imports `analytics.retraining_pipeline.run_scheduled_retrain`
2. Passes `model_type`, `run_id`, `execution_dt`, and `mlflow_uri`
3. Internally: drift detection → challenger training → A/B evaluation → MLflow promotion
4. Pushes result summary to XCom key `retrain_result`

### Task 2 — `mlops_slack_alert_{model}` (PythonOperator / Slack stub)

| Property | Value |
|---|---|
| `trigger_rule` | `all_done` — fires even if `retrain_*` fails |
| Channel | `#mlops-alerts` |
| Webhook source | Airflow Variable `ANALYTICA_SLACK_MLOPS_WEBHOOK_URL` |

**Stub upgrade path:** Replace with `SlackWebhookOperator` from `apache-airflow-providers-slack` once the Airflow Connection `slack_mlops_alerts` is configured:

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

t_slack = SlackWebhookOperator(
    task_id               = "mlops_slack_alert_{model}",
    slack_webhook_conn_id = "slack_mlops_alerts",
    message               = "...",
    channel               = "#mlops-alerts",
)
```

---

## SNS on_failure_callback

`sns_failure_callback(context)` is attached to `DEFAULT_ARGS` and fires on any task failure.

**Behaviour:**
- Reads `ANALYTICA_SNS_FAILURE_TOPIC_ARN` and `ANALYTICA_SNS_REGION` from Airflow Variables
- Publishes a structured JSON message to the SNS topic
- Fails silently (non-blocking) if the topic ARN is not configured

**SNS message schema:**
```json
{
  "agent":          "ANALYTICA",
  "event":          "task_failure",
  "dag_id":         "analytica_retrain_xgb_precip",
  "task_id":        "retrain_xgb_precip",
  "run_id":         "scheduled__2026-06-01T18:00:00+00:00",
  "execution_date": "2026-06-01T18:00:00+00:00",
  "exception":      "...",
  "log_url":        "http://airflow.tropiclimate.id/...",
  "alerted_at":     "2026-06-01T18:05:22Z"
}
```

---

## Airflow Variables (configure before first run)

Set these via Airflow UI → **Admin → Variables** or with:
```bash
airflow variables set ANALYTICA_MLFLOW_TRACKING_URI http://mlflow.tropiclimate.id:5000
airflow variables set ANALYTICA_SNS_FAILURE_TOPIC_ARN arn:aws:sns:ap-southeast-3:123456789:analytica-failures
airflow variables set ANALYTICA_SNS_REGION ap-southeast-3
airflow variables set ANALYTICA_SLACK_MLOPS_WEBHOOK_URL https://hooks.slack.com/services/...
```

| Variable Key | Purpose | Default |
|---|---|---|
| `ANALYTICA_MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `ANALYTICA_SNS_FAILURE_TOPIC_ARN` | SNS topic for failure alerts | *(none — alerts skipped)* |
| `ANALYTICA_SNS_REGION` | AWS region for SNS | `ap-southeast-3` |
| `ANALYTICA_SLACK_MLOPS_WEBHOOK_URL` | Slack incoming webhook for #mlops-alerts | *(none — stub skipped)* |

---

## Deployment

### Copy DAG to Airflow DAGs folder

```bash
cp src/airflow/retraining_dag.py $AIRFLOW_HOME/dags/
# or mount src/airflow/ as the DAGs volume in docker-compose / Kubernetes
```

### Verify DAG load (no import errors)

```bash
airflow dags list | grep analytica_retrain
airflow dags test analytica_retrain_xgb_precip 2026-06-01
```

### Trigger manually

```bash
airflow dags trigger analytica_retrain_xgb_precip
airflow dags trigger analytica_retrain_prophet_seasonal
airflow dags trigger analytica_retrain_cnn_landcover
airflow dags trigger analytica_retrain_lstm_streamflow
```

### Install required providers

```bash
pip install apache-airflow-providers-amazon   # boto3/SNS callback
pip install apache-airflow-providers-slack    # SlackWebhookOperator (when upgrading stub)
```

---

## XCom Reference

| Pushed by task | Key | Type | Description |
|---|---|---|---|
| `retrain_{model}` | `retrain_result` | `dict` | Full retraining summary from `run_scheduled_retrain()` |

**`retrain_result` fields:**

| Field | Type | Description |
|---|---|---|
| `model_type` | str | Model identifier |
| `status` / `action` | str | `promoted`, `archived`, `no_op`, `stub_run` |
| `run_id` | str | MLflow run ID of challenger |
| `new_production_v` | str | New MLflow Production version (if promoted) |
| `drift_detected` | bool | Whether PSI/KS drift triggered retraining |
| `ab_win_rate` | float | Challenger win rate (0–1) across A/B rounds |
| `completed_at` | str | ISO-8601 completion timestamp |

---

## Related Files

| File | Purpose |
|---|---|
| `src/analytics/retraining_pipeline.py` | Core retraining logic, drift detection, A/B evaluation |
| `src/analytics/mlflow_setup.py` | MLflow tracking / registry setup |
| `src/analytics/models.py` | XGBoost, Prophet, CNN, LSTM model definitions |
| `src/airflow/dag_factory.py` | Extended 7-stage DAG factory (sprint 2 advanced version) |
| `docs/model_documentation.md` | Full model documentation (Model Card 2.0) |

---

## Contacts

| Role | Contact |
|---|---|
| ML Engineering | ANALYTICA Agent |
| Infrastructure / Airflow | CLOUD-FORGE Agent |
| Orchestration & Governance | CLIMATE-OS |

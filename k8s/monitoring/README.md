# Monitoring Stack README — CLOUD-FORGE

## Stack
- **kube-prometheus-stack** (Prometheus + Grafana + Alertmanager + node-exporter + kube-state-metrics)
- **Custom PrometheusRule** for Tropi pipeline SLA alerts

## Deploy
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  -f k8s/monitoring/values.yaml

# Apply pipeline SLA alert rules
kubectl apply -f k8s/monitoring/alerts/pipeline-sla.yaml
```

## Pipeline SLA alert rules (`alerts/pipeline-sla.yaml`)

| Alert | Threshold | Severity | Metric |
|---|---|---|---|
| `TropiIngestionLagHigh` | last success > 30 min | warning | `tropi_pipeline_last_ingestion_success_timestamp_seconds` |
| `TropiIngestionLagCritical` | last success > 1h | critical | same |
| `TropiKafkaConsumerLagHigh` | lag > 50k messages | warning | `kafka_consumer_group_lag` |
| `TropiQPEIngestionStale` | QPE pipeline > 30 min | critical | pipeline label filter |
| `TropiModelInferenceP99High` | p99 > 5s | warning | `tropi_model_inference_duration_seconds` |
| `TropiModelInferenceP99Critical` | p99 > 15s | critical | same |
| `TropiModelServingDown` | pod unreachable | critical | `up` |
| `TropiFloodAlertDeliveryLate` | p95 delivery > 2 min | critical | `tropi_flood_alert_delivery_duration_seconds` |
| `TropiFloodAlertBPBDWebhookFailing` | webhook errors > 0 | critical | `tropi_flood_alert_webhook_failures_total` |
| `TropiFloodAlertPipelineStale` | no run > 30 min | critical | pipeline label filter |
| `TropiSLOBurnRateCritical` | 14x budget burn (1h) | critical | `http_requests_total` |
| `TropiSLOBurnRateWarning` | 6x budget burn (6h) | warning | same |

## Required application metrics
Each subsystem must export these Prometheus metrics:

**DATA-FLOW / HYDROLOGIS (Airflow workers):**
```
tropi_pipeline_last_ingestion_success_timestamp_seconds{pipeline="<dag_id>"}
```
Set as a gauge on every successful Airflow task via `airflow.stats` or a custom Pushgateway job.

**ANALYTICA (model serving pods):**
```
tropi_model_inference_duration_seconds_bucket{model="<model_name>",le="..."}
tropi_model_inference_duration_seconds_count{model="<model_name>"}
```
Expose as a Prometheus histogram from the FastAPI serving layer.

**HYDROLOGIS (flood alert pipeline):**
```
tropi_flood_alert_delivery_duration_seconds_bucket{river="ciliwung|brantas|solo",le="..."}
tropi_flood_alert_webhook_failures_total{river="...",endpoint="bpbd"}
```
Measure from threshold breach timestamp to confirmed BPBD webhook 2xx response.

## Alertmanager routing
- `critical` severity → PagerDuty (immediate)
- `warning` severity → Slack `#platform-alerts`
- `TropiFloodAlertDeliveryLate` → dedicated BPBD webhook receiver (0s group wait)

Populate placeholder values in `values.yaml` via Secrets Manager after deploy.

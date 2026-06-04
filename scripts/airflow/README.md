# Airflow Variable Provisioning — CLOUD-FORGE

Two-step process to bring Airflow variables live after `terraform apply`.

## Step 1 — Populate AWS Secrets Manager

Run once per environment after Terraform has created the empty secret shells:

```bash
# Interactive population (prompts for each value securely)
bash scripts/airflow/populate_secrets_template.sh --env prod
```

Secrets written to paths under `/${project}/${env}/airflow/`:

| Airflow Variable | Secret Suffix | Source |
|---|---|---|
| `NASA_EARTHDATA_TOKEN` | `NASA_EARTHDATA_TOKEN` | [Earthdata login](https://urs.earthdata.nasa.gov/) → Generate Token |
| `BMKG_API_KEY` | `BMKG_API_KEY` | BMKG Open Data Portal → API Keys |
| `JWT_SECRET` | `JWT_SECRET` | Generate: `openssl rand -hex 32` |
| `TROPI_API_BASE_URL` | `TROPI_API_BASE_URL` | `https://api.tropi-climate.id` (or staging URL) |
| `KAFKA_BOOTSTRAP_BROKERS` | `KAFKA_BOOTSTRAP_BROKERS` | MSK console → Bootstrap servers (TLS) |

## Step 2 — Push to Airflow

```bash
# Dry run first
bash scripts/airflow/provision_airflow_variables.sh --env prod --dry-run

# Push to Airflow via CLI (on the Airflow host)
bash scripts/airflow/provision_airflow_variables.sh --env prod

# OR via Airflow REST API (remote)
export AIRFLOW_API_BASE_URL=https://airflow.tropi-climate.id
export AIRFLOW_BASIC_AUTH=admin:$(aws secretsmanager get-secret-value \
  --secret-id /tropi-climate/prod/airflow/AIRFLOW_ADMIN_PASSWORD \
  --query 'SecretString' --output text | jq -r '.value')
bash scripts/airflow/provision_airflow_variables.sh --env prod --rest-api
```

## Terraform module

The Secrets Manager module at `infrastructure/terraform/modules/secrets/` provisions:
- KMS key with automatic annual rotation
- Empty secret shells for all Airflow DAG variables + runtime secrets
- IAM policy `airflow_secrets_policy_arn` — attach to Airflow worker task role
- IAM policy `ecs_secrets_policy_arn` — attach to ECS API/ETL task roles

## Secrets rotation

`JWT_SECRET`, `DB_PASSWORD`, `REDIS_AUTH_TOKEN`, `GRAFANA_ADMIN_PASSWORD`, and `ELASTIC_PASSWORD` have `Rotation = true` in the Terraform tags. Wire AWS Secrets Manager rotation Lambda (or use Terraform `aws_secretsmanager_secret_rotation`) after initial deployment.

## Security notes

- Never commit actual secret values — the `populate_secrets_template.sh` reads them interactively
- Secrets Manager uses a dedicated KMS key (`alias/tropi-climate-prod-secrets`) with annual key rotation
- Airflow workers get least-privilege read-only access via the IAM policy attached to their ECS task role
- Recovery window is 7 days — accidental deletions can be recovered

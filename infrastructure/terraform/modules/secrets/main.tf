# ============================================================
# Secrets Manager Module — Tropi Climate Analytics
# Provisions all runtime secrets referenced by Airflow DAGs,
# API services, and ETL workers. Secrets are created as
# empty shells (SecretString = "{}") — values are populated
# out-of-band via CI/CD or manual ops rotation, never in TF.
# ============================================================

variable "environment" {}
variable "project"     {}
variable "aws_region"  { default = "ap-southeast-3" }

locals {
  # ── Airflow DAG variables (HYDROLOGIS pipelines) ──────────
  airflow_secrets = {
    nasa_earthdata_token = {
      name        = "/${var.project}/${var.environment}/airflow/NASA_EARTHDATA_TOKEN"
      description = "NASA Earthdata bearer token for CMR/SMAP/GPM/GRACE-FO API access"
      rotation    = false
    }
    bmkg_api_key = {
      name        = "/${var.project}/${var.environment}/airflow/BMKG_API_KEY"
      description = "BMKG Open Data API key for ground-station gauge ingestion and kriging fusion"
      rotation    = false
    }
    jwt_secret = {
      name        = "/${var.project}/${var.environment}/airflow/JWT_SECRET"
      description = "JWT signing secret for Tropi Climate API authentication (HMAC-SHA256)"
      rotation    = true
    }
    tropi_api_base_url = {
      name        = "/${var.project}/${var.environment}/airflow/TROPI_API_BASE_URL"
      description = "Internal API base URL for inter-service DAG callbacks (e.g. https://api.tropi-climate.id)"
      rotation    = false
    }
    kafka_bootstrap_brokers = {
      name        = "/${var.project}/${var.environment}/airflow/KAFKA_BOOTSTRAP_BROKERS"
      description = "MSK Kafka TLS bootstrap broker string for all Airflow pipeline outputs"
      rotation    = false
    }
  }

  # ── Additional runtime secrets (API, ETL, DB) ─────────────
  runtime_secrets = {
    db_password = {
      name        = "/${var.project}/${var.environment}/rds/DB_PASSWORD"
      description = "Aurora PostgreSQL master password"
      rotation    = true
    }
    redis_auth_token = {
      name        = "/${var.project}/${var.environment}/elasticache/REDIS_AUTH_TOKEN"
      description = "ElastiCache Redis AUTH token (transit encryption)"
      rotation    = true
    }
    kafka_truststore_password = {
      name        = "/${var.project}/${var.environment}/msk/KAFKA_TRUSTSTORE_PASSWORD"
      description = "MSK Kafka TLS truststore JKS password"
      rotation    = false
    }
    grafana_admin_password = {
      name        = "/${var.project}/${var.environment}/monitoring/GRAFANA_ADMIN_PASSWORD"
      description = "Grafana admin console password"
      rotation    = true
    }
    elastic_password = {
      name        = "/${var.project}/${var.environment}/elk/ELASTIC_PASSWORD"
      description = "Elasticsearch superuser password (xpack.security)"
      rotation    = true
    }
    bpbd_alert_webhook_url = {
      name        = "/${var.project}/${var.environment}/integrations/BPBD_ALERT_WEBHOOK_URL"
      description = "BPBD (National Disaster Management Authority) webhook for flood early-warning push"
      rotation    = false
    }
    mlflow_tracking_uri = {
      name        = "/${var.project}/${var.environment}/mlops/MLFLOW_TRACKING_URI"
      description = "MLflow tracking server URI for ANALYTICA model runs"
      rotation    = false
    }
  }

  all_secrets = merge(local.airflow_secrets, local.runtime_secrets)
}

# ── KMS Key for secret encryption ────────────────────────────
resource "aws_kms_key" "secrets" {
  description             = "${var.project}-${var.environment} Secrets Manager KMS key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags = { Name = "${var.project}-${var.environment}-secrets-kms" }
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.project}-${var.environment}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# ── Secret resources ──────────────────────────────────────────
resource "aws_secretsmanager_secret" "secrets" {
  for_each    = local.all_secrets
  name        = each.value.name
  description = each.value.description
  kms_key_id  = aws_kms_key.secrets.arn

  # Prevent accidental deletion of live secrets
  recovery_window_in_days = 7

  tags = {
    SecretType  = each.key
    Component   = startswith(each.key, "nasa") || startswith(each.key, "bmkg") || startswith(each.key, "jwt") || startswith(each.key, "tropi") || startswith(each.key, "kafka_bootstrap") ? "airflow" : "runtime"
    Rotation    = each.value.rotation ? "true" : "false"
  }
}

# ── IAM policy: Airflow workers can read Airflow secrets ──────
resource "aws_iam_policy" "airflow_secrets_read" {
  name        = "${var.project}-${var.environment}-airflow-secrets-read"
  description = "Allow Airflow task workers to read DAG secrets from Secrets Manager"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadAirflowSecrets"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = [
          for k, v in local.airflow_secrets :
          aws_secretsmanager_secret.secrets[k].arn
        ]
      },
      {
        Sid    = "DecryptWithKMS"
        Effect = "Allow"
        Action = ["kms:Decrypt", "kms:GenerateDataKey"]
        Resource = [aws_kms_key.secrets.arn]
      }
    ]
  })
}

# ── IAM policy: ECS task role reads all runtime secrets ───────
resource "aws_iam_policy" "ecs_secrets_read" {
  name        = "${var.project}-${var.environment}-ecs-secrets-read"
  description = "Allow ECS API/ETL tasks to read runtime secrets"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadRuntimeSecrets"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
        Resource = [
          for k, v in local.all_secrets :
          aws_secretsmanager_secret.secrets[k].arn
        ]
      },
      {
        Sid    = "DecryptWithKMS"
        Effect = "Allow"
        Action = ["kms:Decrypt", "kms:GenerateDataKey"]
        Resource = [aws_kms_key.secrets.arn]
      }
    ]
  })
}

# ── Outputs ───────────────────────────────────────────────────
output "kms_key_arn"     { value = aws_kms_key.secrets.arn }
output "kms_key_id"      { value = aws_kms_key.secrets.key_id }
output "secret_arns" {
  description = "Map of secret key → ARN for all provisioned secrets"
  value       = { for k, v in aws_secretsmanager_secret.secrets : k => v.arn }
  sensitive   = true
}
output "airflow_secrets_policy_arn" { value = aws_iam_policy.airflow_secrets_read.arn }
output "ecs_secrets_policy_arn"     { value = aws_iam_policy.ecs_secrets_read.arn }

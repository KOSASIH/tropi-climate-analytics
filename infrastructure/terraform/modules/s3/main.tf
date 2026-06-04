# ============================================================
# S3 Module — Data lake, processed data, backups, logs, artifacts
# ============================================================

variable "environment" {}
variable "project"     {}
variable "aws_region"  { default = "ap-southeast-3" }

locals {
  buckets = {
    satellite_data  = "${var.project}-${var.environment}-satellite-data"
    processed_data  = "${var.project}-${var.environment}-processed-data"
    backup          = "${var.project}-${var.environment}-backups"
    logs            = "${var.project}-${var.environment}-logs"
    ml_artifacts    = "${var.project}-${var.environment}-ml-artifacts"
    terraform_state = "${var.project}-terraform-state"
  }
}

resource "aws_s3_bucket" "buckets" {
  for_each = local.buckets
  bucket   = each.value
  tags     = { Purpose = each.key }
}

resource "aws_s3_bucket_versioning" "buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.buckets[each.key].id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "buckets" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.buckets[each.key].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "buckets" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.buckets[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle: move satellite raw data to Glacier after 90 days
resource "aws_s3_bucket_lifecycle_configuration" "satellite_data" {
  bucket = aws_s3_bucket.buckets["satellite_data"].id
  rule {
    id     = "satellite-data-tiering"
    status = "Enabled"
    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
  }
}

# Cross-region replication to Singapore (ap-southeast-1) for DR
resource "aws_s3_bucket_replication_configuration" "satellite_data_dr" {
  bucket = aws_s3_bucket.buckets["satellite_data"].id
  role   = aws_iam_role.s3_replication.arn
  rule {
    id     = "dr-replication-singapore"
    status = "Enabled"
    destination {
      bucket        = "arn:aws:s3:::${var.project}-${var.environment}-satellite-data-dr-sg"
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_iam_role" "s3_replication" {
  name               = "${var.project}-${var.environment}-s3-replication"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
    }]
  })
}

output "satellite_data_bucket" { value = aws_s3_bucket.buckets["satellite_data"].id }
output "processed_data_bucket" { value = aws_s3_bucket.buckets["processed_data"].id }
output "backup_bucket"         { value = aws_s3_bucket.buckets["backup"].id }
output "ml_artifacts_bucket"   { value = aws_s3_bucket.buckets["ml_artifacts"].id }
output "logs_bucket"           { value = aws_s3_bucket.buckets["logs"].id }

# ============================================================
# Tropi Climate Analytics — Terraform Outputs
# ============================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "public_subnet_ids" {
  description = "Public subnet IDs (ALB)"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "Private subnet IDs (ECS tasks)"
  value       = module.vpc.private_subnet_ids
}

output "database_subnet_ids" {
  description = "Database subnet IDs (RDS)"
  value       = module.vpc.database_subnet_ids
}

output "ecs_cluster_arn" {
  description = "ECS Cluster ARN"
  value       = module.ecs.cluster_arn
}

output "rds_endpoint" {
  description = "RDS Aurora writer endpoint"
  value       = module.rds.writer_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint"
  value       = module.elasticache.primary_endpoint
  sensitive   = true
}

output "msk_bootstrap_brokers" {
  description = "MSK Kafka bootstrap broker string"
  value       = module.msk.bootstrap_brokers_tls
  sensitive   = true
}

output "s3_satellite_data_bucket" {
  description = "S3 bucket for raw satellite data"
  value       = module.s3.satellite_data_bucket
}

output "s3_processed_data_bucket" {
  description = "S3 bucket for processed outputs"
  value       = module.s3.processed_data_bucket
}

output "s3_backup_bucket" {
  description = "S3 bucket for database backups"
  value       = module.s3.backup_bucket
}

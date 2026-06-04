# ============================================================
# ElastiCache Redis Module — Multi-AZ with automatic failover
# ============================================================

variable "environment" {}
variable "project"     {}
variable "vpc_id"      {}
variable "subnet_ids"  { type = list(string) }
variable "node_type"   { default = "cache.r6g.large" }

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project}-${var.environment}-redis-subnet"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "redis" {
  name   = "${var.project}-${var.environment}-redis-sg"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Redis from VPC"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${var.project}-${var.environment}-redis"
  description                = "Tropi Climate Analytics Redis cache"
  node_type                  = var.node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  engine_version             = "7.1"
  num_cache_clusters         = 3
  automatic_failover_enabled = true
  multi_az_enabled           = true
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  snapshot_retention_limit   = 7
  snapshot_window            = "19:00-20:00"  # UTC
  maintenance_window         = "sun:20:00-sun:21:00"
  tags = { Name = "${var.project}-${var.environment}-redis" }
}

output "primary_endpoint" {
  value     = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive = true
}
output "reader_endpoint" {
  value     = aws_elasticache_replication_group.main.reader_endpoint_address
  sensitive = true
}

# ============================================================
# RDS Module — Aurora PostgreSQL + PostGIS, Multi-AZ
# ============================================================

variable "environment"    {}
variable "project"        {}
variable "vpc_id"         {}
variable "subnet_ids"     { type = list(string) }
variable "db_password"    { sensitive = true }
variable "instance_class" { default = "db.r6g.xlarge" }
variable "azs"            { type = list(string) }

resource "aws_db_subnet_group" "main" {
  name       = "${var.project}-${var.environment}-db-subnet-group"
  subnet_ids = var.subnet_ids
  tags       = { Name = "${var.project}-${var.environment}-db-subnet-group" }
}

resource "aws_security_group" "rds" {
  name   = "${var.project}-${var.environment}-rds-sg"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "PostgreSQL from VPC"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_rds_cluster" "main" {
  cluster_identifier      = "${var.project}-${var.environment}"
  engine                  = "aurora-postgresql"
  engine_version          = "15.4"
  database_name           = "tropi_climate"
  master_username         = "tropi_admin"
  master_password         = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.main.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  availability_zones      = var.azs
  backup_retention_period = 7
  preferred_backup_window = "18:00-19:00"  # UTC = 01:00-02:00 WIB
  storage_encrypted       = true
  deletion_protection     = true
  enabled_cloudwatch_logs_exports = ["postgresql"]

  # PostGIS extension enabled via parameter group below
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.main.name

  tags = { Name = "${var.project}-${var.environment}-aurora" }
}

resource "aws_rds_cluster_parameter_group" "main" {
  family      = "aurora-postgresql15"
  name        = "${var.project}-${var.environment}-pg15-params"
  description = "Aurora PostgreSQL 15 params with PostGIS support"
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,pgaudit"
  }
}

# Writer instance
resource "aws_rds_cluster_instance" "writer" {
  identifier           = "${var.project}-${var.environment}-writer"
  cluster_identifier   = aws_rds_cluster.main.id
  instance_class       = var.instance_class
  engine               = aws_rds_cluster.main.engine
  engine_version       = aws_rds_cluster.main.engine_version
  publicly_accessible  = false
  monitoring_interval  = 60
  monitoring_role_arn  = aws_iam_role.rds_enhanced_monitoring.arn
  performance_insights_enabled = true
  tags = { Role = "writer" }
}

# Reader instances (2 readers for HA)
resource "aws_rds_cluster_instance" "reader" {
  count                = 2
  identifier           = "${var.project}-${var.environment}-reader-${count.index}"
  cluster_identifier   = aws_rds_cluster.main.id
  instance_class       = var.instance_class
  engine               = aws_rds_cluster.main.engine
  engine_version       = aws_rds_cluster.main.engine_version
  publicly_accessible  = false
  monitoring_interval  = 60
  monitoring_role_arn  = aws_iam_role.rds_enhanced_monitoring.arn
  performance_insights_enabled = true
  tags = { Role = "reader" }
}

resource "aws_iam_role" "rds_enhanced_monitoring" {
  name               = "${var.project}-${var.environment}-rds-monitoring"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "monitoring.rds.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

output "writer_endpoint"  { value = aws_rds_cluster.main.endpoint sensitive = true }
output "reader_endpoint"  { value = aws_rds_cluster.main.reader_endpoint sensitive = true }
output "cluster_id"       { value = aws_rds_cluster.main.cluster_identifier }

# ============================================================
# Tropi Climate Analytics — Terraform Root Module
# Region: ap-southeast-3 (Jakarta) | Multi-AZ: a, b, c
# Managed by: CLOUD-FORGE
# ============================================================

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket         = "tropi-climate-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-southeast-3"
    dynamodb_table = "tropi-climate-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "tropi-climate-analytics"
      Environment = var.environment
      ManagedBy   = "CLOUD-FORGE"
      Team        = "Platform"
    }
  }
}

# ─── VPC ────────────────────────────────────────────────────────────────
module "vpc" {
  source      = "./modules/vpc"
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
  azs         = var.availability_zones
  project     = var.project_name
}

# ─── ECS Cluster ────────────────────────────────────────────────────────
module "ecs" {
  source             = "./modules/ecs"
  environment        = var.environment
  project            = var.project_name
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
}

# ─── RDS Aurora PostgreSQL + PostGIS ────────────────────────────────────
module "rds" {
  source             = "./modules/rds"
  environment        = var.environment
  project            = var.project_name
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.database_subnet_ids
  db_password        = var.db_password
  instance_class     = var.rds_instance_class
  azs                = var.availability_zones
}

# ─── ElastiCache Redis ──────────────────────────────────────────────────
module "elasticache" {
  source      = "./modules/elasticache"
  environment = var.environment
  project     = var.project_name
  vpc_id      = module.vpc.vpc_id
  subnet_ids  = module.vpc.private_subnet_ids
  node_type   = var.redis_node_type
}

# ─── MSK Kafka ──────────────────────────────────────────────────────────
module "msk" {
  source             = "./modules/msk"
  environment        = var.environment
  project            = var.project_name
  vpc_id             = module.vpc.vpc_id
  subnet_ids         = module.vpc.private_subnet_ids
  instance_type      = var.kafka_instance_type
  kafka_version      = "3.6.0"
  number_of_brokers  = 3
}

# ─── S3 Buckets ─────────────────────────────────────────────────────────
module "s3" {
  source      = "./modules/s3"
  environment = var.environment
  project     = var.project_name
  aws_region  = var.aws_region
}

# ─── Secrets Manager ────────────────────────────────────────────────────
module "secrets" {
  source      = "./modules/secrets"
  environment = var.environment
  project     = var.project_name
  aws_region  = var.aws_region
}

# ==============================================================
# EKS Module Variables — Tropi Climate Analytics
# ==============================================================

variable "environment" {
  description = "Deployment environment (prod / staging / dev)"
  type        = string
}

variable "project" {
  description = "Project name slug used as resource name prefix"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-3"
}

variable "vpc_id" {
  description = "VPC ID for EKS cluster networking"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for EKS control-plane ENIs (private + public)"
  type        = list(string)
}

variable "node_subnet_ids" {
  description = "Private subnet IDs for worker nodes (multi-AZ)"
  type        = list(string)
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

variable "gpu_instance_types" {
  description = "GPU instance types for ANALYTICA training node group"
  type        = list(string)
  default     = ["g4dn.xlarge", "g4dn.2xlarge", "p3.2xlarge"]
}

variable "cpu_instance_types" {
  description = "CPU instance types for ingestion/ETL node group (on-demand)"
  type        = list(string)
  default     = ["m5.2xlarge", "m5.4xlarge", "m5a.2xlarge", "m4.2xlarge"]
}

variable "spot_instance_types" {
  description = "Instance types for spot batch node group"
  type        = list(string)
  default     = ["c5.4xlarge", "c5.2xlarge", "c5a.4xlarge", "c5n.4xlarge", "m5.4xlarge"]
}

variable "system_min_nodes"     { type = number; default = 2 }
variable "ingestion_min_nodes"  { type = number; default = 2 }
variable "ingestion_max_nodes"  { type = number; default = 12 }
variable "gpu_max_nodes"        { type = number; default = 4 }
variable "spot_batch_max_nodes" { type = number; default = 20 }

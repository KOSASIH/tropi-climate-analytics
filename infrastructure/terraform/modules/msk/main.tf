# ============================================================
# MSK Kafka Module — 3 brokers across 3 AZs
# ============================================================

variable "environment"       {}
variable "project"           {}
variable "vpc_id"            {}
variable "subnet_ids"        { type = list(string) }
variable "instance_type"     { default = "kafka.m5.large" }
variable "kafka_version"     { default = "3.6.0" }
variable "number_of_brokers" { default = 3 }

resource "aws_security_group" "msk" {
  name   = "${var.project}-${var.environment}-msk-sg"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 9094
    to_port     = 9094
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Kafka TLS from VPC"
  }
  ingress {
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
    description = "Kafka plaintext from VPC"
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_msk_configuration" "main" {
  kafka_versions = [var.kafka_version]
  name           = "${var.project}-${var.environment}-msk-config"
  server_properties = <<-PROPS
    auto.create.topics.enable=false
    delete.topic.enable=true
    log.retention.hours=168
    num.partitions=6
    default.replication.factor=3
    min.insync.replicas=2
    log.segment.bytes=1073741824
  PROPS
}

resource "aws_msk_cluster" "main" {
  cluster_name           = "${var.project}-${var.environment}-kafka"
  kafka_version          = var.kafka_version
  number_of_broker_nodes = var.number_of_brokers

  broker_node_group_info {
    instance_type  = var.instance_type
    client_subnets = var.subnet_ids
    storage_info {
      ebs_storage_info { volume_size = 500 }
    }
    security_groups = [aws_security_group.msk.id]
  }

  configuration_info {
    arn      = aws_msk_configuration.main.arn
    revision = aws_msk_configuration.main.latest_revision
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  enhanced_monitoring = "PER_TOPIC_PER_BROKER"

  open_monitoring {
    prometheus {
      jmx_exporter  { enabled_in_broker = true }
      node_exporter { enabled_in_broker = true }
    }
  }

  logging {
    broker_logs {
      cloudwatch_logs { enabled = true log_group = "/aws/msk/${var.project}-${var.environment}" }
      s3 { enabled = true bucket = "${var.project}-${var.environment}-logs" prefix = "msk/" }
    }
  }

  tags = { Name = "${var.project}-${var.environment}-msk" }
}

output "bootstrap_brokers_tls" {
  value     = aws_msk_cluster.main.bootstrap_brokers_tls
  sensitive = true
}
output "zookeeper_connect" {
  value     = aws_msk_cluster.main.zookeeper_connect_string
  sensitive = true
}
output "cluster_arn" { value = aws_msk_cluster.main.arn }

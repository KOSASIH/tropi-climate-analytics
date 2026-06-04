# ============================================================
# ECS Module — Fargate cluster with Container Insights
# ============================================================

variable "environment"        {}
variable "project"            {}
variable "vpc_id"             {}
variable "private_subnet_ids" { type = list(string) }
variable "public_subnet_ids"  { type = list(string) }

resource "aws_ecs_cluster" "main" {
  name = "${var.project}-${var.environment}"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = { Name = "${var.project}-${var.environment}-ecs" }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  default_capacity_provider_strategy {
    base              = 2
    weight            = 80
    capacity_provider = "FARGATE"
  }
  default_capacity_provider_strategy {
    weight            = 20
    capacity_provider = "FARGATE_SPOT"
  }
}

# ─── ALB for API traffic ───────────────────────────────────────
resource "aws_lb" "api" {
  name               = "${var.project}-${var.environment}-api-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  tags               = { Name = "${var.project}-${var.environment}-api-alb" }
}

resource "aws_lb_target_group" "api" {
  name        = "${var.project}-${var.environment}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "api_https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

output "cluster_arn"      { value = aws_ecs_cluster.main.arn }
output "cluster_name"     { value = aws_ecs_cluster.main.name }
output "api_alb_dns_name" { value = aws_lb.api.dns_name }
output "api_target_group_arn" { value = aws_lb_target_group.api.arn }

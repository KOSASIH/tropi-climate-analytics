# ==============================================================
# Karpenter Autoscaler — AWS resources
# IRSA, SQS interruption queue, EventBridge rules
# K8s NodePool / EC2NodeClass CRDs live in k8s/karpenter/
# ==============================================================

locals {
  karpenter_namespace       = "karpenter"
  karpenter_service_account = "karpenter"
}

# ── IRSA trust policy ─────────────────────────────────────────
data "aws_iam_policy_document" "karpenter_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.eks.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub"
      values   = ["system:serviceaccount:${local.karpenter_namespace}:${local.karpenter_service_account}"]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "karpenter" {
  name               = "${var.project}-${var.environment}-karpenter"
  assume_role_policy = data.aws_iam_policy_document.karpenter_trust.json
}

resource "aws_iam_policy" "karpenter" {
  name        = "${var.project}-${var.environment}-karpenter-controller"
  description = "Karpenter controller — EC2 node provisioning and interruption handling"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EC2ProvisioningActions"
        Effect = "Allow"
        Action = [
          "ec2:RunInstances", "ec2:CreateFleet",
          "ec2:CreateLaunchTemplate", "ec2:DeleteLaunchTemplate",
          "ec2:TerminateInstances", "ec2:CreateTags",
        ]
        Resource = "*"
      },
      {
        Sid    = "EC2ReadActions"
        Effect = "Allow"
        Action = [
          "ec2:DescribeImages", "ec2:DescribeInstances",
          "ec2:DescribeInstanceTypes", "ec2:DescribeInstanceTypeOfferings",
          "ec2:DescribeAvailabilityZones", "ec2:DescribeSpotPriceHistory",
          "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups",
          "ec2:DescribeLaunchTemplates", "ec2:DescribeTags",
        ]
        Resource = "*"
      },
      {
        Sid      = "IAMPassRole"
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.node.arn]
      },
      {
        Sid      = "SSMReadActions"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = ["arn:aws:ssm:${var.aws_region}::parameter/aws/service/*"]
      },
      {
        Sid      = "PricingRead"
        Effect   = "Allow"
        Action   = ["pricing:GetProducts"]
        Resource = "*"
      },
      {
        Sid      = "InterruptionQueue"
        Effect   = "Allow"
        Action   = ["sqs:DeleteMessage", "sqs:GetQueueUrl", "sqs:GetQueueAttributes", "sqs:ReceiveMessage"]
        Resource = [aws_sqs_queue.karpenter_interruption.arn]
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "karpenter" {
  role       = aws_iam_role.karpenter.name
  policy_arn = aws_iam_policy.karpenter.arn
}

# ── SQS interruption queue ────────────────────────────────────
resource "aws_sqs_queue" "karpenter_interruption" {
  name                      = "${var.project}-${var.environment}-karpenter-interruption"
  message_retention_seconds = 300
  sqs_managed_sse_enabled   = true
  tags                      = { Name = "${var.project}-${var.environment}-karpenter-interruption" }
}

resource "aws_sqs_queue_policy" "karpenter_interruption" {
  queue_url = aws_sqs_queue.karpenter_interruption.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = ["events.amazonaws.com", "sqs.amazonaws.com"] }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.karpenter_interruption.arn
    }]
  })
}

# ── EventBridge interruption rules ───────────────────────────
locals {
  interruption_events = {
    spot_interruption  = { source = "aws.ec2";     detail_type = "EC2 Spot Instance Interruption Warning" }
    instance_rebalance = { source = "aws.ec2";     detail_type = "EC2 Instance Rebalance Recommendation" }
    state_change       = { source = "aws.ec2";     detail_type = "EC2 Instance State-change Notification" }
    health_event       = { source = "aws.health";  detail_type = "AWS Health Event" }
  }
}

resource "aws_cloudwatch_event_rule" "karpenter" {
  for_each    = local.interruption_events
  name        = "${var.project}-${var.environment}-karpenter-${each.key}"
  description = "Karpenter interruption: ${each.value.detail_type}"
  event_pattern = jsonencode({
    source      = [each.value.source]
    detail-type = [each.value.detail_type]
  })
}

resource "aws_cloudwatch_event_target" "karpenter" {
  for_each  = local.interruption_events
  rule      = aws_cloudwatch_event_rule.karpenter[each.key].name
  target_id = "KarpenterSQS"
  arn       = aws_sqs_queue.karpenter_interruption.arn
}

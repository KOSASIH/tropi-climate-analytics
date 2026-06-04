# ==============================================================
# EKS Cluster + Managed Node Groups — Tropi Climate Analytics
# Multi-AZ ap-southeast-3 (Jakarta)
#
# Node groups:
#   system        on-demand m5.xlarge  | taint: dedicated=system
#   cpu-ingestion on-demand m5.2xlarge | taint: dedicated=cpu-ingestion
#   gpu-training  on-demand g4dn/p3    | taint: dedicated=gpu + nvidia.com/gpu
#   spot-batch    spot c5/m5           | taint: dedicated=spot-batch
# ==============================================================

# ── KMS for envelope encryption of K8s secrets ────────────────
resource "aws_kms_key" "eks" {
  description             = "${var.project}-${var.environment} EKS envelope encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = { Name = "${var.project}-${var.environment}-eks-kms" }
}

# ── Cluster IAM Role ──────────────────────────────────────────
resource "aws_iam_role" "cluster" {
  name = "${var.project}-${var.environment}-eks-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cluster_policy" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role_policy_attachment" "cluster_vpc" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
}

# ── EKS Cluster ───────────────────────────────────────────────
resource "aws_eks_cluster" "main" {
  name     = "${var.project}-${var.environment}"
  role_arn = aws_iam_role.cluster.arn
  version  = var.cluster_version

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  encryption_config {
    provider  { key_arn = aws_kms_key.eks.arn }
    resources = ["secrets"]
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy,
    aws_iam_role_policy_attachment.cluster_vpc,
  ]
  tags = { Name = "${var.project}-${var.environment}-eks" }
}

# ── OIDC Provider (IRSA) ──────────────────────────────────────
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

# ── Node IAM Role ─────────────────────────────────────────────
resource "aws_iam_role" "node" {
  name = "${var.project}-${var.environment}-eks-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "node_policies" {
  for_each = toset([
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
    "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy",
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
  ])
  role       = aws_iam_role.node.name
  policy_arn = each.value
}

resource "aws_iam_instance_profile" "node" {
  name = "${var.project}-${var.environment}-eks-node-profile"
  role = aws_iam_role.node.name
}

# ── Node Group: system ────────────────────────────────────────
resource "aws_eks_node_group" "system" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "system"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.node_subnet_ids
  instance_types  = ["m5.xlarge"]
  capacity_type   = "ON_DEMAND"
  disk_size       = 50

  scaling_config { min_size = var.system_min_nodes; desired_size = var.system_min_nodes; max_size = 4 }
  update_config  { max_unavailable = 1 }

  labels = { role = "system", workload = "system" }
  taint  { key = "dedicated"; value = "system"; effect = "NO_SCHEDULE" }

  tags       = { Name = "${var.project}-${var.environment}-node-system" }
  depends_on = [aws_iam_role_policy_attachment.node_policies]
}

# ── Node Group: cpu-ingestion ─────────────────────────────────
# DATA-FLOW, HYDROLOGIS, GEOSPATIAL ETL workers
resource "aws_eks_node_group" "cpu_ingestion" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "cpu-ingestion"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.node_subnet_ids
  instance_types  = var.cpu_instance_types
  capacity_type   = "ON_DEMAND"
  disk_size       = 100

  scaling_config { min_size = var.ingestion_min_nodes; desired_size = var.ingestion_min_nodes; max_size = var.ingestion_max_nodes }
  update_config  { max_unavailable = 1 }

  labels = {
    role     = "ingestion"
    workload = "cpu-ingestion"
    "node.kubernetes.io/workload-type" = "cpu"
  }
  taint { key = "dedicated"; value = "cpu-ingestion"; effect = "NO_SCHEDULE" }

  tags       = { Name = "${var.project}-${var.environment}-node-cpu-ingestion" }
  depends_on = [aws_iam_role_policy_attachment.node_policies]
}

# ── Node Group: gpu-training ──────────────────────────────────
# ANALYTICA CNN/transformer training — scale-to-zero baseline
resource "aws_eks_node_group" "gpu_training" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-training"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = [var.node_subnet_ids[0]]   # single-AZ GPU placement locality
  instance_types  = var.gpu_instance_types
  capacity_type   = "ON_DEMAND"
  disk_size       = 200

  scaling_config { min_size = 0; desired_size = 0; max_size = var.gpu_max_nodes }
  update_config  { max_unavailable = 1 }

  labels = {
    role     = "ml-training"
    workload = "gpu-training"
    "node.kubernetes.io/workload-type" = "gpu"
    "k8s.amazonaws.com/accelerator"    = "nvidia-tesla-t4"
  }
  taint { key = "dedicated";    value = "gpu";     effect = "NO_SCHEDULE" }
  taint { key = "nvidia.com/gpu"; value = "present"; effect = "NO_SCHEDULE" }

  tags       = { Name = "${var.project}-${var.environment}-node-gpu-training" }
  depends_on = [aws_iam_role_policy_attachment.node_policies]
}

# ── Node Group: spot-batch ────────────────────────────────────
# Non-critical batch jobs, satellite tile preprocessing, backfills
resource "aws_eks_node_group" "spot_batch" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "spot-batch"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.node_subnet_ids
  instance_types  = var.spot_instance_types
  capacity_type   = "SPOT"
  disk_size       = 100

  scaling_config { min_size = 0; desired_size = 1; max_size = var.spot_batch_max_nodes }
  update_config  { max_unavailable_percentage = 33 }

  labels = {
    role     = "batch"
    workload = "spot-batch"
    "node.kubernetes.io/workload-type" = "spot"
    "karpenter.sh/capacity-type"       = "spot"
  }
  taint { key = "dedicated"; value = "spot-batch"; effect = "NO_SCHEDULE" }

  tags       = { Name = "${var.project}-${var.environment}-node-spot-batch" }
  depends_on = [aws_iam_role_policy_attachment.node_policies]
}

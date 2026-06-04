# ==============================================================
# EKS Module Outputs — Tropi Climate Analytics
# ==============================================================

output "cluster_name"     { value = aws_eks_cluster.main.name }
output "cluster_endpoint" { value = aws_eks_cluster.main.endpoint; sensitive = true }
output "cluster_ca_data"  { value = aws_eks_cluster.main.certificate_authority[0].data; sensitive = true }
output "cluster_version"  { value = aws_eks_cluster.main.version }

output "oidc_provider_arn" { value = aws_iam_openid_connect_provider.eks.arn }
output "oidc_provider_url" { value = replace(aws_iam_openid_connect_provider.eks.url, "https://", "") }

output "node_role_arn"              { value = aws_iam_role.node.arn }
output "node_instance_profile_name" { value = aws_iam_instance_profile.node.name }

output "karpenter_role_arn"                  { value = aws_iam_role.karpenter.arn }
output "karpenter_interruption_queue_url"    { value = aws_sqs_queue.karpenter_interruption.url }
output "karpenter_interruption_queue_name"   { value = aws_sqs_queue.karpenter_interruption.name }

output "gpu_node_group_name"          { value = aws_eks_node_group.gpu_training.node_group_name }
output "cpu_ingestion_node_group_name" { value = aws_eks_node_group.cpu_ingestion.node_group_name }
output "spot_batch_node_group_name"   { value = aws_eks_node_group.spot_batch.node_group_name }

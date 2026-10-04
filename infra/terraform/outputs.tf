output "cluster_arn" {
  value = aws_ecs_cluster.this.arn
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.job.arn
}

output "ecr_repository_url" {
  description = "Push the job image here: <url>:<image_tag>."
  value       = aws_ecr_repository.app.repository_url
}

output "subnets" {
  value = data.aws_subnets.default.ids
}

output "security_group_id" {
  value = aws_security_group.task.id
}

output "log_bucket" {
  value = aws_s3_bucket.logs.bucket
}

output "sns_topic_arn" {
  value = aws_sns_topic.notify.arn
}

output "ssm_parameters" {
  description = "Set the Alpaca PAPER keys here (aws ssm put-parameter --type SecureString --overwrite)."
  value       = [for p in aws_ssm_parameter.alpaca : p.name]
}

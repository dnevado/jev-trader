# Forward paper trading on AWS: EventBridge Scheduler → ECS Fargate task (python -m jevbt.aws_job trade|report)
# → Alpaca PAPER account, with SNS email notifications and run logs in S3. The container image is built with
# docker/Dockerfile and pushed to the ECR repository created here (tag = var.image_tag), see README.

data "aws_caller_identity" "current" {}

# Guard: never plan/apply this workspace with credentials of another account (e.g. a wrong aws_profile).
resource "terraform_data" "account_guard" {
  input = var.account_id
  lifecycle {
    precondition {
      condition     = data.aws_caller_identity.current.account_id == var.account_id
      error_message = "Credentials are for account ${data.aws_caller_identity.current.account_id}, but this workspace deploys to ${var.account_id}. Check aws_profile."
    }
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

locals {
  ssm_names = {
    key_id = "${var.ssm_prefix}alpaca_api_key_id"
    secret = "${var.ssm_prefix}alpaca_api_secret_key"
  }
  container = "app"
}

# ---------- secrets (values set outside Terraform so they never enter the state) ----------

resource "aws_ssm_parameter" "alpaca" {
  for_each    = local.ssm_names
  name        = each.value
  type        = "SecureString"
  value       = "CHANGE_ME"
  description = "Alpaca PAPER account ${each.key} for ${var.name} (set with aws ssm put-parameter --overwrite)"
  lifecycle {
    ignore_changes = [value]
  }
}

# ---------- logs bucket ----------

resource "aws_s3_bucket" "logs" {
  bucket_prefix = "${var.name}-logs-"
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket                  = aws_s3_bucket.logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------- email notifications ----------

resource "aws_sns_topic" "notify" {
  name = "${var.name}-notify"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.notify.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

resource "aws_sns_topic_policy" "notify" {
  arn = aws_sns_topic.notify.arn
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "EventBridgeFailureAlerts"
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sns:Publish"
      Resource  = aws_sns_topic.notify.arn
      Condition = { ArnEquals = { "aws:SourceArn" = aws_cloudwatch_event_rule.task_failed.arn } }
    }]
  })
}

# ---------- container image ----------

resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

# ---------- ECS ----------

resource "aws_ecs_cluster" "this" {
  name = var.name
}

resource "aws_cloudwatch_log_group" "task" {
  name              = "/ecs/${var.name}"
  retention_in_days = 90
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "task" {
  statement {
    sid       = "ReadAlpacaKeys"
    actions   = ["ssm:GetParameters"]
    resources = [for p in aws_ssm_parameter.alpaca : p.arn]
  }
  statement {
    sid       = "DecryptWithSsmDefaultKey"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.aws_region}.amazonaws.com"]
    }
  }
  statement {
    sid       = "WriteRunLogs"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.logs.arn}/paper/*"]
  }
  statement {
    sid       = "Notify"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.notify.arn]
  }
}

resource "aws_iam_role_policy" "task" {
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.task.json
}

resource "aws_ecs_task_definition" "job" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name      = local.container
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true
    command   = ["trade"]
    environment = [for k, v in {
      JEVBT_TICKERS       = join(",", var.tickers)
      JEVBT_STRATEGY      = var.strategy
      JEVBT_DIRECTION     = var.direction
      JEVBT_MAX_ALLOC     = tostring(var.max_alloc)
      JEVBT_MAX_GROSS     = tostring(var.max_gross)
      JEVBT_TIF           = "opg"
      JEVBT_REBALANCE     = var.rebalance
      JEVBT_LOG_BUCKET    = aws_s3_bucket.logs.bucket
      JEVBT_SNS_TOPIC_ARN = aws_sns_topic.notify.arn
      JEVBT_SSM_PREFIX    = var.ssm_prefix
      JEVBT_DATA_DIR      = "/tmp/data"
      ALPACA_DATA_FEED    = var.data_feed
      AWS_DEFAULT_REGION  = var.aws_region
    } : { name = k, value = v }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.task.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "job"
      }
    }
  }])
}

resource "aws_security_group" "task" {
  name        = "${var.name}-task"
  description = "Outbound only (Alpaca APIs, AWS APIs)"
  vpc_id      = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------- schedules ----------

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.name}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Action    = "ecs:RunTask"
        Resource  = "${replace(aws_ecs_task_definition.job.arn, "/:\\d+$/", "")}:*"
        Condition = { ArnEquals = { "ecs:cluster" = aws_ecs_cluster.this.arn } }
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = [aws_iam_role.execution.arn, aws_iam_role.task.arn]
      },
    ]
  })
}

resource "aws_scheduler_schedule" "runs" {
  for_each = {
    trade  = { expression = var.trade_schedule, retries = 0 } # orders: never retried automatically
    report = { expression = var.report_schedule, retries = 2 }
  }
  name                         = "${var.name}-${each.key}"
  schedule_expression          = each.value.expression
  schedule_expression_timezone = var.schedule_timezone
  flexible_time_window {
    mode = "OFF"
  }
  target {
    arn      = aws_ecs_cluster.this.arn
    role_arn = aws_iam_role.scheduler.arn
    input    = jsonencode({ containerOverrides = [{ name = local.container, command = [each.key] }] })
    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.job.arn
      launch_type         = "FARGATE"
      task_count          = 1
      network_configuration {
        subnets          = data.aws_subnets.default.ids
        security_groups  = [aws_security_group.task.id]
        assign_public_ip = true # default VPC public subnets: reach Alpaca without a NAT gateway
      }
    }
    retry_policy {
      maximum_retry_attempts = each.value.retries
    }
  }
}

# ---------- failure alerts (emails even if the job could not) ----------

resource "aws_cloudwatch_event_rule" "task_failed" {
  name        = "${var.name}-task-failed"
  description = "jevbt paper task stopped with a non-zero exit code or failed to start"
  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      clusterArn = [aws_ecs_cluster.this.arn]
      lastStatus = ["STOPPED"]
      "$or" = [
        { containers = { exitCode = [{ "anything-but" = 0 }] } },
        { stopCode = ["TaskFailedToStart"] },
      ]
    }
  })
}

resource "aws_cloudwatch_event_target" "task_failed" {
  rule      = aws_cloudwatch_event_rule.task_failed.name
  target_id = "email"
  arn       = aws_sns_topic.notify.arn
  input_transformer {
    input_paths = {
      group  = "$.detail.group"
      reason = "$.detail.stoppedReason"
      code   = "$.detail.stopCode"
      task   = "$.detail.taskArn"
    }
    input_template = "\"jevbt paper task FAILED (<group>): <code> - <reason>. Task <task>. Check CloudWatch logs /ecs/${var.name}.\""
  }
}

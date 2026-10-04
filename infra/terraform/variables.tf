variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "account_id" {
  description = "AWS account this workspace deploys to; plan/apply stop if the credentials belong to another account."
  type        = string
}

variable "aws_profile" {
  description = "AWS CLI / SSO profile used to deploy (null = default credentials chain)."
  type        = string
  default     = null
}

variable "name" {
  description = "Prefix for every resource name."
  type        = string
  default     = "jevbt-paper"
}

variable "notification_email" {
  description = "Address that receives order / fill / error emails (confirm the SNS subscription email once)."
  type        = string
}

variable "image_tag" {
  description = "Tag of the job image in the ECR repository (pushed with docker push before the first run)."
  type        = string
  default     = "v1"
}

variable "task_cpu" {
  description = "Fargate task CPU units (512 = 0.5 vCPU)."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 1024
}

variable "tickers" {
  description = "Universe traded on the paper account."
  type        = list(string)
}

variable "strategy" {
  type    = string
  default = "trend"
  validation {
    condition     = contains(["trend", "baseline"], var.strategy)
    error_message = "strategy must be trend or baseline."
  }
}

variable "direction" {
  type    = string
  default = "long"
  validation {
    condition     = contains(["long", "short", "both"], var.direction)
    error_message = "direction must be long, short or both."
  }
}

variable "max_alloc" {
  description = "Weight per position (1/12 ≈ 0.0833)."
  type        = number
  default     = 0.0833
}

variable "max_gross" {
  description = "Cap on gross exposure (1.0 = fully invested, no leverage)."
  type        = number
  default     = 1.0
}

variable "data_feed" {
  description = "Alpaca market data feed (iex on the free plan; sip works for bars older than 15 minutes)."
  type        = string
  default     = "iex"
}

variable "trade_schedule" {
  description = "EventBridge Scheduler cron for the trade run (the code only trades on the first session of the week)."
  type        = string
  default     = "cron(0 9 ? * MON-FRI *)"
}

variable "report_schedule" {
  description = "EventBridge Scheduler cron for the fills report (after the 09:30 open)."
  type        = string
  default     = "cron(0 10 ? * MON-FRI *)"
}

variable "schedule_timezone" {
  type    = string
  default = "America/New_York"
}

variable "ssm_prefix" {
  description = "Prefix of the SecureString parameters alpaca_api_key_id / alpaca_api_secret_key."
  type        = string
  default     = "/jevbt/"
}

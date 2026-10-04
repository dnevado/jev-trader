terraform {
  required_version = ">= 1.10" # S3 backend native locking (use_lockfile)

  # Partial configuration: terraform init -backend-config=backend.hcl (see backend.hcl.example).
  # Each workspace is stored under env:/<workspace>/<key>; one workspace per AWS account.
  backend "s3" {}
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.40"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
  default_tags {
    tags = {
      Project   = "jevbt-paper"
      ManagedBy = "terraform"
    }
  }
}

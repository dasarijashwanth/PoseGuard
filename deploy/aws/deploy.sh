#!/usr/bin/env bash
# Deploy the PoseGuard API to AWS ECS Fargate (Day 10).
#
# Run from the REPO ROOT:
#
#     bash deploy/aws/deploy.sh
#
# What it does, in order:
#   1. Builds the same slim Docker image as Day 9 (render.yaml /
#      experiments/model/Dockerfile, no torch, no mlflow).
#   2. Creates the ECR repo if missing, pushes the image there.
#   3. Backs the trained model artifacts up to S3
#      (s3://poseguard-models-<account-id>/models/), so a retrain
#      can ship without rebuilding the image from scratch.
#   4. Registers the ECS task definition with the real image URI.
#   5. Creates the ECS service on first run, updates it after.
#
# What it needs before it runs (see docs/DEPLOY_AWS.md):
#   - An AWS account, region us-east-1, and the AWS CLI signed in.
#   - The ECS cluster, security group, subnets, and load balancer
#     target group from docs/DEPLOY_AWS.md, with their IDs filled
#     into deploy/aws/service.json (the <...> placeholders).
#   - Docker installed locally. This machine (the build VM) has no
#     Docker, so this script has never been run end to end; the
#     files it consumes are covered by tests/test_aws.py instead.
#
# Nothing in this script touches the Render deploy from Day 9.
set -euo pipefail

REGION="us-east-1"
REPO="poseguard-api"
CLUSTER="poseguard"
SERVICE_FILE="deploy/aws/service.json"
TASK_FILE="deploy/aws/task-definition.json"

command -v aws >/dev/null || { echo "aws CLI not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
BUCKET="poseguard-models-${ACCOUNT_ID}"

echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"

# 1. ECR repo (create it once; skip if it already exists).
if ! aws ecr describe-repositories --repository-names "${REPO}" \
    --region "${REGION}" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "${REPO}" \
    --region "${REGION}" >/dev/null
  echo "Created ECR repo ${REPO}"
fi

# 2. Build and push the image.
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"
docker build -f experiments/model/Dockerfile -t "${REPO}:latest" .
docker tag "${REPO}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
echo "Pushed ${ECR_URI}:latest"

# 3. Back the trained model artifacts up to S3.
if ! aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
  aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}" >/dev/null
  echo "Created S3 bucket ${BUCKET}"
fi
aws s3 sync experiments/model/artifacts/ "s3://${BUCKET}/models/" --delete
echo "Synced model artifacts to s3://${BUCKET}/models/"

# 4. Register the task definition with the real image URI.
# Python does the substitution (no jq dependency): it reads the
# template, swaps the <account-id> placeholder, and writes a
# rendered copy that register-task-definition consumes.
RENDERED_TASK="$(mktemp)"
ACCOUNT_ID="${ACCOUNT_ID}" ECR_URI="${ECR_URI}" python3 - "$TASK_FILE" "$RENDERED_TASK" <<'EOF'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    task = json.load(fh)

task["executionRoleArn"] = task["executionRoleArn"].replace(
    "<account-id>", os.environ["ACCOUNT_ID"])
task["taskRoleArn"] = task["taskRoleArn"].replace(
    "<account-id>", os.environ["ACCOUNT_ID"])
for container in task["containerDefinitions"]:
    container["image"] = os.environ["ECR_URI"] + ":latest"

with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(task, fh, indent=2)
EOF
aws ecs register-task-definition --region "${REGION}" \
  --cli-input-json "file://${RENDERED_TASK}" >/dev/null
echo "Registered task definition poseguard-api"

# 5. Create the service on first run, update it after.
if aws ecs describe-services --cluster "${CLUSTER}" --services "${REPO}" \
    --region "${REGION}" --query 'services[0].status' --output text \
    2>/dev/null | grep -q ACTIVE; then
  aws ecs update-service --cluster "${CLUSTER}" --service "${REPO}" \
    --region "${REGION}" --force-new-deployment >/dev/null
  echo "Updated service ${REPO} (new deployment rolling out)"
else
  aws ecs create-service --region "${REGION}" \
    --cli-input-json "file://${SERVICE_FILE}" >/dev/null
  echo "Created service ${REPO}"
fi

echo "Done. Check status with:"
echo "  aws ecs describe-services --cluster ${CLUSTER} --services ${REPO} --region ${REGION}"

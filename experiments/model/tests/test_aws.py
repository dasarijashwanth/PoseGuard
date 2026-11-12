"""Tests for the Day 10 AWS deploy prep (deploy/aws/ at the repo
root). No AWS account, Docker, or network is available on this VM,
so these tests verify the definitions are consistent instead of
running a deploy: the task and service JSON parse and agree with
each other and with the Dockerfile, and the deploy script contains
every step it claims.
"""

import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODEL_DIR.parents[1]
AWS_DIR = REPO_ROOT / "deploy" / "aws"


def _task() -> dict:
    path = AWS_DIR / "task-definition.json"
    assert path.exists(), "task-definition.json missing"
    return json.loads(path.read_text(encoding="utf-8"))


def _service() -> dict:
    path = AWS_DIR / "service.json"
    assert path.exists(), "service.json missing"
    return json.loads(path.read_text(encoding="utf-8"))


def _deploy_script() -> str:
    path = AWS_DIR / "deploy.sh"
    assert path.exists(), "deploy.sh missing"
    return path.read_text(encoding="utf-8")


def test_task_definition_is_smallest_fargate():
    task = _task()
    assert task["family"] == "poseguard-api"
    assert task["requiresCompatibilities"] == ["FARGATE"]
    assert task["networkMode"] == "awsvpc"
    assert task["cpu"] == "256"
    assert task["memory"] == "512"
    assert task["runtimePlatform"]["cpuArchitecture"] == "X86_64"


def test_task_container_image_is_placeholder():
    task = _task()
    containers = task["containerDefinitions"]
    assert len(containers) == 1
    container = containers[0]
    assert container["name"] == "poseguard-api"
    assert container["essential"] is True
    # The real account ID belongs to his AWS account, never
    # hardcoded in the repo; deploy.sh substitutes it at run time.
    assert container["image"].startswith("<account-id>.dkr.ecr.")
    assert container["image"].endswith("/poseguard-api:latest")
    assert "<account-id>" in task["executionRoleArn"]
    assert "<account-id>" in task["taskRoleArn"]


def test_task_container_port_and_health_check():
    task = _task()
    container = task["containerDefinitions"][0]
    assert container["portMappings"] == [
        {"containerPort": 8000, "protocol": "tcp"}]
    health = container["healthCheck"]
    command = " ".join(health["command"])
    assert "/health" in command
    # The slim image has no curl; the check must use Python.
    assert "python" in command
    assert "curl" not in command
    assert health["interval"] == 30
    assert health["retries"] == 3


def test_task_logs_go_to_cloudwatch():
    task = _task()
    container = task["containerDefinitions"][0]
    logs = container["logConfiguration"]
    assert logs["logDriver"] == "awslogs"
    assert logs["options"]["awslogs-group"] == "/ecs/poseguard-api"
    assert logs["options"]["awslogs-region"] == "us-east-1"


def test_task_port_matches_dockerfile():
    dockerfile = (MODEL_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in dockerfile
    container = _task()["containerDefinitions"][0]
    assert container["portMappings"][0]["containerPort"] == 8000


def test_service_references_task_and_balancer():
    service = _service()
    assert service["cluster"] == "poseguard"
    assert service["serviceName"] == "poseguard-api"
    assert service["taskDefinition"] == "poseguard-api"
    assert service["launchType"] == "FARGATE"
    assert service["desiredCount"] == 1
    balancer = service["loadBalancers"][0]
    assert balancer["containerName"] == "poseguard-api"
    assert balancer["containerPort"] == 8000
    assert balancer["targetGroupArn"].startswith("<")


def test_service_networking_and_rollout():
    service = _service()
    vpc = service["networkConfiguration"]["awsvpcConfiguration"]
    assert vpc["assignPublicIp"] == "ENABLED"
    assert len(vpc["subnets"]) == 2
    assert all(s.startswith("<") for s in vpc["subnets"])
    assert len(vpc["securityGroups"]) == 1
    # A redeploy rolls in with no downtime.
    assert service["deploymentConfiguration"]["minimumHealthyPercent"] == 100
    assert service["deploymentConfiguration"]["maximumPercent"] == 200


def test_deploy_script_fails_fast_and_checks_tools():
    script = _deploy_script()
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    for tool in ("aws", "docker", "python3"):
        assert f"command -v {tool}" in script


def test_deploy_script_covers_ecr_build_push():
    script = _deploy_script()
    assert "aws ecr get-login-password" in script
    assert "docker build -f experiments/model/Dockerfile" in script
    assert "docker push" in script
    assert "create-repository" in script


def test_deploy_script_backs_models_up_to_s3():
    script = _deploy_script()
    assert "aws s3 sync experiments/model/artifacts/" in script
    assert "poseguard-models-" in script
    assert "create-bucket" in script


def test_deploy_script_registers_task_and_service():
    script = _deploy_script()
    assert "register-task-definition" in script
    assert "create-service" in script
    assert "update-service" in script
    assert "--force-new-deployment" in script
    # The image placeholder is substituted with Python, not jq.
    assert "os.environ" in script
    assert "containerDefinitions" in script


def test_aws_deploy_guide_exists_and_states_limits():
    guide = (REPO_ROOT / "docs" / "DEPLOY_AWS.md").read_text(
        encoding="utf-8")
    assert "deploy/aws/deploy.sh" in guide
    assert "roughly $25 a month" in guide
    assert "never been run end to end" in guide
    assert "No live URL exists" in guide

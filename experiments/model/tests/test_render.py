"""Tests for the Day 9 Render Blueprint (render.yaml at the repo
root). Render reads this file to create the service in one click, so
these tests check that it parses, names a Docker web service, points
at files that exist, and health-checks the endpoint the API serves.
"""

import re
from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODEL_DIR.parents[1]


def _blueprint() -> dict:
    path = REPO_ROOT / "render.yaml"
    assert path.exists(), "render.yaml missing from repo root"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _service() -> dict:
    data = _blueprint()
    assert isinstance(data.get("services"), list)
    assert len(data["services"]) == 1
    return data["services"][0]


def test_blueprint_has_one_docker_web_service():
    svc = _service()
    assert svc["type"] == "web"
    assert svc["runtime"] == "docker"
    assert svc["plan"] == "free"


def test_blueprint_service_name_is_valid():
    svc = _service()
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*", svc["name"]), svc["name"]


def test_blueprint_docker_paths_exist():
    svc = _service()
    dockerfile = REPO_ROOT / svc["dockerfilePath"]
    assert dockerfile.exists(), f"missing {dockerfile}"
    assert svc["dockerContext"] == "."
    assert "Dockerfile" in dockerfile.name


def test_blueprint_health_check_matches_api():
    svc = _service()
    assert svc["healthCheckPath"] == "/health"
    api = (MODEL_DIR / "api.py").read_text(encoding="utf-8")
    assert '@app.get("/health")' in api


def test_blueprint_env_vars_are_plain_key_values():
    svc = _service()
    env_vars = svc.get("envVars", [])
    assert isinstance(env_vars, list)
    for var in env_vars:
        assert set(var) == {"key", "value"}
        assert isinstance(var["key"], str) and var["key"]
        assert isinstance(var["value"], str)
    keys = {var["key"] for var in env_vars}
    assert "PYTHONUNBUFFERED" in keys

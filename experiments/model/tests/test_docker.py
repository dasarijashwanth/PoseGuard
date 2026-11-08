"""Smoke tests for the Day 6 Docker packaging (Dockerfile,
docker-compose.yml, .dockerignore). Docker itself is NOT available
on this VM, so these tests verify the definition is consistent
instead of running a build: the base image, the requirements file
it installs, the entrypoint it runs, and the compose services.
"""

from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODEL_DIR.parents[1]


def _read(name: str) -> str:
    return (MODEL_DIR / name).read_text(encoding="utf-8")


def test_dockerfile_base_and_entrypoint():
    text = _read("Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "COPY experiments/eval/requirements.txt" in text
    assert "pip install --no-cache-dir -r requirements.txt" in text
    assert "COPY experiments/ ./experiments/" in text
    assert "EXPOSE 8000" in text
    assert "uvicorn" in text
    assert "experiments.model.api:app" in text
    assert "--host" in text and "0.0.0.0" in text


def test_dockerfile_requirements_exist_and_include_runtime_deps():
    req = (REPO_ROOT / "experiments" / "eval" / "requirements.txt")
    assert req.exists()
    pkgs = {line.split(">")[0].split("=")[0].strip().lower()
            for line in req.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}
    for needed in ("fastapi", "uvicorn", "scikit-learn", "mlflow",
                   "pandas", "numpy"):
        assert needed in pkgs, f"{needed} missing from requirements.txt"


def test_dockerfile_entrypoint_module_exists():
    api = MODEL_DIR / "api.py"
    assert api.exists()
    assert "def create_app" in api.read_text(encoding="utf-8")


def test_compose_services_and_ports():
    compose = yaml.safe_load(_read("docker-compose.yml"))
    api = compose["services"]["api"]
    assert api["build"]["context"] == "../.."
    assert api["build"]["dockerfile"] == "experiments/model/Dockerfile"
    assert "8000:8000" in api["ports"]
    assert "./artifacts:/app/experiments/model/artifacts:ro" in api["volumes"]
    mlflow = compose["services"]["mlflow"]
    assert "5000:5000" in mlflow["ports"]
    assert "mlflow server" in mlflow["command"]
    assert "./mlruns:/mlruns" in mlflow["volumes"]


def test_dockerignore_keeps_local_state_out():
    ignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "experiments/model/mlruns/" in ignore
    assert "__pycache__/" in ignore
    assert ".git/" in ignore

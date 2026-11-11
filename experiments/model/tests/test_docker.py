"""Smoke tests for the Day 6 Docker packaging, updated Day 9 for the
Render deploy (Dockerfile, docker-compose.yml,.dockerignore).
Docker itself is NOT available on this VM, so these tests verify the
definition is consistent instead of running a build: the base image,
the requirements file it installs, the entrypoint it runs, the port
it binds, and the compose services.
"""

from pathlib import Path

import yaml

MODEL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODEL_DIR.parents[1]


def _read(name: str) -> str:
    return (MODEL_DIR / name).read_text(encoding="utf-8")


def _pkgs(path: Path) -> set:
    return {line.split(">")[0].split("=")[0].strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def test_dockerfile_base_and_entrypoint():
    text = _read("Dockerfile")
    assert "FROM python:3.12-slim" in text
    assert "ARG REQUIREMENTS=experiments/model/requirements-api.txt" in text
    assert "COPY ${REQUIREMENTS} ./requirements.txt" in text
    assert "pip install --no-cache-dir -r requirements.txt" in text
    assert "COPY experiments/ ./experiments/" in text
    assert "EXPOSE 8000" in text
    assert "uvicorn" in text
    assert "experiments.model.api:app" in text
    assert "--host" in text and "0.0.0.0" in text


def test_dockerfile_binds_render_port():
    # Render sets $PORT (10000 by default, not 8000). A hardcoded
    # port would start fine and never receive traffic.
    text = _read("Dockerfile")
    assert "${PORT:-8000}" in text
    assert '"--port", "8000"' not in text


def test_dockerfile_requirements_exist_and_include_runtime_deps():
    req = MODEL_DIR / "requirements-api.txt"
    assert req.exists()
    pkgs = _pkgs(req)
    for needed in ("fastapi", "uvicorn", "scikit-learn", "pandas",
                   "numpy", "joblib"):
        assert needed in pkgs, f"{needed} missing from requirements-api.txt"
    # The API never imports torch or mlflow; they made the old image
    # huge (default torch install is the multi-GB CUDA build).
    assert "torch" not in pkgs
    assert "mlflow" not in pkgs


def test_full_requirements_still_exist_for_mlflow():
    req = REPO_ROOT / "experiments" / "eval" / "requirements.txt"
    assert req.exists()
    assert "mlflow" in _pkgs(req)


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


def test_compose_mlflow_uses_full_requirements():
    # The slim API set has no mlflow, so the mlflow service must
    # build with the full eval requirements via the build arg.
    compose = yaml.safe_load(_read("docker-compose.yml"))
    mlflow = compose["services"]["mlflow"]
    assert mlflow["build"]["args"]["REQUIREMENTS"] == (
        "experiments/eval/requirements.txt")
    assert (REPO_ROOT / "experiments" / "eval"
            / "requirements.txt").exists()


def test_dockerignore_keeps_local_state_out():
    ignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "experiments/model/mlruns/" in ignore
    assert "__pycache__/" in ignore
    assert ".git/" in ignore

"""Tests for the Day 13 v2 serving artifacts.

Day 13 retrains the serving models on the gated features and
serves them from artifacts_v2/. The Day 5 artifacts stay in
artifacts/ untouched, so the two versions can be compared.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from experiments.model import api as api_module
from experiments.model.api import ARTIFACTS_DIR, create_app, load_models
from experiments.model.features import FEATURES
from experiments.model.tests.conftest import RUN1_CSV, RUN1_PRESENT
from experiments.model.train import main as train_main

V1_DIR = Path(__file__).parent.parent / "artifacts"
RESULTS_DAY13 = Path(__file__).parent.parent / "results_day13.json"


def test_v2_artifacts_load_by_default():
    models = load_models()
    assert set(models) == {"head", "shoulder"}
    for cue in ("head", "shoulder"):
        assert models[cue]["model_name"] == "logistic_regression"
        assert models[cue]["features"] == FEATURES[cue]
    assert len(models["head"]["features"]) == 6
    assert len(models["shoulder"]["features"]) == 11
    gated = [f for f in models["shoulder"]["features"]
             if f.endswith("_sn")]
    assert gated == ["mean_abs_sn", "max_abs_sn", "std_sn"]


def test_v1_artifacts_untouched():
    """The Day 5 artifacts are still there, on the old features."""
    models = load_models(V1_DIR)
    assert set(models) == {"head", "shoulder"}
    assert len(models["head"]["features"]) == 6
    assert len(models["shoulder"]["features"]) == 8
    assert not any(f.endswith("_sn")
                   for f in models["shoulder"]["features"])


def test_v2_numbers_match_day12():
    """The retrain reproduces the Day 12 new-feature numbers."""
    result = json.loads(RESULTS_DAY13.read_text(encoding="utf-8"))
    shoulder = result["cues"]["shoulder"]
    assert shoulder["best_model"] == "logistic_regression"
    assert shoulder["best_oof_metrics"]["f1_tilted"] == 0.219
    assert shoulder["best_oof_trials"]["detected_trials"] == 5
    assert shoulder["best_oof_trials"]["false_positive_events"] == 3
    head = result["cues"]["head"]
    assert head["best_model"] == "logistic_regression"
    assert head["best_oof_metrics"]["f1_tilted"] == 0.078
    assert head["best_oof_trials"]["detected_trials"] == 1
    assert head["best_oof_trials"]["false_positive_events"] == 6


def test_default_app_serves_v2():
    """The module-level app loads the v2 artifacts on import."""
    assert api_module.app is not None
    c = TestClient(api_module.app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["model_names"] == {
        "head": "logistic_regression",
        "shoulder": "logistic_regression"}


def test_create_app_accepts_explicit_v1_dir():
    """An explicit artifacts dir still works, for comparison."""
    c = TestClient(create_app(artifacts_dir=V1_DIR))
    assert c.get("/health").status_code == 200
    assert len(c.get("/health").json()["models"]) == 2


@pytest.mark.skipif(not RUN1_PRESENT, reason="Run 1 CSV not present")
def test_train_cli_artifacts_dir_flag(tmp_path):
    """--save --artifacts-dir writes the models where asked."""
    out_dir = tmp_path / "v2"
    out_json = tmp_path / "r.json"
    rc = train_main(["--csv", RUN1_CSV, "--save",
                     "--artifacts-dir", str(out_dir),
                     "--out", str(out_json)])
    assert rc == 0
    assert (out_dir / "model_head.joblib").exists()
    assert (out_dir / "model_shoulder.joblib").exists()
    models = load_models(out_dir)
    assert models["shoulder"]["model_name"] == "logistic_regression"
    assert len(models["shoulder"]["features"]) == 11

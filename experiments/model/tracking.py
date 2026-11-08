"""MLflow experiment tracking for the Day 6 classifier runs.

Day 5 left its numbers in one `results.json`: one training run,
no comparison. MLflow is the standard fix for that: every training
run is logged under a name, with its settings (params), its scores
(metrics) and its trained model (artifact), all in one local
file-based store. Nothing here needs a server: the default tracking
directory is `experiments/model/mlruns/`, and the comparison is read
back from that same store.

Usage (from the repo root):
    python -m experiments.model.train --csv <csv> --mlflow
    python -m experiments.model.train --csv <csv> --model random_forest \\
        --mlflow --run-name random_forest_only
    python -m experiments.model.tracking   # writes mlflow_comparison.json

View the runs:
    mlflow ui --backend-store-uri experiments/model/mlruns
    # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_DIR = Path(__file__).parent / "mlruns"
EXPERIMENT_NAME = "poseguard-classifier"
COMPARISON_FILE = Path(__file__).parent / "mlflow_comparison.json"

# Metrics logged per cue (all from the out-of-fold evaluation).
_CUE_METRICS = ("accuracy", "precision_tilted", "recall_tilted",
                "f1_tilted", "calm_only_accuracy")


def tracking_uri(tracking_dir: Path | str | None = None) -> str:
    """File-based tracking URI for the given directory."""
    d = Path(tracking_dir) if tracking_dir else DEFAULT_TRACKING_DIR
    return f"file://{d.resolve()}"


def _cue_metrics(cue_result: dict) -> dict:
    """Flat metric dict for one cue, prefixed with the cue name."""
    m = cue_result["best_oof_metrics"]
    t = cue_result["best_oof_trials"]
    return {
        "accuracy": m["accuracy"],
        "precision_tilted": m["precision_tilted"],
        "recall_tilted": m["recall_tilted"],
        "f1_tilted": m["f1_tilted"],
        "trials_detected": t["detected_trials"],
        "false_positive_events": t["false_positive_events"],
        "work_windows_tilted": t["work_windows_tilted"],
        "calm_only_accuracy": cue_result["calm_only_metrics"]["accuracy"],
    }


def log_training_run(result: dict, run_name: str,
                     tracking_dir: Path | str | None = None,
                     trained: dict | None = None) -> str:
    """Log one full train_session result as an MLflow run.

    Params: which model won per cue, how many windows were tilted,
    and the training choice (auto = best by OOF F1, or a forced
    model). Metrics: the out-of-fold numbers per cue, plus trials
    detected and false-positive events, the same numbers the report
    compares against the rule baseline. Artifacts: the winning model
    per cue (joblib), when `trained` models are passed in.
    Returns the MLflow run id.
    """
    mlflow.set_tracking_uri(tracking_uri(tracking_dir))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        params = {}
        for cue in ("head", "shoulder"):
            r = result["cues"][cue]
            params[f"{cue}_model"] = r["best_model"]
            params[f"{cue}_n_windows"] = r["n_windows"]
            params[f"{cue}_n_tilted_windows"] = r["n_tilted_windows"]
            params[f"{cue}_features"] = ",".join(r["features"])
            for key, value in _cue_metrics(r).items():
                mlflow.log_metric(f"{cue}_{key}", value)
        mlflow.log_params(params)
        if trained:
            with tempfile.TemporaryDirectory() as tmp:
                for cue in ("head", "shoulder"):
                    model = trained[cue][result["cues"][cue]["best_model"]]
                    path = Path(tmp) / f"model_{cue}.joblib"
                    joblib.dump({"cue": cue,
                                 "model_name": result["cues"][cue]["best_model"],
                                 "model": model,
                                 "features": result["cues"][cue]["features"]},
                                path)
                    mlflow.log_artifact(str(path), artifact_path="models")
        return run.info.run_id


def summarize_runs(tracking_dir: Path | str | None = None) -> list[dict]:
    """All runs in the experiment, newest first, as plain dicts.

    Each entry: run name, params, and the per-cue metrics that the
    comparison is about. This is what `mlflow_comparison.json` and
    the report's Day 6 section are built from.
    """
    mlflow.set_tracking_uri(tracking_uri(tracking_dir))
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return []
    runs = client.search_runs(exp.experiment_id,
                              order_by=["attributes.start_time DESC"])
    out = []
    for r in runs:
        metrics = {k: round(v, 3) for k, v in r.data.metrics.items()}
        out.append({"run_name": r.data.tags.get("mlflow.runName",
                                                r.info.run_id[:8]),
                    "run_id": r.info.run_id,
                    "params": dict(r.data.params),
                    "metrics": metrics})
    return out


def write_comparison(tracking_dir: Path | str | None = None,
                     out: Path | None = None) -> dict:
    """Write the run comparison to mlflow_comparison.json and return it.

    The comparison is the honest question Day 6 answers: does forcing
    one model (or any other second run) beat the Day 5 baseline on
    the numbers that matter (trials detected, false positives, F1)?
    """
    runs = summarize_runs(tracking_dir)
    comparison = {"experiment": EXPERIMENT_NAME, "runs": runs}
    path = out if out else COMPARISON_FILE
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> int:
    comparison = write_comparison()
    print(f"{len(comparison['runs'])} run(s) in "
          f"{EXPERIMENT_NAME}; comparison written to {COMPARISON_FILE}")
    for r in comparison["runs"]:
        m = r["metrics"]
        print(f"  {r['run_name']}: head "
              f"{m.get('head_trials_detected')}/5 trials, "
              f"{m.get('head_false_positive_events')} FP, "
              f"F1 {m.get('head_f1_tilted')}  |  shoulder "
              f"{m.get('shoulder_trials_detected')}/5 trials, "
              f"{m.get('shoulder_false_positive_events')} FP, "
              f"F1 {m.get('shoulder_f1_tilted')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train and evaluate the Day 5 posture classifiers.

One classifier per cue (head, shoulder), each a scikit-learn model
over the 1-second window features in `features.py`. Two models are
trained per cue and the better one (by out-of-fold F1) is kept:

    logistic regression (scaled, class-weighted) - small, explainable
    random forest (200 trees, class-weighted)   - catches non-linear
    splits, e.g. "steady AND moderately tilted"

Evaluation is honest about the tiny dataset (Run 1: ~23 tilted windows
per cue, ~360 upright). Two views:

1. A plain stratified 70/30 train/test split (fixed seed 42), with
   accuracy, precision, recall and F1 per class.
2. Grouped 5-fold cross-validation, grouped by trial (hold windows)
   and by work time-block, so a trial's windows never appear on both
   sides. Out-of-fold predictions for EVERY window give the trial
   verdicts: a trial counts as detected when at least half its hold
   windows are predicted tilted, and false positives are runs of 5+
   consecutive tilted work windows (5 s, mirroring the rule). Those
   numbers compare directly with the Day 3-4 rule baseline.

A "calm-only" check is reported too: the same out-of-fold
predictions restricted to windows where the shoulder line was stable
(std_s < 5 deg). The work step in Run 1 contains tracker-noise
episodes (shoulder angle swinging >100 deg frame to frame). If a
classifier's accuracy collapses once those noisy windows are removed,
its signal was tracker noise, not posture, and the report says so.

Usage (from the repo root):
    python -m experiments.model.train --csv <path/to/validation.csv>
    python -m experiments.model.train --csv <path> --save
    python -m experiments.model.train --csv <path> --mlflow
    python -m experiments.model.train --csv <path> --model random_forest \\
        --mlflow --run-name random_forest_only
Writes experiments/model/results.json (and, with --save, the trained
models to experiments/model/artifacts/). With --mlflow the run is
logged to MLflow (see tracking.py) so runs can be compared.
Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GroupKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session
from experiments.eval.signals import WINNER_COL, add_signals

from .features import FEATURES, build_windows
from .tracking import log_training_run

RANDOM_STATE = 42
CALM_STD_S = 5.0   # shoulder line steadier than this = no tracker noise
FP_RUN_WINDOWS = 5  # 5 consecutive tilted 1 s windows = a 5 s false event

# Rule-based baseline, Days 3-4, same Run 1 CSV.
RULE_BASELINE = {
    "head": {"detected_trials": 0, "false_positive_events": 0},
    "shoulder": {"detected_trials": 0, "false_positive_events": 7},
}


def make_models():
    """(name, model) pairs, both deterministic at RANDOM_STATE."""
    return [
        ("logistic_regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced",
                                       max_iter=1000,
                                       random_state=RANDOM_STATE)),
        ])),
        ("random_forest", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=1)),
    ]


def _window_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True,
                                   zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                      labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_tilted": int((y_true == 1).sum()),
        "n_upright": int((y_true == 0).sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision_tilted": round(float(report["1"]["precision"]), 3),
        "recall_tilted": round(float(report["1"]["recall"]), 3),
        "f1_tilted": round(float(report["1"]["f1-score"]), 3),
        "precision_upright": round(float(report["0"]["precision"]), 3),
        "recall_upright": round(float(report["0"]["recall"]), 3),
        "f1_upright": round(float(report["0"]["f1-score"]), 3),
        "confusion": {"tn": int(tn), "fp": int(fp),
                      "fn": int(fn), "tp": int(tp)},
    }


def _trial_verdicts(windows: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Trial-level detection and work false-positive runs, from window
    predictions. A trial counts as detected when >= half its hold
    windows are predicted tilted. A false-positive event is 5+
    consecutive tilted work windows in time order."""
    w = windows.copy()
    w["pred"] = y_pred
    detected = 0
    per_trial = []
    for trial in sorted(w.loc[w["label"] == 1, "group"].unique()):
        t = w[(w["label"] == 1) & (w["group"] == trial)]
        hit = bool(len(t) and t["pred"].mean() >= 0.5)
        detected += int(hit)
        per_trial.append({"trial": int(trial), "windows": int(len(t)),
                          "tilted_predicted": int(t["pred"].sum()),
                          "detected": hit})
    work = w[w["step"] == "work"].sort_values("start_s")
    runs, run = 0, 0
    for p in work["pred"]:
        run = run + 1 if p == 1 else 0
        if run == FP_RUN_WINDOWS:
            runs += 1
    return {"detected_trials": detected, "trials_total": 5,
            "per_trial": per_trial,
            "false_positive_events": runs,
            "work_windows_tilted": int(work["pred"].sum()),
            "work_windows_total": int(len(work))}


def evaluate_cue(windows: pd.DataFrame, cue: str,
                 model_choice: str = "auto") -> dict:
    """Train both models for one cue; return metrics + trained models.

    `model_choice` is "auto" (train both, keep the better by
    out-of-fold F1) or one model name, to force a comparison run
    for MLflow (Day 6). A forced run trains only that model.
    """
    feats = FEATURES[cue]
    X = windows[feats].to_numpy(dtype=float)
    y = windows["label"].to_numpy(dtype=int)
    groups = windows["group"].to_numpy(dtype=int)

    # --- plain 70/30 split, fixed seed ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)

    # --- grouped 5-fold out-of-fold predictions for every window ---
    cv = GroupKFold(n_splits=5)
    models = make_models()
    if model_choice != "auto":
        models = [m for m in models if m[0] == model_choice]
        if not models:
            raise ValueError(f"unknown model {model_choice!r}; "
                             "use 'auto', 'logistic_regression' or "
                             "'random_forest'")
    results, trained = {}, {}
    for name, model in models:
        model.fit(X_tr, y_tr)
        trained[name] = model
        split_pred = model.predict(X_te)
        oof_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
        results[name] = {
            "split_metrics": _window_metrics(y_te, split_pred),
            "oof_metrics": _window_metrics(y, oof_pred),
            "oof_trials": _trial_verdicts(windows, oof_pred),
            "oof_pred": [int(p) for p in oof_pred],
        }
        if name == "random_forest":
            results[name]["feature_importance"] = {
                f: round(float(v), 3)
                for f, v in zip(feats, model.feature_importances_)}
        else:
            # logistic coefficients, on the scaled features
            coef = model.named_steps["clf"].coef_[0]
            results[name]["coefficients"] = {
                f: round(float(v), 3) for f, v in zip(feats, coef)}

    best_name = max(results,
                    key=lambda n: results[n]["oof_metrics"]["f1_tilted"])
    best = results[best_name]
    # Decision-threshold sweep on the winning model: could ANY cutoff
    # make it pass the gate (4/5 trials, 0 false positives)? This is
    # the Day 4 threshold curve, for the classifier.
    best_model = trained[best_name]
    oof_proba = cross_val_predict(best_model, X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
    sweep = []
    for thr in (0.3, 0.5, 0.7, 0.9):
        t = _trial_verdicts(windows, (oof_proba >= thr).astype(int))
        sweep.append({"threshold": thr,
                      "detected_trials": t["detected_trials"],
                      "false_positive_events": t["false_positive_events"]})
    # Calm-only check on the winning model's out-of-fold predictions:
    # does the signal survive once tracker-noise windows are removed?
    calm = windows["std_s"] < CALM_STD_S
    calm_metrics = _window_metrics(y[calm.to_numpy()],
                                   np.array(best["oof_pred"])[calm.to_numpy()])
    return {
        "cue": cue, "features": feats,
        "n_windows": int(len(windows)),
        "n_tilted_windows": int((y == 1).sum()),
        "n_upright_windows": int((y == 0).sum()),
        "models": results,
        "best_model": best_name,
        "best_oof_metrics": best["oof_metrics"],
        "best_oof_trials": best["oof_trials"],
        "threshold_sweep": sweep,
        "calm_only_metrics": calm_metrics,
        "calm_windows": int(calm.sum()),
        "rule_baseline": RULE_BASELINE[cue],
        "trained": trained,
    }


def train_session(csv_path: str, model_choice: str = "auto") -> dict:
    """Full Day 5 run on one validation CSV.

    Day 11: the frames also carry the replacement-signal columns,
    and the winning signal (see experiments.eval.signals.WINNER_COL)
    is exposed as s_dev_new so the shoulder windows include the
    gated-signal features. The head cue is untouched.
    """
    session = load_session(csv_path)
    frames = add_signals(add_deviations(session), session)
    frames["s_dev_new"] = frames[WINNER_COL]
    out = {"csv": csv_path, "frames": int(len(frames)),
           "model_choice": model_choice, "cues": {}}
    for cue in ("head", "shoulder"):
        windows = build_windows(frames, session, cue)
        result = evaluate_cue(windows, cue, model_choice=model_choice)
        trained = result.pop("trained")
        out["cues"][cue] = result
        out.setdefault("_trained", {})[cue] = trained
    return out


def save_artifacts(result: dict, artifacts_dir: Path) -> list[str]:
    """Save the winning model per cue (plus its metadata) with joblib."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for cue, cue_result in result["cues"].items():
        model = result["_trained"][cue][cue_result["best_model"]]
        payload = {"cue": cue, "model_name": cue_result["best_model"],
                   "model": model, "features": cue_result["features"]}
        path = artifacts_dir / f"model_{cue}.joblib"
        joblib.dump(payload, path)
        saved.append(str(path))
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PoseGuard Day 5 classifier")
    parser.add_argument("--csv", required=True, help="validation export CSV")
    parser.add_argument("--save", action="store_true",
                        help="also save trained models to artifacts/")
    parser.add_argument("--model", default="auto",
                        choices=["auto", "logistic_regression",
                                 "random_forest"],
                        help="train both models and keep the better "
                             "(auto), or force one model for a "
                             "comparison run")
    parser.add_argument("--mlflow", action="store_true",
                        help="log this run to MLflow "
                             "(experiments/model/mlruns/)")
    parser.add_argument("--run-name", default=None,
                        help="MLflow run name (default: day5_baseline "
                             "for --model auto, else <model>_only)")
    parser.add_argument("--out", default=None,
                        help="where to write results.json "
                             "(default: results.json next to this module)")
    args = parser.parse_args(argv)

    result = train_session(args.csv, model_choice=args.model)
    trained = result.pop("_trained")
    for cue in ("head", "shoulder"):
        r = result["cues"][cue]
        b = r["rule_baseline"]
        print(f"{cue}: best={r['best_model']}  "
              f"OOF acc={r['best_oof_metrics']['accuracy']} "
              f"F1(tilted)={r['best_oof_metrics']['f1_tilted']} "
              f"recall(tilted)={r['best_oof_metrics']['recall_tilted']} "
              f"precision(tilted)={r['best_oof_metrics']['precision_tilted']}")
        t = r["best_oof_trials"]
        print(f"  trials detected {t['detected_trials']}/5 "
              f"(rule baseline: {b['detected_trials']}/5), "
              f"false-positive events {t['false_positive_events']} "
              f"(rule baseline: {b['false_positive_events']})")
        print("  decision-threshold sweep (trials detected / FP events): "
              + ", ".join(f"{s['threshold']}: "
                          f"{s['detected_trials']}/5, "
                          f"{s['false_positive_events']} FP"
                          for s in r["threshold_sweep"]))
        print(f"  calm-only accuracy "
              f"{r['calm_only_metrics']['accuracy']} "
              f"(on {r['calm_windows']} calm windows)")

    out = Path(args.out) if args.out else Path(__file__).parent / "results.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults written to {out}")
    if args.save:
        result["_trained"] = trained
        saved = save_artifacts(result, Path(__file__).parent / "artifacts")
        for p in saved:
            print(f"Model saved to {p}")
    if args.mlflow:
        run_name = args.run_name or (
            "day5_baseline" if args.model == "auto"
            else f"{args.model}_only")
        run_id = log_training_run(result, run_name, trained=trained)
        print(f"MLflow run logged: {run_name} ({run_id[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

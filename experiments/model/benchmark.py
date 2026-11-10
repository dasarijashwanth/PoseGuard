"""Day 8 benchmark: PyTorch MLP vs the Day 5 scikit-learn models.

Same Run 1 CSV, same window features, same grouped 5-fold CV by
trial for every model, so the numbers compare directly. The rule
baseline from Days 3-4 is the reference in every row: the gate is
4/5 trials detected with 0 false-positive events.

Usage (from the repo root):
    python -m experiments.model.benchmark --csv <path/to/validation.csv>

Writes experiments/model/benchmark.json (full numbers) and
experiments/model/BENCHMARK.md (the comparison table and verdicts).
Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session

from .features import build_windows
from .torch_mlp import evaluate_mlp_cue
from .train import evaluate_cue

GATE_TRIALS = 4
GATE_FP = 0


def _row(model_name: str, oof: dict, trials: dict) -> dict:
    return {
        "model": model_name,
        "accuracy": oof["accuracy"],
        "precision_tilted": oof["precision_tilted"],
        "recall_tilted": oof["recall_tilted"],
        "f1_tilted": oof["f1_tilted"],
        "detected_trials": trials["detected_trials"],
        "false_positive_events": trials["false_positive_events"],
        "work_windows_tilted": trials["work_windows_tilted"],
        "work_windows_total": trials["work_windows_total"],
    }


def _verdict(cue: str, rows: list[dict], rule: dict,
             mlp_calm_acc: float, best_sk_calm_acc: float) -> str:
    mlp = next(r for r in rows if r["model"] == "mlp")
    best_sk = max((r for r in rows if r["model"] != "mlp"),
                  key=lambda r: r["f1_tilted"])
    gate = (mlp["detected_trials"] >= GATE_TRIALS
            and mlp["false_positive_events"] == GATE_FP)
    parts = []
    if mlp["f1_tilted"] > best_sk["f1_tilted"]:
        parts.append(
            f"The MLP beats the best scikit-learn model "
            f"({best_sk['model']}) on F1: {mlp['f1_tilted']} vs "
            f"{best_sk['f1_tilted']}.")
    elif mlp["f1_tilted"] == best_sk["f1_tilted"]:
        parts.append(
            f"The MLP ties the best scikit-learn model "
            f"({best_sk['model']}) on F1 at {mlp['f1_tilted']}.")
    else:
        parts.append(
            f"The MLP does NOT beat the best scikit-learn model "
            f"({best_sk['model']}): F1 {mlp['f1_tilted']} vs "
            f"{best_sk['f1_tilted']}. A neural net does not fix "
            f"this cue on one session of data.")
    if gate:
        parts.append(
            f"It passes the gate: {mlp['detected_trials']}/5 trials "
            f"with {mlp['false_positive_events']} false positives, "
            f"against the rule baseline of "
            f"{rule['detected_trials']}/5 trials and "
            f"{rule['false_positive_events']} false positives.")
    else:
        parts.append(
            f"It does NOT pass the gate ({GATE_TRIALS}/5 trials, "
            f"{GATE_FP} false positives): {mlp['detected_trials']}/5 "
            f"trials with {mlp['false_positive_events']} false "
            f"positives, against the rule baseline of "
            f"{rule['detected_trials']}/5 trials and "
            f"{rule['false_positive_events']} false positives.")
    if mlp_calm_acc < best_sk_calm_acc - 0.05:
        parts.append(
            f"On calm windows only its accuracy is {mlp_calm_acc} "
            f"vs {best_sk_calm_acc} for the best sklearn model: "
            f"the signal leans on tracker noise, not posture.")
    return " ".join(parts)


def run_benchmark(csv_path: str) -> dict:
    session = load_session(csv_path)
    frames = add_deviations(session)
    out = {"csv": csv_path, "frames": int(len(frames)), "cues": {}}
    for cue in ("head", "shoulder"):
        windows = build_windows(frames, session, cue)
        sk = evaluate_cue(windows, cue)
        mlp = evaluate_mlp_cue(windows, cue)
        rows = []
        for name in ("logistic_regression", "random_forest"):
            m = sk["models"][name]
            rows.append(_row(name, m["oof_metrics"], m["oof_trials"]))
        m = mlp["models"]["mlp"]
        rows.append(_row("mlp", m["oof_metrics"], m["oof_trials"]))
        best_sk = max((r for r in rows if r["model"] != "mlp"),
                      key=lambda r: r["f1_tilted"])
        best_sk_calm = sk["calm_only_metrics"]["accuracy"]
        rule = sk["rule_baseline"]
        out["cues"][cue] = {
            "n_windows": int(len(windows)),
            "n_tilted_windows": int((windows["label"] == 1).sum()),
            "rows": rows,
            "rule_baseline": rule,
            "mlp_threshold_sweep": mlp["threshold_sweep"],
            "sk_threshold_sweep": sk["threshold_sweep"],
            "mlp_calm_accuracy": mlp["calm_only_metrics"]["accuracy"],
            "best_sk_calm_accuracy": best_sk_calm,
            "mlp_split_metrics": m["split_metrics"],
            "verdict": _verdict(cue, rows, rule,
                                mlp["calm_only_metrics"]["accuracy"],
                                best_sk_calm),
        }
    return out


def _md_table(rows: list[dict]) -> str:
    lines = [
        "| Model | Accuracy | Precision (tilted) | Recall (tilted) | "
        "F1 (tilted) | Trials detected | False positives | "
        "Work windows flagged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.3f} | "
            f"{r['precision_tilted']:.3f} | {r['recall_tilted']:.3f} | "
            f"{r['f1_tilted']:.3f} | {r['detected_trials']}/5 | "
            f"{r['false_positive_events']} | "
            f"{r['work_windows_tilted']}/{r['work_windows_total']} |")
    return "\n".join(lines)


def write_report(result: dict, md_path: Path) -> None:
    parts = [
        "# Day 8 benchmark: PyTorch MLP vs scikit-learn",
        "",
        "Same Run 1 CSV, same 1-second window features, same grouped "
        "5-fold cross-validation by trial for every model. The rule "
        "baseline from Days 3-4 is the reference in every row. The "
        "gate is 4/5 trials detected with 0 false-positive events.",
        "",
        "The MLP is a small two-hidden-layer net (32, 16) with ReLU "
        "and dropout, Adam with weight decay, early stopping on a "
        "validation split taken from the training groups only, and "
        "inverse-frequency class weights computed from the training "
        "portion only. CPU-only, deterministic at a fixed seed.",
        "",
    ]
    for cue, c in result["cues"].items():
        parts += [
            f"## {cue.capitalize()} "
            f"({c['n_windows']} windows, {c['n_tilted_windows']} tilted)",
            "",
            _md_table(c["rows"]),
            "",
            f"Rule baseline: {c['rule_baseline']['detected_trials']}/5 "
            f"trials, {c['rule_baseline']['false_positive_events']} "
            f"false positives.",
            "",
            f"MLP threshold sweep (trials / false positives): "
            + ", ".join(
                f"{s['threshold']}: {s['detected_trials']}/5, "
                f"{s['false_positive_events']} FP"
                for s in c["mlp_threshold_sweep"]),
            "",
            f"MLP calm-only accuracy: {c['mlp_calm_accuracy']} "
            f"(best sklearn: {c['best_sk_calm_accuracy']}).",
            "",
            f"**Verdict:** {c['verdict']}",
            "",
        ]
    parts += [
        "## What this means",
        "",
        "A neural net does not rescue a dataset of one session with "
        "23 tilted windows. Whatever the MLP wins on F1, check the "
        "trial and false-positive columns before believing it: the "
        "gate, not accuracy, is the honest score. The fix is still "
        "the re-run data (longer, deeper holds, more trials), not a "
        "fancier model.",
        "",
    ]
    md_path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 8 MLP benchmark")
    parser.add_argument("--csv", required=True,
                        help="validation export CSV")
    parser.add_argument("--out", default=None,
                        help="where to write benchmark.json "
                             "(default: benchmark.json next to this module)")
    args = parser.parse_args(argv)

    result = run_benchmark(args.csv)
    here = Path(__file__).parent
    out = Path(args.out) if args.out else here / "benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, here / "BENCHMARK.md")
    print(f"Benchmark written to {out} and {here / 'BENCHMARK.md'}")
    for cue, c in result["cues"].items():
        print(f"\n{cue}:")
        for r in c["rows"]:
            print(f"  {r['model']:20s} acc={r['accuracy']:.3f} "
                  f"F1={r['f1_tilted']:.3f} "
                  f"trials={r['detected_trials']}/5 "
                  f"FP={r['false_positive_events']}")
        print(f"  verdict: {c['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

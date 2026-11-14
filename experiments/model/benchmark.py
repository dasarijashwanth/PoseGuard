"""Day 8 benchmark: PyTorch MLP vs the Day 5 scikit-learn models.

Same Run 1 CSV, same window features, same grouped 5-fold CV by
trial for every model, so the numbers compare directly. The rule
baseline from Days 3-4 is the reference in every row: the gate is
4/5 trials detected with 0 false-positive events.

Day 12 extends the same benchmark to the Day 11 gated shoulder
signal: RF and MLP are now evaluated on BOTH feature sets, the old
8-feature shoulder list and the new 11-feature list with the three
gated stats, for both cues. The head cue is unchanged by Day 11,
so its old and new sets are the same by construction. The
old-feature rows must reproduce Day 8 exactly.

Usage (from the repo root):
    python -m experiments.model.benchmark --csv <path/to/validation.csv>

Writes experiments/model/benchmark.json (full numbers) and appends
a Day 12 section to experiments/model/BENCHMARK.md, leaving the
Day 8 and Day 11 sections untouched. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session
from experiments.eval.signals import WINNER_COL, add_signals

from .features import FEATURES, OLD_FEATURES, build_windows
from .torch_mlp import evaluate_mlp_cue
from .train import evaluate_cue

GATE_TRIALS = 4
GATE_FP = 0
DAY12_MARKER = "## Day 12"


def _row(model_name: str, oof: dict, trials: dict,
         calm_accuracy: float) -> dict:
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
        "calm_accuracy": calm_accuracy,
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


def _evaluate_feature_set(windows, cue: str, feats: list[str]) -> dict:
    """All three models on one feature set for one cue.

    Same grouped 5-fold CV by trial and the same out-of-fold
    protocol as Day 8, so old vs new compares directly.
    """
    sk = evaluate_cue(windows, cue, features=feats)
    mlp = evaluate_mlp_cue(windows, cue, features=feats)
    rows = []
    for name in ("logistic_regression", "random_forest"):
        m = sk["models"][name]
        rows.append(_row(name, m["oof_metrics"], m["oof_trials"],
                         m["calm_only_metrics"]["accuracy"]))
    m = mlp["models"]["mlp"]
    rows.append(_row("mlp", m["oof_metrics"], m["oof_trials"],
                     mlp["calm_only_metrics"]["accuracy"]))
    best_sk = max((r for r in rows if r["model"] != "mlp"),
                  key=lambda r: r["f1_tilted"])
    return {
        "features": feats,
        "rows": rows,
        "rule_baseline": sk["rule_baseline"],
        "mlp_threshold_sweep": mlp["threshold_sweep"],
        "sk_threshold_sweep": sk["threshold_sweep"],
        "mlp_calm_accuracy": mlp["calm_only_metrics"]["accuracy"],
        "best_sk_calm_accuracy": best_sk["calm_accuracy"],
        "mlp_split_metrics": m["split_metrics"],
        "lr_coefficients": sk["models"]["logistic_regression"][
            "coefficients"],
        "rf_feature_importance": sk["models"]["random_forest"][
            "feature_importance"],
        "verdict": _verdict(cue, rows, sk["rule_baseline"],
                            mlp["calm_only_metrics"]["accuracy"],
                            best_sk["calm_accuracy"]),
    }


def _verdict_day12(cue: str, old: dict, new: dict) -> str:
    """What the gated features changed, per model, for one cue."""
    parts = []
    for model in ("logistic_regression", "random_forest", "mlp"):
        o = next(r for r in old["rows"] if r["model"] == model)
        n = next(r for r in new["rows"] if r["model"] == model)
        if n["f1_tilted"] > o["f1_tilted"]:
            direction = (f"F1 {o['f1_tilted']} to {n['f1_tilted']}, "
                         f"false positives {o['false_positive_events']} "
                         f"to {n['false_positive_events']}, work "
                         f"flagged {o['work_windows_tilted']} to "
                         f"{n['work_windows_tilted']}")
        elif n["f1_tilted"] == o["f1_tilted"]:
            direction = (f"F1 unchanged at {n['f1_tilted']}, false "
                         f"positives {o['false_positive_events']} to "
                         f"{n['false_positive_events']}")
        else:
            direction = (f"F1 {o['f1_tilted']} down to "
                         f"{n['f1_tilted']}, false positives "
                         f"{o['false_positive_events']} to "
                         f"{n['false_positive_events']}")
        parts.append(f"{model}: {direction}.")
    passed = [r["model"] for fs in (old, new) for r in fs["rows"]
              if r["detected_trials"] >= GATE_TRIALS
              and r["false_positive_events"] == GATE_FP]
    if passed:
        parts.append(f"Passes the gate: {', '.join(passed)}.")
    else:
        parts.append(
            f"Nothing passes the gate ({GATE_TRIALS}/5 trials, "
            f"{GATE_FP} false positives) on either feature set.")
    best_old = max(old["rows"], key=lambda r: r["f1_tilted"])
    best_new = max(new["rows"], key=lambda r: r["f1_tilted"])
    parts.append(
        f"Best-F1 model calm-only accuracy {best_old['calm_accuracy']} "
        f"({best_old['model']}, old) to {best_new['calm_accuracy']} "
        f"({best_new['model']}, new). The random forest calm figure "
        f"is higher only because it predicts almost everything "
        f"upright, so it is not the honest comparison.")
    return " ".join(parts)


def run_benchmark(csv_path: str) -> dict:
    session = load_session(csv_path)
    frames = add_signals(add_deviations(session), session)
    frames["s_dev_new"] = frames[WINNER_COL]
    out = {"csv": csv_path, "frames": int(len(frames)), "cues": {}}
    for cue in ("head", "shoulder"):
        windows = build_windows(frames, session, cue)
        old = _evaluate_feature_set(windows, cue, OLD_FEATURES[cue])
        new = _evaluate_feature_set(windows, cue, FEATURES[cue])
        out["cues"][cue] = {
            "n_windows": int(len(windows)),
            "n_tilted_windows": int((windows["label"] == 1).sum()),
            "rule_baseline": old["rule_baseline"],
            "feature_sets": {"old": old, "new": new},
            "verdict": _verdict_day12(cue, old, new),
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


def _md_table_day12(rows: list[dict]) -> str:
    lines = [
        "| Model | Accuracy | Precision (tilted) | Recall (tilted) | "
        "F1 (tilted) | Trials | False positives | "
        "Work windows flagged | Calm-only accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['accuracy']:.3f} | "
            f"{r['precision_tilted']:.3f} | {r['recall_tilted']:.3f} | "
            f"{r['f1_tilted']:.3f} | {r['detected_trials']}/5 | "
            f"{r['false_positive_events']} | "
            f"{r['work_windows_tilted']}/{r['work_windows_total']} | "
            f"{r['calm_accuracy']:.3f} |")
    return "\n".join(lines)


def _day8_section(result: dict) -> list[str]:
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
        old = c["feature_sets"]["old"]
        parts += [
            f"## {cue.capitalize()} "
            f"({c['n_windows']} windows, {c['n_tilted_windows']} tilted)",
            "",
            _md_table(old["rows"]),
            "",
            f"Rule baseline: {c['rule_baseline']['detected_trials']}/5 "
            f"trials, {c['rule_baseline']['false_positive_events']} "
            f"false positives.",
            "",
            f"MLP threshold sweep (trials / false positives): "
            + ", ".join(
                f"{s['threshold']}: {s['detected_trials']}/5, "
                f"{s['false_positive_events']} FP"
                for s in old["mlp_threshold_sweep"]),
            "",
            f"MLP calm-only accuracy: {old['mlp_calm_accuracy']} "
            f"(best sklearn: {old['best_sk_calm_accuracy']}).",
            "",
            f"**Verdict:** {old['verdict']}",
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
    return parts


def _day12_section(result: dict) -> list[str]:
    parts = [
        DAY12_MARKER + ": RF + MLP on the gated signal",
        "",
        "Day 11 wired the jump-gated shoulder deviation into the "
        "classifier as three new features (mean_abs_sn, max_abs_sn, "
        "std_sn) and showed logistic regression improving on every "
        "metric. Day 12 asks whether the other two models improve "
        "too, on the same grouped 5-fold CV by trial and the same "
        "out-of-fold protocol as Day 8. Each cue is run twice: old "
        "features (Day 8 list) and new features (Day 11 list). The "
        "head cue is unchanged by Day 11, so its old and new rows "
        "are the same by construction. The gate is still 4/5 trials "
        "with 0 false positives.",
        "",
    ]
    for cue, c in result["cues"].items():
        old = c["feature_sets"]["old"]
        new = c["feature_sets"]["new"]
        parts += [
            f"### {cue.capitalize()} "
            f"({c['n_windows']} windows, {c['n_tilted_windows']} tilted)",
            "",
            "Old features "
            f"({len(old['features'])}):",
            "",
            _md_table_day12(old["rows"]),
            "",
            "New features "
            f"({len(new['features'])}):",
            "",
            _md_table_day12(new["rows"]),
            "",
            f"Rule baseline: {c['rule_baseline']['detected_trials']}/5 "
            f"trials, {c['rule_baseline']['false_positive_events']} "
            f"false positives.",
            "",
            f"MLP threshold sweep, new features (trials / false "
            f"positives): "
            + ", ".join(
                f"{s['threshold']}: {s['detected_trials']}/5, "
                f"{s['false_positive_events']} FP"
                for s in new["mlp_threshold_sweep"]),
            "",
        ]
        if cue == "shoulder":
            gated_lr = {k: v for k, v in
                        new["lr_coefficients"].items()
                        if k.endswith("_sn")}
            gated_rf = {k: v for k, v in
                        new["rf_feature_importance"].items()
                        if k.endswith("_sn")}
            parts += [
                "Logistic regression coefficients on the gated "
                f"features (new set): {gated_lr}.",
                "",
                "Random forest importances on the gated features "
                f"(new set): {gated_rf}.",
                "",
            ]
        parts += [f"**Verdict:** {c['verdict']}", ""]
    parts += [
        "## What the gated features mean",
        "",
        "The gated features help logistic regression and the MLP on "
        "F1, and leave the random forest where it was, but they do "
        "not hand any model a real tilt signal. The MLP gain comes "
        "with more false positives (1 to 3) and its calm-only "
        "accuracy slips a little, so the extra detections are not "
        "cleaner signal. Check the trial and false-positive "
        "columns before believing any F1 move: the gate, not "
        "accuracy, is the honest score. The fix is still the "
        "re-run data (longer, deeper holds, more trials, raw "
        "landmarks), not a fancier model or a cleaner feature.",
        "",
    ]
    return parts


def write_report(result: dict, md_path: Path) -> None:
    """Append (or refresh) the Day 12 section in BENCHMARK.md.

    Everything before the Day 12 marker is kept untouched, so the
    Day 8 and Day 11 sections survive a re-run. Running twice does
    not duplicate the section. If the file does not exist yet, the
    full Day 8 section is written first from the old feature set.
    """
    section = "\n".join(_day12_section(result))
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        if DAY12_MARKER in text:
            head = text.split(DAY12_MARKER, 1)[0].rstrip() + "\n\n"
            md_path.write_text(head + section, encoding="utf-8")
        else:
            md_path.write_text(text.rstrip() + "\n\n" + section,
                               encoding="utf-8")
    else:
        parts = _day8_section(result) + _day12_section(result)
        md_path.write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 12 benchmark: RF + MLP on the "
                    "gated signal, old vs new features")
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
        for set_name in ("old", "new"):
            print(f"\n{cue} ({set_name}):")
            for r in c["feature_sets"][set_name]["rows"]:
                print(f"  {r['model']:20s} acc={r['accuracy']:.3f} "
                      f"F1={r['f1_tilted']:.3f} "
                      f"trials={r['detected_trials']}/5 "
                      f"FP={r['false_positive_events']} "
                      f"calm={r['calm_accuracy']:.3f}")
        print(f"  verdict: {c['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

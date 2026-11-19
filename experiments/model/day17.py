"""Day 17: shoulder cue rework, rule grid plus model comparison.

Same Run 1 CSV, same grouped 5-fold CV by trial, same out-of-fold
protocol as Days 8, 12, and 15, so the numbers compare directly.
Two parts, both shoulder only; the head cue is untouched:

1. The rule grid (experiments.eval.shoulder.evaluate_shoulder_gate):
   thresholds 1-8 deg x persistence 2/3/5 s x velocity cap none /
   5 / 10 / 20 deg/s. Each configuration gets trials detected,
   work false positives, and trial-level F1.
2. The model comparison: logistic regression, random forest, and
   MLP on the old 11 shoulder features vs the new 13 (velocity
   and dwell added). The served v2 artifacts are NOT retrained:
   nothing here passes the gate, so there is nothing better to
   serve.

The gate is 4/5 trials detected with 0 false-positive events.

Usage (from the repo root):
    python -m experiments.model.day17 --csv <path/to/validation.csv>

Writes experiments/model/results_day17.json (full numbers) and
appends a Day 17 section to experiments/model/BENCHMARK.md, leaving
the Day 8, Day 11, Day 12, Day 15, and Day 16 sections untouched.
Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.eval.analysis import add_deviations
from experiments.eval.loader import load_session
from experiments.eval.shoulder import evaluate_shoulder_gate
from experiments.eval.signals import WINNER_COL, add_signals

from .benchmark import (
    GATE_FP,
    GATE_TRIALS,
    _evaluate_feature_set,
    _md_table_day12,
)
from .features import (
    DAY17_SHOULDER_FEATURES,
    SHOULDER_FEATURES,
    build_windows,
)

DAY17_MARKER = "## Day 17"


def _verdict_day17(old: dict, new: dict) -> str:
    """What velocity and dwell changed, per model, for the shoulder cue."""
    parts = []
    for model in ("logistic_regression", "random_forest", "mlp"):
        o = next(r for r in old["rows"] if r["model"] == model)
        n = next(r for r in new["rows"] if r["model"] == model)
        if n["f1_tilted"] > o["f1_tilted"]:
            direction = (f"F1 {o['f1_tilted']} to {n['f1_tilted']}, "
                         f"trials {o['detected_trials']}/5 to "
                         f"{n['detected_trials']}/5, false positives "
                         f"{o['false_positive_events']} to "
                         f"{n['false_positive_events']}, work flagged "
                         f"{o['work_windows_tilted']} to "
                         f"{n['work_windows_tilted']}")
        elif n["f1_tilted"] == o["f1_tilted"]:
            direction = (f"F1 unchanged at {n['f1_tilted']}, trials "
                         f"{o['detected_trials']}/5 to "
                         f"{n['detected_trials']}/5, false positives "
                         f"{o['false_positive_events']} to "
                         f"{n['false_positive_events']}")
        else:
            direction = (f"F1 {o['f1_tilted']} down to "
                         f"{n['f1_tilted']}, trials "
                         f"{o['detected_trials']}/5 to "
                         f"{n['detected_trials']}/5, false positives "
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
        f"({best_new['model']}, new).")
    return " ".join(parts)


def run_day17(csv_path: str) -> dict:
    """Full Day 17 run on one validation CSV. Shoulder cue only."""
    session = load_session(csv_path)
    frames = add_signals(add_deviations(session), session)
    frames["s_dev_new"] = frames[WINNER_COL]
    grid = evaluate_shoulder_gate(frames, session)
    windows = build_windows(frames, session, "shoulder")
    old = _evaluate_feature_set(windows, "shoulder", SHOULDER_FEATURES)
    new = _evaluate_feature_set(windows, "shoulder",
                                DAY17_SHOULDER_FEATURES)
    return {
        "csv": csv_path,
        "frames": int(len(frames)),
        "cue": "shoulder",
        "n_windows": int(len(windows)),
        "n_tilted_windows": int((windows["label"] == 1).sum()),
        "rule_grid": grid,
        "model_comparison": {
            "old_features": SHOULDER_FEATURES,
            "new_features": DAY17_SHOULDER_FEATURES,
            "feature_sets": {"old": old, "new": new},
            "verdict": _verdict_day17(old, new),
        },
    }


def _grid_table(rows: list[dict]) -> str:
    lines = [
        "| Threshold (deg) | Persistence (s) | Velocity cap (deg/s) | "
        "Trials detected | False positives | F1 (trial) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda r: (-r["f1"], r["false_positives"])):
        cap = ("none" if r["vel_cap_deg_s"] is None
               else str(r["vel_cap_deg_s"]))
        lines.append(
            f"| {r['threshold']} | {r['persist_s']} | {cap} | "
            f"{r['detected']}/5 | {r['false_positives']} | "
            f"{r['f1']:.3f} |")
    return "\n".join(lines)


def _day17_section(result: dict) -> list[str]:
    grid = result["rule_grid"]
    comp = result["model_comparison"]
    old = comp["feature_sets"]["old"]
    new = comp["feature_sets"]["new"]
    best = grid["best"]
    parts = [
        DAY17_MARKER + ": shoulder cue rework",
        "",
        "Day 11 left the shoulder cue with the tracker-noise half "
        "fixed (the jump-gated signal cuts false positives from 7 "
        "to 1 at the 6 deg operating threshold) but the level half "
        "untouched: the gated signal still detects 0/5 trials there, "
        "because a deliberate hold moves the shoulder line only "
        "about 2.5 deg while everyday leaning drifts 4-7 deg. Day "
        "17 gives the shoulder cue the Day 15 treatment: it "
        "separates the level (thresholds 1-8 deg), the persistence "
        "(2/3/5 s), and the speed of the shoulder line (velocity "
        "cap none / 5 / 10 / 20 deg/s, where a fast frame breaks a "
        "persistence run). Every configuration is scored the same "
        "way: trials detected, work false positives, and "
        "trial-level F1. Shoulder only; the head cue is untouched. "
        "The shoulder trial alignment is inferred from the head "
        "alignment (Run 1 recorded no shoulder trial summary), so "
        "the grid numbers carry that caveat.",
        "",
        "### Per-hold dwell (Run 1)",
        "",
        "How long each deliberate hold actually stayed tilted, and "
        "how deep it went. Dwell is measured above 2 deg: the "
        "holds live around 2-3 deg, so a 5 deg threshold would be "
        "zero on every window.",
        "",
        "| Trial | Samples | Mean |dev| (deg) | Max |dev| (deg) | "
        "Dwell >1 deg (s) | Dwell >2 deg (s) | Dwell >3 deg (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grid["hold_dwells"]:
        parts.append(
            f"| {row['trial']} | {row['samples']} | "
            f"{row['mean_abs_dev']} | {row['max_abs_dev']} | "
            f"{row['dwell_gt1_s']} | {row['dwell_gt2_s']} | "
            f"{row['dwell_gt3_s']} |")
    parts += [
        "",
        "### Rule grid (best 12 of 72 configurations by F1)",
        "",
        _grid_table(sorted(grid["rows"],
                           key=lambda r: (-r["f1"], r["false_positives"])
                           )[:12]),
        "",
        f"Best configuration by F1: threshold "
        f"{best['threshold']} deg, persistence {best['persist_s']} s, "
        f"velocity cap "
        f"{'none' if best['vel_cap_deg_s'] is None else str(best['vel_cap_deg_s']) + ' deg/s'}, "
        f"{best['detected']}/5 trials, {best['false_positives']} "
        f"false positives, F1 {best['f1']:.3f}.",
        "",
        "### Model comparison (same grouped 5-fold CV as Days 8/12/15)",
        "",
        "Old features "
        f"({len(comp['old_features'])}):",
        "",
        _md_table_day12(old["rows"]),
        "",
        "New features "
        f"({len(comp['new_features'])}): velocity (vel_s, mean |speed| "
        f"of the shoulder line in deg/s) and dwell (dwell_s, longest "
        f"continuous run above 2 deg in seconds) added:",
        "",
        _md_table_day12(new["rows"]),
        "",
        "Logistic regression coefficients on the new features "
        f"(new set): {new['lr_coefficients']}.",
        "",
        "Random forest importances on the new features "
        f"(new set): {new['rf_feature_importance']}.",
        "",
        f"**Verdict:** {comp['verdict']}",
        "",
        "## What the shoulder rework means",
        "",
        "The velocity cap does real work on the false-positive side, "
        "more than it did for the head cue: at 1 deg / 5 s it cuts "
        "false positives from 19 (no cap) to 8 (any cap), because it "
        "kills the tracker jumps that start most work excursions. "
        "But 8 is still 8, and the grid's detection half is bought "
        "at the lowest threshold: 16 configurations detect 4/5 or "
        "5/5 trials, all at 1-2 deg, and the cheapest of them still "
        "carries 8 false positives. At the 6 deg operating "
        "threshold the grid reproduces Day 3/4 exactly, 0/5 trials "
        "with 7 false positives. Two of the four measurable holds "
        "dwell above 2 deg for nearly the whole 6 s (trials 1 and "
        "3), so the holds are not too brief at the low threshold; "
        "they are too small to stand out from work, where the "
        "shoulder sits 4-7 deg off baseline for seconds at a time. "
        "Trial 5 is barely recorded (8 samples), so at most 4/5 "
        "trials are measurable in Run 1 at all, and the best grid "
        "configuration detects exactly those four. Trial 3 carries "
        "a 58 deg tracker spike inside the hold itself, which is "
        "why the velocity cap matters even inside trials.",
        "",
        "The new features make the models worse, not better. "
        "Logistic regression drops from 5/5 trials to 3/5 and F1 "
        "0.219 to 0.172, and the MLP drops from 4/5 to 3/5 with "
        "F1 0.215 to 0.172, 4 false positives, 123 of 318 work "
        "windows flagged, and calm-only accuracy 0.638 to 0.524. "
        "The coefficients contradict the story the features were "
        "built to tell: vel_s is -0.475 and dwell_s is -0.328, so "
        "the model punishes the very speed and persistence it was "
        "meant to reward, and the gated coefficients stay negative "
        "(mean_abs_sn -1.197, max_abs_sn -1.194), so the stillness "
        "artifact from Days 5 and 11 is still doing the work. The "
        "random forest ignores the new features (importances 0.067 "
        "and 0.059) and still detects nothing. The decision-"
        "threshold sweeps agree: at 0.3 the new MLP reaches 5/5 "
        "trials with 11 false positives, so no cutoff passes the "
        "gate either. The served v2 artifacts are unchanged: there "
        "is nothing better to serve. The fix is still the camera "
        "re-run with raw landmarks and longer, deeper holds (Day 14 "
        "tool), not a shorter gate or a cleaner feature.",
        "",
    ]
    return parts


def write_report(result: dict, md_path: Path) -> None:
    """Append (or refresh) the Day 17 section in BENCHMARK.md.

    Everything before the Day 17 marker is kept untouched, so the
    Day 8, Day 11, Day 12, Day 15, and Day 16 sections survive a
    re-run. Running twice does not duplicate the section.
    """
    section = "\n".join(_day17_section(result))
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        if DAY17_MARKER in text:
            head = text.split(DAY17_MARKER, 1)[0].rstrip() + "\n\n"
            md_path.write_text(head + section, encoding="utf-8")
        else:
            md_path.write_text(text.rstrip() + "\n\n" + section,
                               encoding="utf-8")
    else:
        md_path.write_text(section, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 17: shoulder cue rework, rule grid "
                    "plus model comparison")
    parser.add_argument("--csv", required=True,
                        help="validation export CSV")
    parser.add_argument("--out", default=None,
                        help="where to write results_day17.json "
                             "(default: results_day17.json next to this "
                             "module)")
    args = parser.parse_args(argv)

    result = run_day17(args.csv)
    here = Path(__file__).parent
    out = Path(args.out) if args.out else here / "results_day17.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, here / "BENCHMARK.md")
    print(f"Day 17 results written to {out} and {here / 'BENCHMARK.md'}")
    grid = result["rule_grid"]
    print("\nRule grid best:")
    b = grid["best"]
    print(f"  threshold {b['threshold']} deg, persist {b['persist_s']} s, "
          f"vel cap {b['vel_cap_deg_s']}, {b['detected']}/5 trials, "
          f"{b['false_positives']} FP, F1 {b['f1']}")
    print(f"  alignment verified: {grid['alignment_verified']}")
    print("\nHold dwells (longest run above 2 deg per trial): "
          + ", ".join(f"{r['trial']}: {r['dwell_gt2_s']}s"
                      for r in grid["hold_dwells"]))
    comp = result["model_comparison"]
    for set_name in ("old", "new"):
        print(f"\nshoulder ({set_name}):")
        for r in comp["feature_sets"][set_name]["rows"]:
            print(f"  {r['model']:20s} acc={r['accuracy']:.3f} "
                  f"F1={r['f1_tilted']:.3f} "
                  f"trials={r['detected_trials']}/5 "
                  f"FP={r['false_positive_events']} "
                  f"calm={r['calm_accuracy']:.3f}")
    print(f"\nverdict: {comp['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Day 15: head cue rework, rule grid plus model comparison.

Same Run 1 CSV, same grouped 5-fold CV by trial, same out-of-fold
protocol as Days 8 and 12, so the numbers compare directly. Two
parts, both head only; the shoulder cue is untouched:

1. The rule grid (experiments.eval.head.evaluate_head_gate):
   thresholds 4-8 deg x persistence 2/3/5 s x velocity cap none /
   5 / 10 / 20 deg/s. Each configuration gets trials detected,
   work false positives, and trial-level F1.
2. The model comparison: logistic regression, random forest, and
   MLP on the old 6 head features vs the new 8 (velocity and dwell
   added). The served v2 artifacts are NOT retrained: nothing here
   passes the gate, so there is nothing better to serve.

The gate is 4/5 trials detected with 0 false-positive events.

Usage (from the repo root):
    python -m experiments.model.day15 --csv <path/to/validation.csv>

Writes experiments/model/results_day15.json (full numbers) and
appends a Day 15 section to experiments/model/BENCHMARK.md, leaving
the Day 8, Day 11, and Day 12 sections untouched. Exit code 0 on
success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.eval.analysis import add_deviations
from experiments.eval.head import evaluate_head_gate
from experiments.eval.loader import load_session
from experiments.eval.signals import WINNER_COL, add_signals

from .benchmark import (
    GATE_FP,
    GATE_TRIALS,
    _evaluate_feature_set,
    _md_table_day12,
)
from .features import DAY15_HEAD_FEATURES, FEATURES, build_windows

DAY15_MARKER = "## Day 15"


def _verdict_day15(old: dict, new: dict) -> str:
    """What velocity and dwell changed, per model, for the head cue."""
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


def run_day15(csv_path: str) -> dict:
    """Full Day 15 run on one validation CSV. Head cue only."""
    session = load_session(csv_path)
    frames = add_signals(add_deviations(session), session)
    frames["s_dev_new"] = frames[WINNER_COL]
    grid = evaluate_head_gate(frames, session)
    windows = build_windows(frames, session, "head")
    old = _evaluate_feature_set(windows, "head", FEATURES["head"])
    new = _evaluate_feature_set(windows, "head", DAY15_HEAD_FEATURES)
    return {
        "csv": csv_path,
        "frames": int(len(frames)),
        "cue": "head",
        "n_windows": int(len(windows)),
        "n_tilted_windows": int((windows["label"] == 1).sum()),
        "rule_grid": grid,
        "model_comparison": {
            "old_features": FEATURES["head"],
            "new_features": DAY15_HEAD_FEATURES,
            "feature_sets": {"old": old, "new": new},
            "verdict": _verdict_day15(old, new),
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


def _day15_section(result: dict) -> list[str]:
    grid = result["rule_grid"]
    comp = result["model_comparison"]
    old = comp["feature_sets"]["old"]
    new = comp["feature_sets"]["new"]
    best = grid["best"]
    parts = [
        DAY15_MARKER + ": head cue rework",
        "",
        "Day 13 left the head cue as the worst half of the served "
        "model (logistic regression 1/5 trials, 6 false positives, "
        "F1 0.078), and Day 4 said the tilts were real but too brief "
        "for the 5 s gate. Day 15 separates the three things that "
        "claim conflates: the level (thresholds 4-8 deg), the "
        "persistence (2/3/5 s), and the speed of the head line "
        "(velocity cap none / 5 / 10 / 20 deg/s, where a fast frame "
        "breaks a persistence run). Every configuration is scored "
        "the same way: trials detected, work false positives, and "
        "trial-level F1. Head only; the shoulder cue is untouched.",
        "",
        "### Per-hold dwell (Run 1)",
        "",
        "How long each deliberate hold actually stayed tilted, and "
        "how deep it went. The 5th hold is barely recorded (6 "
        "frames), so at most 4/5 trials are measurable in Run 1.",
        "",
        "| Trial | Samples | Mean |dev| (deg) | Max |dev| (deg) | "
        "Dwell >4 deg (s) | Dwell >5 deg (s) | Dwell >6 deg (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grid["hold_dwells"]:
        parts.append(
            f"| {row['trial']} | {row['samples']} | "
            f"{row['mean_abs_dev']} | {row['max_abs_dev']} | "
            f"{row['dwell_gt4_s']} | {row['dwell_gt5_s']} | "
            f"{row['dwell_gt6_s']} |")
    parts += [
        "",
        "### Rule grid (best 12 of 60 configurations by F1)",
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
        "### Model comparison (same grouped 5-fold CV as Days 8/12)",
        "",
        "Old features "
        f"({len(comp['old_features'])}):",
        "",
        _md_table_day12(old["rows"]),
        "",
        "New features "
        f"({len(comp['new_features'])}): velocity (vel_h, mean |speed| "
        f"of the head line in deg/s) and dwell (dwell_h, longest "
        f"continuous run above 5 deg in seconds) added:",
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
        "## What the head rework means",
        "",
        "Shortening the gate does not save the head cue, because "
        "the holds are shallow as well as brief: the deepest hold "
        "peaks at 7.1 deg and the longest run above 4 deg inside "
        "any hold is 2.2 s, while ordinary work contains head "
        "excursions of 10-18 deg sustained for 2-4 s. The velocity "
        "cap helps the false-positive side (it kills the tracker "
        "jumps, including a 104 deg spike) but cannot create "
        "detections the holds never earned. The new features move "
        "the models a little on F1 and not at all on the gate, and "
        "the served v2 artifacts are unchanged: there is nothing "
        "better to serve. The fix is still the camera re-run with "
        "raw landmarks and longer, deeper holds (Day 14 tool), not "
        "a shorter gate or a cleaner feature.",
        "",
    ]
    return parts


def write_report(result: dict, md_path: Path) -> None:
    """Append (or refresh) the Day 15 section in BENCHMARK.md.

    Everything before the Day 15 marker is kept untouched, so the
    Day 8, Day 11, and Day 12 sections survive a re-run. Running
    twice does not duplicate the section.
    """
    section = "\n".join(_day15_section(result))
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        if DAY15_MARKER in text:
            head = text.split(DAY15_MARKER, 1)[0].rstrip() + "\n\n"
            md_path.write_text(head + section, encoding="utf-8")
        else:
            md_path.write_text(text.rstrip() + "\n\n" + section,
                               encoding="utf-8")
    else:
        md_path.write_text(section, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 15: head cue rework, rule grid plus "
                    "model comparison")
    parser.add_argument("--csv", required=True,
                        help="validation export CSV")
    parser.add_argument("--out", default=None,
                        help="where to write results_day15.json "
                             "(default: results_day15.json next to this "
                             "module)")
    args = parser.parse_args(argv)

    result = run_day15(args.csv)
    here = Path(__file__).parent
    out = Path(args.out) if args.out else here / "results_day15.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, here / "BENCHMARK.md")
    print(f"Day 15 results written to {out} and {here / 'BENCHMARK.md'}")
    grid = result["rule_grid"]
    print("\nRule grid best:")
    b = grid["best"]
    print(f"  threshold {b['threshold']} deg, persist {b['persist_s']} s, "
          f"vel cap {b['vel_cap_deg_s']}, {b['detected']}/5 trials, "
          f"{b['false_positives']} FP, F1 {b['f1']}")
    print("\nHold dwells (longest run above 4 deg per trial): "
          + ", ".join(f"{r['trial']}: {r['dwell_gt4_s']}s"
                      for r in grid["hold_dwells"]))
    comp = result["model_comparison"]
    for set_name in ("old", "new"):
        print(f"\nhead ({set_name}):")
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

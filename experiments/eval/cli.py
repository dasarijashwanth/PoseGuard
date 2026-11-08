"""CLI: analyse one validation CSV and print the verdict.

Usage (from the repo root):
    python -m experiments.eval.cli --csv <path/to/validation.csv>
    python -m experiments.eval.cli --csv <path> --out experiments/eval/REPORT.md

Prints a short per-cue verdict plus the threshold sensitivity table,
and writes the full Markdown report to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import add_deviations, verdict
from .loader import load_session
from .report import build_markdown
from .stats import statistical_verdict


def _load_model_results() -> dict | None:
    """Day 5 classifier results, if training has been run."""
    path = Path(__file__).parent.parent / "model" / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tracking_comparison() -> dict | None:
    """Day 6 MLflow run comparison, if tracking runs exist."""
    path = Path(__file__).parent.parent / "model" / "mlflow_comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PoseGuard validation harness")
    parser.add_argument("--csv", required=True, help="validation export CSV")
    parser.add_argument("--out", default=None,
                        help="where to write the Markdown report "
                             "(default: REPORT.md next to this package)")
    args = parser.parse_args(argv)

    session = load_session(args.csv)
    frames = add_deviations(session)
    result = verdict(frames, session)

    print(f"Session: {session.meta['Exported']} ({len(frames)} frames)")
    for cue in ("head", "shoulder"):
        c = result["cues"][cue]
        status = "PASS" if c["pass"] else "FAIL"
        print(f"{cue}: {status}  {c['detected']}/5 detected "
              f"at {c['threshold']} deg, {c['false_positives']} false positives")
        for reason in c["reasons"]:
            print(f"  - {reason}")
    loss = result["loss_test"]
    print(f"landmark-loss test: ran={loss['ran']}, clean={loss['clean']} "
          f"({loss['tracking_lost_frames']} frames lost)")
    print()
    print(f"{'cue':<9} {'thr':>4} {'detected':>9} {'false_pos':>9} {'longest_run_s':>13}")
    for r in result["sensitivity"]:
        print(f"{r['cue']:<9} {r['threshold']:>4} {r['detected']:>9} "
              f"{r['false_positives']:>9} {r['longest_run_s']:>13}")

    print()
    print("Statistical validation (Day 4):")
    stats = statistical_verdict(frames, session)
    for cue in ("head", "shoulder"):
        s = stats["cues"][cue]
        print(f"{cue}: {s['verdict']}  detection 95% CI "
              f"{s['ci_low']:.0%}-{s['ci_high']:.0%}, "
              f"{s['false_positives']} false positives "
              f"(~{s['fp_per_hour']:.0f}/hour)")
        print(f"  {s['verdict_text']}")

    model_results = _load_model_results()
    if model_results is not None:
        print()
        print("Classifier (Day 5):")
        for cue in ("head", "shoulder"):
            r = model_results["cues"][cue]
            t = r["best_oof_trials"]
            print(f"{cue}: {r['best_model']}  "
                  f"trials {t['detected_trials']}/5 "
                  f"(rule: {r['rule_baseline']['detected_trials']}/5), "
                  f"FP events {t['false_positive_events']} "
                  f"(rule: {r['rule_baseline']['false_positive_events']}), "
                  f"F1(tilted) {r['best_oof_metrics']['f1_tilted']}")

    tracking_comparison = _load_tracking_comparison()
    if tracking_comparison is not None:
        print()
        print(f"MLflow tracking (Day 6): "
              f"{len(tracking_comparison['runs'])} run(s) logged")
        for r in tracking_comparison["runs"]:
            m = r["metrics"]
            print(f"{r['run_name']}: head "
                  f"{m.get('head_trials_detected', 0):.0f}/5 trials, "
                  f"F1 {m.get('head_f1_tilted', 0):.3f}  |  shoulder "
                  f"{m.get('shoulder_trials_detected', 0):.0f}/5 trials, "
                  f"F1 {m.get('shoulder_f1_tilted', 0):.3f}")

    out = Path(args.out) if args.out else Path(__file__).parent / "REPORT.md"
    out.write_text(build_markdown(session, result, frames,
                                  model_results=model_results,
                                  tracking_comparison=tracking_comparison),
                   encoding="utf-8")
    print(f"\nFull report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

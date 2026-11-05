"""CLI: analyse one validation CSV and print the verdict.

Usage (from the repo root):
    python -m experiments.eval.cli --csv <path/to/validation.csv>
    python -m experiments.eval.cli --csv <path> --out experiments/eval/REPORT.md

Prints a short per-cue verdict plus the threshold sensitivity table,
and writes the full Markdown report to --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import add_deviations, verdict
from .loader import load_session
from .report import build_markdown


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

    out = Path(args.out) if args.out else Path(__file__).parent / "REPORT.md"
    out.write_text(build_markdown(session, result), encoding="utf-8")
    print(f"\nFull report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

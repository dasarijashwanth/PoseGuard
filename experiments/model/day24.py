"""Day 24: capture-to-benchmark pipeline.

One command takes a Run 2 camera capture (the Day 14 tool,
experiments/rerun-capture.html) all the way to a verdict:

1. Pre-flight: validate the capture CSV (and manifest, when one
   is given) and print what the file holds in plain English:
   frames, tracked fraction, holds per cue, and every warning.
   A file that fails validation stops here with exit code 2 and
   nothing is benchmarked, the same convention as Day 16.
2. The Day 16 re-benchmark (experiments.model.day16.run_day16):
   the Day 15 head grid and model comparison and the Day 11/12
   shoulder signal and model comparison, with Run 2 timings and
   rule baselines counted on this file. The function is called,
   not day16.main, so results_day16.json and BENCHMARK.md are
   left untouched: Day 24 archives its own run instead.
3. The Day 23 combined-cue gate (experiments.eval.combined) on
   the same session, with the Run 2 protocol timings.
4. Archive: experiments/model/runs/<YYYY-MM-DD-HHMM>/results.json
   (the full validation report, Day 16 result, and combined
   result) and summary.md (the plain-English verdict), which is
   also printed.

Usage (from the repo root):
    python -m experiments.model.day24 --csv <path/to/capture.csv>
        [--manifest <path/to/manifest.json>]
        [--source <label for the archive>]
    python -m experiments.model.day24 --drop-dir <folder>
        [--source <label>]

--drop-dir auto-discovers the capture: the newest .csv in the
folder, with the manifest taken as the .json with the same stem
when present, else the newest .json, else none. --out-dir
overrides the archive root (default experiments/model/runs).

Exit code 0 on success, 2 when the capture or manifest fails
validation, matching Day 16.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from experiments.eval.analysis import add_deviations
from experiments.eval.combined import add_combined, evaluate_combined
from experiments.eval.loader import load_session
from experiments.eval.rerun import validate_rerun_csv
from experiments.eval.rerun_protocol import (
    HOLD_S as R2_HOLD_S,
)
from experiments.eval.rerun_protocol import (
    READY_S as R2_READY_S,
)
from experiments.eval.rerun_protocol import (
    REST_S as R2_REST_S,
)
from experiments.eval.signals import add_signals

from .day16 import check_manifest, run_day16

R2_TIMING = {"ready_s": R2_READY_S, "hold_s": R2_HOLD_S, "rest_s": R2_REST_S}
# The combined variants in preference order for the per-cue "best
# variant" pick: the normalized sum is the Day 23 headline
# candidate, so it wins an exact tie over the isolation score.
VARIANT_ORDER = ("sum", "iso", "and_control")
DEFAULT_SOURCE = "Run 2 re-run capture"


def discover_inputs(drop_dir: str) -> tuple[str, str | None]:
    """Find the newest capture CSV (and its manifest) in a folder.

    The CSV is the newest .csv by modification time. The manifest
    is the .json with the same stem when present, else the newest
    .json, else None. Raises ValueError when the folder holds no
    .csv at all.
    """
    folder = Path(drop_dir)
    csvs = sorted(folder.glob("*.csv"),
                  key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise ValueError(f"no .csv capture file found in {drop_dir}")
    csv_path = csvs[-1]
    same_stem = csv_path.with_suffix(".json")
    if same_stem.exists():
        return str(csv_path), str(same_stem)
    jsons = sorted(folder.glob("*.json"),
                   key=lambda p: p.stat().st_mtime)
    return str(csv_path), (str(jsons[-1]) if jsons else None)


def _holds_line(holds: list[dict]) -> str:
    """One line: holds found per cue and how many hit their target."""
    parts = []
    for cue in ("head", "shoulder"):
        cue_holds = [h for h in holds if h["cue"] == cue]
        reached = sum(1 for h in cue_holds if h["reached_target"])
        parts.append(f"{cue} {len(cue_holds)}/5 holds, "
                     f"{reached}/5 reached their depth target")
    return "; ".join(parts)


def format_preflight(csv_path: str, report: dict) -> str:
    """The plain-English pre-flight block for one validation report."""
    lines = [f"Pre-flight: {csv_path}"]
    if report["frames"]:
        lines.append(
            f"  {report['frames']} frames, "
            f"{report['tracked_fraction']:.1%} tracked, "
            f"work sample {report['work_s']} s.")
    if report["holds"]:
        lines.append(f"  Holds: {_holds_line(report['holds'])}.")
    if report["warnings"]:
        lines.append(f"  Warnings ({len(report['warnings'])}):")
        lines += [f"    - {w}" for w in report["warnings"]]
    else:
        lines.append("  Warnings: none.")
    return "\n".join(lines)


def preflight(csv_path: str, manifest_path: str | None = None) -> dict:
    """Validate a capture and print the pre-flight block.

    Returns the validate_rerun_csv report. Raises ValueError with
    every error listed when the CSV (or the manifest, when one is
    given) cannot be used; the caller turns that into exit code 2.
    """
    report = validate_rerun_csv(csv_path)
    print(format_preflight(csv_path, report))
    errors = list(report["errors"])
    if manifest_path is not None and report["ok"]:
        # Only check the manifest against a CSV that loaded: the
        # Day 16 rule, so a bad CSV reports its own reasons.
        session = load_session(csv_path)
        errors += check_manifest(manifest_path, session, report)
    if errors:
        raise ValueError("capture file cannot be used for the "
                         "re-benchmark:\n" + "\n".join(
                             f"  - {e}" for e in errors))
    return report


def run_combined(csv_path: str) -> dict:
    """The Day 23 combined-cue gate on one Run 2 capture."""
    session = load_session(csv_path)
    frames = add_combined(add_signals(add_deviations(session), session))
    return evaluate_combined(frames, session, **R2_TIMING)


def best_variant(cue_result: dict) -> tuple[str, dict]:
    """The best combined variant for one cue.

    Best by F1, ties toward more trials detected, then fewer
    false positives; an exact tie goes to the earlier variant in
    VARIANT_ORDER, so the normalized sum wins over isolation.
    """
    name = max(VARIANT_ORDER,
               key=lambda n: (cue_result["variants"][n]["best"]["f1"],
                              cue_result["variants"][n]["best"]["detected"],
                              -cue_result["variants"][n]["best"]
                              ["false_positives"]))
    return name, cue_result["variants"][name]["best"]


def _pass_word(row: dict) -> str:
    return "passes the gate" if row["pass"] else "does not pass the gate"


def build_summary(source: str, csv_path: str,
                  manifest_path: str | None, validation: dict,
                  day16_result: dict, combined_result: dict) -> str:
    """The plain-English run summary (also written as summary.md)."""
    lines = [
        f"# PoseGuard re-benchmark run: {source}",
        "",
        f"- Capture: {csv_path}",
        f"- Manifest: {manifest_path if manifest_path else 'none'}",
        "",
        "## Validation",
        "",
        f"{validation['frames']} frames, "
        f"{validation['tracked_fraction']:.1%} tracked, "
        f"work sample {validation['work_s']} s. "
        f"Holds: {_holds_line(validation['holds'])}.",
        "",
    ]
    if validation["warnings"]:
        lines += ["Warnings:", ""]
        lines += [f"- {w}" for w in validation["warnings"]]
    else:
        lines += ["Warnings: none."]
    lines += [
        "",
        "## Day 16 verdict",
        "",
        day16_result["verdict"],
        "",
        "## Combined gate (Day 23)",
        "",
        "Best variant per cue by F1 (ties toward more trials, then "
        "fewer false positives; the gate is 4/5 trials with 0 "
        "false positives):",
        "",
    ]
    for cue in ("head", "shoulder"):
        name, best = best_variant(combined_result["cues"][cue])
        lines.append(
            f"- {cue}: {name} at threshold {best['threshold']}, "
            f"{best['detected']}/{best['trials_total']} trials, "
            f"{best['false_positives']} false positives, "
            f"F1 {best['f1']:.3f}: {_pass_word(best)}.")
    lines += [
        "",
        "All variants:",
        "",
        "| Cue | Variant | Threshold | Trials | False positives "
        "| F1 | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cue in ("head", "shoulder"):
        for name in VARIANT_ORDER:
            best = combined_result["cues"][cue]["variants"][name]["best"]
            lines.append(
                f"| {cue} | {name} | {best['threshold']} | "
                f"{best['detected']}/{best['trials_total']} | "
                f"{best['false_positives']} | {best['f1']:.3f} | "
                f"{'yes' if best['pass'] else 'no'} |")
    lines.append("")
    return "\n".join(lines)


def write_archive(archive: dict, out_dir: str | None = None) -> Path:
    """Write results.json + summary.md under a timestamped run dir."""
    root = (Path(out_dir) if out_dir
            else Path(__file__).parent / "runs")
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    run_dir = root / stamp
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps(archive, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(archive["summary"],
                                        encoding="utf-8")
    return run_dir


def run_day24(csv_path: str, manifest_path: str | None = None,
              source: str = DEFAULT_SOURCE,
              out_dir: str | None = None) -> dict:
    """Full Day 24 pipeline on one Run 2 capture.

    Prints the pre-flight and the summary, archives results.json
    and summary.md, and returns the archive dict. Raises
    ValueError when validation fails; nothing is archived then.
    """
    validation = preflight(csv_path, manifest_path)
    day16_result = run_day16(csv_path, manifest_path, source=source)
    combined_result = run_combined(csv_path)
    summary = build_summary(source, csv_path, manifest_path,
                            validation, day16_result, combined_result)
    archive = {
        "csv": csv_path,
        "manifest": manifest_path,
        "source": source,
        "validation": validation,
        "day16": day16_result,
        "combined": combined_result,
        "summary": summary,
    }
    run_dir = write_archive(archive, out_dir)
    archive["run_dir"] = str(run_dir)
    print()
    print(summary)
    print(f"Archived to {run_dir}")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 24: capture-to-benchmark pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", default=None,
                       help="Run 2 capture CSV (Day 14 tool output)")
    group.add_argument("--drop-dir", default=None,
                       help="folder to auto-discover the newest "
                            "capture CSV (and manifest) in")
    parser.add_argument("--manifest", default=None,
                        help="Run 2 manifest JSON (Day 14 tool output); "
                             "overrides --drop-dir discovery")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="label for the data in the archive "
                             "(default: Run 2 re-run capture)")
    parser.add_argument("--out-dir", default=None,
                        help="archive root (default: runs/ next to "
                             "this module)")
    args = parser.parse_args(argv)

    try:
        if args.drop_dir:
            csv_path, manifest_path = discover_inputs(args.drop_dir)
            if args.manifest:
                manifest_path = args.manifest
        else:
            csv_path, manifest_path = args.csv, args.manifest
        run_day24(csv_path, manifest_path, source=args.source,
                  out_dir=args.out_dir)
    except (ValueError, OSError) as e:
        # A missing file is an OSError, not a validation failure:
        # both are user errors with a clear message, exit code 2,
        # and no traceback either way (the Day 16 convention).
        print(f"Day 24 failed:\n{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

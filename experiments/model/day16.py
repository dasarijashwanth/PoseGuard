"""Day 16: Run 2 re-benchmark harness.

The user's own camera re-run (Day 14 tool) has not arrived yet, so
Day 16 builds the harness that will run on it, not the run itself.
Given a Run 2 capture CSV (and optionally its JSON manifest), it:

1. Validates the file with experiments.eval.rerun (schema, structure,
   tracking gaps, hold targets). A bad file is rejected with a clear
   error and a non-zero exit code; nothing is benchmarked.
2. Re-runs the Day 15 head evaluation on the new data: the rule grid
   (thresholds 4-8 deg x persistence 2/3/5 s x velocity cap none /
   5 / 10 / 20 deg/s) and the LR / RF / MLP comparison on the old
   6 vs new 8 head features, same grouped 5-fold CV as Days 8/12.
3. Re-runs the shoulder evaluation: the Day 11 jump-gated signal
   (all three candidates plus the old level) and the Day 12 model
   comparison (old 8 vs new 11 shoulder features).
4. Writes experiments/model/results_day16.json and appends a Day 16
   section to experiments/model/BENCHMARK.md, leaving earlier
   sections untouched.

One thing the earlier days got wrong for this data: every shared
evaluation hardcoded the Run 1 trial timings (3 s ready, 6 s hold,
3 s rest, a 12 s cycle). Run 2 holds are 10 s with 5 s ready and
rest, a 20 s cycle, so the Run 1 windows would drift a full trial
out of phase by the second hold. The shared functions now take the
protocol timings as keyword arguments (Run 1 defaults, so Days
3-15 are unchanged) and this harness passes the Run 2 timings from
experiments.eval.rerun_protocol. The rule baselines are also
recomputed on the new data instead of reusing the Run 1 numbers:
the head baseline is the session-threshold rule (15 deg, 5 s) and
the shoulder baseline is the old level at the session threshold
(8 deg, 5 s), both counted on this file.

The gate is 4/5 trials detected with 0 false-positive events, per
cue. The verdict names every pass and says plainly when nothing
passes. A run on the synthetic fixture (see
experiments/model/tests/test_day16.py) passes the gate on both
cues, because the fixture holds are clean steps to target depth
with flat work; that proves the harness can report a pass, it does
not say anything about the real capture.

Usage (from the repo root):
    python -m experiments.model.day16 --csv <path/to/capture.csv>
        [--manifest <path/to/manifest.json>]
        [--source <label for the BENCHMARK.md section>]

Exit code 0 on success, 2 when the capture file or manifest fails
validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.eval.analysis import (
    add_deviations,
    align_trials,
    analyze_cue,
    false_positives,
    sustained_events,
)
from experiments.eval.head import evaluate_head_gate
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
from experiments.eval.signals import WINNER_COL, add_signals, evaluate_all

from .benchmark import (
    GATE_FP,
    GATE_TRIALS,
    _evaluate_feature_set,
    _md_table_day12,
    _verdict,
    _verdict_day12,
)
from .day15 import _grid_table, _verdict_day15
from .features import DAY15_HEAD_FEATURES, FEATURES, OLD_FEATURES, build_windows

DAY16_MARKER = "## Day 16"
R2_TIMING = {"ready_s": R2_READY_S, "hold_s": R2_HOLD_S, "rest_s": R2_REST_S}
PERSIST_S = 5.0  # the persistence the API gate actually runs


def check_manifest(manifest_path: str, session, validation: dict) -> list[str]:
    """Check a Run 2 manifest JSON against its capture CSV.

    The manifest the Day 14 tool writes carries the protocol name,
    the frame count, the baselines, and one entry per hold. Returns
    a list of mismatch messages (empty when it agrees with the CSV).
    A missing or unparseable manifest is an error too: the harness
    refuses to benchmark a capture whose manifest it cannot read.
    """
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"cannot read manifest {manifest_path}: {e}"]
    if not isinstance(data, dict):
        return [f"manifest {manifest_path} is not a JSON object"]
    errors = []
    if data.get("protocol") != "Run 2 re-run capture":
        errors.append(f"manifest protocol is not a Run 2 capture: "
                      f"{data.get('protocol')!r}")
    manifest_frames = data.get("frames")
    if isinstance(manifest_frames, str):
        # A JSON string holding the same number is the same count,
        # not a mismatch: "5150" and 5150 agree.
        try:
            manifest_frames = int(manifest_frames)
        except ValueError:
            try:
                manifest_frames = float(manifest_frames)
            except ValueError:
                pass
    if manifest_frames != validation["frames"]:
        errors.append(f"manifest frames {data.get('frames')} do not "
                      f"match the CSV ({validation['frames']} frames)")
    for key, meta_key in (("baseline_head_deg", "Baseline head (deg)"),
                          ("baseline_shoulder_deg",
                           "Baseline shoulder (deg)")):
        if key in data and data[key] is not None:
            try:
                if abs(float(data[key])
                       - float(session.meta[meta_key])) > 0.01:
                    errors.append(
                        f"manifest {key} {data[key]} does not match "
                        f"the CSV baseline {session.meta[meta_key]}")
            except (TypeError, ValueError, KeyError):
                errors.append(f"manifest {key} {data[key]!r} is not a "
                              f"number the CSV baseline can be checked "
                              f"against")
    holds = data.get("holds")
    if not isinstance(holds, list) or len(holds) != 10:
        errors.append(f"manifest holds are not 10 entries: "
                      f"{len(holds) if isinstance(holds, list) else holds!r}")
    return errors


def validate_capture(csv_path: str,
                     manifest_path: str | None = None) -> dict:
    """Validate a Run 2 capture CSV and optional manifest.

    Returns the validator report (with the manifest errors appended
    under "manifest_errors"). Raises ValueError with a clear message
    when the file cannot be used at all.
    """
    report = validate_rerun_csv(csv_path)
    manifest_errors: list[str] = []
    if manifest_path is not None and report["ok"]:
        # Only check the manifest against a CSV that loaded: loading
        # the session of a file validation already rejected would
        # raise the loader's raw error and hide the real reasons.
        session = load_session(csv_path)
        manifest_errors = check_manifest(manifest_path, session, report)
    report["manifest_errors"] = manifest_errors
    errors = list(report["errors"]) + manifest_errors
    if not report["ok"] or manifest_errors:
        raise ValueError("capture file cannot be used for the "
                         "re-benchmark:\n" + "\n".join(
                             f"  - {e}" for e in errors))
    return report


def _rule_baseline_head(frames, session) -> dict:
    """The head rule baseline on the new data.

    The session-threshold rule the API would run: the threshold
    recorded in the file (15 deg for Run 2), 5 s persistence, no
    velocity cap. Counted on this file, not carried over from Run 1.
    """
    alignment = align_trials(frames, session, "head", **R2_TIMING)
    a = analyze_cue(frames, session, "head", session.threshold_head,
                    alignment, persist_s=PERSIST_S, **R2_TIMING)
    work = frames[frames["step"] == "work"].sort_values("elapsed_s")
    fp = sustained_events(work, "h_dev", session.threshold_head,
                          persist_s=PERSIST_S)
    return {"detected_trials": a["detected"],
            "false_positive_events": len(fp)}


def _rule_baseline_shoulder(frames, session) -> dict:
    """The shoulder rule baseline on the new data.

    The old level at the session threshold (8 deg for Run 2), 5 s
    persistence: the direct analogue of the Day 3-4 baseline the
    Day 12 comparison used, counted on this file.
    """
    head_align = align_trials(frames, session, "head", **R2_TIMING)
    alignment = align_trials(frames, session, "shoulder",
                             fallback_offset=head_align["offset_s"],
                             **R2_TIMING)
    a = analyze_cue(frames, session, "shoulder",
                    session.threshold_shoulder, alignment,
                    persist_s=PERSIST_S, **R2_TIMING)
    fp = false_positives(frames, "shoulder", session.threshold_shoulder)
    return {"detected_trials": a["detected"],
            "false_positive_events": fp["count"]}


def _feature_sets_with_run2_baseline(windows, cue: str, old_feats,
                                     new_feats, rule_baseline: dict) -> dict:
    """Old vs new feature sets, with the Run 2 rule baseline.

    _evaluate_feature_set stores the Day 3-4 Run 1 baseline and a
    verdict string built on it, so both are rebuilt here with the
    baseline counted on this file. The models, CV, and metrics are
    exactly the Day 8/12 protocol.
    """
    out = {}
    for name, feats in (("old", old_feats), ("new", new_feats)):
        fs = _evaluate_feature_set(windows, cue, feats)
        fs["rule_baseline"] = rule_baseline
        fs["verdict"] = _verdict(cue, fs["rows"], rule_baseline,
                                 fs["mlp_calm_accuracy"],
                                 fs["best_sk_calm_accuracy"])
        out[name] = fs
    return out


def _require_both_classes(windows, cue: str) -> None:
    """A cue needs tilted and upright windows to benchmark.

    A file can pass validation with a cue's holds mostly untracked:
    too little tracked data to form even one tilted window, while
    work windows still exist. The model stage would then fail inside
    sklearn with a message about classes that names nothing about
    the actual problem, so name it here instead.
    """
    if windows["label"].nunique() < 2:
        raise ValueError(
            f"no tilted windows for cue {cue!r}: the {cue} holds "
            f"have too little tracked data to benchmark (the file "
            f"passed validation, but the model stage needs tilted "
            f"and upright windows)")


def run_day16(csv_path: str, manifest_path: str | None = None,
              source: str = "Run 2 re-run capture") -> dict:
    """Full Day 16 run on one validated Run 2 capture CSV."""
    validation = validate_capture(csv_path, manifest_path)
    session = load_session(csv_path)
    frames = add_signals(add_deviations(session), session)
    frames["s_dev_new"] = frames[WINNER_COL]

    head_rule = _rule_baseline_head(frames, session)
    shoulder_rule = _rule_baseline_shoulder(frames, session)

    grid = evaluate_head_gate(frames, session, **R2_TIMING)
    head_windows = build_windows(frames, session, "head", **R2_TIMING)
    _require_both_classes(head_windows, "head")
    head_sets = _feature_sets_with_run2_baseline(
        head_windows, "head", FEATURES["head"], DAY15_HEAD_FEATURES,
        head_rule)
    head = {
        "n_windows": int(len(head_windows)),
        "n_tilted_windows": int((head_windows["label"] == 1).sum()),
        "rule_grid": grid,
        "rule_baseline": head_rule,
        "model_comparison": {
            "old_features": FEATURES["head"],
            "new_features": DAY15_HEAD_FEATURES,
            "feature_sets": head_sets,
            "verdict": _verdict_day15(head_sets["old"], head_sets["new"]),
        },
    }

    signal = evaluate_all(frames, session, **R2_TIMING)
    shoulder_windows = build_windows(frames, session, "shoulder",
                                     **R2_TIMING)
    _require_both_classes(shoulder_windows, "shoulder")
    shoulder_sets = _feature_sets_with_run2_baseline(
        shoulder_windows, "shoulder", OLD_FEATURES["shoulder"],
        FEATURES["shoulder"], shoulder_rule)
    shoulder = {
        "n_windows": int(len(shoulder_windows)),
        "n_tilted_windows": int(
            (shoulder_windows["label"] == 1).sum()),
        "signal": signal,
        "rule_baseline": shoulder_rule,
        "model_comparison": {
            "old_features": OLD_FEATURES["shoulder"],
            "new_features": FEATURES["shoulder"],
            "feature_sets": shoulder_sets,
            "verdict": _verdict_day12("shoulder", shoulder_sets["old"],
                                      shoulder_sets["new"]),
        },
    }

    result = {
        "csv": csv_path,
        "manifest": manifest_path,
        "source": source,
        "validation": {
            "frames": validation["frames"],
            "tracked_fraction": validation["tracked_fraction"],
            "work_s": validation["work_s"],
            "warnings": validation["warnings"],
            "holds": validation["holds"],
        },
        "frames": int(len(frames)),
        "cues": {"head": head, "shoulder": shoulder},
    }
    result["verdict"] = verdict_day16(result)
    return result


def verdict_day16(result: dict) -> str:
    """Per-cue gate verdict on the new data, stated honestly.

    Names every configuration and model that passes the gate
    (4/5 trials, 0 false positives). When nothing passes on a cue,
    it says so instead of softening it: a pass the data did not
    earn is not reported.
    """
    parts = []
    head = result["cues"]["head"]
    head_passes = []
    best = head["rule_grid"]["best"]
    if (best["detected"] >= GATE_TRIALS
            and best["false_positives"] == GATE_FP):
        cap = ("none" if best["vel_cap_deg_s"] is None
               else f"{best['vel_cap_deg_s']} deg/s")
        head_passes.append(
            f"rule grid best ({best['threshold']} deg, "
            f"{best['persist_s']} s, velocity cap {cap})")
    for set_name in ("old", "new"):
        for row in head["model_comparison"]["feature_sets"][set_name]["rows"]:
            if (row["detected_trials"] >= GATE_TRIALS
                    and row["false_positive_events"] == GATE_FP):
                head_passes.append(
                    f"{row['model']} ({set_name} features)")
    if head_passes:
        parts.append("Head passes the gate: "
                     + ", ".join(head_passes) + ".")
    else:
        parts.append(
            f"Head: nothing passes the gate ({GATE_TRIALS}/5 trials, "
            f"{GATE_FP} false positives) on the new data.")
    shoulder = result["cues"]["shoulder"]
    shoulder_passes = []
    gated = shoulder["signal"]["candidates"]["gated"]["best"]
    if (gated["detected"] >= GATE_TRIALS
            and gated["false_positives"] == GATE_FP):
        shoulder_passes.append(
            f"gated signal best ({gated['threshold']} deg)")
    for set_name in ("old", "new"):
        for row in shoulder["model_comparison"]["feature_sets"][
                set_name]["rows"]:
            if (row["detected_trials"] >= GATE_TRIALS
                    and row["false_positive_events"] == GATE_FP):
                shoulder_passes.append(
                    f"{row['model']} ({set_name} features)")
    if shoulder_passes:
        parts.append("Shoulder passes the gate: "
                     + ", ".join(shoulder_passes) + ".")
    else:
        parts.append(
            f"Shoulder: nothing passes the gate ({GATE_TRIALS}/5 "
            f"trials, {GATE_FP} false positives) on the new data.")
    return " ".join(parts)


def _signal_table(candidate: dict) -> str:
    lines = [
        "| Threshold (deg) | Trials detected | False positives | "
        "Longest run (s) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in candidate["threshold_curve"]:
        lines.append(
            f"| {r['threshold']} | {r['detected']}/5 | "
            f"{r['false_positives']} | {r['longest_run_s']} |")
    return "\n".join(lines)


def _day16_section(result: dict) -> list[str]:
    val = result["validation"]
    head = result["cues"]["head"]
    shoulder = result["cues"]["shoulder"]
    grid = head["rule_grid"]
    best = grid["best"]
    comp = head["model_comparison"]
    old = comp["feature_sets"]["old"]
    new = comp["feature_sets"]["new"]
    sig = shoulder["signal"]
    gated = sig["candidates"]["gated"]
    scomp = shoulder["model_comparison"]
    sold = scomp["feature_sets"]["old"]
    snew = scomp["feature_sets"]["new"]
    warnings = val["warnings"]
    parts = [
        DAY16_MARKER + ": Run 2 re-benchmark harness",
        "",
        f"Source: {result['source']}. The Day 14 capture tool has "
        f"not produced a real session yet, so Day 16 is the harness "
        f"itself: it validates a Run 2 capture, then re-runs the "
        f"Day 15 head evaluation and the Day 11/12 shoulder "
        f"evaluation on the new data, with the Run 2 protocol "
        f"timings (5 s ready, 10 s hold, 5 s rest) instead of the "
        f"Run 1 timings the earlier days hardcoded. When the real "
        f"capture arrives, the same command refreshes this section "
        f"with its numbers.",
        "",
        "### Validation",
        "",
        f"{val['frames']} frames, {val['tracked_fraction']:.1%} "
        f"tracked, work sample {val['work_s']} s. "
        f"{len(val['holds'])} holds checked, "
        f"{sum(1 for h in val['holds'] if h['reached_target'])} "
        f"reached their depth target.",
        "",
    ]
    if warnings:
        parts += [
            "Validator warnings:",
            "",
        ] + [f"- {w}" for w in warnings] + [""]
    else:
        parts += ["No validator warnings.", ""]
    parts += [
        "### Head rule grid (best 12 of 60 configurations by F1)",
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
        f"Head rule baseline on this file (session threshold "
        f"{head['rule_baseline']['detected_trials']}/5 trials, "
        f"{head['rule_baseline']['false_positive_events']} false "
        f"positives at the recorded threshold, 5 s persistence).",
        "",
        "### Head model comparison (same grouped 5-fold CV as Days 8/12)",
        "",
        "Old features "
        f"({len(comp['old_features'])}):",
        "",
        _md_table_day12(old["rows"]),
        "",
        "New features "
        f"({len(comp['new_features'])}):",
        "",
        _md_table_day12(new["rows"]),
        "",
        f"**Head verdict:** {comp['verdict']}",
        "",
        "### Shoulder signal (Day 11 re-run)",
        "",
        f"Alignment verified: {sig['alignment_verified']}. "
        f"Old level best: threshold "
        f"{sig['old_level']['best']['threshold']} deg, "
        f"{sig['old_level']['best']['detected']}/5 trials, "
        f"{sig['old_level']['best']['false_positives']} false "
        f"positives. Gated signal best: threshold "
        f"{gated['best']['threshold']} deg, "
        f"{gated['best']['detected']}/5 trials, "
        f"{gated['best']['false_positives']} false positives.",
        "",
        _signal_table(gated),
        "",
        "Gated persistence curve at the best threshold "
        f"({gated['best']['threshold']} deg): "
        + ", ".join(
            f"{p['persist_s']} s: {p['detected']}/5"
            for p in gated["persistence_curve"]),
        "",
        f"Shoulder rule baseline on this file (old level at the "
        f"recorded threshold, 5 s persistence): "
        f"{shoulder['rule_baseline']['detected_trials']}/5 trials, "
        f"{shoulder['rule_baseline']['false_positive_events']} "
        f"false positives.",
        "",
        "### Shoulder model comparison (same grouped 5-fold CV as Days 8/12)",
        "",
        "Old features "
        f"({len(scomp['old_features'])}):",
        "",
        _md_table_day12(sold["rows"]),
        "",
        "New features "
        f"({len(scomp['new_features'])}):",
        "",
        _md_table_day12(snew["rows"]),
        "",
        f"**Shoulder verdict:** {scomp['verdict']}",
        "",
        f"**Verdict:** {result['verdict']}",
        "",
        "## What the Day 16 harness means",
        "",
        "The harness runs end to end on a Run 2 file: validation, "
        "the Day 15 head grid and model comparison, and the Day "
        "11/12 shoulder signal and model comparison, all with the "
        "Run 2 timings and rule baselines counted on the new data. "
        "Nothing here is a claim about the real capture, which has "
        "not arrived: on the synthetic fixture the holds are clean "
        "steps to target depth with flat work, so the gate passes "
        "there by construction. The real question, whether longer, "
        "deeper holds with raw landmarks separate from ordinary "
        "work, is still open until the capture run lands and this "
        "same command runs on it.",
        "",
    ]
    return parts


def write_report(result: dict, md_path: Path) -> None:
    """Append (or refresh) the Day 16 section in BENCHMARK.md.

    Everything before the Day 16 marker is kept untouched, so the
    Day 8, Day 11, Day 12, and Day 15 sections survive a re-run.
    Running twice does not duplicate the section.
    """
    section = "\n".join(_day16_section(result))
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8")
        if DAY16_MARKER in text:
            head = text.split(DAY16_MARKER, 1)[0].rstrip() + "\n\n"
            md_path.write_text(head + section, encoding="utf-8")
        else:
            md_path.write_text(text.rstrip() + "\n\n" + section,
                               encoding="utf-8")
    else:
        md_path.write_text(section, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PoseGuard Day 16: Run 2 re-benchmark harness")
    parser.add_argument("--csv", required=True,
                        help="Run 2 capture CSV (Day 14 tool output)")
    parser.add_argument("--manifest", default=None,
                        help="Run 2 manifest JSON (Day 14 tool output)")
    parser.add_argument("--out", default=None,
                        help="where to write results_day16.json "
                             "(default: results_day16.json next to this "
                             "module)")
    parser.add_argument("--source", default="Run 2 re-run capture",
                        help="label for the data in results_day16.json "
                             "and BENCHMARK.md (default: Run 2 re-run "
                             "capture)")
    args = parser.parse_args(argv)

    try:
        result = run_day16(args.csv, args.manifest, source=args.source)
    except (ValueError, OSError) as e:
        # A missing file is an OSError, not a validation failure:
        # both are user errors with a clear message, exit code 2,
        # and no traceback either way.
        print(f"Day 16 failed:\n{e}", file=sys.stderr)
        return 2
    here = Path(__file__).parent
    out = Path(args.out) if args.out else here / "results_day16.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result, here / "BENCHMARK.md")
    print(f"Day 16 results written to {out} and {here / 'BENCHMARK.md'}")
    val = result["validation"]
    print(f"Validation: {val['frames']} frames, "
          f"{val['tracked_fraction']:.1%} tracked, "
          f"{len(val['warnings'])} warnings")
    head = result["cues"]["head"]
    b = head["rule_grid"]["best"]
    print(f"\nHead rule grid best: {b['threshold']} deg, "
          f"{b['persist_s']} s, cap {b['vel_cap_deg_s']}, "
          f"{b['detected']}/5 trials, {b['false_positives']} FP, "
          f"F1 {b['f1']}")
    for set_name in ("old", "new"):
        print(f"\nhead ({set_name}):")
        for r in head["model_comparison"]["feature_sets"][set_name]["rows"]:
            print(f"  {r['model']:20s} acc={r['accuracy']:.3f} "
                  f"F1={r['f1_tilted']:.3f} "
                  f"trials={r['detected_trials']}/5 "
                  f"FP={r['false_positive_events']} "
                  f"calm={r['calm_accuracy']:.3f}")
    shoulder = result["cues"]["shoulder"]
    g = shoulder["signal"]["candidates"]["gated"]["best"]
    print(f"\nShoulder gated best: {g['threshold']} deg, "
          f"{g['detected']}/5 trials, {g['false_positives']} FP")
    for set_name in ("old", "new"):
        print(f"\nshoulder ({set_name}):")
        for r in shoulder["model_comparison"]["feature_sets"][
                set_name]["rows"]:
            print(f"  {r['model']:20s} acc={r['accuracy']:.3f} "
                  f"F1={r['f1_tilted']:.3f} "
                  f"trials={r['detected_trials']}/5 "
                  f"FP={r['false_positive_events']} "
                  f"calm={r['calm_accuracy']:.3f}")
    print(f"\nverdict: {result['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Render the analysis of one session as a Markdown report."""

from __future__ import annotations

from .loader import SessionData


def _trial_table(cue_result: dict) -> str:
    lines = ["| trial | samples | mean dev (deg) | std | max | detected |",
             "|---:|---:|---:|---:|---:|---|"]
    for i, t in enumerate(cue_result["analysis"]["per_trial"], start=1):
        mean = f"{t['mean']:.2f}" if t["mean"] == t["mean"] else "n/a"
        std = f"{t['std']:.2f}" if t["std"] == t["std"] else "n/a"
        mx = f"{t['max_abs']:.2f}" if t["max_abs"] == t["max_abs"] else "n/a"
        flag = " **SHORT**" if t["samples"] < 30 else ""
        lines.append(f"| {i} | {t['samples']}{flag} | {mean} | {std} | {mx} "
                     f"| {'yes' if t['detected'] else 'no'} |")
    return "\n".join(lines)


def _sens_table(rows: list[dict]) -> str:
    lines = ["| cue | threshold (deg) | detected | false positives | longest run (s) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['cue']} | {r['threshold']} | {r['detected']} | "
                     f"{r['false_positives']} | {r['longest_run_s']} |")
    return "\n".join(lines)


def build_markdown(session: SessionData, result: dict,
                   frames=None) -> str:
    """Full Markdown report for one analysed session.

    `frames` is the deviation-enriched frame table from
    `analysis.add_deviations` (the CLI passes it). Without it, the
    report still renders the Day 3 sections from `session.frames`,
    but the Day 4 statistics are skipped: they need the smoothed
    circular deviations.
    """
    f = frames if frames is not None else session.frames
    steps = {s: int((f["step"] == s).sum())
             for s in ["calib", "head", "shoulder", "work", "loss", "results"]}
    loss = result["loss_test"]
    lines = [
        f"# PoseGuard validation report: {session.meta['Exported']}",
        "",
        f"Engine: {session.meta['Engine']}",
        f"Frames: {len(f)}  |  calibration {steps['calib']}  |  head trials "
        f"{steps['head']}  |  shoulder trials {steps['shoulder']}  |  "
        f"ordinary work {steps['work']}  |  landmark loss {steps['loss']}",
        f"Baselines: head {session.baseline_head} deg, "
        f"shoulder {session.baseline_shoulder} deg",
        "",
        "## Verdict",
        "",
    ]
    for cue in ("head", "shoulder"):
        c = result["cues"][cue]
        status = "PASS" if c["pass"] else "FAIL"
        lines.append(f"### {cue} tilt: {status} "
                     f"({c['detected']}/5 detected at {c['threshold']} deg, "
                     f"{c['false_positives']} false positives)")
        for reason in c["reasons"]:
            lines.append(f"- {reason}")
        align = c["alignment"]
        verified = "verified against the recorded trial summary" \
            if align["verified"] else \
            "INFERRED: no trial summary was recorded for this cue, " \
            "alignment reused from the other cue"
        lines.append(f"- Trial clock offset: {align['offset_s']} s ({verified})")
        lines.append("")
        lines.append(_trial_table(c))
        lines.append("")
    lines += [
        "## Landmark-loss test",
        "",
        f"Tracking lost for {loss['tracking_lost_frames']} frames "
        f"(~{loss['tracking_lost_seconds']} s) out of {loss['frames_total']}.",
        f"Head firings during loss step: {len(loss['head_events'])}; "
        f"shoulder firings: {len(loss['shoulder_events'])}.",
        f"Test ran: {'yes' if loss['ran'] else 'NO, camera never covered'}. "
        f"Clean: {'yes' if loss['clean'] else 'no'}.",
        "",
        "## Threshold sensitivity",
        "",
        "What each threshold would have decided. Detection needs a 5 s "
        "sustained breach inside a 6 s hold.",
        "",
        _sens_table(result["sensitivity"]),
        "",
        "## The wraparound bug, in numbers",
        "",
        "The file's own deviation columns used plain subtraction. "
        "Counting frames where |linear deviation| exceeds 180 degrees "
        "shows how often the old math produced phantom jumps:",
        "",
    ]
    bad_h = int((f["head_deviation"].abs() > 180).sum())
    bad_s = int((f["shoulder_deviation"].abs() > 180).sum())
    lines.append(f"- Head: {bad_h} frames with |linear deviation| > 180 deg")
    lines.append(f"- Shoulder: {bad_s} frames with |linear deviation| > 180 deg")
    lines.append("- Circular deviation is always within [-180, 180] by "
                 "construction, so these phantoms disappear entirely.")
    lines.append("")

    # Day 4: statistical validation. Imported here so report.py stays
    # usable without the stats module for older callers. Skipped when
    # no enriched frames were passed (see docstring).
    if frames is None:
        return "\n".join(lines)

    from .stats import statistical_verdict

    stats = statistical_verdict(f, session)
    lines += [
        "## Statistical validation (Day 4)",
        "",
        "With only 5 trials per cue, a raw 0/5 is not proof the true "
        "rate is zero. The Wilson interval below is the honest range: "
        "the true detection rate is plausibly anywhere inside it. The "
        "gate needs 80%, so a cue passes this check only if the "
        "interval's top end clears 80% AND false positives are zero.",
        "",
    ]
    for cue in ("head", "shoulder"):
        s = stats["cues"][cue]
        lines.append(f"### {cue} tilt: {s['verdict'].replace('_', ' ')}")
        lines.append("")
        lines.append(f"- Detected {s['detected']}/5, 95% Wilson interval "
                     f"{s['ci_low']:.0%} to {s['ci_high']:.0%} "
                     f"(gate needs 80%).")
        lines.append(f"- False positives: {s['false_positives']} in "
                     f"{s['work_seconds'] / 60:.1f} minutes of ordinary "
                     f"work, about {s['fp_per_hour']:.0f} per hour "
                     f"(95% interval {s['fp_per_hour_low']:.0f} to "
                     f"{s['fp_per_hour_high']:.0f}).")
        lines.append(f"- Longest hold above {s['threshold']} deg: "
                     f"{s['longest_run_s']} s (rule needs 5 s).")
        zfp = s["zero_fp_threshold"]
        if zfp is None:
            lines.append("- No threshold up to 30 deg kills all false "
                         "positives.")
        else:
            lines.append(f"- Lowest threshold with zero false positives: "
                         f"{zfp['threshold']} deg, where "
                         f"{zfp['detected']}/5 holds are detected.")
        lines.append("")
        lines.append(f"**Verdict:** {s['verdict_text']}")
        lines.append("")
        lines.append("| threshold (deg) | detected | 95% CI | false positives | FP/hour |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in s["threshold_curve"]:
            lines.append(f"| {r['threshold']} | {r['detected']}/5 | "
                         f"{r['ci_low']:.0%}-{r['ci_high']:.0%} | "
                         f"{r['false_positives']} | {r['fp_per_hour']:.0f} |")
        lines.append("")
        lines.append(f"Persistence curve at {s['persistence_threshold']} "
                     f"deg (lowest tested threshold): what each "
                     f"persistence rule would have decided.")
        lines.append("| persistence rule (s) | detected |")
        lines.append("|---:|---:|")
        for r in s["persistence_curve"]:
            lines.append(f"| {r['persist_s']} | {r['detected']}/5 |")
        lines.append("")
    return "\n".join(lines)

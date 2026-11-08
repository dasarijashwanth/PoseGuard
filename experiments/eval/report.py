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


def _model_section(model_results: dict) -> list[str]:
    """Day 5 classifier results, from experiments/model/results.json."""
    lines = [
        "## Machine-learning classifier (Day 5)",
        "",
        "One scikit-learn classifier per cue, over 1-second windows of "
        "the smoothed circular deviations (10 frames each). Labels come "
        "from the protocol: windows inside a trial hold are 'tilted', "
        "windows from ordinary work and the ready/rest gaps are "
        "'upright'. Logistic regression and a 200-tree random forest "
        "were trained per cue; the better one by out-of-fold F1 is "
        "kept. Evaluation is grouped 5-fold cross-validation, grouped "
        "by trial, so no trial's windows appear on both sides.",
        "",
    ]
    for cue in ("head", "shoulder"):
        r = model_results["cues"][cue]
        m = r["best_oof_metrics"]
        t = r["best_oof_trials"]
        b = r["rule_baseline"]
        lines.append(f"### {cue} tilt: {r['best_model']} "
                     f"({r['n_tilted_windows']} tilted vs "
                     f"{r['n_upright_windows']} upright windows)")
        lines.append("")
        lines.append(f"- Out-of-fold accuracy {m['accuracy']}, "
                     f"precision (tilted) {m['precision_tilted']}, "
                     f"recall (tilted) {m['recall_tilted']}, "
                     f"F1 (tilted) {m['f1_tilted']}. Confusion: "
                     f"TP {m['confusion']['tp']}, FP "
                     f"{m['confusion']['fp']}, FN "
                     f"{m['confusion']['fn']}, TN "
                     f"{m['confusion']['tn']}.")
        lines.append(f"- Trials detected {t['detected_trials']}/5 "
                     f"(rule baseline: {b['detected_trials']}/5), "
                     f"false-positive events "
                     f"{t['false_positive_events']} (rule baseline: "
                     f"{b['false_positive_events']}). A trial counts "
                     f"when at least half its hold windows are "
                     f"predicted tilted; a false-positive event is 5 "
                     f"consecutive tilted work windows.")
        lines.append("- Decision-threshold sweep (trials detected / "
                     "false-positive events): "
                     + ", ".join(f"{s['threshold']}: "
                                 f"{s['detected_trials']}/5, "
                                 f"{s['false_positive_events']} FP"
                                 for s in r["threshold_sweep"])
                     + ". No cutoff passes the gate (4/5 trials, 0 "
                       "false positives).")
        lines.append(f"- Calm-only accuracy "
                     f"{r['calm_only_metrics']['accuracy']} on "
                     f"{r['calm_windows']} windows where the shoulder "
                     f"line was stable (std < 5 deg, i.e. no tracker "
                     f"noise). If the signal were posture, it would "
                     f"survive here.")
        lines.append("")
    lines += [
        "**Verdict:** the classifier does not beat the rule baseline on "
        "either cue. The head model detects 1/5 trials with 6 "
        "false-positive events (the rule: 0/5, 0 FP). The shoulder "
        "model detects 5/5 but with precision 0.10 and 5 "
        "false-positive events (the rule: 0/5, 7 FP), and the calm-only "
        "check shows its separation is tracker stillness, not shoulder "
        "tilt: during the holds you sat still, during work the tracker "
        "was noisy. Worst of all, the head model learned the "
        "relationship backwards: a flat upright window scores 0.56 "
        "tilted, while a steady +6 deg tilt scores 0.33, because in "
        "Run 1 the biggest head deviations happened during noisy desk "
        "work, not during holds. One session of five brief holds is "
        "too little data to train a posture classifier, and protocol "
        "labels are not ground truth: two of the five head holds barely "
        "tilted at all. The classifier is built, tested and served "
        "behind FastAPI (see experiments/model/), but it needs the "
        "re-run data (longer, deeper holds, more trials) before it can "
        "say anything the rule cannot.",
        "",
    ]
    return lines


def _tracking_section(comparison: dict) -> list[str]:
    """Day 6 MLflow comparison, from experiments/model/mlflow_comparison.json."""
    lines = [
        "## MLflow experiment tracking (Day 6)",
        "",
        "Day 5 left its numbers in one `results.json`. Day 6 logs every "
        "training run to MLflow (local file store, "
        "`experiments/model/mlruns/`), so runs can be compared side by "
        "side instead of overwritten. Each run records which model won "
        "per cue, the out-of-fold metrics, trials detected and "
        "false-positive events, plus the trained model as an artifact. "
        "View them with `mlflow ui --backend-store-uri "
        "experiments/model/mlruns` (http://127.0.0.1:5000).",
        "",
        "| run | head trials | head FP | head F1 (tilted) | shoulder trials | shoulder FP | shoulder F1 (tilted) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparison["runs"]:
        m = r["metrics"]
        lines.append(
            f"| {r['run_name']} | {m.get('head_trials_detected', 0):.0f}/5 | "
            f"{m.get('head_false_positive_events', 0):.0f} | "
            f"{m.get('head_f1_tilted', 0):.3f} | "
            f"{m.get('shoulder_trials_detected', 0):.0f}/5 | "
            f"{m.get('shoulder_false_positive_events', 0):.0f} | "
            f"{m.get('shoulder_f1_tilted', 0):.3f} |")
    lines += [
        "",
        "**What the comparison shows:** the random-forest-only run "
        "looks better at first glance (accuracy about 0.93 on both "
        "cues, against 0.45-0.55 for the baseline) but detects 0/5 "
        "trials on both cues: it learned to always answer 'upright', "
        "which is right 93% of the time because 360 of 383 windows "
        "are upright. Accuracy lies on imbalanced data; F1 (tilted) "
        "is the honest score, and it is 0.000. The baseline at least "
        "detects something (head 1/5, shoulder 5/5), at the cost of "
        "false positives. Neither run beats the rule baseline, and no "
        "model choice fixes a dataset of five brief holds. That is "
        "exactly what tracking is for: the comparison is written down "
        "and checkable, instead of a single flattering number.",
        "",
    ]
    return lines


def build_markdown(session: SessionData, result: dict,
                   frames=None, model_results: dict | None = None,
                   tracking_comparison: dict | None = None) -> str:
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

    # Day 5: classifier results, when results.json exists.
    if model_results is not None:
        lines += _model_section(model_results)

    # Day 6: MLflow run comparison, when mlflow_comparison.json exists.
    if tracking_comparison is not None:
        lines += _tracking_section(tracking_comparison)

    return "\n".join(lines)

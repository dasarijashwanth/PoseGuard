"""PoseGuard public demo (Day 18).

Run locally:  streamlit run demo/app.py
Deploy free:  Streamlit Community Cloud, pointed at this repo, main file demo/app.py.

The app shows the real Run 1 validation session, not a simulation:
sample angle traces, gate verdicts per trial, and the model scoreboard.
The honest verdict is the headline: nothing passes the gate yet.
"""

from __future__ import annotations

import altair as alt
import streamlit as st

from demo_data import (
    HEAD_THRESHOLD_DEG,
    SHOULDER_THRESHOLD_DEG,
    VERDICT,
    best_rule,
    gate_verdict,
    load_sample,
    sample_stats,
    scoreboard_frame,
    trials_frame,
)


def main() -> None:
    st.set_page_config(page_title="PoseGuard: on-device posture cues", layout="wide")
    st.title("PoseGuard: can a webcam tell when you slouch?")
    st.write(
        "On-device vision AI that watches head and shoulder angles during desk "
        "work and nudges you to sit up. This demo shows the real validation "
        "session (Run 1, Nov 3, 2026) and every number the project has "
        "measured so far, including the ones that do not flatter it."
    )

    st.error("Honest verdict: " + VERDICT)

    df = load_sample()
    stats = sample_stats(df)

    st.header("The session, as recorded")
    st.write(
        f"Run 1: {stats['rows']} samples at 1 Hz over {stats['duration_s']} s, "
        f"steps {', '.join(stats['steps'])}. Largest head deviation "
        f"{stats['max_abs_head_dev']} deg and largest shoulder deviation "
        f"{stats['max_abs_shoulder_dev']} deg, both at 447 s: a 6-second "
        "tracker meltdown mid-work, not a posture change. The chart axis is "
        "clipped at +/-25 deg so that meltdown does not hide the 2-7 deg "
        "holds; gaps are frames where the tracker was lost or gated out. "
        "Angles are circular deviations from the calibration baseline, "
        "shoulder jump-gated the Day 11 way."
    )
    chart_df = df.melt(
        id_vars=["elapsed_s"],
        value_vars=["head_dev_deg", "shoulder_dev_deg"],
        var_name="cue",
        value_name="dev_deg",
    )
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("elapsed_s", title="Elapsed (s)"),
            y=alt.Y("dev_deg", title="Deviation from baseline (deg)",
                    scale=alt.Scale(domain=[-25, 25])),
            color=alt.Color("cue", title="Cue"),
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Session thresholds: head {HEAD_THRESHOLD_DEG} deg, shoulder "
        f"{SHOULDER_THRESHOLD_DEG} deg. Deliberate holds were 10 s; ordinary "
        "work drifts 4-7 deg on the shoulder line, which is why the holds "
        "hide inside everyday movement."
    )

    st.header("Gate verdicts per trial")
    st.write(
        "The gate is 4/5 trials detected with 0 false positives. Trial 5 is "
        "barely recorded (6 head frames, 8 shoulder frames), so at most 4/5 "
        "trials are measurable in Run 1."
    )
    for cue in ("head", "shoulder"):
        rule = best_rule(cue)
        verdict = gate_verdict(rule["detected"], rule["false_positives"])
        st.subheader(f"{cue.capitalize()} cue: best rule")
        st.write(
            f"{rule['threshold_deg']} deg / {rule['persist_s']} s / "
            f"{rule['vel_cap_deg_s']} deg/s cap: {rule['detected']}/5 trials, "
            f"{rule['false_positives']} false positives, F1 {rule['f1']}. "
            + verdict["reason"]
        )
        st.dataframe(trials_frame(cue), use_container_width=True)
    st.caption(
        "Dwell is the longest run above the dwell threshold (head 4 deg, "
        "shoulder 2 deg). The holds are shallow as well as brief: the deepest "
        "head hold peaks at 7.1 deg against an 8 deg threshold no hold ever reaches."
    )

    st.header("Model scoreboard")
    st.write(
        "Grouped 5-fold CV by trial, same protocol for every row. Latest "
        "feature set per cue (head: Day 15 velocity + dwell; shoulder: Day 17 "
        "velocity + dwell). The new features made both models worse, so the "
        "served models are unchanged."
    )
    st.dataframe(scoreboard_frame(), use_container_width=True)
    st.caption(
        "The random forest looks accurate only because it predicts almost "
        "everything upright: 0/5 trials on both cues. Calm-only accuracy is "
        "accuracy on work windows with no tilt."
    )

    st.header("What would fix it")
    st.write(
        "A camera re-run with raw landmarks and longer, deeper holds "
        "(the Day 14 capture tool: 10 s holds, shoulder 8 deg and head "
        "15 deg targets). The Day 16 harness re-benchmarks everything the "
        "moment that capture lands. Until then the served models stay as "
        "they are, and the gate stays unpassed."
    )
    st.write(
        "Code, walkthroughs, and the full benchmark tables: "
        "github.com/dasarijashwanth/PoseGuard"
    )


if __name__ == "__main__":
    main()

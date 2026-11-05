# Day 3 walkthrough: the evaluation harness, in plain English

## What this is

The webcam experiment writes a CSV every time you run it: one row per
video frame, with the head and shoulder angles it measured. This
harness (`experiments/eval/`) reads that CSV and answers, with numbers,
the question the whole validation is about: **do the two posture cues
actually work?**

Run it from the repo root:

```
python -m experiments.eval.cli --csv <path/to/validation.csv>
```

It prints the verdict and writes the full report to
`experiments/eval/REPORT.md`.

## The four ideas that matter

### 1. Circular angle math: the short way round the circle

Angles live on a circle, not a number line. The shoulder angle is
reported between -180 and +180 degrees. Your calibrated baseline is
+174 degrees. When you lean slightly the other way, the reading flips
to -179 degrees. Plain subtraction says -179 - 174 = -353 degrees,
a phantom jump. The real distance is 7 degrees.

The fix is one line: measure the difference with sine and cosine,
which wrap it into [-180, 180] automatically.

```
circ_dev(angle, baseline) = atan2(sin(angle - baseline), cos(angle - baseline))
```

That is the same fix already in the web experiment. The harness
recomputes every deviation this way, and `REPORT.md` counts the
damage: 1,188 shoulder frames in Run 1 had a linear deviation beyond
180 degrees. All phantoms, all gone with circular math.

### 2. Smoothing: don't trust a single frame

One video frame is noisy. The harness smooths the angle stream with
an exponential moving average (EMA): each new smoothed value is 60%
old value, 40% new reading. A one-frame glitch barely moves it; a
real sustained tilt moves it steadily. Same smoothing the web page
uses, same 0.4 weight.

### 3. Persistence: a cue only fires after 5 steady seconds

A detection is not "one frame past the threshold". The deviation has
to stay beyond the threshold continuously for 5 seconds, with
tracking alive the whole time. If tracking drops, or the angle comes
back inside, the timer resets to zero. That is why a brief big tilt
does not count, and it is the rule the harness applies when it
replays your CSV.

### 4. Threshold sensitivity: what if the threshold had been different?

This is the most useful table the harness produces. It replays the
whole session at several thresholds and shows what each one would
have decided. If dropping the head threshold from 8 to 5 still gives
0/5 detections, the threshold is not the problem. The movement
itself is too small or too brief, and no threshold tweak will save
that cue.

## What Run 1 actually tells us

**Head tilt: FAIL, 0/5.** Not a threshold problem. Even at 5 degrees
the longest unbroken run above threshold was 1.5 seconds, against a
5-second requirement. Your tilts peaked around 4-5 degrees and never
held. Trial 5 only has 6 samples: that hold was cut short, you moved
on after about half a second.

**Shoulder tilt: FAIL, 0/5, and 7 false positives.** Two separate
problems. First, the trial holds never produced a 5-second breach at
any threshold from 4 to 8 degrees. Second, during ordinary work the
shoulder line genuinely sat more than 6 degrees off baseline for
5-20 second stretches, seven times. The wraparound bug made those
look like absurd -350 degree jumps, but fixing the math does not fix
the cue: leaning and reaching while you type looks exactly like a
dropped shoulder to this metric. Trial 3 also shows the landmark
tracker itself going wild for a few seconds (angles swinging over
100 degrees frame to frame), which is sensor noise, not posture.

**Landmark-loss test: never ran.** Tracking was lost for only 18
scattered frames out of 1,828. The camera was never covered. Two head
firings also went off during that step, so it fails the "clean" check
as well.

## The honest conclusion

The head cue needs bigger, longer holds (or a lower bar: a 3-degree
threshold with a shorter persistence would have caught trial 1).
The shoulder cue, as designed, cannot tell a deliberate shoulder
drop from ordinary leaning at a desk. That is a design finding, not
a bug: either the cue needs a different signal, or the product needs
to accept that shoulder tilt alone is not a reliable cue.

## Files

- `circular.py`: the angle math (wrap, deviation, circular mean).
- `loader.py`: parses and validates the export CSV.
- `analysis.py`: smoothing, persistence detection, trial alignment,
  verdicts, sensitivity table.
- `report.py`: renders `REPORT.md`.
- `cli.py`: the command line entry point.
- `tests/`: pytest suite, including a synthetic session with known
  answers and an end-to-end test on the real Run 1 CSV.

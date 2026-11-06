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

## Day 4: the statistics, in plain English

Day 3 told you what happened at each threshold. Day 4 asks whether
*any* setting could have made it work, using three ideas from
statistics. The new code lives in `stats.py`, and the numbers land in
the "Statistical validation" section of `REPORT.md`.

### 1. The Wilson interval: 0/5 does not mean "never"

Five trials is a tiny sample. If you flip a coin 5 times and get zero
heads, you would not conclude the coin *never* lands heads. The
Wilson interval is the honest version of that thought: given 0 hits
out of 5 tries, the true detection rate is plausibly anywhere from 0%
up to about 43%. That is the 95% confidence interval, and it is the
right tool because the sample is small and the result sits at the edge
(0 out of 5), where the usual "average plus or minus" math breaks.

Why it matters: the gate needs 80% detection (4 out of 5). Even the
*optimistic* end of the interval, 43%, is nowhere near 80%. So the
head cue did not fail by bad luck on a small sample. It failed by a
wide margin. No amount of re-running the same 5 trials would plausibly
flip it to a pass.

### 2. False positives per hour: how annoying is the cue?

A raw count like "7 false positives" is hard to judge without knowing
how long the work sample was. Seven false alarms in five minutes of
typing is about 83 per hour, roughly one every 45 seconds. That is the
rate, and the interval around it (about 21 to 145 per hour) says the
true rate is high even at the optimistic end. A reminder that fires
every minute while you work is not a reminder, it is noise. This is
why the shoulder cue fails even before detection is considered.

### 3. Threshold and persistence curves: is there ANY setting that works?

Instead of asking about one threshold, Day 4 sweeps both knobs:

- **Threshold curve:** detection rate and false-positive rate at every
  tested threshold. A usable cue needs some row where detection is
  high *and* false positives are zero. For the shoulder, no such row
  exists: at 4 degrees there are 12 false positives, at 8 degrees
  there are still 4, and detection is 0/5 everywhere.
- **Persistence curve:** what if the firing rule needed 0.5, 1, 1.5,
  2 or 3 seconds instead of 5? At 5 degrees with a 0.5-second rule,
  the head cue would have caught 3 of 5 holds. The tilts were real,
  they just lasted half a second to a second and a half. That is the
  strongest evidence the head cue is salvageable: the movement is
  there, the duration is not.
- **Zero-FP threshold:** the lowest threshold with zero false
  positives. For the shoulder, *no* threshold up to 30 degrees kills
  all false positives, the tracker noise alone fires the cue. For the
  head it is 4 degrees, where detection is still 0/5.

### The verdicts, and what they mean for the re-run

**Head tilt: salvageable.** Clean record (zero false positives at
every threshold), real tilts that were too brief. The re-run needs
longer, deeper holds: aim for a full 6 seconds past 5 degrees, not a
quick dip. If the re-run still shows 0/5, the cue itself is in
question, but Run 1 does not prove that.

**Shoulder tilt: the metric is wrong for desk work.** No threshold
and no persistence rule passes both criteria, and even a 30-degree
threshold still fires during ordinary typing. Leaning at a desk and
dropping a shoulder produce the same angle, so this angle cannot tell
them apart. Day 5's classifier will have to find a different signal
(say, shoulder height difference, or the angle *plus* how long you
have been leaning), because tuning this one further is a dead end.

## Files

- `circular.py`: the angle math (wrap, deviation, circular mean).
- `loader.py`: parses and validates the export CSV.
- `analysis.py`: smoothing, persistence detection, trial alignment,
  verdicts, sensitivity table.
- `stats.py`: Wilson intervals, false-positive rates, threshold and
  persistence curves, zero-FP threshold, per-cue verdicts.
- `report.py`: renders `REPORT.md`.
- `cli.py`: the command line entry point.
- `tests/`: pytest suite, including a synthetic session with known
  answers and an end-to-end test on the real Run 1 CSV.

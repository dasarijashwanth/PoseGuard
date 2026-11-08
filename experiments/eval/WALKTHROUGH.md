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

## Day 5: the classifier, in plain English

Day 3 and 4 used fixed rules: "fire when the angle stays past a
threshold for 5 seconds". Day 5 asks whether a machine-learning
model can do better than the rule. Instead of one angle at one
moment, the model looks at a whole second of angles at a time (10
frames) and learns what a tilt *looks like* from the data itself.

### What the model sees

Each 1-second window becomes a short list of numbers, the features:

- **Head cue:** how far the head is tilted on average, the biggest
  tilt in the window, which direction (the tilts go one way), how
  steady the head line is, and how jumpy it is frame to frame.
- **Shoulder cue:** deliberately NOT the shoulder angle itself. Day 4
  proved that angle cannot tell a dropped shoulder from leaning at a
  desk, so the model is forbidden from seeing it. It gets only
  movement: how jumpy the shoulder line is, how far it swings, how
  fast it drifts, how the head and shoulder move together, and the
  head features as context. There is a test that enforces this ban:
  if anyone adds a shoulder-level feature to the list, the test
  fails.

### How it learns, and how it is graded

The windows are labelled by the protocol, not by a threshold: windows
inside a trial hold are "tilted", windows from ordinary typing are
"upright". Two models are trained per cue, logistic regression (a
straight-line divider, easy to read) and a random forest (200
decision trees, catches curved boundaries). The better one is kept.

Grading is grouped cross-validation, grouped by trial. That means
when the model is tested on trial 4's windows, it has never seen
trial 4 at all, it only trained on the other trials. That is the
honest way to grade with 5 trials: a random split would let the
model memorize a trial's quirks and flatter the score. The same
out-of-fold predictions give the trial verdicts (a trial counts when
at least half its hold windows are called tilted) and the
false-positive events (5 tilted work windows in a row), so the
numbers compare directly with the rule baseline from Days 3 and 4.

Two extra checks keep the grading honest. A decision-threshold sweep
asks whether ANY cutoff could pass the gate (4 of 5 trials, zero
false positives). A calm-only check re-scores the model on windows
where the shoulder line was stable, removing the tracker-noise
episodes: if the signal is posture, it should survive there.

### What Run 1 actually tells us

**The classifier does not beat the rule on either cue.** That is the
finding, and it is worth more than a flattering number would be.

**Head: 1/5 trials, 6 false-positive events.** The rule got 0/5
with zero false positives. The model caught one trial the rule
missed, but at the cost of six false alarms during ordinary work.
Its F1 score for the tilted class is 0.078, which in plain terms
means: when it says "tilted", it is wrong 96 times out of 100.

**Shoulder: 5/5 trials, but precision 0.10 and 5 false-positive
events.** The rule got 0/5 with 7 false positives. The 5/5 looks
like a win until you read the rest: the model calls 134 of 318 work
windows tilted. The calm-only check exposes why: on calm windows
its accuracy falls to 0.45, worse than guessing the majority class.
The model did not learn shoulder tilt. It learned that during the
holds you sat still and during work the tracker was noisy, so
"steady" became its proxy for "tilted". A proxy like that breaks
the moment you sit still at your desk, which is most of the day.

**No decision threshold passes the gate.** At a low cutoff the head
model reaches 4/5 trials but with 9 false alarms; at a high cutoff
the false alarms vanish but so does all detection. The shoulder is
the same story: 5/5 with 16 false alarms at a low cutoff, 0/5 at a
high one. There is no setting where both criteria hold, exactly as
Day 4 found for the rule's thresholds.

**The head model learned the relationship backwards.** This is the
sharpest detail. A perfectly flat upright window scores 0.56
tilted, while a steady 6-degree tilt scores 0.33. The model
genuinely believes big head deviations mean "upright", because in
Run 1 the biggest head deviations happened during noisy desk work,
not during the holds. A model that scores a real tilt lower than
sitting still is not a detector, it is a mirror of the noise in one
afternoon's data.

### Why it failed, and what would fix it

Two reasons, both about data, not about the algorithm. First, one
session of five brief holds is too little data: 23 tilted windows
against 360 upright ones, and the holds were 1.5-second dips, not
the 6-second holds the protocol asked for. Second, protocol labels
are not ground truth: two of the five head holds barely tilted at
all, so some "tilted" training examples show no tilt. No model can
learn a boundary between two groups that overlap almost completely.

The fix is the re-run, not a fancier model: longer, deeper holds,
more trials, and ideally a second session so the model can be tested
on a day it never trained on. The classifier itself is built,
tested (26 new tests) and served behind FastAPI (`GET /health`,
`POST /predict`), and it will be ready to re-train the moment that
data exists. Day 6 (MLflow tracking, Docker) builds on this package.

## Day 5 files

- `experiments/model/features.py`: 1-second window features, with
  the shoulder-level ban.
- `experiments/model/train.py`: training, grouped cross-validation,
  threshold sweep, calm-only check, artifact saving.
- `experiments/model/api.py`: FastAPI app, `/health` and `/predict`.
- `experiments/model/artifacts/`: the trained models (about 2 KB
  each, logistic regression won both cues).
- `experiments/model/results.json`: full metrics, picked up by the
  eval CLI into `REPORT.md`.
- `experiments/model/tests/`: 26 tests, including synthetic
  clean-signal learning and API tests via TestClient.

## Day 6: tracking experiments and shipping the API, in plain English

Day 5 ended with one file, `results.json`, holding one training
run. That is fine for one run and useless for two: the moment you
try a different model, the old numbers are overwritten and you can
no longer say, honestly, which one was better. Day 6 fixes that
with the two tools every ML team actually uses: MLflow for tracking
experiments, and Docker for shipping the API.

### MLflow: a lab notebook that cannot be overwritten

MLflow is a lab notebook for training runs. Every time you train,
one **run** is recorded under a name, with three things:

- **Params:** the settings of that run. Which model won per cue,
  how many windows were tilted, which features were used.
- **Metrics:** the scores. Accuracy, precision, recall, F1 per cue,
  trials detected, false-positive events, calm-only accuracy.
- **Artifacts:** the trained model files themselves, so any run can
  be re-served exactly as it was.

The store is a local folder, `experiments/model/mlruns/`, no server
needed. Log a run with one flag:

```
python -m experiments.model.train --csv <path> --mlflow
python -m experiments.model.train --csv <path> --model random_forest \
    --mlflow --run-name random_forest_only
```

The second command forces one model instead of letting the trainer
pick the better of the two. That is the comparison run: same data,
same grading, one deliberate difference. Read the comparison back
with `python -m experiments.model.tracking`, which writes
`mlflow_comparison.json`, and the eval CLI picks that up into
`REPORT.md`. Or open the web UI:

```
mlflow ui --backend-store-uri experiments/model/mlruns
```

### The comparison, and why accuracy lies

Two runs are logged on Run 1. The baseline (Day 5, best model per
cue) and a random-forest-only run:

| run | head trials | head F1 | shoulder trials | shoulder F1 |
|---|---|---|---|---|
| day5_baseline | 1/5 | 0.078 | 5/5 | 0.179 |
| random_forest_only | 0/5 | 0.000 | 0/5 | 0.000 |

The forest run looks better at first glance: accuracy about 0.93 on
both cues, against 0.45 to 0.55 for the baseline. But it detects
nothing at all. It learned to always answer "upright", which is
right 93% of the time because 360 of 383 windows are upright. That
is the classic imbalanced-data trap: accuracy rewards the lazy
answer, and F1 for the rare class is the honest score. The
baseline at least detects something (head 1/5, shoulder 5/5), at
the cost of false positives. Neither run beats the rule, and no
model choice fixes five brief holds. The point of tracking is that
this comparison is written down and checkable, instead of one
flattering number in a chat message.

### Docker: the API as a sealed box

The FastAPI service runs on your machine today with
`uvicorn experiments.model.api:app`. Docker packs that into a sealed
box: a `Dockerfile` that starts from a slim Python image, installs
the exact requirements, copies the code and the trained models, and
runs the same uvicorn command on port 8000. Anyone with Docker can
run the same box, on any machine, and get the same API:

```
docker build -f experiments/model/Dockerfile -t poseguard-api .
docker run -p 8000:8000 poseguard-api
```

`docker-compose.yml` runs two boxes at once: the API on port 8000
and the MLflow web UI on port 5000, so the tracking store is
browsable while the API serves. A fresh retrain can be served
without rebuilding the image: the compose file mounts the local
`artifacts/` folder over the baked-in models.

One honest limit: Docker is not installed on this VM, so the image
was **not** built here. The Dockerfile and compose file are verified
by tests that check the definition is consistent (base image,
requirements file, entrypoint, ports), not by a real build. Before
trusting the box, build it once on a machine with Docker.

## Day 6 files

- `experiments/model/tracking.py`: MLflow logging, run comparison,
  `mlflow_comparison.json`.
- `experiments/model/train.py`: new `--model`, `--mlflow` and
  `--run-name` flags.
- `experiments/model/Dockerfile`: slim Python image, API on port 8000.
- `experiments/model/docker-compose.yml`: API + MLflow UI services.
- `.dockerignore` (repo root): keeps the tracking store, caches and
  `.git` out of the image.
- `experiments/model/mlflow_comparison.json`: the two logged runs,
  picked up by the eval CLI into `REPORT.md`.
- `experiments/model/tests/test_tracking.py` and `test_docker.py`:
  11 new tests.

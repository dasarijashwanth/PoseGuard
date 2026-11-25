# Camera re-run capture checklist

One session, about 8.6 minutes, no installs. The tool is
`experiments/rerun-capture.html`: one file, the pose model loads
from the CDN, all processing stays in the browser tab. Full
background: `experiments/WALKTHROUGH_DAY14.md`.

## Before you start

1. Download `experiments/rerun-capture.html` from the repo and
   save it in a folder by itself, for example `C:\poseguard\`.
2. Open PowerShell in that folder and paste the block printed at
   the top of the page. It serves the folder on
   `http://localhost:8000/` with the built-in `HttpListener`.
   Leave the PowerShell window open while recording.
3. Open `http://localhost:8000/rerun-capture.html` in Chrome or
   Edge. Do not open the file directly with `file://`; the
   camera is blocked outside a secure context, and localhost
   counts as one.
4. Sit where the camera sees your head, shoulders, **and hips**
   (step back if the tool says hips left the frame; a frame
   without hips cannot test the shoulder-vs-hip signal). Good
   light on your face and shoulders, plain background if you can.

## During the session (guided, about 8.6 minutes)

1. Press **Load pose engine**, then **Enable camera** and allow
   access, then **Start session** and follow the prompts.
2. Calibration (5 s): sit normally, upright, looking at the
   screen.
3. Five head trials, then five shoulder trials. Each hold is
   10 s with a live depth target: **head 15 deg, shoulder 8
   deg**. The target bar turns green past the target; hold deep
   enough to keep it green for the full 10 s.
4. Work sample (5 min): just work normally at the desk. Do not
   hold still on purpose; ordinary drift is the point.
5. Landmark-loss test (10 s): follow the prompt (cover or leave
   the camera as instructed).

## After

1. Press **Download CSV** and **Download manifest**. Send both
   files back.
2. If the tool reported tracking problems (hips out of frame,
   holds below target), note it when you send the files; the
   pre-flight will flag the same things as warnings.

## What happens next

One command runs the whole re-benchmark on the capture:

```
python -m experiments.model.day24 --csv <capture.csv> --manifest <manifest.json>
```

It prints a pre-flight (frames, tracked %, holds, warnings),
then the verdict per cue: the Day 16 re-benchmark (head grid and
models, shoulder signal and models) and the Day 23 combined-cue
gate, each against the same gate (4/5 trials, 0 false
positives). The full numbers land in
`experiments/model/runs/<date-time>/results.json` with a
plain-English `summary.md` next to it. If the pre-flight
reports errors, the session needs a re-run; warnings are noted
in the summary and the benchmark still runs.

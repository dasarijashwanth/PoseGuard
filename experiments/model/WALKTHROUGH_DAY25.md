# Day 25 walkthrough: capture-session launcher

Day 24 made the re-benchmark one command, but getting a capture
into it still took three manual steps: serve the Day 14 tool on
localhost (the checklist's PowerShell block), follow the
checklist at the desk, then find the two downloaded files and
point Day 24 at them. Day 25 removes the seams: one command
serves the capture page, walks the checklist in the terminal,
watches for the downloads, and hands them to the Day 24
pipeline the moment both land. The user's own capture has still
not arrived, so Day 25, like Days 16 and 24, is machinery,
proven by tests that need no camera.

## What was built

**The launcher: `experiments/model/day25.py`.** One command:

```
python -m experiments.model.day25
```

The flow:

1. **Serve.** The Day 14 capture page
   (`experiments/rerun-capture.html`) is served on
   `http://localhost:8000/rerun-capture.html` from a background
   thread. Localhost is a secure context, so the camera works;
   a hosted page never gets camera access, which is why the
   launcher serves the file itself instead of linking anywhere.
   The handler answers only `/` and `/rerun-capture.html` and
   reads the file per request. `--port` changes the port; a
   taken port exits 2 with a clear message and the fallback
   command.
2. **Walk.** `experiments/CAPTURE_CHECKLIST.md` is parsed and
   printed in order (before, during, after), followed by the
   session timeline built from
   `experiments/eval/rerun_protocol.py`: calibration 5 s, five
   head holds and five shoulder holds at 5 s ready + 10 s hold
   (targets 15 deg head, 8 deg shoulder) + 5 s rest each, the
   300 s work sample, and the 10 s landmark-loss test, 515 s
   total, the checklist's "about 8.6 minutes". Because the
   numbers come from the protocol module, the walk cannot
   drift from what the benchmark scores against. The launcher
   replaces the checklist's serve-it-yourself step and says so.
3. **Watch.** The drop directory (the browser's `~/Downloads`
   by default, `--drop-dir` to change) is polled every 2 s.
   State changes print once: nothing yet, CSV found but the
   manifest still missing (press Download manifest too), and
   so on. Pairing reuses Day 24's `discover_inputs`: newest
   `.csv`, manifest by same stem else newest `.json`. The
   capture tool names its downloads
   `poseguard-rerun-<time>.csv` and
   `poseguard-rerun-manifest-<time>.json`, different stems, so
   in practice the newest-`.json` fallback does the pairing.
4. **Hand off.** When both files are present the server stops
   and `day24.run_day24` is called as a function, the same
   pattern Day 24 uses for `run_day16`, so Day 25 benchmarks
   and archives nothing itself. Day 24 prints its pre-flight
   and full summary; the launcher adds the closing lines:
   archive location, the Day 16 verdict, and the best combined
   variant per cue.
5. **No-files state.** If the pair never lands, the launcher
   times out (30 minutes by default, `--timeout` to change, 0
   waits until Ctrl+C) and prints exactly what is missing plus
   the manual fallback, `python -m experiments.model.day24
   --drop-dir <folder>`, so the session can always be finished
   by hand. `--no-wait` skips the server entirely: one check,
   pipeline if both files are there, instructions and exit
   code 2 if not. If the files are already in the folder when
   the launcher starts, it goes straight to the pipeline.

Exit codes follow the Day 16/24 convention: 0 when the pipeline
ran, 2 when it could not (files missing, port taken, or the
capture failed Day 24 validation, which still archives
nothing), 130 on Ctrl+C.

## How it connects to Day 24

Day 25 is a front door, not a second pipeline. Everything after
the handoff is Day 24 unchanged: pre-flight validation with
exit code 2 on a bad file, the Day 16 re-benchmark called as a
function (`results_day16.json` and `BENCHMARK.md` untouched),
the Day 23 combined gate, and the archive under
`experiments/model/runs/<date-time>/`. `--source` and
`--out-dir` pass straight through. The manual Day 24 command
stays valid at every point; the launcher only removes the
serving, checklist-reading, and file-finding around it.

## Tests

11 new Day 25 tests in `experiments/model/tests/test_day25.py`,
all fast, no camera, no network beyond loopback:

- The HTTP handler serves the real capture page at `/` and
  `/rerun-capture.html` (200, `text/html`, byte-identical to
  the file) and 404s everything else.
- Checklist parsing against the real
  `CAPTURE_CHECKLIST.md`: section order (before, during,
  after), step counts (4, 5, 2), and the load-bearing phrases
  (hips in frame, head 15 deg, shoulder 8 deg, Download CSV,
  Download manifest), plus desk order inside the session
  (calibration, holds, work sample).
- The session timeline: calibration first, head holds before
  shoulder holds, work sample, landmark-loss test last, 515 s
  total.
- Drop-dir states: empty (including a missing folder),
  CSV only, manifest only, and ready with the capture tool's
  different-stem filenames paired by the newest-`.json`
  fallback.
- The wait loop with an injected clock: files appearing
  mid-wait are picked up; a timeout returns cleanly and says
  what is missing.
- The handoff with `run_day24` mocked: arguments pass through
  unchanged, and the closing lines name the archive, the
  Day 16 verdict, and the combined best per cue.
- `main --no-wait` both ways: ready folder runs the pipeline
  and exits 0; empty folder exits 2 and prints the exact
  fallback command.

## Files changed

- `experiments/model/day25.py` (new): the launcher, HTTP
  server, checklist parser, session timeline, drop-dir watch,
  Day 24 handoff.
- `experiments/model/tests/test_day25.py` (new): 11 tests.
- `experiments/model/WALKTHROUGH_DAY25.md` (new): this file.
- Root `README.md`: status line to Day 25, Day 25 Build
  Progress entry, "What comes next" item 1 names the Day 25
  launcher command.

## What is still open

- The user's real capture run, now one command
  (`python -m experiments.model.day25`), then the real
  re-benchmark through the Day 24 pipeline it hands off to.
- Whether anything passes the gate on real data, per cue.
- The live deploy (Render or AWS clicks) is still his.

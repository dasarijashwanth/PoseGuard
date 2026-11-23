# WALKTHROUGH_DAY21.md

## Day 21 (Nov 23): harness stress test

Day 16 built the re-benchmark harness that runs the moment the
camera capture CSV lands, and it was only ever exercised on the
clean synthetic fixture. A real capture is not clean: the camera
drops frames, the file gets edited by hand, a column goes missing,
a value comes back infinite. Day 21 is the stress test: feed the
harness malformed CSVs, missing columns, NaN and inf landmark
values, very short sessions, empty files, out-of-order timestamps,
duplicate rows, and a 19 MB file, and make it fail gracefully with
a clear error in every case, never a raw traceback.

## Method

A scratch probe script (not shipped) built variants of the Day 16
synthetic fixture (`experiments/eval/tests/test_rerun.py`,
`build_rerun_csv`, 5,150 frames) and ran each through
`validate_rerun_csv`, `validate_capture`, and `run_day16`. The probe
ran four times: the first run had a bug of its own (row mutations
applied to a slice copy and never written back to the file, so three
cases silently tested the unmutated fixture), which direct
verification exposed and the later runs fixed. The probe is not in
the repo; its cases live on as the 22 tests in
`experiments/model/tests/test_day21.py`.

One environment note: the probe wrote about 500 MB of variant
files into /tmp across its runs and filled the 512 MB /tmp tmpfs,
which then failed unrelated tests with "No space left on device".
The probe dirs were deleted after the run. The lesson is recorded
below: keep probe output under the workspace, not /tmp, when the
files are large.

## What broke, and what was fixed

### 1. Unreadable files raised raw OSErrors (loader, validator)

A nonexistent path raised a raw FileNotFoundError from
`validate_rerun_csv`, and a directory path raised a raw
IsADirectoryError. Fix: `validate_rerun_csv` now catches OSError
around the load and returns ok=False with "cannot read capture
file {path}: {error}". `day16.main` catches OSError as well as
ValueError and prints "Day 16 failed:" instead of "Day 16
validation failed:", exit code 2 either way.

### 2. hold_id "abc" crashed int(), hold_id 1.5 silently truncated

The label check did `int(h)` on every hold_id, so "abc" raised a
raw ValueError, and 1.5 truncated to 1 and was then reported
misleadingly as "missing hold: head 1..5" (pandas parses hold_id
as float, so 1.5 matched nothing). Fix: a `_safe_hold_id` helper
that returns None for non-integers, and the error now names the
bad values: "hold_id values must be integers 1-5, got: ['abc']"
or "['1.5']". The `sorted()` calls for bad phase, cue, and hold
values also take `key=str`, since a numeric cue or phase (123, 7)
would otherwise crash the sort with a TypeError on mixed types.
That path is unreachable through `read_csv` (mixed columns stay
strings), but the guard costs one argument.

### 3. Ragged rows silently shifted every column

A frame row with more fields than the header did not raise:
pandas treated the first column as an index and shifted every
value, so the second row's elapsed_s came back 0.0 instead of
0.1. Fix: `load_session` now checks every frame row's field
count against the header before calling `read_csv`, and raises
"frame row N has M fields, expected K". `low_memory=False` was
added to the `read_csv` call to stop the mixed-type DtypeWarning
the huge-file probe showed (the loader is called three times per
run, so the warning printed three times).

### 4. Out-of-order timestamps were silently sorted (loader)

The loader sorted frames by elapsed_s and then checked
monotonicity, so the check was dead code: any out-of-order file
passed. Fix: the monotonic check now runs before the sort, with
the message "elapsed_s is not monotonic increasing in file
order". Negative elapsed_s still passes, deliberately: all
analysis is relative to step starts, so a shifted clock is
harmless. A garbage timestamp with valid elapsed_s also still
passes: the timestamp becomes NaT and nothing uses it.

### 5. Infinite elapsed_s and tilt values passed silently

An elapsed_s of inf in one row was accepted (sorted last, and it
breaks gradient math downstream). An inf head_tilt on tracked
frames passed validation, and one inf poisons the EMA in
`ema_smooth` for the rest of the session: `circ_dev(inf)` is NaN
and the smooth value stays NaN. Fix: the loader rejects any
non-finite elapsed_s ("elapsed_s has missing or non-finite
values", which replaces the old "elapsed_s has missing values"
message; no test asserted the old wording). The validator errors
on inf tilt or deviation values on tracked frames, and
`ema_smooth` now treats any non-finite angle as missing, so the
smoothing run resumes from the last good value instead of
staying NaN.

### 6. Missing values on tracked frames were silent

A blank head_tilt on tracked frames passed with only warnings
from elsewhere, and blank landmark values on tracked frames
passed entirely (the benchmark does not consume landmarks yet).
Fix: the validator now warns with counts for missing
tilt/deviation values and for missing or non-finite landmark
values on tracked frames, and it warns for duplicate elapsed_s
values (adjacent duplicates; a duplicated block pasted out of
place is caught by the monotonic check instead). Blank landmarks
on untracked frames stay silent, as they should: the capture
tool writes blanks there by design.

### 7. A cue with no usable tracked frames crashed run_day16

A file whose head step was entirely untracked passed validation
and then crashed in `_rule_baseline_head` with IndexError: the
alignment score went NaN (an empty-window mean is NaN, and `nan
or 0` evaluates to nan), so no offset ever won the grid search,
`trials` stayed empty, and `analyze_cue` indexed
`alignment["trials"][k]`. Two fixes. First, the validator now
rejects a file with no usable tracked frames for a cue
("no usable tracked frames for cue head: the head step has no
tracked data to benchmark"); a per-hold tracked_fraction of 0
stays a warning, because one empty hold out of five still leaves
a benchmarkable cue and rejecting the whole file would be too
strict. Second, the alignment itself is fixed so the crash
cannot recur: `align_trials.score` sanitizes NaN means and
stds, adds a +100 penalty when a hold window has 0 samples, and
seeds the best from `score(0.0)` so `trials` is never empty.

### 8. A cue with too little tracked data crashed the model stage

A file can pass validation with a cue's holds mostly tracked but
too thin to form even one tilted window (five tracked frames per
hold, where a window needs eight). The model stage then failed
inside sklearn with a message about classes that names nothing
about the actual problem. Fix: `run_day16` now checks after each
`build_windows` that both label classes exist, and raises
"no tilted windows for cue 'head': the head holds have too
little tracked data to benchmark (the file passed validation,
but the model stage needs tilted and upright windows)".

### 9. Empty hold windows leaked NaN into results JSON

`_window_stats` returned NaN mean/std/max_abs for a window with
no data, and `hold_dwells` did the same, so `results_day16.json`
could contain NaN tokens, which are not valid strict JSON. Fix:
both return None instead of NaN when a window has no data. The
markdown writers use plain `{}` formatting, so None renders
safely, and the existing fixture tests assert finite values and
are unaffected. A `run_day16` on a file with one fully untracked
hold now produces a result that `json.dumps(..., allow_nan=False)`
accepts.

### 10. validate_capture hid the real error behind the loader's

When the CSV was bad and a manifest was supplied,
`validate_capture` loaded the session to check the manifest
first, so the loader's raw error replaced the validator's
aggregated "capture file cannot be used" message. Fix: the
manifest is only checked when the CSV report is ok. The same
function now also tolerates a manifest `frames` given as a
string ("5150" and 5150 agree), and `check_manifest` still
requires the 10 holds entries and matching baselines.

### 11. An unclosed file handle (loader)

The probe output showed a ResourceWarning for an unclosed file
in the loader. Fix: `open()` replaced with `with open(...)`.

## What was already graceful (no fix needed)

Empty file ("empty file"), binary garbage (UnicodeDecodeError,
caught since it subclasses ValueError), plain text and missing
blank line ("no blank line after metadata block"), no frame
header, missing tracking column, elapsed_s "abc", truncated
session (clear "missing hold" errors), and every manifest error
(missing file, invalid JSON, JSON array, frames mismatch,
directory) already raised the aggregated ValueError. A 3000 s
work session (32,150 frames, 19.3 MB) validates in 1.0 s and
runs the benchmark in 113.8 s with no crash and no performance
fix needed. The `experiments/eval/cli.py` Run 1 CLI was
deliberately left out: its missing-file traceback is out of the
Day 16 harness scope.

## Verification

- 324/324 tests passing in one combined run (80 eval + 244
  model, including 22 new Day 21 tests in
  `experiments/model/tests/test_day21.py`).
- The probe re-run after the fixes: every case above now fails
  with a named error or passes with a warning, and no case
  raises a raw traceback.

## Numbers (unchanged)

Best shoulder rule 4/5 trials with 8 false positives (F1 0.471),
best head rule 2/5 with 3 false positives (F1 0.400), best models
F1 0.172 shoulder and 0.105 head. Nothing passes the gate, and Day
21 changes no model, no threshold, and no gate logic: it only
makes the harness fail gracefully on the messy inputs a real
capture will eventually bring.

## What still cannot be verified without a real capture

- That a real camera CSV (not the synthetic fixture) exercises
  these paths the same way. The real capture CSV has still not
  arrived; the Day 14 capture tool is the path to it.
- That the huge-file timing (113.8 s for 32,150 frames) holds on
  the user's machine. No performance fix was needed at this size.
- That the live video stream runs in a real browser: unchanged
  from Day 20, it needs his camera.

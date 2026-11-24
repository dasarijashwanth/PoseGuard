# Day 23 walkthrough: the combined-cue joint score

Day 22 put the whole project on the live page. Day 23 goes back to
the signal, and asks the question every earlier day left alone:
the gate scores each cue separately, but the two cues are measured
on the same body at the same time. Does scoring them together do
better than either alone?

## The design, and why it is a joint score and not an AND

The gate scores detection per cue: 4/5 head holds and 4/5 shoulder
holds, each performed alone, and zero false positives during
ordinary work. A naive AND rule, fire only when both cues deviate
at once, would detect 0 of the single-cue deliberate holds and
fail the gate by construction. It is evaluated below as a control,
and it does exactly that: 0/5 on both cues, on Run 1 and on the
clean synthetic fixture alike.

So the combination is a joint score, computed per frame from the
two deviation series, and run through the same Day 3-4 protocol
(threshold sweep at the 5 s persistence, persistence curve at the
best threshold) on each cue's own step frames:

- **Normalized sum** (`j_dev_sum`): |head deviation| / 5 +
  |gated shoulder deviation| / 1. Each cue is scaled by its own
  operating threshold (head 5 deg, shoulder 1 deg), so 1.0 means
  "one cue at its operating level". A hold moves both angles a
  little, and the two small deviations add up to cross a low
  threshold neither cue sustains alone. This is the real
  candidate.
- **Isolation** (`j_dev_s`, `j_dev_h`): |shoulder| minus |head|
  (and the mirror image). The idea is that whole-body lean moves
  both cues together, so subtracting the other cue should cancel
  it. It does not work on Run 1; see the numbers.
- **AND control**: the base cue's own deviation with a validity
  column requiring the other cue above its operating threshold
  (head above 5 deg, shoulder above 1 deg). One configuration
  per cue.

One implementation detail matters. The isolation scores are
floored at 0: a negative difference (shoulder dominating on the
head score) is not a tilt in the other direction, it is the other
cue firing, and `sustained_events` takes the absolute value, so
an unclamped negative score would fire the wrong cue. Clamping
changed the head isolation result: 4 false positives at 2 deg
became 0. The shoulder numbers are unchanged by it.

## The numbers (Run 1, 5 s gate persistence)

| Cue | Variant | Best threshold | Trials | False positives | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Shoulder | gated level (Day 11/17) | 1 deg | 4/5 | 8 | 0.471 |
| Shoulder | isolation | 0.5 deg | 3/5 | 9 | 0.353 |
| Shoulder | **normalized sum** | **1.0** | **5/5** | **8** | **0.556** |
| Shoulder | AND control | 1 deg | 0/5 | 0 | 0.000 |
| Head | level, 5 s (Day 15 grid) | 4-8 deg | 0/5 | 0 | 0.000 |
| Head | level, 2 s (Day 15 best) | 4 deg | 2/5 | 3 | 0.400 |
| Head | isolation | 2.0 deg | 1/5 | 0 | 0.333 |
| Head | **normalized sum** | **1.0** | **4/5** | **8** | **0.471** |
| Head | AND control | 4 deg | 0/5 | 0 | 0.000 |

Full threshold curves and persistence curves are in
`experiments/model/results_day23.json`.

**The sum is the best rule on Run 1 by F1 on both cues.**
Shoulder 5/5 trials at F1 0.556 beats the gated 4/5 at 0.471;
head 4/5 at F1 0.471 beats the Day 15 best 2/5 at 0.400, and it
does it at the gate's own 5 s persistence, where the head level
alone detects 0/5 at every threshold 4-8 deg. The mechanism is
pooling: a hold moves both angles a little, and the sum of two
small deviations crosses 1.0 where neither sustains it alone.
The threshold is stable: 1.0 is best on both cues, 0.5 and 0.75
cost one more false positive each, 1.25 already loses a trial.

**It still fails the gate, on the same 8 false positives.**
The sum's 8 work false positives are the same shoulder-drift
episodes the gated rule fires on (7 start at the exact same
second, the 8th within one frame), and the head cue inherits
all 8 of them: the head level at 4 deg / 5 s has 0 false
positives on its own, so every one of the sum's head false
positives is imported shoulder drift. The sum buys detection,
not clean work. The threshold that catches the holds (1.0) sits
below the work drift (4-7 deg on the shoulder alone), so no
threshold on this score separates them either.

**The isolation scores are worse than the single-cue rules.**
Shoulder isolation best is 3/5 with 9 false positives (F1
0.353), head isolation 1/5 with 0 false positives (F1 0.333).
The lean hypothesis is wrong for this data: during a shoulder
hold the head moves a little too, so subtracting it eats the
2.5 deg hold, while during ordinary work the head stays
relatively still (mean |head deviation| 0.9-3.3 deg inside the
gated rule's 8 false-positive episodes), so the subtraction
barely touches the false positives. The one thing isolation
does buy is a clean head cue: at 2 deg it is the only head
configuration with 0 false positives that detects anything at
all (1/5 at 5 s, 3/5 at 0.5 s), but 1/5 is nowhere near the
4/5 the gate needs.

**The AND control fails by construction, as predicted.** 0/5
trials on both cues, 0 false positives, on Run 1 and on the
clean fixture. A conjunction can never see a single-cue hold.

## The fixture check: the machinery can pass

On the Day 16 synthetic fixture (clean holds to target depth,
flat work, Run 2 timings 5/10/5), the isolation and sum scores
both pass the gate on both cues (5/5 trials, 0 false positives,
F1 1.0), and the AND control still detects 0/5. That proves the
evaluation can report a pass; the Run 1 failure is the data,
not the harness. The fixture stays labeled synthetic everywhere.

## Honest verdict

Nothing passes the gate. The joint score is the first real
improvement to the best rule since Day 17: the shoulder rule
goes from 4/5 to 5/5 trials and the head rule from 2/5 to 4/5
at the 5 s persistence, both at F1 above the old bests. But
the false-positive half is untouched at 8, and the gate needs
0. The isolation idea is dead for this data, and the AND idea
was dead by construction. The fix is still the camera re-run
with raw landmarks and longer, deeper holds, not a cleverer
combination of the same two angle columns: every combination
tried so far is bounded by the same fact, that a deliberate
shoulder hold moves 2.5 deg while ordinary work drifts 4-7.

## What changed, file by file

- `experiments/eval/combined.py` (new): `add_combined` adds
  the three joint columns (`j_dev_s`, `j_dev_h`, `j_dev_sum`,
  all floored / normalized as described) and the two AND
  boolean columns; `evaluate_combined` runs the Day 3-4
  protocol per cue (threshold sweeps at 5 s, persistence
  curves, AND control), with the head alignment verified and
  the shoulder alignment inferred from it, exactly as Days
  3-4 and 11 did. Timing args are the protocol timings, Run 1
  by default, Run 2 (5 / 10 / 5) for a re-run capture.
- `experiments/model/results_day23.json` (new): the full Run 1
  evaluation, the fixture result, and the verdict, generated
  from the same evaluation code.
- `experiments/eval/tests/test_combined.py` (new, 11 tests):
  unit tests for the three scores (normalized sum, isolation
  on single-cue holds, lean cancellation, the zero clamp,
  NaN propagation, AND thresholds) and the evaluation on the
  eval fixture (AND 0/5 both cues, sum detects the head trial
  and the work excursion, isolation keeps the head cue clean).
- `experiments/model/tests/test_day23.py` (new, 7 tests): the
  Run 1 bests pinned to `results_day23.json` and recomputed
  from the module, nothing passes the gate, the sum beats the
  Day 22 bests in the demo's `BEST_RULES` on both cues, the
  sum passes on the fixture at threshold 1.0 while AND still
  detects nothing, the demo verdict names the joint best, and
  the sum's false positives are the same shoulder-drift
  episodes as the gated rule's.
- `demo/demo_data.py`: `VERDICT` now names the Day 23 joint
  rule as the best rule (5/5 shoulder, 4/5 head, 8 false
  positives). `BEST_RULES` is deliberately unchanged: it is
  the best per-cue rule-grid configuration in degrees, and the
  sum's threshold is in normalized units, not degrees, so it
  does not fit that display; the per-cue grid bests are still
  the right numbers there. The journey table is unchanged too:
  Day 22's section covers Days 1-22 and its test pins exactly
  22 rows.
- Root README: status line and test-count line to Day 23, and
  a Day 23 entry leading Build Progress.

## Verification

360/360 tests passing in one combined run (80 + 11 = 91
eval + 262 + 7 = 269 model). The 18 new Day 23 tests pin the
joint-score columns, the fixture behavior, every Run 1 best
number to `results_day23.json` and to a recomputation from the
module, the gate verdict, and the demo verdict string.
`combined.py` compiles cleanly and the demo still imports.

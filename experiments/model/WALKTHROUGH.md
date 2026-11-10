# Day 8 walkthrough: PyTorch MLP benchmark

Day 8 asks one question: does a neural net beat the scikit-learn
models from Day 5 on the same data? The answer is no, and the
numbers below say exactly how close it came.

## What was built

- `experiments/model/torch_mlp.py`: a small multi-layer perceptron
  (MLP) per cue, CPU-only, trained in PyTorch.
- `experiments/model/benchmark.py`: runs all three models (logistic
  regression, random forest, MLP) on the same Run 1 data with the
  same grouped 5-fold cross-validation, and writes
  `benchmark.json` plus `BENCHMARK.md`.
- `experiments/model/tests/test_torch_mlp.py`: 9 tests for the MLP.
- `experiments/model/BENCHMARK.md`: the comparison table and the
  honest verdict per cue.

## How the MLP works (one paragraph)

Each 1-second window of angle data is already a list of numbers
(the Day 5 features: how far the head tilted, how steady it was,
how jumpy the shoulder line was, and so on). The MLP is a small
chain of layers: the features go in, each layer mixes them into
new combinations and keeps only the positive part (ReLU), some
values are randomly dropped during training (dropout, so the net
cannot lean on one feature), and the last layer outputs two scores,
one for "upright" and one for "tilted". Training nudges the layer
weights so the right score wins, using Adam (a standard optimizer)
and a loss that counts a missed tilt about 8 times more than a
missed upright window, because tilted windows are that rare. A
separate validation split watches the loss and stops training when
it stops improving (early stopping). At test time dropout is off
and the higher score wins.

## How to run it

From the repo root, with the venv active (torch is CPU-only):

```bash
python -m experiments.model.benchmark --csv <path/to/validation.csv>
python -m pytest experiments/model/tests/test_torch_mlp.py -q
```

The benchmark prints the table and writes `benchmark.json` and
`BENCHMARK.md` next to it. The full suite is
`python -m pytest experiments/ -q` (79 tests: 33 eval, 37 model,
9 MLP).

## What the numbers mean

The honest score is F1 on the tilted class plus the gate: 4 of 5
trials detected with 0 false-positive events. Accuracy lies on
this data because most windows are upright (a model that always
says "upright" scores about 93% accuracy and detects nothing).

Head (383 windows, 23 tilted):

| Model | F1 | Trials | False positives | Work flagged |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.078 | 1/5 | 6 | 172/318 |
| Random forest | 0.000 | 0/5 | 0 | 1/318 |
| MLP | 0.060 | 1/5 | 4 | 100/318 |

Shoulder (385 windows, 29 tilted):

| Model | F1 | Trials | False positives | Work flagged |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.179 | 5/5 | 5 | 134/318 |
| Random forest | 0.000 | 0/5 | 0 | 0/318 |
| MLP | 0.133 | 2/5 | 1 | 54/318 |

The rule baseline is 0/5 trials with 0 false positives on head
and 7 on shoulder. No model passes the gate on either cue.

Two real differences are still worth keeping. The MLP is far less
trigger-happy than logistic regression: it flags 100 of 318 work
windows on head (vs 172) and 54 on shoulder (vs 134). And its
signal survives the calm-only check: on windows where the shoulder
line was steady, MLP accuracy is 0.685 on head and 0.663 on
shoulder, against 0.427 and 0.447 for logistic regression, whose
accuracy collapsed once tracker noise was removed. The MLP learned
something less noise-driven. At threshold 0.3 the head MLP reaches
4/5 trials with 2 false positives, the closest any model has come
to the gate. Close is still not a pass.

One warning about the plain 70/30 split numbers: on that split the
head MLP scores F1 0.5, which looks good. It flatters every model,
because a trial's windows land on both sides of the split. The
grouped cross-validation numbers above are the honest ones.

## What to try next

- Re-run data (longer, deeper holds, more trials). That is still
  the fix, not a fancier model. With more tilted windows the MLP
  has something real to learn from, and the comparison should be
  re-run on the new session.
- If the re-run still favors logistic regression, the MLP was a
  useful negative result: it closes the PyTorch gap on the resume
  and proves the team checked the obvious next model.
- The live deploy (Render) and the AWS deploy are still open as
  later days; they do not depend on the classifier beating the
  baseline, they serve whatever the current best model is.

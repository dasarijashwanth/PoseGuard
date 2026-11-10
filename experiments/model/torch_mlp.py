"""PyTorch MLP classifier for the Day 8 benchmark.

One small multi-layer perceptron per cue (head, shoulder), trained on
the same 1-second window features and the same grouped 5-fold CV by
trial as the Day 5 scikit-learn models, so the comparison is
apples-to-apples. The question is simple: does a neural net beat
logistic regression and the random forest on this data, and does
anything beat the rule baseline?

Architecture, kept small on purpose: the whole dataset is one session
(about 383 windows, 23 tilted), so a big net would just memorize
it. Two hidden layers (32, 16) with ReLU and dropout, Adam with
weight decay, early stopping on a validation split. Class weights
(inverse frequency, computed from the training portion only) handle
the imbalance honestly; nothing is oversampled, so no window can leak
across a fold boundary.

Leakage rules, same as Day 5:
- The feature scaler is fit on the training portion of each fold
  only, never on the held-out fold.
- Early stopping watches a validation split taken from the training
  groups only (GroupShuffleSplit on the outer-train groups), never
  the held-out fold.
- The shoulder model trains on FEATURES["shoulder"] only, which
  contains no shoulder angle level (see features.py).

Determinism: fixed seeds, one CPU thread, no shuffling beyond a
seeded generator. Training twice on the same windows gives the same
out-of-fold predictions (covered by a test).

Usage (from the repo root):
    python -m experiments.model.benchmark --csv <path/to/validation.csv>
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.model.features import FEATURES
from experiments.model.train import (
    CALM_STD_S,
    RANDOM_STATE,
    RULE_BASELINE,
    _trial_verdicts,
    _window_metrics,
)

HIDDEN_SIZES = (32, 16)
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20


class MLP(nn.Module):
    """Two hidden layers, ReLU, dropout. Small by design."""

    def __init__(self, n_features: int,
                 hidden: tuple[int, ...] = HIDDEN_SIZES,
                 dropout: float = DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights, from the training portion only.

    Weight for class c is n / (2 * n_c), so the rare tilted class
    counts more per window. Computed from the training labels alone,
    never from a held-out fold.
    """
    y = np.asarray(y_train, dtype=int)
    n = len(y)
    weights = []
    for c in (0, 1):
        n_c = int((y == c).sum())
        if n_c == 0:
            raise ValueError("training portion has no windows of class "
                             f"{c}; cannot weight classes")
        weights.append(n / (2.0 * n_c))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_model(X_tr: np.ndarray, y_tr: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    seed: int = RANDOM_STATE,
                    hidden: tuple[int, ...] = HIDDEN_SIZES,
                    dropout: float = DROPOUT,
                    max_epochs: int = MAX_EPOCHS,
                    patience: int = PATIENCE) -> tuple[MLP, dict]:
    """Train one MLP with early stopping on the validation split.

    Inputs must already be scaled (scaler fit on the training
    portion only). Returns the model at its best validation loss,
    plus a small history dict (epochs run, best val loss).
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = MLP(X_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr))
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    x_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val_t))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history = {"epochs_run": int(best_epoch + 1),
               "best_val_loss": round(float(best_loss), 4)}
    return model, history


def _predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.argmax(logits, dim=1).numpy().astype(int)


def _predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()[:, 1]


def _scale_fit(X_tr: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_tr)
    return scaler


def evaluate_mlp_cue(windows: pd.DataFrame, cue: str,
                     seed: int = RANDOM_STATE) -> dict:
    """Grouped 5-fold out-of-fold evaluation of the MLP for one cue.

    Same protocol as train.evaluate_cue: a plain 70/30 split for a
    quick view, then GroupKFold by trial group for the honest
    out-of-fold numbers, a decision-threshold sweep on the OOF
    probabilities, and the calm-only check. Returns the same shape
    as evaluate_cue (minus the sklearn model list), with "mlp" as
    the single model entry.
    """
    feats = FEATURES[cue]
    X = windows[feats].to_numpy(dtype=float)
    y = windows["label"].to_numpy(dtype=int)
    groups = windows["group"].to_numpy(dtype=int)

    # --- plain 70/30 split, fixed seed (same split as Day 5) ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    scaler = _scale_fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
    # early-stopping split taken from the train portion only
    idx_tr, idx_val = train_test_split(
        np.arange(len(X_tr_s)), test_size=0.20, stratify=y_tr,
        random_state=seed)
    split_model, split_hist = train_one_model(
        X_tr_s[idx_tr], y_tr[idx_tr], X_tr_s[idx_val], y_tr[idx_val],
        seed=seed)
    split_pred = _predict(split_model, X_te_s)

    # --- grouped 5-fold out-of-fold predictions for every window ---
    cv = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_models = []
    fold_histories = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        torch.manual_seed(seed + fold)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20,
                                random_state=seed + fold)
        inner_tr, inner_val = next(gss.split(X[tr_idx], y[tr_idx],
                                             groups[tr_idx]))
        fold_scaler = _scale_fit(X[tr_idx][inner_tr])
        X_fold_tr = fold_scaler.transform(X[tr_idx][inner_tr])
        X_fold_val = fold_scaler.transform(X[tr_idx][inner_val])
        X_fold_te = fold_scaler.transform(X[te_idx])
        model, hist = train_one_model(
            X_fold_tr, y[tr_idx][inner_tr],
            X_fold_val, y[tr_idx][inner_val],
            seed=seed + fold)
        oof_pred[te_idx] = _predict(model, X_fold_te)
        oof_proba[te_idx] = _predict_proba(model, X_fold_te)
        fold_models.append(model)
        fold_histories.append(hist)

    oof_metrics = _window_metrics(y, oof_pred)
    oof_trials = _trial_verdicts(windows, oof_pred)
    sweep = []
    for thr in (0.3, 0.5, 0.7, 0.9):
        t = _trial_verdicts(windows, (oof_proba >= thr).astype(int))
        sweep.append({"threshold": thr,
                      "detected_trials": t["detected_trials"],
                      "false_positive_events": t["false_positive_events"]})
    calm = windows["std_s"] < CALM_STD_S
    calm_metrics = _window_metrics(y[calm.to_numpy()],
                                   oof_pred[calm.to_numpy()])
    return {
        "cue": cue, "features": feats,
        "n_windows": int(len(windows)),
        "n_tilted_windows": int((y == 1).sum()),
        "n_upright_windows": int((y == 0).sum()),
        "models": {"mlp": {
            "split_metrics": _window_metrics(y_te, split_pred),
            "oof_metrics": oof_metrics,
            "oof_trials": oof_trials,
            "oof_pred": [int(p) for p in oof_pred],
            "oof_proba": [round(float(p), 3) for p in oof_proba],
            "split_history": split_hist,
            "fold_histories": fold_histories,
        }},
        "best_model": "mlp",
        "best_oof_metrics": oof_metrics,
        "best_oof_trials": oof_trials,
        "threshold_sweep": sweep,
        "calm_only_metrics": calm_metrics,
        "calm_windows": int(calm.sum()),
        "rule_baseline": RULE_BASELINE[cue],
        "trained": {"mlp_split": split_model, "mlp_folds": fold_models},
    }

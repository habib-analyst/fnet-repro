"""Evaluation metrics for binary real/fake classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from .data import FAKE_SUBTYPES, SUBTYPES


def evaluate(logits: np.ndarray, y: np.ndarray, subtype: np.ndarray) -> dict:
    """Accuracy/AUC at threshold 0, plus a per-subtype breakdown.

    Per-subtype numbers are the fraction of clips of that subtype that
    are classified correctly (for fake subtypes: flagged as fake).
    """
    logits = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (logits >= 0).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    per_subtype = {}
    for name in SUBTYPES:
        mask = subtype == SUBTYPES.index(name)
        per_subtype[name] = {
            "n": int(mask.sum()),
            "correct": int((pred[mask] == y[mask]).sum()),
            "accuracy": float((pred[mask] == y[mask]).mean()) if mask.any() else None,
        }
    return {
        "n": int(len(y)),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "accuracy": float((pred == y).mean()),
        "n_correct": int((pred == y).sum()),
        "auc": float(roc_auc_score(y, probs)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_subtype": per_subtype,
        "fake_subtypes": list(FAKE_SUBTYPES),
    }

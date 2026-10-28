"""Small deterministic training loop used by the experiment scripts."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .metrics import evaluate


def _to_tensor(split: dict) -> dict:
    return {
        "audio": torch.from_numpy(split["audio"]),
        "visual": torch.from_numpy(split["visual"]),
        "y": torch.from_numpy(split["y"]).float(),
        "subtype": split["subtype"],
        "y_np": split["y"],
    }


@torch.no_grad()
def _logits(model: nn.Module, tensors: dict, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(tensors["y"]), batch):
        out.append(
            model(tensors["audio"][i : i + batch], tensors["visual"][i : i + batch])
            .cpu()
            .numpy()
        )
    return np.concatenate(out)


def train_model(
    model: nn.Module,
    dataset: dict,
    epochs: int = 14,
    lr: float = 1e-3,
    batch: int = 32,
    seed: int = 7,
) -> tuple[nn.Module, dict]:
    """Train with BCE, keep the checkpoint with the best validation AUC."""
    torch.manual_seed(seed)
    train = _to_tensor(dataset["train"])
    val = _to_tensor(dataset["val"])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train["y"])
    best_auc, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            logits = model(train["audio"][idx], train["visual"][idx])
            loss = loss_fn(logits, train["y"][idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        val_metrics = evaluate(_logits(model, val), val["y_np"], val["subtype"])
        history.append(
            {"epoch": epoch + 1, "train_loss": total / n, "val_auc": val_metrics["auc"]}
        )
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_auc": best_auc, "history": history}


def test_logits(model: nn.Module, dataset: dict) -> np.ndarray:
    return _logits(model, _to_tensor(dataset["test"]))

"""Train Multimodal-FNet (full) and the no-inter-mixer ablation on
SynthAV, evaluate both on the identity-disjoint test split, and write
``results/results.json`` plus the full model's checkpoint.

Run from the repo root:

    python experiments/run_experiment.py

Data note: SynthAV is a *synthetic* dataset generated in code (see
``src/fnetrepro/data.py``). These numbers measure whether the
architecture trains and separates the planted real/fake failure modes
at small scale -- they are not FakeAVCeleb numbers and are not
comparable to the paper's reported results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnetrepro.data import build_dataset  # noqa: E402
from fnetrepro.metrics import evaluate  # noqa: E402
from fnetrepro.model import MultimodalFNet  # noqa: E402
from fnetrepro.train import test_logits, train_model  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
EPOCHS = 30


def main() -> None:
    torch.manual_seed(7)
    t0 = time.time()
    dataset = build_dataset()
    print(f"dataset built in {time.time() - t0:.1f}s")
    for name, split in dataset.items():
        print(f"  {name}: n={len(split['y'])} real={int((split['y'] == 0).sum())} "
              f"fake={int((split['y'] == 1).sum())}")

    results = {
        "dataset": "SynthAV (synthetic, generated in src/fnetrepro/data.py)",
        "config": {"epochs": EPOCHS, "d_model": 64, "n_intra": 2, "n_inter": 2,
                   "seed": 7, "device": "cpu"},
        "splits": {name: int(len(split["y"])) for name, split in dataset.items()},
        "models": {},
    }
    for label, use_inter in (("full", True), ("ablation_no_inter", False)):
        t1 = time.time()
        torch.manual_seed(7)
        model = MultimodalFNet(use_inter=use_inter)
        n_params = sum(p.numel() for p in model.parameters())
        model, info = train_model(model, dataset, epochs=EPOCHS)
        metrics = evaluate(test_logits(model, dataset), dataset["test"]["y"],
                           dataset["test"]["subtype"])
        results["models"][label] = {
            "use_inter": use_inter,
            "n_params": int(n_params),
            "best_val_auc": info["best_val_auc"],
            "train_seconds": round(time.time() - t1, 1),
            "test": metrics,
        }
        print(f"{label}: params={n_params} test acc={metrics['accuracy']:.4f} "
              f"auc={metrics['auc']:.4f} ({metrics['n_correct']}/{metrics['n']})")
        if label == "full":
            torch.save(model.state_dict(), RESULTS / "model_full.pt")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULTS / 'results.json'}  (total {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

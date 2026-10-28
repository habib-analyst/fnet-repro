"""Integrated-gradients explainability demo.

Trains the full model (same seed/config as ``run_experiment.py``),
takes one ``fake_sync`` and one ``fake_audio`` test clip that the model
classifies correctly, and attributes the fake logit back to the two
input streams with integrated gradients. Writes
``results/ig_summary.json`` (modality attribution shares, most
attributed mel bins / frames, mouth-region share of the visual
attribution, and the IG completeness error) and ``results/ig_example.png``
(attribution maps for the first demo clip).

Run from the repo root:

    python experiments/run_explain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fnetrepro.data import MOUTH_BOX, SUBTYPES, build_dataset  # noqa: E402
from fnetrepro.ig import integrated_gradients  # noqa: E402
from fnetrepro.model import MultimodalFNet  # noqa: E402
from fnetrepro.train import train_model  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
IG_STEPS = 48


def summarise(model, split, idx: int) -> dict:
    audio = torch.from_numpy(split["audio"][idx : idx + 1])
    visual = torch.from_numpy(split["visual"][idx : idx + 1])
    with torch.no_grad():
        logit = float(model(audio, visual)[0])
    attr_a, attr_v = integrated_gradients(model, audio, visual, steps=IG_STEPS)
    attr_a = attr_a[0].numpy()  # (1, F, T)
    attr_v = attr_v[0].numpy()  # (T, 1, H, W)
    abs_a = np.abs(attr_a).sum(axis=(0, 1))  # per time step
    abs_v_frames = np.abs(attr_v).sum(axis=(1, 2, 3))  # per frame
    total = float(np.abs(attr_a).sum() + np.abs(attr_v).sum())
    r0, r1, c0, c1 = MOUTH_BOX
    mouth = float(np.abs(attr_v[:, 0, r0:r1, c0:c1]).sum())
    mel_attr = np.abs(attr_a[0]).sum(axis=1)  # per mel bin
    completeness = float(attr_a.sum() + attr_v.sum())  # baseline is zero
    return {
        "test_index": int(idx),
        "subtype": SUBTYPES[int(split["subtype"][idx])],
        "logit": logit,
        "pred": "fake" if logit >= 0 else "real",
        "ig_steps": IG_STEPS,
        "completeness": {
            # With a zero baseline, sum(attr) should approximate
            # logit(x) - logit(0); the exact comparison for the first
            # demo clip is added by main().
            "sum_attributions": completeness,
        },
        "modality_share_abs_attr": {
            "audio": float(np.abs(attr_a).sum() / total),
            "visual": float(np.abs(attr_v).sum() / total),
        },
        "top_mel_bins": [int(i) for i in np.argsort(mel_attr)[-5:][::-1]],
        "top_frames": [int(i) for i in np.argsort(abs_v_frames)[-5:][::-1]],
        "mouth_share_of_visual_abs_attr": float(mouth / np.abs(attr_v).sum()),
        "_attr_audio": attr_a,
        "_attr_visual": attr_v,
    }


def main() -> None:
    torch.manual_seed(7)
    dataset = build_dataset()
    model = MultimodalFNet(use_inter=True)
    ckpt = RESULTS / "model_full.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded checkpoint {ckpt}")
    else:
        model, _ = train_model(model, dataset, epochs=30)
        print("trained model (no checkpoint found)")
    model.eval()
    split = dataset["test"]
    with torch.no_grad():
        logits = model(
            torch.from_numpy(split["audio"]), torch.from_numpy(split["visual"])
        ).numpy()

    demos = []
    for wanted in ("fake_sync", "fake_audio"):
        subtype_idx = SUBTYPES.index(wanted)
        candidates = [
            i
            for i in range(len(split["y"]))
            if split["subtype"][i] == subtype_idx and logits[i] >= 0
        ]
        if not candidates:
            candidates = [
                i for i in range(len(split["y"])) if split["subtype"][i] == subtype_idx
            ]
        demos.append(summarise(model, split, candidates[0]))

    # Exact completeness check for the first demo: sum(attr) vs
    # logit(x) - logit(0 baseline).
    first = demos[0]
    idx = first["test_index"]
    audio = torch.from_numpy(split["audio"][idx : idx + 1])
    visual = torch.from_numpy(split["visual"][idx : idx + 1])
    with torch.no_grad():
        base_logit = float(model(torch.zeros_like(audio), torch.zeros_like(visual))[0])
    first["completeness"] = {
        "sum_attributions": first["completeness"]["sum_attributions"],
        "logit_minus_baseline": first["logit"] - base_logit,
        "abs_error": abs(first["completeness"]["sum_attributions"] - (first["logit"] - base_logit)),
    }

    # Figure for the first demo clip.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    attr_a = first.pop("_attr_audio")[0]  # (F, T)
    attr_v = first.pop("_attr_visual")  # (T, 1, H, W)
    for d in demos[1:]:
        d.pop("_attr_audio", None)
        d.pop("_attr_visual", None)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].imshow(attr_a, aspect="auto", origin="lower", cmap="RdBu_r")
    axes[0].set_title("audio IG attribution (mel x time)")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("mel bin")
    vmap = np.abs(attr_v[:, 0]).mean(axis=0)
    axes[1].imshow(vmap, cmap="magma")
    r0, r1, c0, c1 = MOUTH_BOX
    axes[1].add_patch(
        Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, edgecolor="cyan")
    )
    axes[1].set_title("visual |IG| (mean over frames), mouth box in cyan")
    axes[2].bar(range(len(attr_v)), np.abs(attr_v).sum(axis=(1, 2, 3)))
    axes[2].set_title("visual |IG| per frame")
    axes[2].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(RESULTS / "ig_example.png", dpi=120)
    print(f"wrote {RESULTS / 'ig_example.png'}")

    with open(RESULTS / "ig_summary.json", "w") as f:
        json.dump({"demos": demos}, f, indent=2)
    for d in demos:
        print(
            f"{d['subtype']}: logit={d['logit']:.3f} "
            f"audio share={d['modality_share_abs_attr']['audio']:.3f} "
            f"visual share={d['modality_share_abs_attr']['visual']:.3f} "
            f"mouth share of visual={d['mouth_share_of_visual_abs_attr']:.3f}"
        )
    print(f"wrote {RESULTS / 'ig_summary.json'}")


if __name__ == "__main__":
    main()

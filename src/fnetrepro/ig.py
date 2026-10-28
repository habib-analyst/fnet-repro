"""Integrated gradients (IG) for MultimodalFNet, implemented directly
on autograd (no external attribution library needed).

Attributions are computed against an all-zeros baseline for both
modalities, with a straight-line path and a Riemann-sum approximation.
For a zero baseline, the attributions of the two inputs sum (up to the
Riemann error) to the model's logit, which the demo and the tests use
as a completeness check.
"""

from __future__ import annotations

import torch


def integrated_gradients(
    model: torch.nn.Module,
    audio: torch.Tensor,
    visual: torch.Tensor,
    steps: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (audio_attr, visual_attr) with the same shapes as the inputs.

    ``model`` is called as ``model(audio, visual) -> logits (B,)`` and is
    expected to be in eval mode. Inputs are single-batch tensors.
    """
    if audio.shape[0] != 1 or visual.shape[0] != 1:
        raise ValueError("integrated_gradients expects a batch of exactly 1")
    model.eval()
    total_a = torch.zeros_like(audio)
    total_v = torch.zeros_like(visual)
    for s in range(1, steps + 1):
        alpha = s / steps
        a_in = (audio * alpha).detach().requires_grad_(True)
        v_in = (visual * alpha).detach().requires_grad_(True)
        logit = model(a_in, v_in).sum()
        grad_a, grad_v = torch.autograd.grad(logit, [a_in, v_in])
        total_a += grad_a.detach()
        total_v += grad_v.detach()
    attr_a = audio * (total_a / steps)
    attr_v = visual * (total_v / steps)
    return attr_a, attr_v

import torch

from fnetrepro.ig import integrated_gradients
from fnetrepro.model import MultimodalFNet


def _model_and_inputs():
    torch.manual_seed(0)
    model = MultimodalFNet()
    model.eval()
    audio = torch.randn(1, 1, 40, 16)
    visual = torch.rand(1, 16, 1, 32, 32)
    return model, audio, visual


def test_ig_shapes_and_finiteness():
    model, audio, visual = _model_and_inputs()
    attr_a, attr_v = integrated_gradients(model, audio, visual, steps=8)
    assert attr_a.shape == audio.shape
    assert attr_v.shape == visual.shape
    assert torch.isfinite(attr_a).all()
    assert torch.isfinite(attr_v).all()


def test_ig_completeness():
    """sum(attributions) ~= logit(x) - logit(zero baseline)."""
    model, audio, visual = _model_and_inputs()
    attr_a, attr_v = integrated_gradients(model, audio, visual, steps=64)
    with torch.no_grad():
        logit_x = float(model(audio, visual)[0])
        logit_0 = float(model(torch.zeros_like(audio), torch.zeros_like(visual))[0])
    total = float(attr_a.sum() + attr_v.sum())
    target = logit_x - logit_0
    assert abs(total - target) <= 0.05 * max(1.0, abs(target))


def test_ig_rejects_batches():
    model, audio, visual = _model_and_inputs()
    try:
        integrated_gradients(
            model, audio.repeat(2, 1, 1, 1), visual.repeat(2, 1, 1, 1, 1), steps=2
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for batch > 1")

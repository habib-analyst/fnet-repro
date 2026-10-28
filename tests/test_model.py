import torch

from fnetrepro.model import FourierBlock, MultimodalFNet, fourier_mix


def _inputs(b=2, t=16):
    torch.manual_seed(0)
    audio = torch.randn(b, 1, 40, t)
    visual = torch.randn(b, t, 1, 32, 32)
    return audio, visual


def test_forward_shape_and_finiteness():
    model = MultimodalFNet()
    audio, visual = _inputs()
    logits = model(audio, visual)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_fourier_mix_preserves_shape_and_is_real():
    x = torch.randn(3, 16, 24)
    y = fourier_mix(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype  # real in, real out


def test_intra_mixer_actually_mixes_across_time():
    block = FourierBlock(d_model=16)
    block.eval()
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        base = block(x)
        perturbed = x.clone()
        # Perturb one time step with a *non-constant* vector: a constant
        # shift across channels would be erased by the block's LayerNorm
        # before the FFT ever sees it.
        perturbed[0, 0] += torch.linspace(-3.0, 5.0, 16)
        moved = block(perturbed)
    # A spectral mixer's output at *other* time steps must change too;
    # a position-wise block would leave them untouched.
    assert not torch.allclose(base[0, 4:], moved[0, 4:], atol=1e-5)


def test_inter_mixer_couples_modalities():
    torch.manual_seed(0)
    audio, visual = _inputs(b=1)
    model = MultimodalFNet(use_inter=True)
    model.eval()
    with torch.no_grad():
        a1, v1 = model.encode(audio, visual)
        a2, v2 = model.encode(audio + 1.5, visual)
    # Changing only the audio must move the *visual* stream when the
    # inter-modality mixer is active.
    assert not torch.allclose(v1, v2, atol=1e-5)
    assert not torch.allclose(a1, a2, atol=1e-5)


def test_ablation_decouples_modalities():
    torch.manual_seed(0)
    audio, visual = _inputs(b=1)
    model = MultimodalFNet(use_inter=False)
    model.eval()
    with torch.no_grad():
        _, v1 = model.encode(audio, visual)
        _, v2 = model.encode(audio + 1.5, visual)
    # Without the inter mixer the visual stream cannot see the audio.
    assert torch.allclose(v1, v2, atol=1e-6)


def test_gradients_flow_to_both_encoders():
    model = MultimodalFNet()
    audio, visual = _inputs()
    loss = model(audio, visual).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
    assert model.audio_encoder.proj.weight.grad.abs().sum() > 0
    assert model.visual_encoder.proj.weight.grad.abs().sum() > 0


def test_one_training_step_reduces_loss_on_fixed_batch():
    torch.manual_seed(0)
    model = MultimodalFNet()
    audio, visual = _inputs(b=8)
    y = torch.tensor([0.0, 1.0] * 4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(6):
        opt.zero_grad()
        loss = loss_fn(model(audio, visual), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]

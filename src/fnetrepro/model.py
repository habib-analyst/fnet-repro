"""Multimodal-FNet, implemented from the method description of:

    Habib Ur Rehman, "Multimodal-FNet" (under review at IEEE TPAMI).

Architecture (as described in the paper):

1.  Per-modality encoders turn the raw streams into token sequences of
    a shared width ``d_model`` -- a small CNN over mel-spectrogram
    frames for audio, a small per-frame CNN for the face-crop sequence.
2.  **Fourier intra-modality mixers** (FNet-style blocks) mix tokens
    *within* each modality in the frequency domain: an FFT is applied
    along the time axis and along the channel axis, the real part is
    kept, and the result enters through residual connections and a
    feed-forward network. There are no attention weights; token mixing
    is entirely spectral.
3.  **Fourier inter-modality mixers** stack the two modalities and
    apply a joint FFT across the modality, time and channel axes, so
    audio and visual tokens exchange information in the frequency
    domain before being split back into per-modality streams (each with
    its own feed-forward network).
4.  A classification head mean-pools both streams, concatenates the
    pooled vectors and predicts real vs fake.

``use_inter=False`` gives the ablation: the inter-modality mixers are
replaced by identity maps, so the modalities only meet at the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fourier_mix(x: torch.Tensor) -> torch.Tensor:
    """FNet-style spectral mixing over the last two dims (time, channel)."""
    return torch.fft.fft2(x, dim=(-2, -1)).real


class FourierBlock(nn.Module):
    """One intra-modality Fourier mixer block (pre-norm, residual)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + fourier_mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class InterFourierBlock(nn.Module):
    """One inter-modality Fourier mixer block.

    The two token streams are stacked into a (batch, modality, time,
    channel) tensor and mixed with a joint FFT over the modality, time
    and channel axes, then split back and passed through per-modality
    feed-forward networks.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn_audio = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
        )
        self.ffn_visual = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

    def forward(
        self, audio: torch.Tensor, visual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.stack([audio, visual], dim=1)  # (B, 2, T, D)
        mixed = torch.fft.fftn(self.norm(z), dim=(1, 2, 3)).real
        z = z + mixed
        a_out, v_out = z[:, 0], z[:, 1]
        a_out = a_out + self.ffn_audio(self.norm_a(a_out))
        v_out = v_out + self.ffn_visual(self.norm_v(v_out))
        return a_out, v_out


class AudioEncoder(nn.Module):
    """Mel-spectrogram (B, 1, F, T) -> tokens (B, T, D)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=(2, 1), padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(32 * 20, d_model)  # freq pooled 40 -> 20

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)  # (B, 32, 20, T)
        h = h.permute(0, 3, 1, 2).flatten(2)  # (B, T, 32*20)
        return self.proj(h)


class VisualEncoder(nn.Module):
    """Frames (B, T, 1, H, W) -> tokens (B, T, D) via a shared frame CNN."""

    def __init__(self, d_model: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[0], x.shape[1]
        h = self.cnn(x.reshape(b * t, *x.shape[2:]))  # (B*T, 32, 1, 1)
        h = h.reshape(b, t, -1)
        return self.proj(h)


class MultimodalFNet(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_intra: int = 2,
        n_inter: int = 2,
        dropout: float = 0.1,
        use_inter: bool = True,
    ):
        super().__init__()
        self.use_inter = use_inter
        self.audio_encoder = AudioEncoder(d_model)
        self.visual_encoder = VisualEncoder(d_model)
        self.audio_intra = nn.ModuleList(FourierBlock(d_model, dropout=dropout) for _ in range(n_intra))
        self.visual_intra = nn.ModuleList(FourierBlock(d_model, dropout=dropout) for _ in range(n_intra))
        self.inter = nn.ModuleList(InterFourierBlock(d_model, dropout=dropout) for _ in range(n_inter))
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode(
        self, audio: torch.Tensor, visual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the post-fusion (audio, visual) token streams."""
        a = self.audio_encoder(audio)
        v = self.visual_encoder(visual)
        for block in self.audio_intra:
            a = block(a)
        for block in self.visual_intra:
            v = block(v)
        if self.use_inter:
            for block in self.inter:
                a, v = block(a, v)
        return a, v

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        a, v = self.encode(audio, visual)
        pooled = torch.cat([a.mean(dim=1), v.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)

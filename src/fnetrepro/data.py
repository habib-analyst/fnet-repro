"""SynthAV: a principled synthetic audio-visual deepfake dataset.

Why synthetic: FakeAVCeleb (the dataset used in the Multimodal-FNet
paper) is distributed under a request form and is far too large for a
CPU-only smoke-scale run. SynthAV instead *generates* paired audio-visual
clips from a parametric model in which "real" and "fake" are defined by
the same failure modes the real detectors face:

* ``real``      -- one identity generates both streams from one shared
                   phoneme plan, so voice, face and lip motion agree.
* ``fake_audio``  -- voice-conversion style: the audio is re-rendered
                   with a *different* identity's voice (pitch, formant
                   scaling, timbre) while the video keeps the original
                   identity and lip plan. Adds a splice artefact.
* ``fake_video``  -- face-swap style: the face is re-rendered with a
                   different identity's face geometry while the audio
                   keeps the original identity. Adds a blending ring at
                   the swap boundary.
* ``fake_sync``   -- lip-sync style (Wav2Lip-like): the mouth is driven
                   by an *independent* phoneme plan, so lip motion and
                   audio disagree in time; the mouth region is blurred,
                   as re-rendered mouths typically are.
* ``fake_both``   -- fake audio and fake video combined.

Audio is a mel-scale spectrogram-like matrix synthesised from a harmonic
comb filtered by vowel formants; the frame energy is tied to the mouth
aperture, which is what makes real clips audio-visually *coupled*.
Visual frames are small face-crop-like renders (ellipse head, eyes, and
a mouth whose opening follows the phoneme plan).

Splits are identity-disjoint: no synthetic identity appears in two
splits, and swap/conversion partners are drawn from the same split.
Nothing here is real media and no number measured on SynthAV says
anything about FakeAVCeleb performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Clip geometry (shared with the explainability demo: the mouth box is
# where the generator draws the mouth).
T_FRAMES = 16
MEL_BINS = 40
FRAME_H = 32
FRAME_W = 32
MOUTH_BOX = (19, 29, 9, 24)  # rows [r0, r1), cols [c0, c1)

SUBTYPES = ("real", "fake_audio", "fake_video", "fake_sync", "fake_both")
FAKE_SUBTYPES = SUBTYPES[1:]

_VOWELS = np.array(
    [
        [730.0, 1090.0],  # a
        [270.0, 2290.0],  # i
        [300.0, 870.0],  # u
        [530.0, 1840.0],  # e
        [440.0, 1020.0],  # o
    ]
)


@dataclass
class Identity:
    f0: float  # base pitch, Hz
    formant_scale: float
    timbre: np.ndarray  # harmonic weights, len 8
    skin: float  # face fill intensity
    head_rx: float
    head_ry: float
    eye_dx: float
    eye_y: float
    mouth_w: float
    brightness: float  # background level


def make_identities(n: int, seed: int) -> list[Identity]:
    rng = np.random.default_rng(seed)
    ids = []
    for _ in range(n):
        timbre = rng.uniform(0.25, 1.0, size=8)
        timbre = timbre / timbre.max()
        ids.append(
            Identity(
                f0=float(rng.uniform(90.0, 220.0)),
                formant_scale=float(rng.uniform(0.85, 1.2)),
                timbre=timbre.astype(np.float64),
                skin=float(rng.uniform(0.35, 0.8)),
                head_rx=float(rng.uniform(10.5, 12.5)),
                head_ry=float(rng.uniform(12.5, 14.5)),
                eye_dx=float(rng.uniform(4.5, 6.0)),
                eye_y=float(rng.uniform(12.0, 14.0)),
                mouth_w=float(rng.uniform(4.2, 5.8)),
                brightness=float(rng.uniform(0.15, 0.4)),
            )
        )
    return ids


def _phoneme_plan(rng: np.random.Generator, t_frames: int = T_FRAMES):
    """A segment plan: per-frame vowel targets and mouth apertures."""
    n_seg = int(rng.integers(3, 6))
    cuts = np.sort(rng.choice(np.arange(1, t_frames), size=n_seg - 1, replace=False))
    bounds = np.concatenate([[0], cuts, [t_frames]])
    vowels = rng.integers(0, len(_VOWELS), size=n_seg)
    apertures = rng.uniform(0.15, 1.0, size=n_seg)
    pitch = rng.uniform(0.95, 1.08, size=n_seg)
    vowel_t = np.zeros(t_frames, dtype=int)
    aper_t = np.zeros(t_frames)
    pitch_t = np.zeros(t_frames)
    for s in range(n_seg):
        lo, hi = bounds[s], bounds[s + 1]
        vowel_t[lo:hi] = vowels[s]
        aper_t[lo:hi] = apertures[s]
        pitch_t[lo:hi] = pitch[s]
    # Smooth the aperture so the mouth opens/closes continuously.
    kernel = np.array([0.25, 0.5, 0.25])
    aper_s = np.convolve(np.pad(aper_t, 1, mode="edge"), kernel, mode="valid")
    return vowel_t, aper_s, pitch_t


def _mel_centers(n_bins: int = MEL_BINS, f_max: float = 4000.0) -> np.ndarray:
    mel = np.linspace(0.0, 2595.0 * np.log10(1.0 + f_max / 700.0), n_bins)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


_MEL_CENTERS = _mel_centers()


def _render_audio(identity: Identity, vowel_t, aper_t, pitch_t, rng) -> np.ndarray:
    """Mel-spectrogram-like matrix (MEL_BINS, T) for one identity + plan."""
    t_frames = len(vowel_t)
    spec = np.zeros((MEL_BINS, t_frames))
    phase = rng.uniform(0, 2 * np.pi)
    for t in range(t_frames):
        f0 = identity.f0 * pitch_t[t] * (1.0 + 0.03 * np.sin(2 * np.pi * 2 * t / t_frames + phase))
        f1, f2 = _VOWELS[vowel_t[t]] * identity.formant_scale
        freqs = _MEL_CENTERS
        comb = np.zeros(MEL_BINS)
        for h in range(1, 9):
            comb += identity.timbre[h - 1] * np.exp(
                -((freqs - h * f0) ** 2) / (2 * 45.0**2)
            )
        formant = 0.3 + np.exp(-((np.log(freqs + 1) - np.log(f1)) ** 2) / (2 * 0.25**2)) + 0.6 * np.exp(
            -((np.log(freqs + 1) - np.log(f2)) ** 2) / (2 * 0.25**2)
        )
        envelope = 0.25 + 0.75 * aper_t[t]
        spec[:, t] = comb * formant * envelope
    spec += 0.02 * rng.random(spec.shape)
    return np.log1p(spec * 3.0).astype(np.float32)


def _ellipse(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2


def _box_blur(img: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    crop = img[r0:r1, c0:c1]
    padded = np.pad(crop, 1, mode="edge")
    out = np.zeros_like(crop)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + crop.shape[0], dx : dx + crop.shape[1]]
    img = img.copy()
    img[r0:r1, c0:c1] = out / 9.0
    return img


def _render_visual(identity: Identity, aper_t, rng, swapped_from: Identity | None = None) -> np.ndarray:
    """Face-crop-like frames (T, 1, H, W). If ``swapped_from`` is given,
    the inner face uses that identity (face swap) while the surround
    keeps ``identity``'s background, with a blending ring artefact."""
    t_frames = len(aper_t)
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(np.float64)
    frames = np.zeros((t_frames, 1, FRAME_H, FRAME_W), dtype=np.float32)
    face_id = swapped_from if swapped_from is not None else identity
    jx = np.cumsum(rng.normal(0, 0.25, size=t_frames))
    jy = np.cumsum(rng.normal(0, 0.25, size=t_frames))
    jx = np.clip(jx - jx.mean(), -1.0, 1.0)
    jy = np.clip(jy - jy.mean(), -1.0, 1.0)
    for t in range(t_frames):
        cx, cy = 16.0 + jx[t], 15.5 + jy[t]
        img = np.full((FRAME_H, FRAME_W), identity.brightness)
        img += 0.05 * (yy / FRAME_H)
        head = _ellipse(yy, xx, cy, cx, face_id.head_ry, face_id.head_rx)
        img = np.where(head <= 1.0, face_id.skin, img)
        if swapped_from is not None:
            ring = (head > 0.80) & (head <= 1.0)
            img = np.where(ring, img * 0.92, img)
        for sgn in (-1, 1):
            eye = _ellipse(yy, xx, face_id.eye_y + jy[t], cx + sgn * face_id.eye_dx, 1.2, 2.2)
            img = np.where(eye <= 1.0, 0.08, img)
        mouth = _ellipse(yy, xx, 23.0 + jy[t] * 0.5, cx, 0.6 + 2.6 * aper_t[t], face_id.mouth_w)
        img = np.where(mouth <= 1.0, 0.12, img)
        img += rng.normal(0, 0.02, size=img.shape)
        frames[t, 0] = np.clip(img, 0.0, 1.0)
    return frames


def _splice_artefact(spec: np.ndarray, rng) -> np.ndarray:
    spec = spec.copy()
    t_b = int(rng.integers(3, spec.shape[1] - 2))
    spec[:, t_b] += 0.5 * rng.random(spec.shape[0]).astype(np.float32)
    spec[int(MEL_BINS * 0.6) :, t_b + 1] += 0.3 * rng.random(MEL_BINS - int(MEL_BINS * 0.6)).astype(np.float32)
    return spec


def _make_sample(kind: str, identity: Identity, partner: Identity, rng):
    vowel_t, aper_t, pitch_t = _phoneme_plan(rng)
    audio_id, visual_id, swap_id = identity, identity, None
    aper_visual = aper_t
    audio_artefact = False
    mouth_blur = False
    if kind == "fake_audio":
        audio_id = partner
        audio_artefact = True
    elif kind == "fake_video":
        visual_id = partner
        swap_id = partner
    elif kind == "fake_sync":
        _, aper_visual, _ = _phoneme_plan(rng)
        mouth_blur = True
    elif kind == "fake_both":
        audio_id = partner
        audio_artefact = True
        visual_id = partner
        swap_id = partner
    spec = _render_audio(audio_id, vowel_t, aper_t, pitch_t, rng)
    if audio_artefact:
        spec = _splice_artefact(spec, rng)
    frames = _render_visual(visual_id, aper_visual, rng, swapped_from=swap_id)
    if mouth_blur:
        r0, r1, c0, c1 = MOUTH_BOX
        for t in range(frames.shape[0]):
            frames[t, 0] = _box_blur(frames[t, 0], r0, r1, c0, c1)
    label = 0 if kind == "real" else 1
    return spec[np.newaxis, :, :], frames, label


def generate_split(identities: list[Identity], n_per_identity: int, seed: int):
    """Generate a balanced split. Returns dict of numpy arrays:
    audio (N,1,MEL_BINS,T), visual (N,T,1,H,W), y (N,), subtype (N,)
    as indices into SUBTYPES, identity (N,) as index into ``identities``.
    """
    rng_master = np.random.default_rng(seed)
    audios, visuals, labels, subtypes, idents = [], [], [], [], []
    for ii, identity in enumerate(identities):
        partners = [j for j in range(len(identities)) if j != ii]
        for k in range(n_per_identity):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            kind = "real" if k % 2 == 0 else FAKE_SUBTYPES[(k // 2) % len(FAKE_SUBTYPES)]
            partner = identities[int(rng.choice(partners))]
            spec, frames, label = _make_sample(kind, identity, partner, rng)
            audios.append(spec)
            visuals.append(frames)
            labels.append(label)
            subtypes.append(SUBTYPES.index(kind))
            idents.append(ii)
    order = rng_master.permutation(len(labels))
    return {
        "audio": np.stack(audios)[order],
        "visual": np.stack(visuals)[order],
        "y": np.array(labels, dtype=np.int64)[order],
        "subtype": np.array(subtypes, dtype=np.int64)[order],
        "identity": np.array(idents, dtype=np.int64)[order],
    }


def build_dataset(seed: int = 17, n_identities: int = 24) -> dict[str, dict]:
    """Full SynthAV dataset with identity-disjoint splits.

    24 identities: 14 train / 4 val / 6 test. Per-identity sample counts
    give 728 train, 240 val, 300 test clips, each split 50% real.
    """
    identities = make_identities(n_identities, seed)
    pools = {
        "train": identities[:14],
        "val": identities[14:18],
        "test": identities[18:24],
    }
    counts = {"train": 52, "val": 60, "test": 50}
    return {
        name: generate_split(pool, counts[name], seed + 1000 * (i + 1))
        for i, (name, pool) in enumerate(pools.items())
    }

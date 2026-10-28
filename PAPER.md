# fnet-repro: A Runnable Implementation of Multimodal-FNet, with a Small-Scale Synthetic Evaluation

**Habib Ur Rehman** · 2026-10-28 · Implementation of the author's own
method paper: "Multimodal-FNet" (under review at IEEE TPAMI).
Payload: `~/workspace/build/fnet-repro/repo_staging/` (model, data
generator, training/evaluation/explainability code, tests, raw results
in `results/results.json` and `results/ig_summary.json`).

## Abstract

The Multimodal-FNet paper (under review at IEEE TPAMI) proposes
Fourier intra- and inter-modality mixers for audio-visual deepfake
detection and reports 92.35% accuracy on FakeAVCeleb, but it has no
public runnable code. This build implements the architecture cleanly
from the method description — per-modality CNN encoders, FNet-style
Fourier mixers within each modality, joint-FFT mixers across the two
modalities, and an integrated-gradients explainability path — and
evaluates it at small scale on **SynthAV**, a synthetic audio-visual
dataset generated in code with planted real/fake failure modes
(voice-conversion-style, face-swap-style, lip-sync-style, and combined
fakes) and identity-disjoint splits (728 train / 240 val / 300 test).
On the SynthAV test split the full model scores **0.727 accuracy
(218/300), AUC 0.809**; a no-inter-mixer ablation scores 0.757
(227/300), AUC 0.813. The per-subtype pattern is the informative part:
audio-carried fakes are detected perfectly by both variants (42/42 and
36/36); the inter-modality mixer more than doubles lip-sync detection
(22/36 = 0.611 vs 10/36 = 0.278), the one subtype visible only across
modalities, at the cost of real-clip accuracy at a fixed threshold
(107/150 vs 130/150); face-swap-style fakes are hard for both (0.31 /
0.25), a scale limit of learning voice↔face identity correspondence
from 728 clips. **No number here is a FakeAVCeleb number**; the paper's
reported result is neither reproduced nor tested by this build.

## 1. Why this build

A method paper without code cannot be inspected, extended, or
challenged. The kill-gate check (payload `KILLGATE.md`, 2026-10-28)
confirmed no official or author code for Multimodal-FNet exists
publicly: GitHub search returns only unrelated medical-fusion "FNet"
repos, and web search returns other groups' audio-visual detectors.
fnet-repro closes the inspectability gap for the author's own paper.

## 2. Method (as implemented)

* **Encoders.** Audio: Conv2d stack over the (40 mel bins × 16 frames)
  spectrogram-like input → per-time-step projection to `d_model = 64`.
  Visual: a shared 2-layer CNN over 32×32 face-crop-like frames →
  per-frame projection to 64. Output: two token sequences (B, 16, 64).
* **Intra-modality Fourier mixers (×2 per modality).** FNet-style:
  FFT over the time and channel axes, real part, pre-norm residuals,
  GELU feed-forward (expansion 4). No attention anywhere.
* **Inter-modality Fourier mixers (×2).** The streams are stacked to
  (B, 2, 16, 64) and mixed by a joint FFT over the modality, time and
  channel axes, then split back, each stream passing its own
  feed-forward network.
* **Head.** Mean-pool both streams, concatenate, 2-layer MLP → logit.
  327,553 parameters total. The ablation (`use_inter=False`) removes
  the inter mixers; unit tests verify the streams are then exactly
  decoupled, and coupled when the mixers are present.
* **Explainability.** Integrated gradients against a zero baseline
  (48 steps in the demo), implemented directly on autograd.

## 3. Data: SynthAV

Generated parametrically (`src/fnetrepro/data.py`). An identity is a
voice (base pitch 90–220 Hz, formant scale, harmonic timbre) plus a
face (geometry, skin intensity). A shared phoneme plan (3–5 vowel
segments) drives both the formant-filtered harmonic audio — whose
frame energy follows the mouth aperture — and the rendered mouth.
Fakes: `fake_audio` re-renders the voice with another identity and
adds a splice burst; `fake_video` re-renders the face with another
identity's geometry and a blending ring; `fake_sync` drives the mouth
from an independent plan and blurs the mouth box; `fake_both`
combines audio and video fakes. Splits are identity-disjoint
(14/4/6 of 24 identities), partners drawn within a split, each split
50% real. 1,268 clips total. This is deliberately *not* a FakeAVCeleb
proxy: the artefacts are simpler than Faceswap/FSGAN/Wav2Lip/RTVC
output, and results transfer neither way.

## 4. Results

Config: 30 epochs, Adam lr 1e-3, batch 32, seed 7, checkpoint by
validation AUC (full 0.836, ablation 0.804), CPU (2 cores): ~99 s /
~76 s of training.

| model | acc | AUC | real | fake_audio | fake_video | fake_sync | fake_both |
|---|---|---|---|---|---|---|---|
| full | 0.727 (218/300) | 0.809 | 107/150 | 42/42 | 11/36 | 22/36 | 36/36 |
| no inter mixer | 0.757 (227/300) | 0.813 | 130/150 | 42/42 | 9/36 | 10/36 | 36/36 |

Reading: (1) the audio channel dominates the easy fakes; (2) the
inter-modality Fourier mixer earns its place exactly where the method
claims it should — cross-modal temporal consistency (fake_sync
0.611 vs 0.278) — but at threshold 0 it trades real-clip accuracy, so
headline accuracy does not favour it at this scale and data size;
(3) identity-correspondence fakes (fake_video) are essentially
unlearned by both at 728 training clips. All three statements are
properties of this small synthetic regime, reported as measured.

## 5. Explainability

IG demos on one correctly classified clip per subtype of interest
(`results/ig_summary.json`, `results/ig_example.png`):

* fake_sync clip (logit 1.620): attribution sums to 0.447 against a
  logit-minus-baseline of 0.426 (abs. error 0.021, 48 steps) — the
  completeness axiom holds to Riemann accuracy.
* Summed |attribution| shares: visual 95.4% / audio 4.6% (fake_sync)
  and 94.9% / 5.1% (fake_audio). The visual input carries 16,384
  values vs 640 audio values, so these shares are dimensionality-
  inflated and must not be read as "the model ignores audio" — the
  per-subtype results prove it does not.
* Mouth-region share of visual |attribution|: 17.8% and 21.0% vs a
  14.6% area share — mild concentration, reported without embellishment.

## 6. Limits

Synthetic, planted artefacts; 1,268 clips; one seed; threshold fixed
at 0 for the headline numbers; architecture choices where the method
description leaves freedom are this implementation's own. The build's
claim is narrow: the Multimodal-FNet architecture is now runnable,
tested (14 pytest tests: mixer coupling/decoupling, IG completeness,
generator determinism and split hygiene, a training smoke test), and
its small-scale synthetic behaviour — including where it fails — is
measured and on record.

## 7. Provenance

Implemented from the method description of the author's own paper
under review. Kill-gate: no prior public code (see payload
`KILLGATE.md`). Nothing was pushed or posted by this build.

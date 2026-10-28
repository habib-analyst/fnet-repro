# fnet-repro — Multimodal-FNet, runnable

This is a runnable implementation of the method from **Multimodal-FNet
(Habib Ur Rehman, under review)** — the paper is currently under review
at IEEE TPAMI. The paper has no public code; this repository implements
its architecture from the method description so the work is inspectable
and reproducible.

> **What this is not.** This repository does **not** reproduce the
> paper's reported result (92.35% accuracy on FakeAVCeleb). FakeAVCeleb
> is distributed under a request form and is far too large for a
> CPU-only run, so the experiment below is a small-scale training and
> evaluation run on **SynthAV, a synthetic audio-visual dataset
> generated in code** (`src/fnetrepro/data.py`), with planted real/fake
> failure modes. Every number in this README was measured on SynthAV
> and says nothing about FakeAVCeleb performance.

## Architecture

```
 audio mel (B,1,40,16)                          frames (B,16,1,32,32)
        │                                              │
 ┌──────▼───────┐                              ┌───────▼────────┐
 │ audio CNN    │                              │ per-frame CNN  │
 │ encoder      │                              │ encoder        │
 └──────┬───────┘                              └───────┬────────┘
        │ tokens (B,16,64)                             │ tokens (B,16,64)
 ┌──────▼───────────────┐                  ┌───────────▼────────────┐
 │ Fourier intra mixer  │  x2              │ Fourier intra mixer    │  x2
 │ (FFT over time+chan) │                  │ (FFT over time+chan)   │
 └──────┬───────────────┘                  └───────────┬────────────┘
        │                                              │
        └──────────────┬───────────────────────────────┘
                       │ stack -> (B, 2, 16, 64)
              ┌────────▼─────────────────────────────┐
              │ Fourier inter-modality mixer  x2     │
              │ joint FFT over (modality, time,      │
              │ channel); split back; per-modality   │
              │ feed-forward networks                │
              └────────┬─────────────────────────────┘
                       │ mean-pool both streams, concat
              ┌────────▼────────┐
              │ MLP head        │ -> real/fake logit
              └─────────────────┘
```

* **Encoders** — a small CNN over the mel spectrogram and a small
  per-frame CNN over face-crop-like frames produce token sequences of
  shared width `d_model = 64`.
* **Fourier intra-modality mixers** — FNet-style blocks: an FFT along
  the time axis and the channel axis, real part kept, residual
  connection, LayerNorm, feed-forward network. Token mixing inside each
  modality is entirely spectral; there is no attention.
* **Fourier inter-modality mixers** — the two streams are stacked and
  mixed with a joint FFT across the modality, time and channel axes,
  so audio and visual tokens exchange information in the frequency
  domain before being split back into per-modality streams.
* **Head** — mean-pooled audio and visual tokens are concatenated and
  classified by a small MLP.
* **Explainability** — integrated gradients (implemented directly on
  autograd, `src/fnetrepro/ig.py`) attribute the fake logit back to
  both input streams.

The model has 327,553 parameters. `use_inter=False` ablates the
inter-modality mixers (the modalities then meet only at the head).

## Data: SynthAV (synthetic — read this before citing any number)

SynthAV clips are generated from a parametric model in which one
synthetic identity produces both streams from a shared phoneme plan:
formant-filtered harmonic audio whose frame energy follows the mouth
aperture, and face-crop-like frames whose mouth opens with the same
plan. Fake clips plant the failure modes real detectors face:

| subtype | construction |
|---|---|
| `real` | one identity, one shared phoneme plan for both streams |
| `fake_audio` | voice-conversion style: audio re-rendered with another identity's voice + a splice artefact |
| `fake_video` | face-swap style: face re-rendered with another identity's geometry + a blending ring |
| `fake_sync` | lip-sync style: mouth driven by an independent phoneme plan (audio-visual desync) + mouth blur |
| `fake_both` | fake audio and fake video combined |

Splits are **identity-disjoint** (24 identities: 14 train / 4 val /
6 test; swap and conversion partners come from the same split):
728 train, 240 validation, 300 test clips, each split 50% real.

## Measured results (SynthAV test split, 300 clips, CPU)

Config: 30 epochs, Adam (lr 1e-3), batch 32, seed 7, checkpoint
selected by validation AUC. Full run: ~99 s of training on 2 CPU
cores; ablation ~76 s. Raw output: `results/results.json`.

| model | accuracy | AUC | real | fake_audio | fake_video | fake_sync | fake_both |
|---|---|---|---|---|---|---|---|
| Multimodal-FNet (full) | 0.727 (218/300) | 0.809 | 107/150 | 42/42 | 11/36 | 22/36 | 36/36 |
| ablation (no inter mixer) | 0.757 (227/300) | 0.813 | 130/150 | 42/42 | 9/36 | 10/36 | 36/36 |

What the numbers actually say:

1. **Audio-carried fakes are easy here.** Both variants detect every
   `fake_audio` and `fake_both` clip; the planted splice artefact and
   voice change are strong signals.
2. **The inter-modality mixer is what catches lip-sync fakes** — the
   one subtype that is only visible *across* modalities: 22/36 (0.611)
   vs 10/36 (0.278) without it. At a fixed decision threshold it costs
   real-clip accuracy (107/150 vs 130/150), so overall accuracy is
   slightly *lower* for the full model at this scale. That trade-off is
   reported, not hidden.
3. **Face-swap-style fakes are hard for both** (0.31 / 0.25).
   Learning a voice-to-face identity correspondence from 728 training
   clips is beyond this scale; the result is a scale limit, not
   evidence about the architecture at paper scale.

## Explainability demo

`experiments/run_explain.py` attributes the fake logit of two
correctly classified test clips with integrated gradients (48 steps,
zero baseline). Output: `results/ig_summary.json`,
`results/ig_example.png`.

* Completeness check (fake_sync demo): summed attributions 0.447 vs
  logit-minus-baseline 0.426 — absolute error 0.021.
* Summed |attribution| is dominated by the visual stream in both demos
  (95.4% and 94.9%). Read that with care: the visual input has 16,384
  values vs 640 for audio, so raw |attribution| shares overstate the
  visual stream's role.
* The mouth box holds 17.8% (fake_sync) and 21.0% (fake_audio) of the
  visual |attribution|, against 14.6% of the frame area — a mild
  concentration on the mouth region, not a smoking gun.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests/                 # 14 tests
python experiments/run_experiment.py    # train + eval -> results/results.json
python experiments/run_explain.py       # IG demo -> results/ig_summary.json, ig_example.png
```

## Honest limits

* Synthetic data only. SynthAV's fakes are planted, parametric
  artefacts — much simpler than Faceswap/FSGAN/Wav2Lip/RTVC output.
  Nothing here validates or refutes the paper's FakeAVCeleb numbers.
* Small scale throughout: 1,268 clips, 328k parameters, CPU minutes.
  The per-subtype pattern (point 3 above) is a scale statement.
* The architecture follows the paper's method description; where the
  description leaves freedom (encoder sizes, block counts, norms), this
  implementation makes its own documented choices.
* IG modality shares are not comparable across inputs of very
  different dimensionality without the caveat above.

## Files

```
src/fnetrepro/model.py    MultimodalFNet: encoders, Fourier intra/inter mixers, head
src/fnetrepro/data.py     SynthAV generator (identity-disjoint splits)
src/fnetrepro/ig.py       integrated gradients
src/fnetrepro/train.py    deterministic training loop, val-AUC checkpointing
src/fnetrepro/metrics.py  accuracy / AUC / per-subtype evaluation
experiments/run_experiment.py   full vs ablation training + evaluation
experiments/run_explain.py      integrated-gradients demo
results/                        measured outputs of the runs above
tests/                          model, data and IG tests (14)
KILLGATE.md                     prior-code check for this build
```

## Status

Implementation of the author's own method paper: Habib Ur Rehman,
"Multimodal-FNet", under review at IEEE TPAMI. The paper reports
92.35% accuracy on FakeAVCeleb; this repository makes the method
runnable and inspectable at small scale and reports only its own
measured SynthAV numbers.

MIT license (see `LICENSE`).

# Kill-gate check — fnet-repro

**Question:** does official or author code for *Multimodal-FNet* (Habib Ur
Rehman, under review at IEEE TPAMI — Fourier intra/inter-modality mixers
for audio-visual deepfake detection) already exist publicly?

**Checked 2026-10-28:**

- GitHub repository search for `Multimodal-FNet`: 2 results, both
  unrelated — DM-FNet (multimodal *medical image fusion*, TMM 2025) and
  M²FNet (multimodal fusion for stroke LVO detection). Neither is
  audio-visual deepfake detection; neither is by Rehman.
- Web search for `"Multimodal-FNet" FakeAVCeleb Rehman` and for
  `Multimodal-FNet Fourier audio-visual deepfake detection GitHub code`:
  no repository, project page, or code release for the paper. Results
  returned *other* audio-visual deepfake methods (AVPF, FTFDNet, AVTENet,
  AVFakeNet, TSFF-Net, MM-DDL) — different architectures by different
  authors, none implementing Fourier intra/inter-modality mixers, none
  claiming to be Multimodal-FNet code.
- Related-but-different prior art (not this paper's code): FNet (Lee-Thorp
  et al., Fourier token mixing for text), F3Net (frequency-domain face
  forgery detection), FakeAVCeleb dataset code (hasamkhalid/FakeAVCeleb —
  the dataset, not a Multimodal-FNet implementation).

**Verdict: PASS.** No official or author code for Multimodal-FNet exists
publicly. The paper is under review and has no public runnable code, so a
clean from-the-method-description implementation does not duplicate any
existing release. Build proceeds.

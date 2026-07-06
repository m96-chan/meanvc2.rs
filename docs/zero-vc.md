# Zero-VC (engine candidate)

> **Candidate engine — not yet implemented.** Evaluation notes for **Zero-VC:
> Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization**
> ([arXiv:2606.20218](https://arxiv.org/abs/2606.20218), Li et al., 2026;
> accepted to Interspeech 2026), parked/tracked in
> [#31](https://github.com/m96-chan/babiniku.rs/issues/31).

[![Paper](https://img.shields.io/badge/arXiv-2606.20218-b31b1b.svg)](https://arxiv.org/abs/2606.20218)
[![Status](https://img.shields.io/badge/status-evaluation-blue.svg)](#status)

Zero-VC is a zero-shot streaming voice conversion system built as a **strictly
causal, zero-lookahead network**: 20 ms frame shift, **20 ms algorithmic
latency**, no future context at all. Its trick is not a new decoder but a
training-time perturbation — **speaker anonymization (SA)** applied to the
source audio — that strips source timbre while preserving prosody, which in
turn removes the generator's dependence on future frames.

## Why we care: the latency-first candidate

The project's latency ladder is MeanVC v1 at a ~270 ms floor, MeanVC 2 in the
~110 ms class, target **<100 ms**. Zero-VC's 20 ms algorithmic latency plus
CPU RTF 0.063 (Intel Xeon Platinum 8468V, 2.4 GHz) is the most aggressive
published point we know of in this space — the paper's own comparison lists
DualVC3 at 40 ms, RT-VC at 47 ms, StreamVC at 60 ms. If its quality holds up,
it is the natural engine for the sub-100 ms target.

## Key ideas

- **Speaker anonymization as perturbation** — during training the source is
  first passed through an off-the-shelf SA module
  ([DigitalPhonetics/speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization),
  Meyer et al.), mapping the voice to a pseudo-speaker while preserving
  temporal alignment, prosodic contours, and phonetic content. The content
  encoder therefore learns representations inherently devoid of source timbre.
  On raw audio SA reaches source-similarity 0.119 (vs. 0.704 for
  LSCodec-style perturbation) while keeping prosody (FPC 0.718).
- **Zero lookahead falls out of SA** — ablations show the SA-trained model
  saturates at 0–20 ms of future context (<3 % relative gain even with 80 ms),
  whereas the non-SA baseline needs 40–60 ms of future frames to stabilize.
  Timbre-clean, prosody-stable inputs remove the need to peek ahead.
- **Strictly causal decoder** — a HiFi-GAN-based generator with every standard
  convolution replaced by a causal one; streaming keeps a state buffer (cache)
  of only the receptive-field past, so per-frame compute is O(1). MSD/MPD
  discriminators and the SA module are training-time only and discarded at
  inference.
- **Attention-pooled timbre conditioning** — WavLM-large layer-7 features from
  the reference are attention-pooled to a single vector and injected into the
  generator as a residual offset through a three-layer Conv1D.

## Architecture

```text
              (training only)
 source wav ──► SA module ──► streaming content encoder ──► content feats ─┐
               (discarded     (distilled streaming                         ▼
                at inference)  w2v-bert-2.0)                     ┌──────────────┐
                                                                 │ causal       │
 reference ──► WavLM-large (layer 7) ──► attention pool ────────►│ HiFi-GAN     │──► wav
    wav                                  ──► Conv1D offset       │ generator    │
                                                                 └──────────────┘
              training: MSD + MPD discriminators (discarded at inference)
```

| Component | Role | Provenance |
|---|---|---|
| SA module | training-time timbre perturbation | external, public ([DigitalPhonetics](https://github.com/DigitalPhonetics/speaker-anonymization)) |
| Streaming content encoder | linguistic features, causal | distilled streaming w2v-bert-2.0 (paper) |
| WavLM-large (layer 7) + attention pooling | timbre embedding | external, public ([microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm)) |
| Causal HiFi-GAN generator | waveform synthesis, O(1)/frame | trained (paper); parameter count not stated |

## Reported numbers (paper)

- **Latency:** 20 ms frame shift, 20 ms algorithmic latency; RTF 0.063 on an
  Intel Xeon Platinum 8468V 2.4 GHz CPU.
- **Zero-shot quality (seed-tts-eval subset):** source-similarity 0.171
  (lower = less leakage; best of compared systems), target-similarity 0.521
  (best), WER 3.96 % (Seed-VC-Small: 2.47 %), FPC 0.688 (best),
  DNSMOS OVRL 3.044. Subjective: SMOS 3.88 ± 0.05 (highest), NMOS 3.81 ± 0.07.
- **Training:** LibriTTS only (~460 h English after filtering, 16 kHz), 1.2 M
  steps — so, like MeanVC v1's Mandarin lock in reverse, it is
  **English-trained**; Japanese behavior is TBD (Phase 0).

## Code / checkpoint availability

**Not public as of this writing (checked 2026-07).** The paper and the
[demo page](https://amphionteam.github.io/Zero-VC-demo/) provide audio samples
only — no code repository or checkpoints are linked, and the paper makes no
release statement. The authors are affiliated with the Amphion team, so a
release inside [Amphion](https://github.com/open-mmlab/Amphion) is plausible
but unconfirmed. Only the external SA module and WavLM-large are public.
A port would currently mean **training from scratch** (feasible in principle:
single-corpus LibriTTS recipe, but the "distilled streaming w2v-bert-2.0"
distillation procedure is only sketched in the paper).

## Status

**Phase 0 — evaluation parked**, tracked in
[#31](https://github.com/m96-chan/babiniku.rs/issues/31), pending a code or
checkpoint release (or a decision to reproduce from the paper). Nothing in
this repo implements Zero-VC yet.

- [ ] Upstream code/checkpoint release — watching; none found so far
- [ ] Perceptual check against MeanVC v1 output (demo samples) — TBD (Phase 0)
- [ ] Japanese / non-English behavior — TBD (Phase 0)
- [ ] Reproduction cost estimate if no release materializes — TBD (Phase 0)

## Planned integration

If unblocked, Zero-VC follows the house pattern (see
[docs/meanvc.md](meanvc.md)): pure-candle port with an upstream-mirroring
module tree, frozen externals (WavLM-large, content encoder) behind the
`vc-core` (`crates/vc-core/src/encoders.rs`)-style traits, and golden tests against the reference
implementation. Its strictly causal, cache-based streaming maps cleanly onto
the existing chunk-driver design, and the CPU RTF suggests it fits the
driver-only-GPU rule; both claims are TBD (Phase 0) until we can run it.

## Citation

```bibtex
@article{li2026zerovc,
  title   = {Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker
             Anonymization},
  author  = {Li, Yudong and Fang, Zihao and Qiu, Junwen and Jing, Ruihai and
             Hang, Ruixiang and Shen, Yingda and Wu, Zhizheng},
  journal = {arXiv preprint arXiv:2606.20218},
  year    = {2026},
  note    = {Accepted to Interspeech 2026}
}
```

## Acknowledgements

- [Zero-VC](https://arxiv.org/abs/2606.20218) (Li et al., 2026) — all model
  ideas belong to the original authors; audio samples at the
  [demo page](https://amphionteam.github.io/Zero-VC-demo/).
- [Speaker anonymization](https://github.com/DigitalPhonetics/speaker-anonymization)
  (Meyer et al.) and [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm)
  (Microsoft) — the public components the method builds on.

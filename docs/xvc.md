# X-VC (engine candidate)

> **Candidate engine — not yet implemented.** Evaluation notes for **X-VC:
> Zero-shot Streaming Voice Conversion in Codec Space**
> ([arXiv:2604.12456](https://arxiv.org/abs/2604.12456), Zheng et al., 2026),
> under Phase 0 evaluation in [#30](https://github.com/m96-chan/babiniku.rs/issues/30).

[![Paper](https://img.shields.io/badge/arXiv-2604.12456-b31b1b.svg)](https://arxiv.org/abs/2604.12456)
[![Status](https://img.shields.io/badge/status-evaluation-blue.svg)](#status)

X-VC is a zero-shot streaming voice conversion system that performs **one-step
conversion directly in the latent space of a pretrained neural codec** (SAC),
rather than in mel-spectrogram space. A compact dual-conditioning converter
(44 M params; 539 M with the frozen codec/encoders) rewrites source codec
latents toward a target speaker, and the codec decoder emits the waveform.
Code and checkpoints are **public** ([Jerrister/X-VC](https://github.com/Jerrister/X-VC),
MIT; weights on [Hugging Face](https://huggingface.co/chenxie95/X-VC)).

## Why we care: breaking the Mandarin lock

MeanVC v1's content pipeline is a Mandarin-trained Fast-U2++ BNF extractor, so
conversion quality is effectively **Mandarin-locked**
([#28](https://github.com/m96-chan/babiniku.rs/issues/28) tracks Japanese).
X-VC's content representation is instead the semantic branch of a codec fed by
the **GLM-4-Voice tokenizer**
([zai-org/glm-4-voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer)),
trained bilingually (EN/ZH, ~10k h from Emilia + LibriTTS plus ~20k h of
Seed-VC-generated pairs) and evaluated **cross-lingually** (EN→ZH WER 2.67 %,
ZH→EN 2.15 %) — the paper describes the codec-space design as naturally
supporting cross-lingual conversion. Japanese coverage of the tokenizer is
unverified: **TBD (Phase 0)**.

## Key ideas

- **Codec-space conversion** — source audio is encoded once by the frozen SAC
  codec (16 kHz, 62.5 Hz latent rate, 1024-dim latents); conversion is a
  single non-iterative pass over latents (no diffusion loop, no separate
  vocoder stage — the codec decoder is the vocoder).
- **Dual-conditioning converter** — a 6-layer / 8-head transformer (hidden
  512) jointly attends over the source latent sequence and a frame-level
  acoustic condition (128-dim mel), with a 192-dim utterance-level speaker
  embedding (ERes2Net) injected via adaptive normalization.
- **Generated-pair training** — Seed-VC synthesizes paired data; standard /
  reconstruction / reversed roles are mixed at (0.4, 0.2, 0.4) so the model
  sees diverse input–output distributions.
- **Chunkwise streaming with overlap smoothing** — inference windows are
  history + current + overlap + optional future context; a cosine cross-fade
  over the overlap removes boundary discontinuities.

## Architecture

```text
 source wav ──► SAC encoder ──► codec latents ────────────┐
               (semantic: GLM-4-Voice tokenizer branch)   ▼
                                                   ┌──────────────┐
 frame-level acoustic condition (mel, 128-d) ─────►│ dual-cond    │
                                                   │ transformer  │──► converted
 reference ──► ERes2Net ──► spk emb (192-d) ──────►│ (adaLN, 6L)  │    latents
    wav        (ModelScope)                        └──────────────┘      │
                                                                         ▼
                                          converted wav ◄── SAC decoder ─┘
```

| Component | Role | Provenance |
|---|---|---|
| SAC codec (encoder/decoder, semantic + acoustic VQ) | latent space + waveform synthesis | frozen, pretrained ([HF](https://huggingface.co/chenxie95/X-VC)) |
| GLM-4-Voice tokenizer | semantic tokenization | frozen ([zai-org/glm-4-voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer)) |
| ERes2Net speaker encoder | 192-dim speaker embedding | frozen (ModelScope) |
| Dual-conditioning converter (44 M) | codec-latent conversion | trained; `ckpts/xvc.pt` |

Streaming behavior is controlled by the official repo's `chunk` / `current` /
`future` / `smooth` parameters (`configs/xvc.yaml`, `scripts/infer_single.sh`;
`current=0` selects offline mode).

## Reported numbers (paper, v2)

- **Model-induced latency 240 ms** = 120 ms current segment + 20 ms overlap +
  100 ms future context; **computation latency 58.17 ms** (encode + convert +
  decode). Hardware for the compute figure and whether CPU real-time holds:
  **TBD (Phase 0)**.
- Offline **RTF 0.014** (vs. Seed-VC tiny 0.069, MeanVC 0.094, same setup).
- Streaming: SIM 0.62 (EN) / 0.72 (ZH), WER 3.14 % (EN) / 2.65 % (ZH),
  UTMOS 3.07 (EN) / 2.35 (ZH); SMOS 3.98 (EN) / 3.89 (ZH).

Note the 240 ms model-induced latency is above MeanVC 2's ~110 ms class; X-VC
trades latency for language coverage. Whether `future` can be shrunk without
quality collapse is a Phase 0 question.

## Status

**Phase 0 — evaluation in progress**, tracked in
[#30](https://github.com/m96-chan/babiniku.rs/issues/30): run the official
PyTorch implementation, measure Japanese conversion quality and CPU latency,
and decide go/no-go on a Rust port. Nothing in this repo implements X-VC yet.

- [ ] Official pipeline reproduced (offline + streaming)
- [ ] Japanese source/target quality assessed (tokenizer coverage) — TBD (Phase 0)
- [ ] CPU real-time feasibility (539 M total params vs. driver-only GPU rule) — TBD (Phase 0)
- [ ] Streaming-parameter sweep (`current` / `future` vs. quality) — TBD (Phase 0)

## Planned integration

If Phase 0 passes, X-VC follows the house pattern (see
[docs/meanvc.md](meanvc.md)): a weight-compatible pure-candle port whose module
tree mirrors the upstream implementation, frozen externals behind the
`src/encoders.rs`-style traits (SAC codec, GLM-4-Voice tokenizer, ERes2Net),
safetensors conversion of the official checkpoints, and stage-by-stage golden
tests against the official implementation before it is wired into the
streaming demo. The 539 M-parameter footprint makes CPU viability the gating
question — quantization or GPU-feature gating may be required, both TBD
(Phase 0).

## Citation

```bibtex
@article{zheng2026xvc,
  title   = {X-VC: Zero-shot Streaming Voice Conversion in Codec Space},
  author  = {Zheng, Qixi and Zhao, Yuxiang and Wang, Tianrui and Chen, Wenxi and
             Xu, Kele and Li, Yikang and Chen, Qinyuan and Qiu, Xipeng and
             Yu, Kai and Chen, Xie},
  journal = {arXiv preprint arXiv:2604.12456},
  year    = {2026}
}
```

## Acknowledgements

- [X-VC](https://arxiv.org/abs/2604.12456) (Zheng et al., 2026) — all model
  ideas belong to the original authors; official code and weights at
  [Jerrister/X-VC](https://github.com/Jerrister/X-VC) (MIT).
- [GLM-4-Voice tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer)
  (Zhipu AI) and ERes2Net (ModelScope) — the frozen front-ends X-VC builds on.

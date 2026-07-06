# X-VC

> **Ported and working (quality parity verified); real-time on an idle
> 8-thread CPU (pipelined driver) and on GPU (`--features cuda`, RTF
> ≈ 0.10) — CPU under desktop co-load still falls behind, tracked in
> [#38](https://github.com/m96-chan/babiniku.rs/issues/38).** Notes for
> **X-VC: Zero-shot Streaming Voice Conversion in Codec Space** ([arXiv:2604.12456](https://arxiv.org/abs/2604.12456), Zheng et
> al., 2026), ported to `crates/xvc` in
> [#30](https://github.com/m96-chan/babiniku.rs/issues/30) — see
> [Status](#status) for measured parity and performance.

[![Paper](https://img.shields.io/badge/arXiv-2604.12456-b31b1b.svg)](https://arxiv.org/abs/2604.12456)
[![Status](https://img.shields.io/badge/status-ported-green.svg)](#status)

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
supporting cross-lingual conversion. Japanese coverage of the tokenizer was
**verified in Phase 0**: offline conversion of Japanese speech preserves the
transcript exactly (whisper-small ASR) while locking onto the target F0 —
the Mandarin lock is gone.

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
  decode). Measured locally: GPU trivially real-time (Rust cuda build RTF
  ≈ 0.10); CPU needs the 640/240 preset plus the pipelined driver and an
  otherwise-idle box — see [Status](#status).
- Offline **RTF 0.014** (vs. Seed-VC tiny 0.069, MeanVC 0.094, same setup).
- Streaming: SIM 0.62 (EN) / 0.72 (ZH), WER 3.14 % (EN) / 2.65 % (ZH),
  UTMOS 3.07 (EN) / 2.35 (ZH); SMOS 3.98 (EN) / 3.89 (ZH).

Note the 240 ms model-induced latency is above MeanVC 2's ~110 ms class; X-VC
trades latency for language coverage. Whether `future` can be shrunk without
quality collapse is a Phase 0 question.

## Status

**Ported (Phases 0–1 + demo integration done)**, tracked in
[#30](https://github.com/m96-chan/babiniku.rs/issues/30). `crates/xvc` is a
weight-compatible pure-candle port of every inference stage, each verified
against the official implementation with skip-if-absent golden tests
(`cargo test -p xvc`):

| stage | module | golden parity (max abs vs official CPU fp32) |
|---|---|---|
| preprocessing (volume-norm / 40 Hz HP / pad) + Whisper 128-mel | `xvc::preprocess` | wav bit-exact, mel 3.0e-5 |
| GLM-4-Voice tokenizer (343.6 M) + 12.5→50 Hz semantic adapter | `xvc::tokenizer` | token ids exact, adapter 6.7e-6 |
| ERes2Net speaker encoder | `xvc::speaker` | embedding 3.2e-5 (cos ≈ 1.0) |
| SAC codec (DAC encoder / FVQ / decoder) | `xvc::codec` | codes exact, wav 4.0e-6 |
| prenet fusion (65.3 M, `Decoder_with_upsample`, ratios `[1,1]`) | `xvc::pipeline` | 6.2e-6 |
| MMDiT acoustic converter (42.4 M) | `xvc::converter` | 4.7e-6 |
| frame-condition dB-mel (torchaudio `MelSpectrogram` + `AmplitudeToDB`) | `xvc::preprocess::FrameMelExtractor` | 3.4e-4 |
| **one full streaming window** (chain fixture) | `xvc::pipeline` | wav 1.4e-5 |
| **offline end to end** (out.wav → test.wav) | `XvcEngine::convert_offline` | 1.2e-5, corr 1.000000 |
| **full CPU-preset stream** (640/240/100/20) | `XvcStream` | 7.7e-5, corr 1.000000 (no VQ flips) |

Usage:

- offline: `cargo run --release -p xvc --example convert_xvc -- <source.wav> <reference.wav> <out.wav>`
- live: `cargo run --release -p vc-demo --bin babiniku-demo -- --engine xvc --reference her_voice.wav`
- weights: convert the official checkpoints once with
  `tools/convert_xvc_tokenizer.py`, `tools/convert_xvc_speaker.py`,
  `tools/convert_xvc_generator.py` →
  `ckpt/xvc_{tokenizer,speaker,codec,converter,prenet}.safetensors`
  (~2.1 GiB fp32 total).

**Performance (fp32, 640/240/100/20 streaming preset)** — issue
[#38](https://github.com/m96-chan/babiniku.rs/issues/38) status:

| mode | measured | late |
|---|---|---|
| offline (CPU 8 threads) | RTF **0.69** | — |
| streaming, serial driver (CPU 8 threads) | Σ ≈ 1.75 (semantic 0.50 / convert 0.66 / decode 0.61) | every hop |
| streaming, **pipelined** driver (CPU 8 threads, idle box) | bottleneck stage ≈ **0.92** (stages overlap on 3 threads) | **0** sustained |
| streaming, pipelined (CPU, desktop co-load ≈ 4 cores) | bottleneck ≈ 1.0 | ⚠️ hops go late |
| streaming (**GPU**, `--features cuda`, RTX 5090) | RTF ≈ **0.10** (semantic 0.06 / convert 0.04 / decode 0.01) | **0**, incl. under desktop load |

`XvcPipelinedStream` runs the three window stages on dedicated threads
(bit-identical output to the serial driver; on accelerators it degrades
to a single worker thread — concurrent candle op submission from several
host threads corrupts results, and GPU kernels serialize on one stream
anyway). The cuda build needs the depthwise-conv fast path in
`xvc::tokenizer` (candle's grouped conv1d launches one kernel per
channel). CPU remains the zero-setup default; **on a busy desktop the CPU
margin is not enough for the live mic** — use the cuda build there, or
wait for the causal-caching redesign (#38): bit-close caching of the
official windowed driver is blocked by the tokenizer's window-relative
positional embeddings, so the remaining CPU headroom requires a
numerics-changing causal semantic branch (quality re-validation needed).
q8 quantization is off the table per the maintainer (fp32 only).

- [x] Official pipeline reproduced (offline + streaming)
- [x] Japanese source/target quality assessed — **PASS** (identical ASR
      transcript offline, target F0 locked in all pairs incl. cross-gender)
- [x] CPU real-time feasibility — offline yes (0.69); streaming yes on an
      idle 8-thread box (pipelined driver, late 0), not yet under desktop
      co-load (#38)
- [x] GPU real-time — `-p vc-demo --features cuda`: RTF ≈ 0.10, late 0
      under load (CUDA 13.3, sm_120)
- [x] Streaming-parameter sweep — 640 ms window / 240 ms hop chosen as the
      CPU preset (quality holds; official 2400/120 preset available via
      `StreamConfig::official()`)

Listening references in `ckpt/` (gitignored): `xvc_rust_ja_out_to_test_offline.wav`
(ja → test.wav), `xvc_rust_M06_to_out_offline.wav` (babiniku direction:
male JVS M06 → out.wav voice), `xvc_demo_out.wav` (live-demo streaming
output), plus the Phase-0 official outputs `xvc_ja_*.wav` for A/B.

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

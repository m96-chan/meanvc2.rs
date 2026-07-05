# meanvc2.rs

> Unofficial Rust implementation of the **MeanVC family** of streaming zero-shot voice conversion systems — **MeanVC** ([arXiv:2510.08392](https://arxiv.org/abs/2510.08392)) and **MeanVC 2** ([arXiv:2606.09050](https://arxiv.org/abs/2606.09050)) — built on [candle](https://github.com/huggingface/candle).

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Paper](https://img.shields.io/badge/arXiv-2606.09050-b31b1b.svg)](https://arxiv.org/abs/2606.09050)
[![Status](https://img.shields.io/badge/status-experimental-yellow.svg)](#project-status)

MeanVC 2 (Ma et al., 2026) is a streaming zero-shot voice conversion system that converts a source speaker's voice into that of an arbitrary unseen target speaker with a **110 ms end-to-end first-packet latency** on a single CPU core. It achieves this by combining:

- **Mean flows** — the decoder regresses the *average* velocity along the ODE trajectory, enabling high-quality mel-spectrogram synthesis with a **single network evaluation (1-NFE)** instead of an iterative diffusion loop.
- **Future-receptive chunking (FRC)** — a layer-wise attention-mask schedule for the DiT decoder that grants each 40 ms chunk a bounded receptive field (6 past chunks + 1 future chunk with the paper's defaults), stabilizing short-chunk streaming without autoregressive teacher forcing.
- **Universal timbre token encoder (UTTE)** — instead of extracting timbre from reference mel-spectrograms (which is sensitive to reference audio quality), a set of learnable universal timbre tokens is modulated by a global speaker embedding, and bottleneck features query them via cross-attention to produce *timbre-aware* content features.

This crate implements the trainable core of MeanVC 2 — UTTE, the FRC-scheduled DiT decoder, the mean-flows objective/sampler, and a chunk-by-chunk streaming driver — in pure Rust, plus candle ports of the frozen external models.

## MeanVC (v1) support

The original **MeanVC** (Ma et al., 2025) shares the recognition–synthesis skeleton and the mean-flows decoder, but uses **chunk-wise autoregressive denoising (CARD)** instead of FRC and a reference-mel **MRTE timbre encoder** instead of UTTE. Unlike MeanVC 2, its [official implementation and pretrained checkpoints are public](https://github.com/ASLP-lab/MeanVC) (VC model, Fast-U2++, and a 16 kHz Vocos on [Hugging Face](https://huggingface.co/ASLP-lab/MeanVC)) — so this repo targets a **weight-compatible v1 port** as the fastest path to real conversion, while the v2 implementation stands ready for the official v2 release. Progress is tracked in [#12](https://github.com/m96-chan/meanvc2.rs/issues/12); the model lives in the `v1` module (`cargo run --release --example v1_demo`).

## Architecture

```text
 source wav ──► streaming ASR ──► BNFs ─────────────┐
                (Fast-U2++, external)               ▼
                                              ┌──────────┐    timbre-aware
 reference ──► speaker encoder ──► spk emb ──►│   UTTE   │──► BNFs
    wav        (ECAPA-TDNN, external)    │    └──────────┘      │
                                         │                      ▼
                              noise ε ───┼──────────────► ┌────────────┐
                              (r, t) ────┴───────────────►│ DiT decoder│
                                                          │ (FRC masks)│
                                                          └─────┬──────┘
                                                       1-NFE    ▼
                                        converted wav ◄── vocoder ◄── mel
                                                          (Vocos, external)
```

| Module | Source | Paper section |
|---|---|---|
| FRC attention-mask scheduling | [`src/frc.rs`](src/frc.rs) | §3.2 |
| Universal timbre token encoder | [`src/model/utte.rs`](src/model/utte.rs) | §3.3 |
| DiT decoder (adaLN-Zero blocks) | [`src/model/dit.rs`](src/model/dit.rs), [`src/model/decoder.rs`](src/model/decoder.rs) | §3.1 |
| Mean-flows loss & 1-NFE sampling | [`src/meanflow.rs`](src/meanflow.rs) | §2.1–2.2 |
| Streaming chunk driver | [`src/streaming.rs`](src/streaming.rs) | §3.2 |
| Log-mel front-end | [`src/audio/mel.rs`](src/audio/mel.rs) | §4.1 |
| External-model traits | [`src/encoders.rs`](src/encoders.rs) | §3.1 |

Default hyper-parameters follow §4.1 of the paper: 4 DiT blocks (hidden 512, 2 heads), 32 universal timbre tokens (hidden 256, 4-head cross-attention), 40 ms chunks on 16 kHz audio, and FRC receptive fields `P = [2, 2, 1, 1]`, `F = [1, 0, 0, 0]`.

## Installation

```toml
[dependencies]
meanvc2 = { git = "https://github.com/m96-chan/meanvc2.rs" }
```

The crate currently builds against the [m96-chan/candle](https://github.com/m96-chan/candle) fork of candle (v0.11, `feat/forward-ad-jvp` branch), which adds the forward-mode AD used by the mean-flows objective.

CUDA / Metal acceleration comes from candle's backends:

```sh
cargo build --release --features cuda   # or --features metal
```

## Quick start

### Offline (full-utterance) conversion

```rust
use candle_core::{Device, Tensor};
use meanvc2::{MeanVc2, MeanVc2Config};

let device = Device::Cpu;
let cfg = MeanVc2Config::default();
let model = MeanVc2::load(cfg, "meanvc2.safetensors", &device)?;

// BNFs from your streaming ASR (upsampled to the mel frame rate) and a
// speaker embedding from your speaker encoder — see `meanvc2::encoders`.
let bnf: Tensor = semantic_encoder.extract(&source_wav, 16_000)?;   // [1, time, bnf_dim]
let speaker: Tensor = speaker_encoder.embed(&reference_wav, 16_000)?; // [1, speaker_dim]

let mel = model.convert(&bnf, &speaker)?; // [1, time, 80], 1-NFE
// feed `mel` to a vocoder (e.g. Vocos) to obtain the waveform
```

### Streaming conversion

```rust
use meanvc2::StreamingConverter;

let mut converter = StreamingConverter::new(&model, &speaker)?;
for bnf_chunk in bnf_chunks {           // [1, 4, bnf_dim] per 40 ms chunk
    for mel_chunk in converter.push(&bnf_chunk)? {
        vocoder.synthesize(&mel_chunk)?; // emitted with 1 chunk look-ahead
    }
}
converter.finish()?;                     // flush trailing chunks
```

A runnable end-to-end demo with random weights:

```sh
cargo run --release --example streaming_demo
```

### Training objective

```rust
use meanvc2::meanflow::{mean_flow_loss, sample_rt, JvpMode};

let cond = model.timbre_aware_bnf(&bnf, &speaker)?;
let masks = model.decoder.frc_masks(time, &device)?;   // chunked training
let rt = sample_rt(batch, 0.75, &device)?;             // r = t 75% of the time
let out = mean_flow_loss(&model, &mel, &cond, &speaker, Some(&masks), &rt, JvpMode::Exact)?;
out.loss.backward()?;
```

The JVP inside the mean-flows target is computed **exactly** with forward-mode AD (`candle_core::forward_ad::jvp`, added on the candle fork's `feat/forward-ad-jvp` branch). `JvpMode::FiniteDifference(delta)` is kept for cross-checking.

## Real-time TUI demo with a virtual microphone (Linux)

`meanvc-demo` converts your microphone in real time and exposes the result as a **virtual microphone** (`MeanVC-Virtual-Mic`, via a PipeWire/PulseAudio null sink) selectable from any app:

```sh
cargo run --release --features demo --bin meanvc-demo -- \
    --reference target_voice.wav --voice-print voice_print.safetensors
```

TUI shows level meters, per-stage RTF, and supports `p` (passthrough A/B), `l` (loopback monitor — hear the converted voice on your speakers), `q` (quit; removes the virtual device). `--monitor` starts with the loopback on. `--wav file.wav` streams a file instead of the mic, `--headless` / `--out out.wav` / `--duration N` support scripted runs. Requires the checkpoints under `ckpt/` (see `examples/convert_v1.rs`) and `pactl`. Measured on a single CPU: VC stage RTF ≈ 0.57 and vocoder RTF ≈ 0.57 running pipelined — sustained real time with ~0.6 s latency.

## External components

MeanVC 2 trains only the UTTE and the DiT decoder. The three frozen components are pretrained external models, abstracted as traits in [`src/encoders.rs`](src/encoders.rs) with pure-candle backends in [`src/backends/`](src/backends/):

| Component | Trait | Backend | Used in the paper |
|---|---|---|---|
| Semantic (BNF) extractor | `SemanticEncoder` | `backends::FastU2pp` (WeNet U2++ conformer port) | [Fast-U2++](https://github.com/wenet-e2e/wenet) (WeNet), 80 ms chunks, 40 ms frames |
| Speaker encoder | `SpeakerEncoder` | `backends::Ecapa` (SpeechBrain-layout port) | [ECAPA-TDNN](https://github.com/speechbrain/speechbrain), 192-dim |
| Vocoder | `Vocoder` | `backends::Vocos` (ConvNeXt + ISTFT port) | [Vocos](https://github.com/gemelo-ai/vocos), 16 kHz |

Each backend's module tree mirrors its upstream implementation so converted safetensors checkpoints map 1:1 (`FastU2pp::load` / `Ecapa::load` / `Vocos::load`); checkpoint conversion scripts and golden-output validation are tracked in [#4](https://github.com/m96-chan/meanvc2.rs/issues/4). The full wav-to-wav pipeline runs end to end with random weights:

```sh
cargo run --release --example pipeline_demo
```

## Project status

This is an **unofficial, experimental** implementation written from the paper — the official source code and weights had not been released at the time of writing (the authors state they will release both; audio samples are [here](https://aslp-lab.github.io/MeanVC2/)).

- [x] FRC layer-wise attention masks (§3.2)
- [x] UTTE with learnable tanh-fused priors (§3.3)
- [x] DiT decoder with adaLN-Zero and dual-timestep `(r, t)` conditioning
- [x] Mean-flows loss (Eq. 4) and 1-NFE sampling
- [x] Streaming converter with bounded look-ahead and cached per-chunk noise
- [x] Log-mel front-end
- [x] Vocos / Fast-U2++ / ECAPA-TDNN inference backends (candle ports; checkpoint conversion + golden tests in [#8](https://github.com/m96-chan/meanvc2.rs/issues/8))
- [x] MeanVC v1 model (`meanvc2::v1`): MRTE, CARD, RoPE + rms-qk-norm ChunkDiT — official parameter tree, 14.1M params vs the paper's 14M ([#12](https://github.com/m96-chan/meanvc2.rs/issues/12))
- [x] MeanVC v1 official weights load + real wav-to-wav example (`cargo run --release --example convert_v1`; perceptual quality pending the Python A/B check in [#14](https://github.com/m96-chan/meanvc2.rs/issues/14))
- [ ] MeanVC 2 pretrained weights (official release pending; training loop [#3](https://github.com/m96-chan/meanvc2.rs/issues/3) as fallback)

Known deviations from the paper (details in the module docs):

- The JVP inside the mean-flows target (Eq. 3) is computed exactly with forward-mode AD implemented on the candle fork (`feat/forward-ad-jvp` branch, tracked in [#1](https://github.com/m96-chan/meanvc2.rs/issues/1)); a finite-difference mode remains available for cross-checking.
- Details the paper leaves unspecified (MLP activations, adaLN layout, mel settings, how the speaker embedding conditions the decoder) follow common DiT/flow-matching practice; the default config counts ~25 M parameters vs. the paper's 18 M.

## Development

```sh
cargo test          # unit + integration tests (shapes, masks, streaming)
cargo clippy --all-targets
cargo run --release --example streaming_demo
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{ma2026meanvc2,
  title   = {MeanVC 2: Robust Low-Latency Streaming Zero-Shot Voice Conversion},
  author  = {Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and
             Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng},
  journal = {arXiv preprint arXiv:2606.09050},
  year    = {2026}
}
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

## Acknowledgements

- [MeanVC 2](https://arxiv.org/abs/2606.09050) by the ASLP@NPU group — all model ideas belong to the original authors.
- [candle](https://github.com/huggingface/candle) — the ML framework this crate is built on (via the [m96-chan/candle](https://github.com/m96-chan/candle) fork).
- [MeanFlow](https://arxiv.org/abs/2505.13447) (Geng et al., 2025) — the one-step generative modeling formulation.

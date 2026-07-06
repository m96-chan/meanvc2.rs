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
| FRC attention-mask scheduling | [`crates/meanvc/src/frc.rs`](../crates/meanvc/src/frc.rs) | §3.2 |
| Universal timbre token encoder | [`crates/meanvc/src/model/utte.rs`](../crates/meanvc/src/model/utte.rs) | §3.3 |
| DiT decoder (adaLN-Zero blocks) | [`crates/meanvc/src/model/dit.rs`](../crates/meanvc/src/model/dit.rs), [`crates/meanvc/src/model/decoder.rs`](../crates/meanvc/src/model/decoder.rs) | §3.1 |
| Mean-flows loss & 1-NFE sampling | [`crates/meanvc/src/meanflow.rs`](../crates/meanvc/src/meanflow.rs) | §2.1–2.2 |
| Streaming chunk driver | [`crates/meanvc/src/streaming.rs`](../crates/meanvc/src/streaming.rs) | §3.2 |
| Log-mel front-end | [`crates/vc-core/src/audio/mel.rs`](../crates/vc-core/src/audio/mel.rs) | §4.1 |
| External-model traits | [`crates/vc-core/src/encoders.rs`](../crates/vc-core/src/encoders.rs) | §3.1 |

Default hyper-parameters follow §4.1 of the paper: 4 DiT blocks (hidden 512, 2 heads), 32 universal timbre tokens (hidden 256, 4-head cross-attention), 40 ms chunks on 16 kHz audio, and FRC receptive fields `P = [2, 2, 1, 1]`, `F = [1, 0, 0, 0]`.

## Installation

```toml
[dependencies]
# package `meanvc`, library name `meanvc2` (use meanvc2::…)
meanvc = { git = "https://github.com/m96-chan/babiniku.rs" }
```

The crate lives in the `crates/meanvc` member of the babiniku.rs cargo workspace (shared traits/DSP in `crates/vc-core`, re-exported as `meanvc2::{audio, encoders, Error, Result}`). It currently builds against the [m96-chan/candle](https://github.com/m96-chan/candle) fork of candle (v0.11, `feat/forward-ad-jvp` branch), which adds the forward-mode AD used by the mean-flows objective.

CUDA / Metal acceleration comes from candle's backends:

```sh
cargo build --release -p meanvc --features cuda   # or --features metal
```

## Quick start

The APIs below are the **MeanVC 2** interfaces and run with random weights until the official v2 release; for conversion that works today with the released **v1** checkpoints, see [MeanVC v1 usage](#meanvc-v1-usage) below and the [TUI demo](#real-time-tui-demo-with-a-virtual-microphone-linux).

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

## MeanVC v1 usage

Offline (full utterance):

```rust
use meanvc2::v1::{MeanVc1, MeanVc1Config};

let model = MeanVc1::load(MeanVc1Config::default(), "ckpt/model_200ms.safetensors", &device)?;
let timbre_src = model.timbre_cond(&bnf, &prompt_mel, &voice_print)?; // MRTE
let mel = model.sample(&bnf, &prompt_mel, &voice_print)?; // chunked CARD, 1-NFE
```

Online (chunk-by-chunk streaming, the path the TUI demo uses):

```rust
use meanvc2::v1::{KvCache, MelV1, KaldiFbank, interpolate_linear};
use meanvc2::backends::{FastU2pp, FastU2ppConfig};

let asr = FastU2pp::load(FastU2ppConfig::official_meanvc1(), "ckpt/fastu2pp.safetensors", &device)?;
let mut asr_state = asr.stream();               // incremental fbank/BNF decode
let mut kv = KvCache::default();                // per-block CARD attention cache
let mut prev_mel = None;
for (q, samples_200ms) in mic_chunks.enumerate() {
    // Returns Some once a full 23-frame analysis window is buffered.
    let Some(bnf) = asr.forward_chunk(&fbank.compute(&samples_200ms, &device)?, &mut asr_state)? else { continue };
    let cond = interpolate_linear(&bnf, 4)?;
    let timbre = model.timbre_cond(&cond, &prompt_mel, &voice_print)?;
    let noise = Tensor::randn(0f32, 1f32, (1, 20, 80), &device)?;
    let u = model.forward_stream(&noise, &timbre, &voice_print, prev_mel.as_ref(), q * 20, &mut kv)?;
    let mel = (noise - u)?;                      // -> Vocos ((mel+1)/2) -> 200 ms of audio
    prev_mel = Some(mel);
}
```

See [`crates/vc-demo/src/bin/demo.rs`](../crates/vc-demo/src/bin/demo.rs) for the complete real-time loop (gating, denoising, pitch shift, virtual mic) and [`crates/meanvc/examples/convert_v1.rs`](../crates/meanvc/examples/convert_v1.rs) for the offline pipeline.

## Real-time TUI demo with a virtual microphone (Linux)

`babiniku-demo` converts your microphone in real time and exposes the result as a **virtual microphone** (`Babiniku-Virtual-Mic`, via a PipeWire/PulseAudio null sink) selectable from any app:

```sh
cargo run --release -p vc-demo --features wavlm --bin babiniku-demo -- \
    --reference target_voice.wav
```

With the `wavlm` feature the 256-dim voice print is computed natively from the reference audio (export the ONNX model once with `tools/export_wavlm_onnx.py`); without it, pass a precomputed `--voice-print file.safetensors`.

TUI shows level meters, per-stage RTF, and supports `p` (passthrough A/B), `l` (loopback monitor — hear the converted voice on your speakers), `[` / `]` (pitch shift ±0.5 semitone, Signalsmith Stretch after the vocoder — useful when the target F0 sits outside the model's training range), `,` / `.` (in-process RNNoise mix ±10 %), `q` (quit; removes the virtual device). `--monitor` starts with the loopback on. `--denoise` inserts PipeWire's WebRTC noise suppression in front of the microphone (recommended for noisy mics; `--input-device` selects a specific source). `--wav file.wav` streams a file instead of the mic, `--headless` / `--out out.wav` / `--duration N` support scripted runs. Requires the checkpoints under `ckpt/` at the workspace root (see `crates/meanvc/examples/convert_v1.rs`) and `pactl`. The ASR stage streams incrementally with `FastU2pp::forward_chunk` (per-layer attention K/V + conv caches, exact WeNet `forward_chunk` parity — issue #9). Measured with `RAYON_NUM_THREADS=8`: per-stage RTF ≈ 0.06 (ASR) / 0.04 (VC) / 0.06 (vocoder), late = 0 sustained.

## Performance notes

candle's CPU matmuls run on a rayon pool that defaults to **all logical cores**; on SMT (hyper-threaded) machines the resulting contention makes the small per-chunk workloads ~3–10× slower. Set `RAYON_NUM_THREADS` to the number of **physical** cores for the best CPU latency — measured on an 8c/16t machine ([#19](https://github.com/m96-chan/meanvc2.rs/issues/19)): demo vocoder chunk RTF 0.57 → 0.06, offline `convert_v1` end-to-end 456 ms → 226 ms.

- `babiniku-demo` and the `convert_v1` example pin the pool to physical cores automatically when `RAYON_NUM_THREADS` is unset.
- The other offline examples, and the library itself, use candle's default; set the variable yourself, e.g. `RAYON_NUM_THREADS=8 cargo run --release --example streaming_demo`.
- When embedding the crate, set `RAYON_NUM_THREADS` (or configure the global rayon pool) before the first tensor op — the pool size is fixed at first use.

## External components

MeanVC 2 trains only the UTTE and the DiT decoder. The three frozen components are pretrained external models, abstracted as traits in [`crates/vc-core/src/encoders.rs`](../crates/vc-core/src/encoders.rs) with pure-candle backends in [`crates/meanvc/src/backends/`](../crates/meanvc/src/backends/):

| Component | Trait | Backend | Used in the paper |
|---|---|---|---|
| Semantic (BNF) extractor | `SemanticEncoder` | `backends::FastU2pp` (WeNet U2++ conformer port) | [Fast-U2++](https://github.com/wenet-e2e/wenet) (WeNet), 80 ms chunks, 40 ms frames |
| Speaker encoder | `SpeakerEncoder` | `backends::Ecapa` (SpeechBrain-layout port) | [ECAPA-TDNN](https://github.com/speechbrain/speechbrain), 192-dim |
| Voice print (v1) | `SpeakerEncoder` | `backends::WavLmSv` (ONNX Runtime, feature `wavlm`) | WavLM-Large SV ([UniSpeech](https://github.com/microsoft/UniSpeech)), 256-dim |
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
- [x] MeanVC v1 official weights load + real wav-to-wav example (`cargo run --release --example convert_v1`), with stage-by-stage PyTorch parity locked in by golden tests against the official implementation — DiT forward, KV-cache streaming, mel, Vocos, kaldi fbank, and the BNF pipeline ([#14](https://github.com/m96-chan/meanvc2.rs/issues/14); fixtures via `tools/gen_v1_fixtures.py`, see [`tools/README.md`](tools/README.md))
- [x] Incremental Fast-U2++ streaming: `FastU2pp::forward_chunk` with per-layer attention-K/V and depthwise-conv caches, matching the official TorchScript chunked decode to 3e-6 (`tests/asr_streaming.rs`; chunk-cached Vocos still open in [#9](https://github.com/m96-chan/meanvc2.rs/issues/9))
- [ ] MeanVC 2 pretrained weights (official release pending; training loop [#3](https://github.com/m96-chan/meanvc2.rs/issues/3) as fallback)

Known deviations from the paper (details in the module docs):

- The JVP inside the mean-flows target (Eq. 3) is computed exactly with forward-mode AD implemented on the candle fork (`feat/forward-ad-jvp` branch, tracked in [#1](https://github.com/m96-chan/meanvc2.rs/issues/1)); a finite-difference mode remains available for cross-checking.
- Details the paper leaves unspecified (MLP activations, adaLN layout, mel settings, how the speaker embedding conditions the decoder) follow common DiT/flow-matching practice; the default config counts ~25 M parameters vs. the paper's 18 M.

## Development

```sh
cargo test --workspace          # unit + integration tests (shapes, masks, streaming)
cargo clippy --workspace --all-targets
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

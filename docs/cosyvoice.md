# CosyVoice2

A weight-compatible pure-candle port of the **VC path** of
[CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)
([arXiv:2412.10117](https://arxiv.org/abs/2412.10117)) — the FSQ
speech-token tokenizer, causal conformer + CFM flow, and HiFT vocoder,
with the text LLM entirely bypassed (`inference_vc` in the official repo
never touches it). Ported checkpoint: `CosyVoice2-0.5B` (VC-path ≈ 265 M
params, 24 kHz).

Unlike [Seed-VC](seedvc.md), **code and weights are Apache-2.0**
(verified in issue [#71](https://github.com/m96-chan/babiniku.rs/issues/71)),
so `crates/cosyvoice` ships in the **default** MIT OR Apache-2.0 build —
no GPL feature gate.

## Why we care: a permissive-license alternative to Seed-VC

Issue #71's Phase 0 recon ran the official implementation through the
issue-#42 needle torture protocol and measured:

- **Needle-clean**, same as Seed-VC's BigVGAN line — HiFT (NSF +
  iSTFT) has no window-local decoder pathology, confirmed again on the
  Rust port (`cargo run -p cosyvoice --example offline_convert`, scanned
  with the project's `|y|/3ms-RMS > 4` detector: strict-tier clean on
  every torture file, same as the official run).
- **Apache-2.0 throughout** — the headline difference from Seed-VC.
- GPU-bound like Seed-VC (CPU RTF 1.3+, not real-time); needs
  `--features cuda` or `--features metal` for live use.

## Architecture (VC path only — the LLM/TTS path is not ported)

| stage | module | shape notes |
|---|---|---|
| tokenizer | FSQ speech tokenizer (`speech_tokenizer_v2`), S3Tokenizer-v2 architecture: FSMN-attention Whisper-style encoder + FSQ head (8×3-level ⇒ vocab 6561) | whisper 128-mel @ 16 kHz → 25 Hz tokens; **full (non-causal) attention** — no incremental tokenization |
| speaker | **CAM++** (fresh port from the Apache-2.0 3D-Speaker reference, *not* copied from `crates/seedvc`'s GPL copy) | kaldi fbank 80 → FCM + D-TDNN/CAM → 192-d x-vector |
| encoder | `UpsampleConformerEncoder`: 3-token pre-lookahead conv, 6-layer rel-pos conformer (no macaron/conv modules), nearest ×2 upsample, 4 more layers | 25 Hz tokens → 50 Hz → 80-d `mu` via a linear proj |
| flow | `CausalConditionalCFM`: 10-step Euler ODE, CFG 0.7, **fixed seed-0 noise** (shipped in the checkpoint as `rand_noise`, sliced per-length — bit-comparable with the official run) | causal U-Net estimator, channels=[256], 4+12+4 resnet/transformer blocks |
| vocoder | **HiFT** (NSF source-filter + iSTFT HiFi-GAN, arXiv:2309.09493) | mel 80 @ 24 kHz → 24 kHz audio, `ConvRNN` F0 predictor, 9-harmonic NSF source |

Known deviation, documented in `crates/cosyvoice/src/campplus.rs`: the
official `campplus.onnx` was traced at a fixed 200-frame input, so it
drifts from the true dynamic CAM++ model at other lengths (cos ≈
0.91–0.998 depending on length). This port implements the correct
dynamic model — exact at 200 frames, cosine-close everywhere else.

## Status: golden parity (`cargo test -p cosyvoice`, skip-if-absent)

| stage | tolerance vs official |
|---|---|
| whisper mel-128 / kaldi fbank-80 | max abs < 2e-4 / < 2e-3 |
| FSQ tokenizer | ≤ 1 % stray tokens (FSQ rounding sits on hard decision boundaries) |
| CAM++ | exact at the ONNX trace length (200 frames); cos > 0.995 at full length |
| conformer encoder (`mu`) | cos > 0.9999, max abs < 2e-2 |
| CFM mel (offline + one streaming chunk) | cos > 0.999 |
| HiFT (F0 / NSF source / audio) | max abs < 1e-2 / < 1e-3 / < 2e-2, corr > 0.999 |
| **offline end-to-end wiring** (official tensors through our flow+hift) | corr > 0.999 |

The offline-pipeline test suite deliberately checks *wiring* (token
concatenation order, argument threading) against exact official tensors
separately from *feature-extraction fidelity* (cosine similarity) — see
`crates/cosyvoice/src/pipeline.rs` module docs for why: HiFT's NSF source
integrates phase via `cumsum`, so a single misjudged voiced/unvoiced
frame permanently shifts the harmonic phase for the rest of the clip — a
sub-0.01 % mel deviation (e.g. a different but still-correct resampler)
can collapse whole-clip audio correlation even though every stage is
individually correct. This is a property of the reference algorithm, not
a bug in the port.

## Streaming (live TUI)

CosyVoice2's **own** `inference_vc(..., stream=True)` doesn't apply to
live mic input: it renders mel/audio incrementally from an
already-fully-known source token sequence — the non-causal FSQ tokenizer
runs once over the *entire* source clip before any chunked rendering
begins. That's a dead end for unbounded live audio (flagged as an open
risk in the #71 recon).

`CosyVoiceStream` instead follows the same shape as
[`SeedVcStream`](seedvc.md): a sliding source window (3.0 s context +
1.0 s new block) is **re-tokenized and re-rendered from scratch every
hop**, and the newly-settled tail is crossfaded (80 ms, raised cosine) at
HiFT's native 24 kHz before an exact ×2 resample to 48 kHz. Unlike
Seed-VC's DiT, HiFT's GAN decoder has no run-to-run diffusion variance,
so a plain crossfade — no SOLA phase search — is enough to hide the seam
(verified needle-clean on the TUI's own output).

Cost: the flow reruns over the whole window every hop (not incremental),
so this is **CUDA/Metal-only for live use**, same as Seed-VC.

**Field bug, fixed:** an early version drained the *entire* rolling
buffer on every hop instead of only the new block, which silently zeroed
`context` out of the pipeline — every hop tokenized/encoded a bare,
history-free 1.0 s slice. An ambiguous slice (near-silence / background
noise with no anchoring speech) was enough for HiFT's F0 predictor to
lock onto a spurious low frequency and produce a loud drone. Reproduced
with a synthetic speech→room-tone wav (`stream_tail_probe` example):
per-hop RMS jumped 0.04→0.31→0.41 and the spectral peak collapsed to
33–37 Hz the moment the window went silence-dominated — in the streaming
path *only*; neither the official implementation nor this crate's own
offline single-pass conversion showed it on the same audio, confirming
it was a streaming-driver bug, not a model or porting issue. Fixed by
tracking buffered-but-unconsumed input separately from the sliding
context window (mirroring `crates/seedvc/src/stream.rs`'s `pending`
counter) and narrowing each hop's emission to the new block only instead
of the whole re-rendered window.

## Usage

```sh
# one-time: convert the official CosyVoice2-0.5B checkpoint
python tools/convert_cosyvoice.py --cosyvoice-dir <CosyVoice2-0.5B snapshot> --out ckpt
# → ckpt/cosyvoice_{tokenizer,campplus,flow,hift,mel}.safetensors (~1.1 GB fp32)

# build + run (default build — no feature flag, Apache-2.0)
cargo run --release -p babiniku --features cuda --bin babiniku -- \
    --engine cosyvoice --reference her_voice.wav --monitor --denoise
```

- Golden fixtures for development: `tools/gen_cosyvoice_fixtures.py`
  (runs the official implementation — per CLAUDE.md it stays Python by
  design). Debug/bench example: `offline_convert`.
- Block/context/crossfade are tuned via `cosyvoice::StreamConfig`
  (no CLI knob yet — see issue #75 for follow-ups).

## Performance (RTX-class GPU, fp32)

| mode | measured (RTX 5090) |
|---|---|
| offline (`offline_convert`) | RTF ≈ 0.12–0.17 |
| streaming, 1.0 s block, 3.0 s context | RTF ≈ 0.27–0.29 per hop, `late 0` |
| algorithmic latency | ~1.0 s block + 80 ms crossfade |

CPU is **not** real-time (RTF 1.3+ offline, ~5 streaming); `meanvc`
remains the CPU baseline and `xvc` the low-latency CUDA alternative.

## Citation

```bibtex
@article{du2024cosyvoice2,
  title={CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}
```

## Acknowledgements

- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) — the
  official implementation and released weights (Apache-2.0).
- [xingchensong/S3Tokenizer](https://github.com/xingchensong/S3Tokenizer) —
  the Apache-2.0 torch reference for the FSQ tokenizer architecture.
- [alibaba-damo-academy/3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker) —
  the Apache-2.0 CAM++ reference this port's speaker encoder is written
  from (fresh implementation, weight-compatible with `campplus_cn_common`).

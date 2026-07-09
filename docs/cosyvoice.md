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
  design). Debug/bench examples: `offline_convert`, `stream_tail_probe`
  (per-hop RMS/spectral inspection), `speaker_sim_probe` (CAM++
  cosine-similarity check: does the output actually move toward the
  reference?), `ref_align_probe` (prompt token/mel frame-count
  self-consistency), `similarity_over_time` / `pairwise_similarity`
  (per-window CAM++ trace and self-consistency matrix — field-debugging
  a recorded session), `context_ab` (A/B two `context` values on the
  same material).
- `--cosyvoice-context-s <seconds>` (default 3.0) and
  `--cosyvoice-crossfade-ms <ms>` (default 80) override
  `cosyvoice::StreamConfig` from the CLI. More context can stabilize
  speaker conditioning on harder material at roughly proportional extra
  compute per hop — field-tested mixed results (§below), not a
  guaranteed win.

## Performance (RTX-class GPU, fp32)

| mode | measured (RTX 5090) |
|---|---|
| offline (`offline_convert`) | RTF ≈ 0.12–0.17 |
| streaming, 1.0 s block, 3.0 s context | RTF ≈ 0.27–0.4 per hop, `late 0` |
| algorithmic latency | ~1.0 s block + 80 ms crossfade |

CPU is **not** real-time (RTF 1.3+ offline, ~5 streaming); `meanvc`
remains the CPU baseline and `xvc` the low-latency CUDA alternative.

GPU load is genuinely higher than a minimal design: the flow reruns
CFM over the **whole** `context + block` window every hop (§Streaming
above) — roughly 4× the compute a hop-incremental design would need.
This is a real, current cost of the sliding-window approach, not a
leak; lowering `context` in `StreamConfig` trades this down at the
cost of encoder/tokenizer stability at the block boundary (no CLI knob
yet, see issue #75).

**Speaker-similarity check (field report 2026-07):** a report that
converted output "sounds natural but far from the reference" led to a
quantitative check with `speaker_sim_probe` — CAM++ embedding cosine
similarity between the converted output and the reference vs. the
original source. On a cross-gender pair (F19→M06 JVS clips), output-vs-
reference similarity was **0.72–0.74** against a 0.06 source-vs-
reference baseline — i.e. speaker conditioning is demonstrably pulling
the output strongly toward the target, both offline and through the
live TUI streaming path. A real (if minor) self-consistency bug was
found and fixed along the way — `feat_len()` could drift a frame or two
from `tokens_len() * 2` since the two are computed via independently
resampled/STFT'd signals, shifting the prompt/source boundary
`Flow::cfm` expects — but it was not large enough to explain the
report. **Current read: CosyVoice2's zero-shot speaker-cloning fidelity
in this VC path is probably inherently weaker than Seed-VC's** (never
formally A/B'd for timbre similarity in the #71 recon, which focused on
needle-scanning and RTF); a longer/cleaner reference clip and a direct
`--engine seedvc` comparison on the same material are the next things
to try.

**Follow-up (same field report):** a real 80 s live session recording,
analyzed with the new `similarity_over_time` / `pairwise_similarity`
examples against the *actual* reference used (`ref_trimmed.wav`, not
`ref_stage1.wav` — an earlier round of this investigation used the
wrong file), confirmed the user's impression: similarity dipped and
recovered throughout, correlating with **voiced ratio (r ≈ +0.49)** and
**F0 stability (r ≈ −0.44)** per 3 s window — clearly-voiced, pitch-
stable speech converts closer to the reference; consonant-heavy,
pitch-wavering, or otherwise less-clearly-voiced speech (fast/casual
delivery, laughing, trailing intonation) converts with weaker
conditioning and reads as closer to the source's own voice. This lines
up with a `--wav`-driven control test on clean, comparable material
consistently landing at 0.6–0.8 similarity through the identical
streaming code path — the *mechanism* isn't broken (no literal
passthrough leakage either: high-frequency energy above 12 kHz, where
raw 48 kHz mic content would show but synthesized 24 kHz-native HiFT
output wouldn't, stayed near zero throughout the recording) — but
**live, expressive/casual speech is a harder regime for this VC path's
speaker conditioning to hold onto than clean read-speech**, and the
from-scratch sliding-window re-tokenization (§Streaming) plausibly
amplifies this versus a hypothetical longer-context/incremental design.
Not a discrete bug to fix; a characteristic to keep tuning around
(candidates: larger `context`, a cleaner/more expressive reference
clip, or revisiting whether Seed-VC's DiT line is simply more robust
here).

**`context` A/B (same field report):** tested 3.0 s (default) vs. 6.0 s
on noisy material with `context_ab` — mean similarity barely moved
(0.673 → 0.679) but the worst-case dip improved (0.500 → 0.569) *and*
a couple of previously-fine windows got mildly worse at 6.0 s. Net:
inconsistent, not a clear win, at ~2× the compute per hop. Shipped
`--cosyvoice-context-s`/`--cosyvoice-crossfade-ms` anyway so this is a
quick CLI experiment rather than a rebuild — worth trying per-reference
material, but don't expect it to be a decisive fix on its own.

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

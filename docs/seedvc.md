# Seed-VC

A weight-compatible pure-candle port of
[Seed-VC](https://github.com/Plachtaa/seed-vc)
([arXiv:2411.09943](https://arxiv.org/abs/2411.09943)) — zero-shot voice
conversion with a diffusion transformer sampled by conditional flow
matching. The ported checkpoint is
`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned` (98.5 M,
22.05 kHz), the model that won the maintainer's blind A/B against X-VC.

> **License — read this first.** The upstream code **and** the released
> weights are **GPL-3.0**, so [`crates/seedvc`](../crates/seedvc) is
> GPL-3.0 and strictly opt-in behind the `seedvc` cargo feature. Every
> binary built *without* the feature stays MIT OR Apache-2.0; a binary
> built *with* it is GPL when distributed. The upstream GitHub
> repository is archived (read-only) — we port knowledge, not a
> dependency.

## Why we care: the needle-free, most-natural engine

Two independent reasons, both field-verified (issue
[#49](https://github.com/m96-chan/babiniku.rs/issues/49)):

1. **No decoder-needle pathology.** The issue-#42 カチカチ saga traced
   to the X-VC SAC decoder emitting window-local tanh-scale pulses —
   present in its official implementation, fought here with a
   three-layer defense stack. Seed-VC's BigVGAN line produces **zero
   needles on the same torture protocol** (amplitude-stepped speech,
   sokuon bursts): this engine ships with *no* needle guard and *no*
   cross-window verification, because it needs none.
2. **More natural by ear.** Continuous Whisper-encoder features (vs
   X-VC's 16384-entry VQ tokens) preserve micro-prosody, and the
   10-step CFM buys quality that a one-step decoder cannot — the
   maintainer's A/B and VRChat listeners agree.

The trade: it needs a GPU (the DiT runs ~20 forwards per hop) and the
GPL feature gate.

## Architecture (as shipped in the checkpoint)

| stage | module | shape notes |
|---|---|---|
| content | Whisper-small **encoder** (fp32) | 16 kHz → 50 Hz × 768; pad-to-30 s, trimmed to `n/320+1` frames |
| speaker | **CAM++** (funasr `campplus_cn_common`, 6.9 M) | kaldi fbank 80 → D-TDNN → 192-d embedding |
| length regulation | `InterpolateRegulator` (4.9 M) | 768→512, nearest-interpolated to the mel grid, 4×[conv k3 + GroupNorm(1) + Mish] |
| converter | **DiT** 512 dim × 13 layers × 8 heads (U-ViT skips, AdaptiveRMSNorm on t, SwiGLU) + 8-layer gated **WaveNet** refiner | CFM Euler sampling, 10 steps offline / 6 live, cfg 0.7 as one stacked B=2 forward |
| mel | 80 bins @ 22 050 Hz, n_fft 1024 / hop 256, **fmax "None" = Nyquist** | HiFiGAN-style, `ln(clamp(x, 1e-5))` |
| vocoder | **BigVGAN v2** `22khz_80band_256x` | alias-free Snake (kaiser-sinc 2× up/down), 6 stages, clamp(−1, 1) |

Note the checkpoint **disagrees with the preset yml** shipped alongside
it (which describes a 384×9 model with `time/style_as_token: true`);
the weight shapes are authoritative and the port follows them.

## Status: golden parity (`cargo test -p seedvc`, skip-if-absent)

| stage | max abs vs official |
|---|---|
| mel front-end | < 1e-3 |
| Whisper-small encoder | front-end 2.7e-4 / encoder < 2e-3 |
| length regulator | 1.1e-4 |
| CAM++ | 5.4e-6 (cos 0.9999994) |
| DiT + CFM, **full 10-step trajectory** | 4.7e-4, correlation 1.000000 |
| BigVGAN | 6.7e-6 |
| **offline end-to-end** (raw audio → wave) | correlation 0.9927 with in-test stage attribution |

Residuals against the e2e fixture are TF32-noise-limited (the fixture
was generated on CUDA with TF32 convolutions; strict-fp32 parity is
1e-4-class everywhere). Porting traps found and documented in code:
PyTorch's nearest interpolation uses a **float32** scale factor; the
checkpoint's `style_encoder` is dead weight (the real speaker encoder
is the standalone CAM++ bin); an upstream `padding=` kwarg is silently
discarded (reflect padding applies); torchaudio's resample kernel has
an **asymmetric tap range**.

## Streaming (live TUI)

`SeedVcStream` follows the official real-time GUI's scheme with the
lessons of #42/#50 baked in:

- sliding context: **2.5 s for Whisper**, but only **0.5 s + the 320 ms
  block** flows through the regulator/CFM/vocoder per hop (running the
  full context through the DiT tripled the sequence and blew the
  real-time budget);
- reference prompt capped at **4 s** (it occupies the DiT sequence on
  every step of every hop);
- **fixed CFM noise** per stream — fresh noise per hop decorrelated the
  texture at block joints;
- **SOLA splicing** (10 ms search) — adjacent diffusion renders agree
  in envelope but not phase, and a plain crossfade knocks at the block
  rate;
- output is genuine 22.05 kHz bandwidth resampled **straight to
  48 kHz** — no BWE upsampler, no declick stack.

Steady state on CUDA: **~0.25 s per 0.32 s block** at 6 CFM steps
(`--cfm-steps` trades quality vs headroom), `late 0`.

## Usage

```sh
# one-time: convert the official checkpoints (inside a seed-vc clone's env)
python tools/convert_seedvc.py --seedvc-dir <seed-vc clone> --out ckpt
# → ckpt/seedvc_{dit,regulator,campplus,bigvgan,whisper}.safetensors (~1.2 GB fp32)

# build + run (GPL build!)
cargo run --release -p babiniku --features cuda,seedvc --bin babiniku -- \
    --engine seedvc --reference her_voice_48k.wav --monitor --denoise
```

- **Use a 48 kHz reference.** The engine resamples it internally, but a
  16 kHz file starves the timbre prompt above 8 kHz (measured: the
  8–11 kHz output deficit drops from +23 dB to +5 dB with a 48 k
  reference) and caps the voice-profile EQ target
  ([#62](https://github.com/m96-chan/babiniku.rs/issues/62)).
- Knobs with particular relevance here: `--cfm-steps` (live default 6),
  `--out-denoise` / `<` `>` (RNNoise on the output; the CFM leaves a
  faint noise bed), `--profile-eq` / `(` `)` (adapts the output
  spectrum toward the reference's real LTAS).
- Golden fixtures for development:
  `tools/gen_seedvc_fixtures.py`, `gen_seedvc_whisper_fixture.py`,
  `gen_seedvc_dit_fixture.py`, `gen_seedvc_bigvgan_fixture.py` (all run
  the official implementation — per CLAUDE.md they stay Python by
  design). Debug/bench examples: `stream_probe`, `offline_convert`.

## Performance (RTX-class GPU, fp32)

| mode | measured |
|---|---|
| offline (`offline_convert`, 10 steps) | RTF ≈ 0.18 for the whole chain |
| streaming, 6 steps, 320 ms block | 0.24–0.25 s per block (≈ 40 % headroom), `late 0` |
| algorithmic latency | block 320 ms + SOLA/crossfade ≈ 40 ms + output resample |

CPU is **not** real-time for this engine (a single hop costs seconds);
`meanvc` remains the CPU baseline and `xvc` the low-latency CUDA
alternative.

## Citation

```bibtex
@article{liu2024seedvc,
  title={Zero-shot Voice Conversion with Diffusion Transformers},
  author={Liu, Songting},
  journal={arXiv preprint arXiv:2411.09943},
  year={2024}
}
```

## Acknowledgements

- [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — the
  official implementation and released weights (GPL-3.0).
- OpenAI Whisper (encoder weights, Apache-2.0), NVIDIA BigVGAN v2
  (MIT), FunASR CAM++.

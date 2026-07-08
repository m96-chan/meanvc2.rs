<div align="center">

<img src="assets/header.png" width="100%" alt="babiniku.rs — a high-performance voice conversion library in Rust" />

**Your avatar has a face. Now give it a voice.**

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

</div>

## バ美肉 — babiniku

*Babiniku* (バーチャル美少女受肉, "incarnating as a virtual girl") is the art of becoming your avatar — not wearing it, **being** it. Trackers move her hands. Shaders light her hair. And then you unmute, and your own voice walks in and breaks the spell.

babiniku.rs is the missing last mile: a **real-time zero-shot voice changer toolkit in pure Rust**. Give it a few seconds of your character's voice, speak, and a **virtual microphone** delivers her voice to Discord, Zoom, OBS — anything with a mic picker. On a plain CPU. No Python, no CUDA Toolkit, no cloud: your voice never leaves your machine.

```sh
cargo run --release -p babiniku --features wavlm --bin babiniku -- \
    --reference her_voice.wav --monitor --denoise
```

Live TUI knobs while you speak: pitch (`[` `]`), noise suppression (`,` `.`), input gate (`-` `=`), bandwidth extension (`;` `'`), output noise reduction (`<` `>`), voice-profile EQ (`(` `)`), passthrough A/B (`p`), self-monitor (`l`).

The virtual mic runs at **48 kHz** and `--out` recordings are written at 48 kHz: the 16 kHz engines (meanvc, xvc) are upsampled in-process (exact ×3 windowed-sinc), while Seed-VC synthesizes at 22.05 kHz and is resampled straight to 48 kHz. On top of that, `--bwe <0-100>` (or the `;`/`'` knob, off by default) blends in a pure-DSP **harmonic exciter** that synthesizes the missing 8–16 kHz band (sibilance/"air") from the 3–8 kHz band — it lifts the "gauzy" veil of 16 kHz output at zero added latency, on every engine ([#42](https://github.com/m96-chan/babiniku.rs/issues/42)).

Quit with `q` — or Ctrl-C / SIGTERM, which run the same clean teardown of the virtual devices; stale `babiniku` devices left by a killed run are recovered automatically at the next startup.

## Install

`cargo install --git` is the supported install channel ([#69](https://github.com/m96-chan/babiniku.rs/issues/69)) — a plain Rust toolchain is all you need, and CI smoke-tests this exact path on every change:

```sh
# CPU baseline — no CUDA Toolkit, no Python, works everywhere
cargo install --git https://github.com/m96-chan/babiniku.rs babiniku
# with a GPU / extras
cargo install --git https://github.com/m96-chan/babiniku.rs babiniku --features cuda        # NVIDIA
cargo install --git https://github.com/m96-chan/babiniku.rs babiniku --features metal       # Apple Silicon
cargo install --git https://github.com/m96-chan/babiniku.rs babiniku --features wavlm       # native voice print from the reference wav
cargo install --git https://github.com/m96-chan/babiniku.rs babiniku --features cuda,seedvc # + Seed-VC (GPL-3.0 build!)
```

| Feature | What it adds | Build-time needs |
|---|---|---|
| *(default)* | CPU real-time baseline, MeanVC + X-VC engines | none beyond Rust (Linux: `libpulse` headers) |
| `wavlm` | voice print computed from the reference wav (ONNX Runtime) | downloads the prebuilt ONNX runtime |
| `cuda` | NVIDIA GPU inference (X-VC live mic, Seed-VC) | CUDA Toolkit **at build time only** — at runtime the driver alone suffices |
| `metal` | Apple-Silicon GPU inference | Xcode Command Line Tools |
| `seedvc` | Seed-VC engine — **the binary becomes GPL-3.0 when distributed** | GPU recommended at runtime |

`babiniku --version` prints the compiled feature set (and the GPL notice on `seedvc` builds), so a build's flavor is always auditable.

An installed binary looks for checkpoints in the platform data directory — `~/.local/share/babiniku/ckpt` (Linux, honoring `$XDG_DATA_HOME`), `~/Library/Application Support/babiniku/ckpt` (macOS), `%APPDATA%\babiniku\ckpt` (Windows) — overridable with `--ckpt-dir <dir>` or `BABINIKU_CKPT_DIR`; from a repo checkout, `./ckpt` keeps working as before. Checkpoint setup per engine: [docs/meanvc.md](docs/meanvc.md) · [docs/xvc.md](docs/xvc.md) · [docs/seedvc.md](docs/seedvc.md) (a `babiniku-fetch` downloader is planned, [#65](https://github.com/m96-chan/babiniku.rs/issues/65)).

Publishing to crates.io is blocked on the candle-fork **git dependency** (crates.io forbids git deps) until the forward-AD patch is upstreamed ([#10](https://github.com/m96-chan/babiniku.rs/issues/10)); until then every crate stays `publish = false` and `--git` is the way.

Pick the engine with `--engine meanvc` (default), `--engine xvc`
(multilingual, incl. Japanese — needs the converted X-VC checkpoints in
`ckpt/`, see [docs/xvc.md](docs/xvc.md); real-time on an idle 8-thread
CPU via the pipelined driver, comfortably real-time on a GPU with a
`--features cuda` build), or `--engine seedvc` (the most natural voice
of the three by ear — needs a build with **`--features seedvc`**, which
is **GPL-3.0 when distributed**, and a GPU; see
[docs/seedvc.md](docs/seedvc.md)). The TUI shows the active engine and its
per-stage RTF.

## Use cases

- **VTuber / streaming** — stay in character on stream, including impromptu collabs.
- **Meetings & Discord as your avatar** — the mic named `Babiniku-Virtual-Mic` is just… you.
- **Game voice chat** — squad hears the character, not the tired human at 2 a.m.
- **Voice privacy** — speak publicly without publishing your real voice.

## Engines

| Engine | Status | Notes |
|---|---|---|
| [MeanVC v1](docs/meanvc.md) | ✅ working, official weights | ~0.14 RTF end-to-end on CPU, ≈0.6 s latency; Mandarin-trained ([#28](https://github.com/m96-chan/babiniku.rs/issues/28) tracks Japanese) |
| [MeanVC 2](docs/meanvc.md) | ⏳ implemented, awaiting official weights | 40 ms chunks → ~110 ms latency class |
| [X-VC](docs/xvc.md) | ✅ working, official weights | Japanese-native quality; **live mic needs the CUDA build** (`--features cuda`, CUDA Toolkit at build time only — RTF ≈ 0.10 on GPU; CPU ≈ 0.9+ falls behind on a busy desktop) |
| [Seed-VC](docs/seedvc.md) | ✅ working, official weights (**GPL-3.0, opt-in `seedvc` feature**) | Most natural by ear; 22.05 kHz BigVGAN line with **no decoder-needle pathology** (no declick stack needed); sliding-context streaming with SOLA joins, ~0.25 s/0.32 s block on GPU; adaptive voice-profile EQ toward the reference's real spectrum ([#49](https://github.com/m96-chan/babiniku.rs/issues/49)/[#50](https://github.com/m96-chan/babiniku.rs/issues/50)/[#62](https://github.com/m96-chan/babiniku.rs/issues/62)) |
| [Zero-VC](docs/zero-vc.md) | 🔍 evaluation | zero-lookahead (20 ms algorithmic latency) — latency-first candidate; no public code yet ([#31](https://github.com/m96-chan/babiniku.rs/issues/31)) |

Every engine is ported weight-compatible and verified stage-by-stage against its official implementation with golden tests (`cargo test --workspace`). Deep dive, APIs, checkpoint setup, performance notes: [docs/meanvc.md](docs/meanvc.md). Issues are labeled by architecture (`meanvc`, `meanvc2`, `xvc`, `seedvc`, `tui`, `infra`).

## Platform support

The engine core is pure Rust and portable; the platform surface — capture/playback and the **virtual microphone** — lives behind an audio backend layer in `crates/babiniku` ([#51](https://github.com/m96-chan/babiniku.rs/issues/51), [#52](https://github.com/m96-chan/babiniku.rs/issues/52)).

| Platform | Capture / playback | Virtual mic | Status |
|---|---|---|---|
| Linux | PulseAudio/PipeWire | ✅ null sink + remap (`babiniku_mic`) | ✅ working |
| [Windows](docs/windows.md) | WASAPI (`cpal`) | routed to VB-CABLE / VoiceMeeter (auto-detected or `--output-device`) | ✅ merged, CI-verified — live VB-CABLE routing awaiting field reports ([#53](https://github.com/m96-chan/babiniku.rs/issues/53)) |
| [macOS](docs/macos.md) | CoreAudio (`cpal`) | routed to BlackHole (auto-detected or `--output-device`) | ✅ merged, CI-verified — field reports welcome ([#54](https://github.com/m96-chan/babiniku.rs/issues/54)) |
| Android / iOS | AAudio / AVAudioEngine | in-app routing (library-first) | 📋 planned ([#56](https://github.com/m96-chan/babiniku.rs/issues/56) / [#57](https://github.com/m96-chan/babiniku.rs/issues/57)) |

Verify a platform's audio stack in seconds — no model checkpoints needed (lists devices, creates the virtual-mic route, plays a tone through it, captures 1 s of mic audio, tears down):

```sh
cargo run --release -p babiniku --example audio_probe
```

## Workspace layout

The repo is a cargo workspace — one crate per engine on a shared foundation:

| Crate | What it is |
|---|---|
| [`crates/vc-core`](crates/vc-core) | Engine-agnostic foundation: encoder/speaker/vocoder traits, log-mel front-end, BWE post-processing (`bwe::Upsampler3x`, `bwe::Exciter`), `Error`/`Result` |
| [`crates/meanvc`](crates/meanvc) | MeanVC v1 + MeanVC 2 engines (library name `meanvc2`), examples, golden tests |
| [`crates/babiniku`](crates/babiniku) | The `babiniku` real-time TUI / virtual-mic binary, plus the per-platform audio backends (`babiniku::audio`: Pulse on Linux, cpal/WASAPI/CoreAudio elsewhere) and the `audio_probe` example |
| [`crates/xvc`](crates/xvc) | X-VC engine: GLM-4-Voice tokenizer, ERes2Net, SAC codec, prenet, MMDiT converter + the `XvcEngine` offline/streaming pipeline ([#30](https://github.com/m96-chan/babiniku.rs/issues/30)) |
| [`crates/seedvc`](crates/seedvc) | Seed-VC engine (**GPL-3.0**, feature-gated): Whisper-small content, CAM++ speaker, DiT+WaveNet CFM, BigVGAN + `SeedVcEngine`/`SeedVcStream` ([#50](https://github.com/m96-chan/babiniku.rs/issues/50)) |

Checkpoints stay at the repo root (`ckpt/`), as do `tools/` and `docs/`.

## Contributing

Fork + pull request only — see [CONTRIBUTING.md](CONTRIBUTING.md) (issue-first, TDD, demo-before-push; **no code as issue attachments**).

## License

MIT OR Apache-2.0, at your option — except [`crates/seedvc`](crates/seedvc), which is **GPL-3.0** (upstream code and weights) and strictly opt-in: binaries built without the `seedvc` cargo feature carry no GPL obligations. Model weights belong to their original authors ([ASLP-lab/MeanVC](https://github.com/ASLP-lab/MeanVC) et al.). The avatar above is the maintainer's own — bring yours. Header/avatar artwork: a personal modification of a model by [こまど (Komado)](https://komado.booth.pm/items) — shown for illustration only; all rights to the original model belong to its creator.

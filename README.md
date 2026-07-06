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
cargo run --release -p vc-demo --features wavlm --bin babiniku-demo -- \
    --reference her_voice.wav --monitor --denoise
```

Live TUI knobs while you speak: pitch (`[` `]`), noise suppression (`,` `.`), input gate (`-` `=`), bandwidth extension (`;` `'`), passthrough A/B (`p`), self-monitor (`l`).

The engines synthesize at 16 kHz, so the demo upsamples the converted voice to **48 kHz in-process** (exact ×3 windowed-sinc) before playback — the virtual mic runs at 48 kHz and `--out` recordings are written at 48 kHz. On top of that, `--bwe <0-100>` (or the `;`/`'` knob, off by default) blends in a pure-DSP **harmonic exciter** that synthesizes the missing 8–16 kHz band (sibilance/"air") from the 3–8 kHz band — it lifts the "gauzy" veil of 16 kHz output at zero added latency, on every engine ([#42](https://github.com/m96-chan/babiniku.rs/issues/42)).

Quit with `q` — or Ctrl-C / SIGTERM, which run the same clean teardown of the virtual devices; stale `babiniku` devices left by a killed run are recovered automatically at the next startup.

Pick the engine with `--engine meanvc` (default) or `--engine xvc`
(multilingual, incl. Japanese — needs the converted X-VC checkpoints in
`ckpt/`, see [docs/xvc.md](docs/xvc.md); real-time on an idle 8-thread
CPU via the pipelined driver, and comfortably real-time on a GPU with a
`--features cuda` build — see the table below). The TUI shows the active
engine and its per-stage RTF.

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
| [Zero-VC](docs/zero-vc.md) | 🔍 evaluation | zero-lookahead (20 ms algorithmic latency) — latency-first candidate; no public code yet ([#31](https://github.com/m96-chan/babiniku.rs/issues/31)) |

Every engine is ported weight-compatible and verified stage-by-stage against its official implementation with golden tests (`cargo test --workspace`). Deep dive, APIs, checkpoint setup, performance notes: [docs/meanvc.md](docs/meanvc.md). Issues are labeled by architecture (`meanvc`, `meanvc2`, `demo`, `infra`).

## Workspace layout

The repo is a cargo workspace — one crate per engine on a shared foundation:

| Crate | What it is |
|---|---|
| [`crates/vc-core`](crates/vc-core) | Engine-agnostic foundation: encoder/speaker/vocoder traits, log-mel front-end, BWE post-processing (`bwe::Upsampler3x`, `bwe::Exciter`), `Error`/`Result` |
| [`crates/meanvc`](crates/meanvc) | MeanVC v1 + MeanVC 2 engines (library name `meanvc2`), examples, golden tests |
| [`crates/vc-demo`](crates/vc-demo) | The `babiniku-demo` real-time TUI / virtual-mic binary |
| [`crates/xvc`](crates/xvc) | X-VC engine: GLM-4-Voice tokenizer, ERes2Net, SAC codec, prenet, MMDiT converter + the `XvcEngine` offline/streaming pipeline ([#30](https://github.com/m96-chan/babiniku.rs/issues/30)) |

Checkpoints stay at the repo root (`ckpt/`), as do `tools/` and `docs/`.

## License

MIT OR Apache-2.0, at your option. Model weights belong to their original authors ([ASLP-lab/MeanVC](https://github.com/ASLP-lab/MeanVC) et al.). The avatar above is the maintainer's own — bring yours. Header/avatar artwork: a personal modification of a model by [こまど (Komado)](https://drive.google.com/file/d/1DuVNYmahJTelmDbZ1RVXXKQ7wuYPYy3u/view) — shown for illustration only; all rights to the original model belong to its creator.

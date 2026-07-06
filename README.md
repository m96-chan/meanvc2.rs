<div align="center">

<img src="assets/avatar.png" width="280" alt="babiniku avatar" />

# babiniku.rs

**Your avatar has a face. Now give it a voice.**

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

</div>

## バ美肉 — babiniku

*Babiniku* (バーチャル美少女受肉, "incarnating as a virtual girl") is the art of becoming your avatar — not wearing it, **being** it. Trackers move her hands. Shaders light her hair. And then you unmute, and your own voice walks in and breaks the spell.

babiniku.rs is the missing last mile: a **real-time zero-shot voice changer toolkit in pure Rust**. Give it a few seconds of your character's voice, speak, and a **virtual microphone** delivers her voice to Discord, Zoom, OBS — anything with a mic picker. On a plain CPU. No Python, no CUDA Toolkit, no cloud: your voice never leaves your machine.

```sh
cargo run --release --features demo,wavlm --bin meanvc-demo -- \
    --reference her_voice.wav --monitor --denoise
```

Live TUI knobs while you speak: pitch (`[` `]`), noise suppression (`,` `.`), input gate (`-` `=`), passthrough A/B (`p`), self-monitor (`l`).

## Use cases

- **VTuber / streaming** — stay in character on stream, including impromptu collabs.
- **Meetings & Discord as your avatar** — the mic named `MeanVC-Virtual-Mic` is just… you.
- **Game voice chat** — squad hears the character, not the tired human at 2 a.m.
- **Voice privacy** — speak publicly without publishing your real voice.

## Engines

| Engine | Status | Notes |
|---|---|---|
| [MeanVC v1](docs/meanvc.md) | ✅ working, official weights | ~0.14 RTF end-to-end on CPU, ≈0.6 s latency; Mandarin-trained ([#28](https://github.com/m96-chan/babiniku.rs/issues/28) tracks Japanese) |
| [MeanVC 2](docs/meanvc.md) | ⏳ implemented, awaiting official weights | 40 ms chunks → ~110 ms latency class |
| [X-VC](docs/xvc.md) | 🔍 evaluation | codec-space, multilingual — candidate for language-agnostic conversion ([#30](https://github.com/m96-chan/babiniku.rs/issues/30)) |
| [Zero-VC](docs/zero-vc.md) | 🔍 evaluation | zero-lookahead (20 ms algorithmic latency) — latency-first candidate; no public code yet ([#31](https://github.com/m96-chan/babiniku.rs/issues/31)) |

Every engine is ported weight-compatible and verified stage-by-stage against its official implementation with golden tests (`cargo test`). Deep dive, APIs, checkpoint setup, performance notes: [docs/meanvc.md](docs/meanvc.md). Issues are labeled by architecture (`meanvc`, `meanvc2`, `demo`, `infra`).

## License

MIT OR Apache-2.0, at your option. Model weights belong to their original authors ([ASLP-lab/MeanVC](https://github.com/ASLP-lab/MeanVC) et al.). The avatar above is the maintainer's own — bring yours. Avatar artwork: a personal modification of a model by [こまど (Komado)](https://drive.google.com/file/d/1DuVNYmahJTelmDbZ1RVXXKQ7wuYPYy3u/view) — shown for illustration only; all rights to the original model belong to its creator.

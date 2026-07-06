# babiniku.rs

> Become your avatar. A real-time voice conversion toolkit in pure Rust — virtual microphone included.

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**babiniku.rs** turns your voice into any target speaker's voice from a few seconds of reference audio, in real time on a CPU, and exposes the result as a **virtual microphone** that Discord / Zoom / OBS can use directly. Built on [candle](https://github.com/huggingface/candle); **no Python and no CUDA Toolkit required** — everything runs in real time on a plain CPU (GPU backends exist as opt-in cargo features but are never a setup requirement).

```sh
cargo run --release --features demo,wavlm --bin meanvc-demo -- \
    --reference target_voice.wav --monitor --denoise
```

The TUI gives you live knobs: pitch shift (`[` `]`), noise-suppression mix (`,` `.`), input gate (`-` `=`), passthrough A/B (`p`), loopback monitor (`l`).

## Engines

| Engine | Status | Notes |
|---|---|---|
| [MeanVC v1](docs/meanvc.md) | ✅ working, official weights | mean-flows CARD DiT, ~0.14 RTF end-to-end, ≈0.6 s latency; Mandarin-trained (accent on other languages, see [#28](https://github.com/m96-chan/babiniku.rs/issues/28)) |
| [MeanVC 2](docs/meanvc.md) | ⏳ implemented, awaiting official weights | 40 ms FRC chunks → ~110 ms latency class |
| X-VC | 🔍 evaluation | codec-space, multilingual semantic tokenizer — candidate for language-agnostic conversion |

Every ported model is validated stage-by-stage against its official implementation with golden tests (`cargo test`); see [docs/meanvc.md](docs/meanvc.md) for the full architecture, API usage, checkpoint setup, and performance notes.

Issues are labeled by architecture (`meanvc`, `meanvc2`, `demo`, `infra`).

## License

MIT OR Apache-2.0, at your option. Model weights belong to their original authors ([ASLP-lab/MeanVC](https://github.com/ASLP-lab/MeanVC) et al.).

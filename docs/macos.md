# macOS

> **Implemented, awaiting verification on a real Mac** — the CoreAudio
> backend is build-, unit- and CI-verified (this repo's dev container is
> Linux), and [#54](https://github.com/m96-chan/babiniku.rs/issues/54)
> tracks the live run. Windows is the same backend with a different
> loopback driver ([#53](https://github.com/m96-chan/babiniku.rs/issues/53)).

On macOS the demo uses the portable [`cpal`](https://github.com/RustAudio/cpal)
backend (`crates/vc-demo/src/audio/cpal_backend.rs`): capture and playback go
through **CoreAudio**. macOS has no OS-level null sink, so the "virtual
microphone" is a **route** — the converted voice plays into the input end of a
user-installed loopback driver ([BlackHole](https://existential.audio/blackhole/)),
and your app records the driver's output end (`BlackHole 2ch`).

## Prerequisites

- **Rust** via [rustup](https://rustup.rs) (stable, 1.80+).
- **Xcode Command Line Tools** (the linker and CoreAudio headers):

  ```sh
  xcode-select --install
  ```

No CUDA Toolkit, no Python, no Homebrew audio libraries — `cpal` talks to
CoreAudio directly.

## Build

```sh
cargo build --release -p vc-demo
```

CPU is the baseline target and real-time on Apple Silicon. Optionally, the
`metal` feature builds the engines against the Metal backend of candle:

```sh
cargo build --release -p vc-demo --features metal
```

## Verify the audio stack (no checkpoints needed)

```sh
cargo run --release -p vc-demo --example audio_probe
```

Expected output: `backend: cpal`, the capture/playback device lists CoreAudio
sees, a `virtual mic:` status line (see below), 2 s of a 440 Hz tone played
into the route, `capture ok — rms …` from 1 s of microphone audio, and
`audio probe OK`. The first capture triggers the microphone permission prompt
(see [Microphone permission](#microphone-permission-tcc)).

- With BlackHole installed, the status line reads
  `routing the converted voice to "BlackHole 2ch" …` and the tone is
  *inaudible* on your speakers — record from `BlackHole 2ch` (e.g. QuickTime
  audio recording) to hear it.
- Without a loopback device it reads `no loopback device found — playing on
  the default output …` and the tone plays on your speakers.

`--output-device <name>` / `--input-device <name>` force devices (substring
match), `--no-sink` / `--no-capture` skip legs (used by CI, which has no audio
devices).

## Install BlackHole (the virtual mic)

```sh
brew install blackhole-2ch
```

(or the installer from <https://existential.audio/blackhole/>). BlackHole is a
zero-latency loopback driver: whatever plays into the `BlackHole 2ch` output
appears on the `BlackHole 2ch` input. The demo **auto-detects** it in the
output-device list (`pick_route_device` in `crates/vc-demo/src/audio/mod.rs`);
to force a specific device, pass `--output-device "BlackHole 2ch"`.

## Run the demo

With the checkpoints under `ckpt/` (setup: [docs/meanvc.md](meanvc.md); X-VC:
[docs/xvc.md](xvc.md)):

```sh
cargo run --release -p vc-demo --features wavlm --bin babiniku-demo -- \
    --reference her_voice.wav
```

The optional Seed-VC engine (`--engine seedvc`) needs a build with
`--features seedvc`; note that such builds are **GPL-3.0 when
distributed** (see `crates/seedvc`).

Then pick **`BlackHole 2ch` as the microphone** in Discord / OBS / Zoom — that
input carries the converted voice. The TUI knobs work as on Linux; `--denoise`
has no OS-level facility on macOS (the in-process RNNoise knob `,` `.` still
works).

## Microphone permission (TCC)

The first capture makes macOS prompt *"&lt;your terminal&gt; would like to
access the microphone"* — this is per-app (Terminal, iTerm2, VS Code, …), so
grant it to the app you run the demo from. If it was denied (capture then
returns silence or fails), re-enable it under **System Settings › Privacy &
Security › Microphone** and restart the terminal app; the prompt itself only
appears once per app.

## Self-monitor (`l` key)

Toggling `l` opens a **second CoreAudio stream on the default output** so you
can hear the converted voice while it streams into BlackHole (skipped when the
route already *is* the default output — it would duplicate the audio).
Alternative: build a **Multi-Output Device** in *Audio MIDI Setup* combining
BlackHole and your speakers, and pass it as `--output-device` — one stream,
both destinations, at the cost of the demo no longer controlling the monitor.

## CI

[.github/workflows/macos.yml](../.github/workflows/macos.yml) builds and tests
`vc-demo` on an Apple Silicon runner, checks the `metal` feature still
compiles, and smoke-runs `audio_probe --no-sink --no-capture` (backend init +
device enumeration; CI runners have no audio devices).

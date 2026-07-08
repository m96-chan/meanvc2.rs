# Windows

> Windows support for `babiniku` via the cpal/WASAPI audio backend
> ([#52](https://github.com/m96-chan/babiniku.rs/issues/52),
> [#53](https://github.com/m96-chan/babiniku.rs/issues/53)). Build- and
> unit-verified; live verification on a real Windows machine is tracked
> in [#53](https://github.com/m96-chan/babiniku.rs/issues/53).

Unlike Linux (where the demo creates a PulseAudio/PipeWire null sink),
Windows has no OS-level virtual audio device, so the virtual microphone
is a **route**: the converted voice plays into the input end of a
user-installed loopback driver (VB-CABLE), and you select the driver's
output end as the microphone in your app.

```text
 mic ──► babiniku ──► "CABLE Input"  ═ VB-CABLE ═  "CABLE Output" ──► Discord/OBS/Zoom
         (WASAPI capture)  (playback dev)   (loopback)   (recording dev)    (select as mic)
```

## Prerequisites

- **Rust** via [rustup](https://rustup.rs/) — the default Windows host
  toolchain (`x86_64-pc-windows-msvc`, stable).
- **Visual Studio Build Tools** with the *Desktop development with C++*
  workload (the MSVC linker; the rustup installer offers to set this up).
- No audio SDK: the backend talks to WASAPI, which ships with Windows.

## Build

```powershell
cargo build --release -p babiniku
```

## Verify the audio stack (no checkpoints needed)

```powershell
cargo run --release -p babiniku --example audio_probe
```

The probe exercises the whole audio layer in seconds. Expected output:

1. `backend: cpal`, then the capture and playback device lists WASAPI
   sees (your mic, speakers, and — once installed — `CABLE Input` /
   `CABLE Output`);
2. `virtual mic: routing the converted voice to "CABLE Input (VB-Audio
   Virtual Cable)" …` when VB-CABLE is installed and auto-detected —
   otherwise a hint that no loopback device was found and the tone will
   play on the default output;
3. a 2 s 440 Hz tone into the route (record from `CABLE Output` in any
   app to hear it; without VB-CABLE it plays on your speakers);
4. `capture ok — rms 0.xxxx` after 1 s of 16 kHz mic capture (speak into
   the mic: the RMS should be clearly above 0);
5. `audio probe OK`.

`--output-device <name>` / `--input-device <name>` (case-insensitive
substring) override the route and the mic; `--no-sink` / `--no-capture`
skip the playback / capture halves (used by the headless CI smoke run in
[`.github/workflows/windows.yml`](../.github/workflows/windows.yml)).

## Install VB-CABLE (the virtual mic)

1. Download VB-CABLE from <https://vb-audio.com/Cable/> (donationware).
2. Extract and run `VBCABLE_Setup_x64.exe` **as administrator**; reboot
   if the installer asks.
3. Windows gains a playback device **`CABLE Input`** and a recording
   device **`CABLE Output`**.

The demo auto-detects the route: any playback device whose name contains
`CABLE Input`, `VB-Audio`, or `VoiceMeeter Input` is picked automatically
(`pick_route_device` in `crates/babiniku/src/audio/mod.rs`). To force a
specific device:

```powershell
--output-device "CABLE Input"
```

A `--output-device` that matches nothing is an **error** listing the
available devices — a typo never silently falls back to the speakers.
With no loopback driver installed at all, the demo plays on the default
output and prints a setup hint.

## Run the demo

Checkpoints go under `ckpt/` at the workspace root (installed binaries: `%APPDATA%\babiniku\ckpt`, or `--ckpt-dir` — #69) — setup per
[docs/meanvc.md](meanvc.md) (MeanVC v1, the default engine) or
[docs/xvc.md](xvc.md) (`--engine xvc`). Then:

```powershell
cargo run --release -p babiniku --features wavlm --bin babiniku -- `
    --reference her_voice.wav
```

(Without the `wavlm` feature, pass a precomputed
`--voice-print file.safetensors` — see [docs/meanvc.md](meanvc.md).)

The optional Seed-VC engine (`--engine seedvc`) needs a build with
`--features seedvc`; note that such builds are **GPL-3.0 when
distributed** (see `crates/seedvc`).

The TUI reports the resolved route (`routing the converted voice to
"CABLE Input …"`). In **Discord / OBS / Zoom**, open the audio settings
and select **`CABLE Output (VB-Audio Virtual Cable)`** as the microphone
— that stream is the converted voice.

### Self-monitor (`l` key)

`l` toggles hearing the converted voice yourself. On Windows this opens
a **second WASAPI shared-mode stream on the default output** and mirrors
the converted audio there; it is fed lossily, so a glitching monitor can
never stall the virtual-mic route. When no loopback driver is installed
(the route already *is* the default output), the toggle is skipped —
it would double the same audio on the same device.

## Troubleshooting

- **Capture opens but is silent (rms 0.0000) or fails** — check the
  microphone privacy settings: *Settings > Privacy & security >
  Microphone*, and make sure *Let desktop apps access your microphone*
  is on.
- **Device won't open / wrong sample rate** — the backend opens WASAPI
  in shared mode at the device's mix format and resamples in both
  directions in-process, so any shared-mode rate (44.1/48 kHz) works.
  If another app holds the device exclusively, open the device's
  *Properties > Advanced* in the Sound Control Panel and untick *Allow
  applications to take exclusive control of this device*.
- **VoiceMeeter instead of VB-CABLE** — works the same way:
  `--output-device "VoiceMeeter Input"` (auto-detected too), then select
  *VoiceMeeter Output* as the mic in your app.
- **Which names does WASAPI actually expose?** — run the
  [`audio_probe`](#verify-the-audio-stack-no-checkpoints-needed) example;
  it prints the exact device names the demo matches against.

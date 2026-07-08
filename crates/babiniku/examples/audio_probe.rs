//! Audio backend probe — the per-platform demo for issue #51 (#52/#53/#54).
//!
//! Exercises the whole `babiniku::audio` surface without any model
//! checkpoints, so a new platform's audio stack can be verified in
//! seconds before touching the full demo:
//!
//! 1. lists the capture/playback devices the backend sees,
//! 2. creates the virtual-mic route (Linux: `babiniku_mic` null-sink pair;
//!    Windows/macOS: VB-CABLE / BlackHole routing),
//! 3. plays 2 s of a 440 Hz tone at 48 kHz into the route — select the
//!    virtual mic in any app (or `pactl list sources` / a recorder) to
//!    hear it,
//! 4. captures 1 s of 16 kHz microphone audio and reports its RMS,
//! 5. tears the route down again.
//!
//! ```sh
//! cargo run --release -p babiniku --example audio_probe -- \
//!     [--output-device <name>] [--input-device <name>] [--no-capture] [--no-sink]
//! ```

use babiniku::audio::{self, BackendOptions};

fn main() -> anyhow::Result<()> {
    let mut output_device = None;
    let mut input_device = None;
    let mut no_capture = false;
    let mut no_sink = false;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--output-device" => output_device = Some(it.next().expect("--output-device <name>")),
            "--input-device" => input_device = Some(it.next().expect("--input-device <name>")),
            "--no-capture" => no_capture = true,
            "--no-sink" => no_sink = true,
            other => anyhow::bail!("unknown flag {other}"),
        }
    }

    let backend = audio::default_backend(BackendOptions { output_device });
    println!("backend: {}", backend.name());

    let devices = backend.list_devices()?;
    println!("capture devices:");
    for d in &devices.inputs {
        println!("  - {d}");
    }
    println!("playback devices:");
    for d in &devices.outputs {
        println!("  - {d}");
    }

    backend.recover_stale();
    if !no_sink {
        let status = backend.create_virtual_mic()?;
        println!("virtual mic: {status}");

        // 2 s of 440 Hz at 48 kHz into the route: audible proof the
        // converted-voice path reaches the virtual microphone.
        let mut play = backend.open_playback(48_000)?;
        println!("playing a 2 s 440 Hz tone into the route…");
        let chunk: Vec<f32> = (0..4_800)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
            .collect();
        for _ in 0..20 {
            play.write(&chunk)?;
        }
        println!("tone done");
    }

    if !no_capture {
        let mut cap = backend.open_capture(input_device.as_deref(), 16_000, 3_200)?;
        println!("capturing 1 s from the microphone…");
        let mut buf = vec![0f32; 3_200];
        let mut sumsq = 0f64;
        for _ in 0..5 {
            cap.read(&mut buf)?;
            sumsq += buf.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>();
        }
        let rms = (sumsq / 16_000.0).sqrt();
        println!("capture ok — rms {rms:.4}");
    }

    if !no_sink {
        backend.destroy_virtual_mic();
        println!("virtual mic removed");
    }
    println!("audio probe OK");
    Ok(())
}

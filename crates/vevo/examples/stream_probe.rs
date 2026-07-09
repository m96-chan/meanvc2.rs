//! Debug/bench probe for the streaming driver (#74): loads the engine
//! (CUDA when available), feeds a real reference/source pair through
//! `VevoStream`, prints per-hop wall times, and writes the emitted
//! 48 kHz audio to `ckpt/vevo_stream_demo.wav` for a real listening
//! check (rule 2 — not golden-parity, which `cargo test` covers).
//!
//! ```sh
//! cargo run --release -p vevo --features cuda --example stream_probe
//! ```

use candle_core::Device;
use std::time::Instant;
use vevo::pipeline::{resample, VevoEngine};
use vevo::stream::StreamConfig;

fn read_wav(path: &std::path::Path, target_hz: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
    };
    Ok(if spec.sample_rate as usize == target_hz {
        samples
    } else {
        resample(&samples, spec.sample_rate as usize, target_hz)
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
    let device = if std::env::var("VEVO_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("device: {device:?}");

    let t0 = Instant::now();
    let engine = VevoEngine::load(&dir, &device)?;
    println!("load: {:.1}s", t0.elapsed().as_secs_f32());

    let ref16 = read_wav(&dir.join("ref_stage1_48k.wav"), 16_000)?;
    let ref24 = resample(&ref16, 16_000, 24_000);
    let src16 = read_wav(&dir.join("ref_trimmed.wav"), 16_000)?;

    let steps: usize = std::env::var("VEVO_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(StreamConfig::default().steps);
    let cfg = StreamConfig { steps, ..StreamConfig::default() };
    let t0 = Instant::now();
    let mut stream = engine.stream(&ref24, cfg)?;
    println!("reference prep: {:.2}s", t0.elapsed().as_secs_f32());

    let mut out: Vec<f32> = Vec::new();
    let mut fed = 0;
    let mut n = 0;
    while fed + cfg.block <= src16.len() && n < 12 {
        stream.push(&src16[fed..fed + cfg.block]);
        fed += cfg.block;
        while stream.ready() {
            let t0 = Instant::now();
            let chunk = stream.step()?;
            let dt = t0.elapsed().as_secs_f32();
            n += 1;
            let len = chunk.as_ref().map(|c| c.len()).unwrap_or(0);
            println!("hop {n}: {dt:.2}s (budget {:.2}s) out {len}", cfg.block as f32 / 16_000.0);
            if let Some(c) = chunk {
                out.extend(c);
            }
        }
    }

    let out_path = dir.join("vevo_stream_demo.wav");
    let mut w = hound::WavWriter::create(
        &out_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for s in &out {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    println!("wrote {} ({} samples, {:.2}s)", out_path.display(), out.len(), out.len() as f32 / 48_000.0);
    Ok(())
}

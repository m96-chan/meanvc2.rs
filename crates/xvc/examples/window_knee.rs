//! Window-size knee search for issue #42: streams sources at several
//! `chunk_ms` window sizes and reports the decoder-needle count plus the
//! worst-case per-window forward time (the real-time deadline metric —
//! the mean RTF hides compute spikes).
//!
//! ```sh
//! cargo run --release -p xvc --features cuda --example window_knee -- \
//!     <source.wav> <reference.wav> [chunk_ms ...]
//! ```

use std::time::Instant;

use candle_core::Device;
use xvc::{StreamConfig, XvcEngine};

fn read_wav_16k(path: &str) -> anyhow::Result<Vec<f64>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    anyhow::ensure!(spec.sample_rate == 16_000, "need 16 kHz: {path}");
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f64;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.map(|s| s as f64 / scale))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.map(|s| s as f64))
            .collect::<Result<_, _>>()?,
    })
}

/// Needle count: |y| > 0.25 and > 4x the 3 ms local rms, clustered.
fn needles(y: &[f32], sr: usize) -> usize {
    let w = 3 * sr / 1000;
    let mut count = 0;
    let mut last_hit: isize = -(sr as isize);
    for i in 0..y.len() {
        let a = y[i].abs();
        if a < 0.25 {
            continue;
        }
        let lo = i.saturating_sub(w);
        let hi = (i + w).min(y.len());
        let rms = (y[lo..hi].iter().map(|s| s * s).sum::<f32>() / (hi - lo) as f32).sqrt();
        if a > 4.0 * (rms + 1e-6) {
            if i as isize - last_hit > (sr / 100) as isize {
                count += 1;
            }
            last_hit = i as isize;
        }
    }
    count
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args.next().expect("source.wav");
    let refp = args.next().expect("reference.wav");
    let windows: Vec<usize> = {
        let w: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();
        if w.is_empty() {
            vec![640, 960, 1280, 1920, 2400]
        } else {
            w
        }
    };
    let device = if std::env::var("XVC_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let engine = XvcEngine::load("ckpt", &device)?;
    let ref_prep = engine.preprocess(&read_wav_16k(&refp)?);
    let reference = engine.prepare_reference(&ref_prep)?;
    let source = xvc::preprocess::preprocess(&read_wav_16k(&src)?, &Default::default());
    println!(
        "source {:.1}s, device {:?}",
        source.len() as f32 / 16_000.0,
        device
    );

    for w in windows {
        let cfg = StreamConfig {
            chunk_ms: w,
            ..Default::default()
        };
        let mut stream = engine.stream(reference.clone(), cfg)?;
        let mut out: Vec<f32> = Vec::with_capacity(source.len());
        let mut worst_ms = 0f32;
        let mut sum_ms = 0f32;
        let mut n_win = 0u32;
        for chunk in source.chunks(cfg.current_len()) {
            stream.push(chunk);
            loop {
                let t0 = Instant::now();
                let Some(step) = stream.step()? else { break };
                let ms = t0.elapsed().as_secs_f32() * 1e3;
                worst_ms = worst_ms.max(ms);
                sum_ms += ms;
                n_win += 1;
                out.extend_from_slice(&step.samples);
            }
        }
        out.extend_from_slice(&stream.finish()?);
        let n = needles(&out, 16_000);
        println!(
            "chunk_ms {w:5}: 針 {n:3}  worst forward {worst_ms:7.1} ms  mean {:6.1} ms  (budget {} ms)",
            sum_ms / n_win.max(1) as f32,
            cfg.current_ms
        );
    }
    Ok(())
}

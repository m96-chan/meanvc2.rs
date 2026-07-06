//! Offline X-VC conversion: source wav + reference wav → converted wav.
//!
//! ```sh
//! cargo run --release -p xvc --example convert_xvc -- \
//!     <source.wav> <reference.wav> <out.wav>
//! ```
//!
//! Both inputs must be 16 kHz (mono, or the first channel is taken).
//! Loads the converted official checkpoints from `ckpt/` and mirrors the
//! official `bins/infer_single.py` offline path (volume normalization,
//! 40 Hz high-pass, one full-utterance forward).

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use xvc::XvcEngine;

/// Reads a 16 kHz wav as float64 samples in `[-1, 1]` (soundfile-style).
fn read_wav_16k(path: &str) -> anyhow::Result<Vec<f64>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    anyhow::ensure!(
        spec.sample_rate == 16_000,
        "expected a 16 kHz wav: {path} is {} Hz",
        spec.sample_rate
    );
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f64;
            r.samples::<i32>()
                .map(|v| v.map(|s| s as f64 / scale))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .map(|v| v.map(|s| s as f64))
            .collect::<Result<_, _>>()?,
    };
    Ok(samples
        .into_iter()
        .step_by(spec.channels as usize)
        .collect())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source_path, reference_path, out_path] = args.as_slice() else {
        anyhow::bail!("usage: convert_xvc <source.wav> <reference.wav> <out.wav>");
    };

    let ckpt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
    eprintln!("loading X-VC engine from {} …", ckpt.display());
    let t0 = Instant::now();
    let engine = XvcEngine::load(&ckpt, &Device::Cpu)?;
    eprintln!("loaded in {:.1} s", t0.elapsed().as_secs_f64());

    let source = read_wav_16k(source_path)?;
    let reference = read_wav_16k(reference_path)?;
    let src_secs = source.len() as f64 / 16_000.0;

    let t0 = Instant::now();
    let target = engine.preprocess(&reference);
    let conditions = engine.prepare_reference(&target)?;
    let t_ref = t0.elapsed();

    let t0 = Instant::now();
    let converted = engine.convert_offline(&source, &conditions)?;
    let t_conv = t0.elapsed();
    eprintln!(
        "reference {:.2} s prep, conversion {:.2} s for {:.2} s audio (RTF {:.2})",
        t_ref.as_secs_f64(),
        t_conv.as_secs_f64(),
        src_secs,
        t_conv.as_secs_f64() / src_secs,
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(out_path, spec)?;
    for s in &converted {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    writer.finalize()?;
    eprintln!("wrote {out_path} ({} samples)", converted.len());
    Ok(())
}

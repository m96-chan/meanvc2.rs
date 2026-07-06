//! Demo: one real X-VC streaming step through the ported SAC codec and
//! MMDiT converter (issue #30).
//!
//! Loads the converted official weights and the chain fixture (one
//! 640/240/100/20 streaming window of Japanese speech), then runs
//! encode → convert → decode, reports per-stage parity vs the official
//! PyTorch output plus wall times, and writes the synthesized window to
//! `ckpt/xvc_chain_rust.wav` for listening.
//!
//! ```sh
//! cargo run --release -p xvc --example codec_chain_demo
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{Device, Tensor};
use xvc::{AcousticConverter, SacCodec};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap()
}

fn main() -> anyhow::Result<()> {
    let ckpt = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
    let device = Device::Cpu;

    let codec = SacCodec::load(ckpt.join("xvc_codec.safetensors"), &device)?;
    let conv = AcousticConverter::load(ckpt.join("xvc_converter.safetensors"), &device)?;
    let fx = candle_core::safetensors::load(ckpt.join("xvc_chain_fixture.safetensors"), &device)?;

    // Encode the 640 ms source chunk to the quantized acoustic latent.
    let t0 = Instant::now();
    let enc = codec.encode(&fx["chunk_wav"])?;
    let t_enc = t0.elapsed();

    // Convert the fused latent (fixture prenet output — the semantic and
    // prenet stages are ported separately) with the real target
    // speaker/mel conditions.
    let t0 = Instant::now();
    let out = conv.forward(
        &fx["prenet_out"],
        &fx["frame_condition"],
        &fx["speaker_condition"],
    )?;
    let t_conv = t0.elapsed();

    // Decode the converted latent straight to the 16 kHz waveform.
    let t0 = Instant::now();
    let wav = codec.decode(&out)?;
    let t_dec = t0.elapsed();

    println!("stage parity vs official PyTorch (max abs diff):");
    println!(
        "  encode zq     {:.2e}",
        max_abs_diff(&enc.zq, &fx["acoustic_zq"])
    );
    println!(
        "  converter out {:.2e}",
        max_abs_diff(&out, &fx["converter_out"])
    );
    println!("  decoded wav   {:.2e}", max_abs_diff(&wav, &fx["wav_out"]));
    println!(
        "wall time: encode {:.1} ms, convert {:.1} ms, decode {:.1} ms (640 ms window)",
        t_enc.as_secs_f64() * 1e3,
        t_conv.as_secs_f64() * 1e3,
        t_dec.as_secs_f64() * 1e3,
    );

    let samples: Vec<f32> = wav.flatten_all()?.to_vec1()?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let path = ckpt.join("xvc_chain_rust.wav");
    let mut writer = hound::WavWriter::create(&path, spec)?;
    for s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    writer.finalize()?;
    println!("wrote {}", path.display());
    Ok(())
}

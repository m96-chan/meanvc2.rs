//! Demo: 192-d speaker embedding of a reference wav through the X-VC
//! ERes2Net speaker encoder (issue #30).
//!
//! ```sh
//! cargo run --release -p xvc --example speaker_embed -- [wav] [weights]
//! ```
//!
//! Defaults: `ckpt/test.wav` and `ckpt/xvc_speaker.safetensors` at the
//! workspace root (see `tools/convert_xvc_speaker.py`). When
//! `ckpt/xvc_speaker_fixture.safetensors` is present, also reports the
//! cosine similarity against the official reference embedding.

use std::path::PathBuf;
use std::time::Instant;

use candle_core::Device;
use xvc::SpeakerEncoder;

fn workspace_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let wav_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_path("ckpt/test.wav"));
    let weights = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_path("ckpt/xvc_speaker.safetensors"));

    let mut reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    anyhow::ensure!(spec.sample_rate == 16_000, "expected 16 kHz input");
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| Ok(s? as f32 / 32_768.0))
        .collect::<anyhow::Result<_>>()?;
    println!(
        "input: {} ({:.2} s at 16 kHz)",
        wav_path.display(),
        samples.len() as f32 / 16_000.0
    );

    let t = Instant::now();
    let encoder = SpeakerEncoder::load(&weights, &Device::Cpu)?;
    println!("loaded {} in {:.0?}", weights.display(), t.elapsed());

    let t = Instant::now();
    let emb = encoder.embed(&samples)?; // [1, 192]
    let elapsed = t.elapsed();
    let v: Vec<f32> = emb.flatten_all()?.to_vec1()?;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "embedding: shape {:?}, l2 norm {norm:.3}, computed in {elapsed:.0?} \
         (rtf {:.4})",
        emb.dims(),
        elapsed.as_secs_f32() / (samples.len() as f32 / 16_000.0),
    );
    println!("first 8 dims: {:?}", &v[..8]);

    let fixture = workspace_path("ckpt/xvc_speaker_fixture.safetensors");
    if fixture.exists() {
        let fx = candle_core::safetensors::load(&fixture, &Device::Cpu)?;
        let reference: Vec<f32> = fx["embedding"].flatten_all()?.to_vec1()?;
        println!(
            "cosine vs official reference embedding (preprocessed wav): {:.6}",
            cosine(&v, &reference)
        );
    }
    Ok(())
}

//! Quantitative speaker-similarity check: CAM++ cosine similarity between
//! a converted output and (a) the target reference and (b) the original
//! source, to verify conversion actually moves timbre toward the target.
//! cargo run --release -p cosyvoice --features cuda --example speaker_sim_probe -- \
//!     <source.wav> <reference.wav> <converted.wav>
use candle_core::Device;
use cosyvoice::CosyVoiceEngine;
use vc_core::profile::resample_analysis;

fn read16(p: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let sc = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.unwrap() as f32 / sc)
                .collect()
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.unwrap())
            .collect(),
    };
    if spec.sample_rate == 16_000 {
        audio
    } else {
        resample_analysis(&audio, spec.sample_rate as usize, 16_000)
    }
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dev = Device::cuda_if_available(0).unwrap();
    let eng = CosyVoiceEngine::load("ckpt", &dev).unwrap();

    let embed = |path: &str| -> Vec<f32> {
        let audio = read16(path);
        eng.embed_for_debug(&audio).unwrap()
    };

    let src_emb = embed(&a[0]);
    let ref_emb = embed(&a[1]);
    let out_emb = embed(&a[2]);

    println!(
        "src vs ref (how different are the two speakers): {:.4}",
        cos(&src_emb, &ref_emb)
    );
    println!(
        "out vs ref (should be HIGH if conversion worked): {:.4}",
        cos(&out_emb, &ref_emb)
    );
    println!(
        "out vs src (should be LOW-ish if conversion worked): {:.4}",
        cos(&out_emb, &src_emb)
    );
}

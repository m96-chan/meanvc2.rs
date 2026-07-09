//! A/B test: does a larger `context` improve conditioning fidelity on
//! harder (noisy/expressive) material? Streams the same source through
//! two StreamConfigs and reports per-hop CAM++ similarity to the
//! reference for each.
//!
//! ```sh
//! cargo run --release -p cosyvoice --features cuda --example context_ab -- \
//!     <source.wav> <reference.wav> [context_a_s] [context_b_s]
//! ```
use candle_core::Device;
use cosyvoice::{CosyVoiceEngine, StreamConfig};
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

fn run(eng: &CosyVoiceEngine, src: &[f32], rf: &[f32], cfg: StreamConfig) -> Vec<f32> {
    let mut stream = eng.stream(rf, 16_000, cfg).unwrap();
    let mut out = Vec::new();
    for chunk in src.chunks(4_000) {
        stream.push(chunk, 16_000);
        while stream.ready() {
            if let Some(block) = stream.step().unwrap() {
                out.extend(block);
            }
        }
    }
    if let Some(tail) = stream.finish() {
        out.extend(tail);
    }
    out
}

fn trace(eng: &CosyVoiceEngine, label: &str, audio48k: &[f32], ref_emb: &[f32]) {
    let audio16k = resample_analysis(audio48k, 48_000, 16_000);
    let win = 3 * 16_000;
    let hop = 16_000 + 16_000 / 2;
    let mut i = 0;
    while i + win <= audio16k.len() {
        let seg = &audio16k[i..i + win];
        let rms = (seg.iter().map(|s| s * s).sum::<f32>() / seg.len() as f32).sqrt();
        if rms > 5e-4 {
            let emb = eng.embed_for_debug(seg).unwrap();
            println!(
                "{label} t={:5.1}s sim={:.3}",
                i as f32 / 16_000.0,
                cos(&emb, ref_emb)
            );
        }
        i += hop;
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let ctx_a: f32 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(3.0);
    let ctx_b: f32 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(6.0);

    let dev = Device::cuda_if_available(0).unwrap();
    let eng = CosyVoiceEngine::load("ckpt", &dev).unwrap();
    let src = read16(&a[0]);
    let rf = read16(&a[1]);
    let ref_emb = eng.embed_for_debug(&rf).unwrap();

    let base = StreamConfig::default();
    let cfg_a = StreamConfig {
        context: (ctx_a * 16_000.0) as usize,
        ..base
    };
    let cfg_b = StreamConfig {
        context: (ctx_b * 16_000.0) as usize,
        ..base
    };

    println!("=== context={ctx_a}s ===");
    let out_a = run(&eng, &src, &rf, cfg_a);
    trace(&eng, "A", &out_a, &ref_emb);

    println!("=== context={ctx_b}s ===");
    let out_b = run(&eng, &src, &rf, cfg_b);
    trace(&eng, "B", &out_b, &ref_emb);
}

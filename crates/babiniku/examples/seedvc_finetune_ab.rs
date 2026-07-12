//! PoC: does official-trainer fine-tuning actually improve Seed-VC's
//! reference-following on the target voice? (issue #80). Converts the
//! same source/reference pair with the base engine and a fine-tuned
//! engine, then scores both against an *independent* recording of the
//! target voice (not the clip used as the conversion reference) with
//! CosyVoice2's CAM++ as an external yardstick — same technique as
//! `vevo_vs_seedvc_similarity`.
//!
//! ```sh
//! cargo run --release -p babiniku --features cuda,seedvc \
//!     --example seedvc_finetune_ab -- <src.wav> <ref.wav> <eval-target.wav> <finetuned-ckpt-dir>
//! ```

use candle_core::{Device, Tensor};
use cosyvoice::CosyVoiceEngine;
use seedvc::pipeline::{resample, SeedVcEngine};

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.unwrap() as f32 / scale)
                .collect()
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.unwrap())
            .collect(),
    };
    (s, spec.sample_rate)
}

fn write_wav(path: &str, samples: &[f32], sr: u32) {
    let mut w = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn convert(engine: &SeedVcEngine, dev: &Device, src22: &[f32], ref22: &[f32]) -> Vec<f32> {
    let mel_len = |n: usize| (n.saturating_sub(256)) / 256 + 1;
    let t = mel_len(ref22.len()) + mel_len(src22.len());
    let noise = Tensor::randn(0f32, 1f32, (1, 80, t), dev).unwrap();
    engine.convert_offline(src22, ref22, 10, 0.7, &noise).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: seedvc_finetune_ab <src.wav> <ref.wav> <eval-target.wav> <finetuned-ckpt-dir>");
        std::process::exit(2);
    }
    let dev = Device::cuda_if_available(0).unwrap();

    let (src, src_sr) = read_wav(&a[0]);
    let (rf, ref_sr) = read_wav(&a[1]);
    let src22 = resample(&src, src_sr as usize, 22_050);
    let ref22 = resample(&rf, ref_sr as usize, 22_050);

    let base = SeedVcEngine::load("ckpt", &dev).unwrap();
    let base_out = convert(&base, &dev, &src22, &ref22);
    drop(base);
    write_wav("/tmp/seedvc_ab_base.wav", &base_out, 22_050);

    let ft = SeedVcEngine::load(&a[3], &dev).unwrap();
    let ft_out = convert(&ft, &dev, &src22, &ref22);
    drop(ft);
    write_wav("/tmp/seedvc_ab_finetune.wav", &ft_out, 22_050);
    eprintln!("wrote /tmp/seedvc_ab_base.wav and /tmp/seedvc_ab_finetune.wav for listening");

    // ---- CAM++ (CosyVoice2's, used purely as an independent yardstick) ----
    let judge = CosyVoiceEngine::load("ckpt", &dev).unwrap();
    let (eval_raw, eval_sr) = read_wav(&a[2]);
    let eval16 = resample(&eval_raw, eval_sr as usize, 16_000);
    let ref16 = resample(&rf, ref_sr as usize, 16_000);
    let base_out16 = resample(&base_out, 22_050, 16_000);
    let ft_out16 = resample(&ft_out, 22_050, 16_000);
    let embed = |audio: &[f32]| judge.embed_for_debug(audio).unwrap();

    let eval_e = embed(&eval16);
    let ref_e = embed(&ref16);
    let base_e = embed(&base_out16);
    let ft_e = embed(&ft_out16);

    println!("ref (conversion prompt) vs eval (independent same-voice clip): {:.4}", cos(&ref_e, &eval_e));
    println!("BASE      out vs eval (independent same-voice clip): {:.4}", cos(&base_e, &eval_e));
    println!("FINE-TUNE out vs eval (independent same-voice clip): {:.4}", cos(&ft_e, &eval_e));
    println!("BASE      out vs ref  (the conversion prompt itself): {:.4}", cos(&base_e, &ref_e));
    println!("FINE-TUNE out vs ref  (the conversion prompt itself): {:.4}", cos(&ft_e, &ref_e));
}

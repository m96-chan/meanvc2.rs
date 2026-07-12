//! Diagnostic for issue #82 (sibilant grit): dumps the CFM's predicted
//! mel (pre-vocoder) and a re-analysis mel of the actual vocoded output
//! (post-vocoder) as raw f32 binaries, so the two can be compared in the
//! fricative region to tell CFM-prediction failure from BigVGAN
//! reconstruction failure.
//!
//! ```sh
//! cargo run --release -p babiniku --features cuda,seedvc \
//!     --example seedvc_mel_probe -- <src.wav> <ref.wav> <ckpt-dir> <out-prefix>
//! ```
//! Writes `<out-prefix>_pred.bin` / `<out-prefix>_post.bin` (80 x T
//! row-major f32) plus a `<out-prefix>_shape.txt` with `T` per file, and
//! `<out-prefix>_wave.wav` (the vocoded output, for cross-reference).

use candle_core::{Device, Tensor};
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

fn dump_mel(mel: &Tensor, path: &str) -> usize {
    let (_, bins, frames) = mel.dims3().unwrap();
    let flat: Vec<f32> = mel.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(flat.len(), bins * frames);
    let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
    frames
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: seedvc_mel_probe <src.wav> <ref.wav> <ckpt-dir> <out-prefix> [steps]");
        std::process::exit(2);
    }
    let steps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(10);
    let dev = Device::cuda_if_available(0).unwrap();
    let engine = SeedVcEngine::load(&a[2], &dev).unwrap();

    let (src, src_sr) = read_wav(&a[0]);
    let (rf, ref_sr) = read_wav(&a[1]);
    let src22 = resample(&src, src_sr as usize, 22_050);
    let ref22 = resample(&rf, ref_sr as usize, 22_050);
    let src16 = resample(&src22, 22_050, 16_000);
    let ref16 = resample(&ref22, 22_050, 16_000);

    let s_alt = engine.whisper_features(&src16).unwrap();
    let s_ori = engine.whisper_features(&ref16).unwrap();

    let mel_src = engine.mel22(&src22).unwrap();
    let mel_ref = engine.mel22(&ref22).unwrap();
    let t_src = mel_src.dim(2).unwrap();
    let t_ref = mel_ref.dim(2).unwrap();

    let fb = engine.ref_fbank(&ref16);
    let style = engine.campplus_embed(&fb).unwrap();

    let cond = engine.regulate(&s_alt, t_src).unwrap();
    let prompt = engine.regulate(&s_ori, t_ref).unwrap();
    let cat = Tensor::cat(&[&prompt, &cond], 1).unwrap();

    let mel_len = |n: usize| (n.saturating_sub(256)) / 256 + 1;
    let t = mel_len(ref22.len()) + mel_len(src22.len());
    let noise = Tensor::randn(0f32, 1f32, (1, 80, t), &dev).unwrap();

    let vc_mel = engine
        .cfm_inference(&cat, &mel_ref, &style, &noise, steps, 0.7)
        .unwrap();
    // predicted mel for just the source (post-vocoder-input) region
    let pred_src_mel = vc_mel.narrow(2, t_ref, vc_mel.dim(2).unwrap() - t_ref).unwrap();

    let wave = engine.vocode(&pred_src_mel).unwrap();
    write_wav(&format!("{}_wave.wav", a[3]), &wave, 22_050);

    // re-analyze the actual vocoded output through the same mel extractor
    let post_mel = engine.mel22(&wave).unwrap();

    let pred_frames = dump_mel(&pred_src_mel, &format!("{}_pred.bin", a[3]));
    let post_frames = dump_mel(&post_mel, &format!("{}_post.bin", a[3]));
    std::fs::write(
        format!("{}_shape.txt", a[3]),
        format!("pred_frames={pred_frames}\npost_frames={post_frames}\nbins=80\n"),
    )
    .unwrap();
    eprintln!("wrote {}_pred.bin ({pred_frames} frames), {}_post.bin ({post_frames} frames)", a[3], a[3]);
}

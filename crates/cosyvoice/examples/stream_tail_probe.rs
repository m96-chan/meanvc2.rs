//! Per-hop probe for `CosyVoiceStream`: drives the stream and prints
//! RMS/peak/spectral-peak for every emitted hop, writing each hop to its
//! own wav — built to catch a real field bug (a droning artifact on
//! silence-dominated windows, traced to `context` never actually
//! accumulating; see `crates/cosyvoice/src/stream.rs` module docs) and
//! kept as a regression aid for the same class of problem.
//!
//! ```sh
//! cargo run --release -p cosyvoice --features cuda --example stream_tail_probe -- \
//!     <source.wav> [reference.wav] [out_dir]
//! ```
use candle_core::Device;
use cosyvoice::{CosyVoiceEngine, StreamConfig};

fn read16(p: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16_000);
    r.samples::<i32>()
        .step_by(spec.channels as usize)
        .map(|v| v.unwrap() as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32)
        .collect()
}

fn write_wav(path: &std::path::Path, audio: &[f32]) {
    let mut w = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: cosyvoice::stream::STREAM_OUT_SR,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for s in audio {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let src_path = args
        .first()
        .expect("usage: <source.wav> [reference.wav] [out_dir]");
    let ref_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("ckpt/F19_01_16k.wav");
    let out_dir = args.get(2).map(String::as_str).unwrap_or(".");

    let dev = Device::cuda_if_available(0).unwrap();
    let eng = CosyVoiceEngine::load("ckpt", &dev).unwrap();
    let src = read16(src_path);
    let rf = read16(ref_path);

    let cfg = StreamConfig::default();
    let mut stream = eng.stream(&rf, 16_000, cfg).unwrap();
    let mut hop_i = 0;
    for chunk in src.chunks(4_000) {
        stream.push(chunk, 16_000);
        while stream.ready() {
            let Some(out) = stream.step().unwrap() else {
                break;
            };
            let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
            let peak = out.iter().cloned().fold(0f32, |m, s| m.max(s.abs()));
            write_wav(
                &std::path::Path::new(out_dir).join(format!("hop_{hop_i}.wav")),
                &out,
            );
            println!("hop {hop_i}: rms={rms:.5} peak={peak:.4} len={}", out.len());
            hop_i += 1;
        }
    }
    if let Some(tail) = stream.finish() {
        println!("final tail: len={}", tail.len());
        write_wav(
            &std::path::Path::new(out_dir).join("hop_final_tail.wav"),
            &tail,
        );
    }
}

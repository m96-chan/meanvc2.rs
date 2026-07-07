//! Offline (single-pass) conversion for A/B against the streaming path.
//! cargo run --release -p seedvc --features cuda --example offline_convert -- <src.wav> <ref.wav> <out.wav> [steps]
use candle_core::Device;
use seedvc::pipeline::{resample, SeedVcEngine};

fn read16(p: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let sc = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().step_by(spec.channels as usize).map(|v| v.unwrap() as f32 / sc).collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().step_by(spec.channels as usize).map(|v| v.unwrap()).collect(),
    };
    (s, spec.sample_rate)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let steps: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(10);
    let dev = Device::cuda_if_available(0).unwrap();
    let eng = SeedVcEngine::load(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt"), &dev).unwrap();
    let (src, sr1) = read16(&a[0]);
    let (rf, sr2) = read16(&a[1]);
    let src22 = resample(&src, sr1 as usize, 22_050);
    let ref22 = resample(&rf, sr2 as usize, 22_050);
    // 適当な固定ノイズ(melフレーム数から): mel = (len-1024)/256+1 …エンジン内計算と一致させるため一度melを取る
    // MelExtractor pads (n_fft-hop)/2 both sides: frames = (n-256)/256+1.
    let mel_len = |n: usize| (n.saturating_sub(256)) / 256 + 1;
    let t = mel_len(ref22.len()) + mel_len(src22.len());
    let noise = candle_core::Tensor::randn(0f32, 1f32, (1, 80, t), &dev).unwrap();
    let out = eng.convert_offline(&src22, &ref22, steps, 0.7, &noise).unwrap();
    let out48 = seedvc::pipeline::resample_width(&out, 22_050, 48_000, 16);
    let mut w = hound::WavWriter::create(&a[2], hound::WavSpec { channels: 1, sample_rate: 48_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int }).unwrap();
    for s in &out48 { w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap(); }
    w.finalize().unwrap();
    println!("wrote {} ({} samples)", a[2], out48.len());
}

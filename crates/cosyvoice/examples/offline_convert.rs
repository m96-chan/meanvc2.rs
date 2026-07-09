//! Offline (single-pass) conversion for A/B and needle-scan material.
//! cargo run --release -p cosyvoice --features cuda --example offline_convert -- <src.wav> <ref.wav> <out.wav>
use candle_core::Device;
use cosyvoice::CosyVoiceEngine;

fn read(p: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
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
    (s, spec.sample_rate)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dev = Device::cuda_if_available(0).unwrap();
    println!("device: {dev:?}");
    let eng = CosyVoiceEngine::load(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt"),
        &dev,
    )
    .unwrap();
    let (src, src_sr) = read(&a[0]);
    let (rf, ref_sr) = read(&a[1]);
    let t0 = std::time::Instant::now();
    let reference = eng.prepare_reference(&rf, ref_sr).unwrap();
    let out = eng.convert_offline(&src, src_sr, &reference).unwrap();
    let elapsed = t0.elapsed().as_secs_f32();
    let out_dur = out.len() as f32 / cosyvoice::MEL_SR as f32;
    println!(
        "RTF {:.3} ({:.2}s in {:.2}s)",
        elapsed / out_dur,
        out_dur,
        elapsed
    );
    let mut w = hound::WavWriter::create(
        &a[2],
        hound::WavSpec {
            channels: 1,
            sample_rate: cosyvoice::MEL_SR,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for s in &out {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
    println!("wrote {} ({} samples)", a[2], out.len());
}

fn read_wav_16k(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    assert_eq!(spec.sample_rate, 16_000);
    r.samples::<i32>()
        .step_by(spec.channels as usize)
        .map(|v| v.unwrap() as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32)
        .collect()
}

fn main() {
    let device = candle_core::Device::cuda_if_available(0).unwrap();
    println!("device: {device:?}");
    let engine = cosyvoice::CosyVoiceEngine::load("ckpt", &device).unwrap();
    let reference_audio = read_wav_16k("ckpt/F19_01_16k.wav");
    let source_audio = read_wav_16k("ckpt/ref_trimmed.wav");

    let cfg = cosyvoice::StreamConfig::default();
    let mut stream = engine.stream(&reference_audio, 16_000, cfg).unwrap();

    let t0 = std::time::Instant::now();
    let mut out = Vec::new();
    for chunk in source_audio.chunks(4_000) {
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
    let elapsed = t0.elapsed();
    let out_dur = out.len() as f32 / cosyvoice::stream::STREAM_OUT_SR as f32;
    println!(
        "streamed {:.2}s in {:.2}s wall, RTF={:.3}",
        out_dur,
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / out_dur
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: cosyvoice::stream::STREAM_OUT_SR,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create("ckpt/cosyvoice_stream_demo.wav", spec).unwrap();
    for s in &out {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();
}

//! Real wav-to-wav MeanVC v1 conversion with the official checkpoints,
//! mirroring the official `infer_ref.py`:
//!
//! source wav → kaldi fbank → Fast-U2++ (BNFs, ×4 linear interp) ─┐
//! reference wav → MelV1 prompt mel + precomputed voice print ────┼→ CARD DiT
//!                                                    (mel+1)/2 → Vocos → wav
//!
//! Setup (see issue #16):
//!   ckpt/model_200ms.safetensors   (HF ASLP-lab/MeanVC, loads as-is)
//!   ckpt/vocos.safetensors         (tools/convert_official.py)
//!   ckpt/fastu2pp.safetensors      (tools/convert_official.py)
//!   ckpt/voice_print_test.safetensors  (WavLM-Large SV embedding of the
//!       reference wav; Rust backend is issue #15, precomputed for now)
//!
//! ```sh
//! cargo run --release --example convert_v1 -- source.wav reference.wav out.wav
//! ```

use candle_core::Device;
use meanvc2::backends::{FastU2pp, FastU2ppConfig, Vocos, VocosConfig};
use meanvc2::encoders::Vocoder;
use meanvc2::v1::{interpolate_linear, KaldiFbank, MeanVc1, MeanVc1Config, MelV1};

fn read_wav(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    anyhow::ensure!(spec.sample_rate == 16_000, "expected 16 kHz wav, got {}", spec.sample_rate);
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / scale).collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    // Mono: take the first channel.
    Ok(samples.into_iter().step_by(spec.channels as usize).collect())
}

fn main() -> anyhow::Result<()> {
    // candle's CPU gemm uses a rayon pool that defaults to all logical
    // cores; on SMT machines the contention roughly triples small-chunk
    // latency (measured: vocoder chunk RTF 0.57 -> 0.06 with the pool
    // pinned to physical cores). Must run before the first tensor op.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", num_cpus::get_physical().to_string());
    }
    let args: Vec<String> = std::env::args().collect();
    let (src_path, ref_path, out_path) = match args.as_slice() {
        [_, s, r, o] => (s.clone(), r.clone(), o.clone()),
        _ => (
            "ckpt/test.wav".into(),
            "ckpt/test.wav".into(),
            "ckpt/converted.wav".into(),
        ),
    };
    let device = Device::Cpu;
    let total = std::time::Instant::now();

    // Models (all official weights).
    let model = MeanVc1::load(MeanVc1Config::default(), "ckpt/model_200ms.safetensors", &device)?;
    let asr = FastU2pp::load(FastU2ppConfig::official_meanvc1(), "ckpt/fastu2pp.safetensors", &device)?;
    let vocos = Vocos::load(VocosConfig::official_meanvc1(), "ckpt/vocos.safetensors", &device)?;
    println!("models loaded ({:.0?})", total.elapsed());

    let source = read_wav(&src_path)?;
    let reference = read_wav(&ref_path)?;
    let audio_secs = source.len() as f64 / 16_000.0;

    // Content: kaldi fbank -> Fast-U2++ (chunked-causal) -> x4 to 10 ms.
    let t0 = std::time::Instant::now();
    let fbank = KaldiFbank::new().compute(&source, &device)?.unsqueeze(0)?;
    let bnf = asr.forward(&fbank)?;
    let bnf = interpolate_linear(&bnf, 4)?;
    println!("BNFs: {:?} ({:.0?})", bnf.dims(), t0.elapsed());

    // Timbre: prompt mel (official [-1, 1] domain) + precomputed voice print.
    let prompt_mel = MelV1::new().compute(&reference, &device)?.unsqueeze(0)?;
    let vp = candle_core::safetensors::load("ckpt/voice_print_test.safetensors", &device)?;
    let spks = vp["voice_print"].unsqueeze(0)?;
    println!("prompt mel: {:?}, voice print: {:?}", prompt_mel.dims(), spks.dims());

    // Trim BNFs to whole 200 ms chunks and convert.
    let cs = model.config().chunk_size;
    let n = (bnf.dim(1)? / cs) * cs;
    let bnf = bnf.narrow(1, 0, n)?;
    let t0 = std::time::Instant::now();
    let mel = model.sample(&bnf, &prompt_mel, &spks)?;
    let vc_time = t0.elapsed();

    // Vocos expects (mel + 1) / 2.
    let t0 = std::time::Instant::now();
    let mel01 = ((mel.squeeze(0)? + 1.0)? / 2.0)?;
    let wav = vocos.synthesize(&mel01)?;
    let voc_time = t0.elapsed();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&out_path, spec)?;
    let peak = wav.iter().fold(0f32, |m, s| m.max(s.abs())).max(1e-9);
    for s in &wav {
        writer.write_sample(((s / peak.max(1.0)) * 32_767.0) as i16)?;
    }
    writer.finalize()?;

    let rms = (wav.iter().map(|s| s * s).sum::<f32>() / wav.len() as f32).sqrt();
    println!(
        "wrote {out_path}: {} samples ({:.2} s), peak {peak:.3}, rms {rms:.4}",
        wav.len(),
        wav.len() as f32 / 16_000.0,
    );
    println!(
        "VC {vc_time:.0?} (RTF {:.3}) + vocoder {voc_time:.0?} (RTF {:.3}); total {:.0?} for {audio_secs:.2} s input",
        vc_time.as_secs_f64() / audio_secs,
        voc_time.as_secs_f64() / audio_secs,
        total.elapsed(),
    );
    Ok(())
}

//! End-to-end wav-to-wav pipeline demo with randomly initialized weights:
//!
//! source wav → Fast-U2++ (BNFs) ─┐
//!                                ├→ UTTE → FRC-DiT (1-NFE mel) → Vocos → wav
//! reference wav → ECAPA-TDNN ────┘
//!
//! With trained checkpoints (`FastU2pp::load`, `Ecapa::load`,
//! `MeanVc2::load`, `Vocos::load`) this becomes actual voice conversion;
//! here it validates shapes, chunk flow, and latency accounting.
//!
//! ```sh
//! cargo run --release --example pipeline_demo
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use meanvc2::backends::{Ecapa, EcapaConfig, FastU2pp, FastU2ppConfig, Vocos, VocosConfig};
use meanvc2::encoders::{upsample_bnf, SemanticEncoder, SpeakerEncoder, Vocoder};
use meanvc2::{MeanVc2, MeanVc2Config, StreamingConverter};

fn sine(freq: f32, secs: f32, sr: usize) -> Vec<f32> {
    (0..(secs * sr as f32) as usize)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * 0.3)
        .collect()
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let sr = 16_000;

    // Random-weight models (paper-scale configs).
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cfg = MeanVc2Config::default();
    let model = MeanVc2::new(cfg.clone(), vb.pp("meanvc2"))?;
    let asr = FastU2pp::new(FastU2ppConfig::default(), vb.pp("asr"))?;
    let spk_enc = Ecapa::new(EcapaConfig::default(), vb.pp("spk"))?;
    let vocoder = Vocos::new(VocosConfig::for_mel(&cfg.mel), vb.pp("vocos"))?;

    // Synthetic "recordings".
    let source = sine(220.0, 2.0, sr);
    let reference = sine(440.0, 2.0, sr);

    let t0 = std::time::Instant::now();
    let speaker = spk_enc.embed(&reference, sr)?; // [192]
    println!("speaker embedding: {:?} ({:.0?})", speaker.dims(), t0.elapsed());

    let t0 = std::time::Instant::now();
    let bnf = asr.extract(&source, sr)?; // [1, T40ms, 256]
    let (_, num_bnf, _) = bnf.dims3()?;
    println!("BNFs: {:?} ({:.0?})", bnf.dims(), t0.elapsed());

    // Stream: one 40 ms BNF frame per push, upsampled to the mel frame rate.
    let mut converter = StreamingConverter::new(&model, &speaker)?;
    let mut mel_chunks: Vec<Tensor> = Vec::new();
    let mut first_packet = None;
    let mut vc_time = std::time::Duration::ZERO;
    let mut voc_time = std::time::Duration::ZERO;
    let mut wav: Vec<f32> = Vec::new();
    for i in 0..num_bnf {
        let frame = bnf.narrow(1, i, 1)?;
        let chunk = upsample_bnf(&frame, cfg.decoder.chunk_frames)?;
        let t = std::time::Instant::now();
        let ready = converter.push(&chunk)?;
        vc_time += t.elapsed();
        if first_packet.is_none() && !ready.is_empty() {
            first_packet = Some((i + 1, vc_time));
        }
        for mel in ready {
            // Per-chunk vocoding is wasteful (full ConvNeXt receptive field
            // recomputed each 40 ms); a streaming vocoder cache is follow-up
            // work in issue #4. Real deployments also need a mel overlap to
            // avoid conv edge effects.
            let t = std::time::Instant::now();
            wav.extend(vocoder.synthesize(&mel.squeeze(0)?)?);
            voc_time += t.elapsed();
            mel_chunks.push(mel);
        }
    }
    mel_chunks.extend(converter.finish()?);

    if let Some((pushes, at)) = first_packet {
        println!(
            "first packet after {pushes} x 40 ms chunks (compute {at:.0?}); \
             paper accounting: ASR 80 ms buffering + 40 ms look-ahead ≈ 110 ms first-packet latency"
        );
    }
    let audio_secs = num_bnf as f64 * 0.04;
    println!(
        "VC module: {vc_time:.0?} (RTF ≈ {:.3}); per-chunk vocoder: {voc_time:.0?} (RTF ≈ {:.3})",
        vc_time.as_secs_f64() / audio_secs,
        voc_time.as_secs_f64() / audio_secs,
    );

    // Full-utterance vocoding for comparison (vocoder throughput without
    // the per-chunk recompute overhead).
    let mel_full = Tensor::cat(&mel_chunks, 1)?.squeeze(0)?;
    let t = std::time::Instant::now();
    let wav_full = vocoder.synthesize(&mel_full)?;
    println!(
        "full-utterance vocoder: {:.0?} (RTF ≈ {:.3}), {} samples ({:.2} s)",
        t.elapsed(),
        t.elapsed().as_secs_f64() / audio_secs,
        wav_full.len(),
        wav_full.len() as f32 / sr as f32,
    );
    assert!(wav.iter().all(|s| s.is_finite()));
    assert!(wav_full.iter().all(|s| s.is_finite()));
    Ok(())
}

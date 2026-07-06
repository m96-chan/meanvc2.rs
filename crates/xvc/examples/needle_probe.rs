//! Debug probe for issue #42 needle pulses: streams a wav and reports,
//! per window, the peak/rms of the input window, the raw decoded window
//! waveform, and the emitted `current` slice — so a needle can be
//! attributed to a pipeline stage.
//!
//! ```sh
//! cargo run --release -p xvc --features cuda --example needle_probe -- \
//!     <source.wav> <reference.wav>
//! ```

use candle_core::Device;
use xvc::{StreamConfig, XvcEngine};

fn read_wav_16k(path: &str) -> anyhow::Result<Vec<f64>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    anyhow::ensure!(spec.sample_rate == 16_000, "need 16 kHz: {path}");
    Ok(match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f64;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.map(|s| s as f64 / scale))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.map(|s| s as f64))
            .collect::<Result<_, _>>()?,
    })
}

fn stats(x: &[f32]) -> (f32, f32) {
    let peak = x.iter().fold(0f32, |m, s| m.max(s.abs()));
    let rms = (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt();
    (peak, rms)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let src = args.next().expect("source.wav");
    let refp = args.next().expect("reference.wav");
    let device = if std::env::var("XVC_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let engine = XvcEngine::load("ckpt", &device)?;
    // Match the demo: the reference goes through the official preprocess
    // (volume normalization + high-pass) before embedding.
    let ref_prep = engine.preprocess(&read_wav_16k(&refp)?);
    let reference = engine.prepare_reference(&ref_prep)?;
    let source = xvc::preprocess::preprocess(&read_wav_16k(&src)?, &Default::default());

    let cfg = StreamConfig::default();
    let cur = cfg.current_len();
    let hist = cfg.history_len();
    let total = cfg.chunk_len();
    println!("window layout: hist {hist} cur {cur} total {total}");
    // Mirror the windower: window i covers [i*cur - hist, i*cur - hist + total)
    let n_windows = source.len().div_ceil(cur);
    for i in 0..n_windows {
        let start = (i * cur) as i64 - hist as i64;
        let mut window = vec![0f32; total];
        for (k, slot) in window.iter_mut().enumerate() {
            let idx = start + k as i64;
            if idx >= 0 && (idx as usize) < source.len() {
                *slot = source[idx as usize];
            }
        }
        if window.iter().all(|&s| s == 0.0) {
            continue;
        }
        let out = engine.forward_window(&window, &reference)?;
        let wav: Vec<f32> = out.wav.flatten_all()?.to_vec1()?;
        let emitted = &wav[hist..hist + cur];
        let (wp, wr) = stats(&window);
        let (dp, dr) = stats(&wav);
        let (ep, er) = stats(emitted);
        let argmax = wav
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(k, _)| k)
            .unwrap_or(0);
        // Per-frame (320-sample / 50 Hz) L2 norms of each stage tensor:
        // which stage first spikes at the needle frame?
        let frame_norms = |t: &candle_core::Tensor| -> anyhow::Result<(usize, f32)> {
            let t = t.squeeze(0)?; // [T, C] or [C, T]
            let (a, b) = t.dims2()?;
            // Time is the SHORT dim (≈32 frames vs ≥512 channels).
            let t = if a > b { t.transpose(0, 1)? } else { t }; // [T, C]
            let n = t.sqr()?.sum(1)?.sqrt()?.to_vec1::<f32>()?;
            let mut sorted = n.clone();
            sorted.sort_by(|x, y| x.total_cmp(y));
            let med = sorted[sorted.len() / 2] + 1e-6;
            let (kmax, vmax) = n
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .unwrap();
            Ok((kmax, vmax / med))
        };
        let (cf, cr) = frame_norms(&out.converter_out)?;
        let (pf, pr) = frame_norms(&out.prenet_out)?;
        let (zf, zr) = frame_norms(&out.acoustic_zq)?;
        let (sf, sr2) = frame_norms(&out.sem_adapter_out)?;
        let needle_frame = argmax / 320;
        println!(
            "      needle_frame {needle_frame:2}  conv @{cf:2} x{cr:.1}  prenet @{pf:2} x{pr:.1}  zq @{zf:2} x{zr:.1}  sem @{sf:2} x{sr2:.1}"
        );
        // Needle score: emitted peak vs emitted rms.
        let crest = ep / (er + 1e-6);
        let mark = if crest > 5.0 { "  <-- NEEDLE?" } else { "" };
        println!(
            "w{i:3} t={:6.2}s  in p{wp:.3}/r{wr:.3}  dec p{dp:.3}/r{dr:.3}@{argmax:5}  emit p{ep:.3}/r{er:.3} crest {crest:4.1}{mark}",
            i as f32 * cur as f32 / 16000.0
        );
    }
    Ok(())
}

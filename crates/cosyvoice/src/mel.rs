//! Audio front-ends for the CosyVoice2 VC path.
//!
//! Three distinct feature extractors feed the pipeline (§2.2 of the
//! CosyVoice 2 paper; shapes follow `[batch, time, dim]` unless noted):
//!
//! * [`whisper_mel128`] — Whisper-style log-mel, 128 bins @ 16 kHz
//!   (n_fft 400, hop 160, center reflect-pad, last frame dropped) — the
//!   FSQ tokenizer input. Returns `[1, 128, frames]`.
//! * [`hifigan_mel80`] — HiFi-GAN mel, 80 bins @ 24 kHz (n_fft 1920,
//!   hop 480, `(n_fft-hop)/2` reflect pad, natural log, clamp 1e-5) —
//!   prompt features / vocoder-domain mel. Returns `[1, frames, 80]`.
//! * [`kaldi_fbank80`] — Kaldi-compatible fbank, 80 bins @ 16 kHz
//!   (povey window, pre-emphasis 0.97, DC removal, 512-point FFT) with
//!   per-bin mean subtraction — the CAM++ speaker-encoder input.
//!   Returns `[frames, 80]`.
//!
//! The slaney/whisper mel filterbanks are precomputed by
//! `tools/convert_cosyvoice.py` into `cosyvoice_mel.safetensors`; the Kaldi
//! banks are cheap and synthesised here.

use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};
use std::path::Path;
use vc_core::Result;

/// Loaded mel filterbanks (from `cosyvoice_mel.safetensors`).
pub struct MelFrontend {
    /// `[128, 201]` whisper filters for 16 kHz / n_fft 400.
    whisper_fb: Vec<f32>,
    /// `[80, 961]` slaney filters for 24 kHz / n_fft 1920 / fmax 8000.
    hifi_fb: Vec<f32>,
}

fn stft_mag(
    audio: &[f32],
    n_fft: usize,
    hop: usize,
    window: &[f32],
    center: bool,
    power: bool,
) -> (Vec<f32>, usize) {
    // reflect-pad
    let pad = if center { n_fft / 2 } else { (n_fft - hop) / 2 };
    let mut x = Vec::with_capacity(audio.len() + 2 * pad);
    for i in (1..=pad).rev() {
        x.push(audio[i.min(audio.len() - 1)]);
    }
    x.extend_from_slice(audio);
    for i in 2..=pad + 1 {
        x.push(audio[audio.len().saturating_sub(i)]);
    }
    let n_bins = n_fft / 2 + 1;
    let frames = if x.len() >= n_fft {
        1 + (x.len() - n_fft) / hop
    } else {
        0
    };
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut out = vec![0f32; frames * n_bins];
    let mut buf = vec![Complex32::default(); n_fft];
    for f in 0..frames {
        for i in 0..n_fft {
            buf[i] = Complex32::new(x[f * hop + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for b in 0..n_bins {
            let m2 = buf[b].re * buf[b].re + buf[b].im * buf[b].im;
            out[f * n_bins + b] = if power { m2 } else { (m2 + 1e-9).sqrt() };
        }
    }
    (out, frames)
}

/// Periodic Hann window of length `n` (`torch.hann_window(n)`).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (std::f32::consts::PI * i as f32 / n as f32).sin();
            v * v
        })
        .collect()
}

impl MelFrontend {
    /// Load the precomputed filterbanks from `cosyvoice_mel.safetensors`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let t = candle_core::safetensors::load(path.as_ref(), &Device::Cpu)?;
        let whisper_fb = t["whisper_mel_fb_128_16k"]
            .flatten_all()?
            .to_vec1::<f32>()?;
        let hifi_fb = t["mel_fb_80_24k"].flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            whisper_fb,
            hifi_fb,
        })
    }

    /// Whisper log-mel for the FSQ tokenizer. `audio` is 16 kHz mono in
    /// [-1, 1]. Returns `[1, 128, frames]` on `device`.
    pub fn whisper_mel128(&self, audio: &[f32], device: &Device) -> Result<Tensor> {
        let n_fft = 400;
        let hop = 160;
        let window = hann_periodic(n_fft);
        let (mag2, frames) = stft_mag(audio, n_fft, hop, &window, true, true);
        // whisper drops the final stft frame
        let frames = frames.saturating_sub(1);
        let n_bins = n_fft / 2 + 1;
        let n_mels = 128;
        let mut mel = vec![0f32; n_mels * frames];
        for m in 0..n_mels {
            let fb = &self.whisper_fb[m * n_bins..(m + 1) * n_bins];
            for f in 0..frames {
                let s = &mag2[f * n_bins..(f + 1) * n_bins];
                let v: f32 = fb.iter().zip(s).map(|(a, b)| a * b).sum();
                mel[m * frames + f] = v.max(1e-10).log10();
            }
        }
        let max = mel.iter().cloned().fold(f32::MIN, f32::max);
        for v in mel.iter_mut() {
            *v = (v.max(max - 8.0) + 4.0) / 4.0;
        }
        Ok(Tensor::from_vec(mel, (1, n_mels, frames), device)?)
    }

    /// HiFi-GAN mel (prompt feats / vocoder domain). `audio` is 24 kHz mono.
    /// Returns `[1, frames, 80]` on `device`.
    pub fn hifigan_mel80(&self, audio: &[f32], device: &Device) -> Result<Tensor> {
        let n_fft = 1920;
        let hop = 480;
        let window = hann_periodic(n_fft);
        let (mag, frames) = stft_mag(audio, n_fft, hop, &window, false, false);
        let n_bins = n_fft / 2 + 1;
        let n_mels = 80;
        let mut mel = vec![0f32; frames * n_mels];
        for f in 0..frames {
            let s = &mag[f * n_bins..(f + 1) * n_bins];
            for m in 0..n_mels {
                let fb = &self.hifi_fb[m * n_bins..(m + 1) * n_bins];
                let v: f32 = fb.iter().zip(s).map(|(a, b)| a * b).sum();
                mel[f * n_mels + m] = v.max(1e-5).ln();
            }
        }
        Ok(Tensor::from_vec(mel, (1, frames, n_mels), device)?)
    }
}

/// Kaldi-compatible 80-bin log fbank with per-bin mean subtraction —
/// the CAM++ input (`torchaudio.compliance.kaldi.fbank` with `dither=0`,
/// `num_mel_bins=80`, 16 kHz). Returns `[frames, 80]` on `device`.
pub fn kaldi_fbank80(audio: &[f32], device: &Device) -> Result<Tensor> {
    let frame_len = 400;
    let hop = 160;
    let n_fft = 512; // rounded up to a power of two
    let n_mels = 80;
    if audio.len() < frame_len {
        return Ok(Tensor::zeros((0, n_mels), candle_core::DType::F32, device)?);
    }
    let frames = 1 + (audio.len() - frame_len) / hop;
    // povey window = symmetric hann ^ 0.85
    let window: Vec<f32> = (0..frame_len)
        .map(|i| {
            let h =
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (frame_len - 1) as f32).cos();
            h.powf(0.85)
        })
        .collect();
    // kaldi mel banks: 80 triangles over bins 0..256, mel = 1127 ln(1+f/700),
    // low 20 Hz, high 8000 Hz (nyquist)
    let mel_of = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
    let (mel_lo, mel_hi) = (mel_of(20.0), mel_of(8000.0));
    let mel_delta = (mel_hi - mel_lo) / (n_mels + 1) as f32;
    let fft_bin_width = 16000.0 / n_fft as f32;
    let n_bins = n_fft / 2; // kaldi banks exclude the nyquist bin
    let mut banks = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = mel_lo + m as f32 * mel_delta;
        let center = mel_lo + (m + 1) as f32 * mel_delta;
        let right = mel_lo + (m + 2) as f32 * mel_delta;
        for b in 0..n_bins {
            let mel = mel_of(fft_bin_width * b as f32);
            if mel > left && mel < right {
                banks[m * n_bins + b] = if mel <= center {
                    (mel - left) / (center - left)
                } else {
                    (right - mel) / (right - center)
                };
            }
        }
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut out = vec![0f32; frames * n_mels];
    let mut frame = vec![0f32; frame_len];
    let mut buf = vec![Complex32::default(); n_fft];
    for fi in 0..frames {
        let src = &audio[fi * hop..fi * hop + frame_len];
        // DC removal
        let mean: f32 = src.iter().sum::<f32>() / frame_len as f32;
        for i in 0..frame_len {
            frame[i] = src[i] - mean;
        }
        // pre-emphasis 0.97 (first sample against itself, kaldi-style)
        for i in (1..frame_len).rev() {
            frame[i] -= 0.97 * frame[i - 1];
        }
        frame[0] -= 0.97 * frame[0];
        for i in 0..n_fft {
            buf[i] = if i < frame_len {
                Complex32::new(frame[i] * window[i], 0.0)
            } else {
                Complex32::default()
            };
        }
        fft.process(&mut buf);
        for m in 0..n_mels {
            let fb = &banks[m * n_bins..(m + 1) * n_bins];
            let mut v = 0f32;
            for b in 0..n_bins {
                v += fb[b] * (buf[b].re * buf[b].re + buf[b].im * buf[b].im);
            }
            out[fi * n_mels + m] = v.max(f32::EPSILON).ln();
        }
    }
    // per-bin mean subtraction (CosyVoice frontend)
    for m in 0..n_mels {
        let mean: f32 = (0..frames).map(|f| out[f * n_mels + m]).sum::<f32>() / frames as f32;
        for f in 0..frames {
            out[f * n_mels + m] -= mean;
        }
    }
    Ok(Tensor::from_vec(out, (frames, n_mels), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn ckpt(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt")
            .join(name);
        p.exists().then_some(p)
    }

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let p = ckpt("cosyvoice_e2e_fixture.safetensors")?;
        candle_core::safetensors::load(p, &Device::Cpu).ok()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn whisper_mel_matches_official() {
        let (Some(fx), Some(mp)) = (fixture(), ckpt("cosyvoice_mel.safetensors")) else {
            return;
        };
        let fe = MelFrontend::load(mp).unwrap();
        let audio = fx["prompt_16k"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mel = fe.whisper_mel128(&audio, &Device::Cpu).unwrap();
        let d = max_abs_diff(&mel, &fx["prompt_mel128"]);
        assert!(d < 2e-4, "whisper mel max abs diff {d}");
    }

    #[test]
    fn kaldi_fbank_matches_official() {
        let Some(fx) = fixture() else { return };
        let audio = fx["prompt_16k"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let fb = kaldi_fbank80(&audio, &Device::Cpu).unwrap();
        let d = max_abs_diff(&fb, &fx["prompt_fbank"]);
        assert!(d < 2e-3, "kaldi fbank max abs diff {d}");
    }
}

//! Mel front-end matching Amphion's `models/codec/melvqgan/melspec.py`
//! `MelSpectrogram`: 128 bins @ 24 kHz, n_fft 1920 / hop 480 / win 1920,
//! `center=False` (reflect-padded by `(n_fft - hop) / 2` each side),
//! `fmin=0` / `fmax=12000` (Nyquist), `ln(clamp(x, 1e-5))`.

use candle_core::{DType, Device, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};

use vc_core::Result;

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub n_fft: usize,
    pub num_mels: usize,
    pub sample_rate: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub fmin: f32,
    pub fmax: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_fft: 1920,
            num_mels: 128,
            sample_rate: 24_000,
            hop_size: 480,
            win_size: 1920,
            fmin: 0.0,
            fmax: 12_000.0,
        }
    }
}

/// Slaney-style librosa mel filterbank (matches `librosa.filters.mel`
/// with the default `norm="slaney"`, `htk=False`).
fn mel_filterbank(cfg: &MelConfig) -> Vec<Vec<f32>> {
    let n_freqs = cfg.n_fft / 2 + 1;
    let hz_to_mel = |hz: f32| -> f32 {
        let f_min = 0f32;
        let f_sp = 200.0 / 3.0;
        let mut mel = (hz - f_min) / f_sp;
        let min_log_hz = 1000f32;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if hz >= min_log_hz {
            mel = min_log_mel + (hz / min_log_hz).ln() / logstep;
        }
        mel
    };
    let mel_to_hz = |mel: f32| -> f32 {
        let f_min = 0f32;
        let f_sp = 200.0 / 3.0;
        let mut hz = f_min + f_sp * mel;
        let min_log_hz = 1000f32;
        let min_log_mel = (min_log_hz - f_min) / f_sp;
        let logstep = (6.4f32).ln() / 27.0;
        if mel >= min_log_mel {
            hz = min_log_hz * (logstep * (mel - min_log_mel)).exp();
        }
        hz
    };

    let n_mels = cfg.num_mels;
    let min_mel = hz_to_mel(cfg.fmin);
    let max_mel = hz_to_mel(cfg.fmax);
    let mel_pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_pts: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32)
        .collect();

    let mut weights = vec![vec![0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        let (f_left, f_center, f_right) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
        let enorm = 2.0 / (f_right - f_left);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - f_left) / (f_center - f_left);
            let upper = (f_right - f) / (f_right - f_center);
            let w = lower.min(upper).max(0.0);
            weights[m][k] = w * enorm;
        }
    }
    weights
}

pub struct MelSpectrogram {
    cfg: MelConfig,
    filterbank: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    device: Device,
}

impl MelSpectrogram {
    pub fn new(cfg: MelConfig, device: &Device) -> Self {
        let filterbank = mel_filterbank(&cfg);
        // torch.hann_window default (periodic=True): sin^2(pi*n/N), n=0..N-1.
        let window: Vec<f32> = (0..cfg.win_size)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / cfg.win_size as f32;
                x.sin().powi(2)
            })
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(cfg.n_fft);
        Self {
            cfg,
            filterbank,
            window,
            fft,
            device: device.clone(),
        }
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    /// `wav`: mono samples. Returns `[num_mels, frames]`.
    pub fn forward(&self, wav: &[f32]) -> Result<Tensor> {
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.hop_size;
        let pad = (n_fft - hop) / 2;

        // Reflect padding (matches torch's `mode="reflect"`).
        let mut padded = Vec::with_capacity(wav.len() + 2 * pad);
        for i in (1..=pad).rev() {
            padded.push(wav[i.min(wav.len().saturating_sub(1))]);
        }
        padded.extend_from_slice(wav);
        for i in 1..=pad {
            let idx = wav.len().saturating_sub(1).saturating_sub(i);
            padded.push(wav[idx]);
        }

        let frames = if padded.len() >= n_fft {
            (padded.len() - n_fft) / hop + 1
        } else {
            0
        };

        let n_freqs = n_fft / 2 + 1;
        let mut mel = vec![0f32; self.cfg.num_mels * frames];
        let mut buf = vec![Complex32::default(); n_fft];
        for t in 0..frames {
            let start = t * hop;
            for i in 0..n_fft {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            let mags: Vec<f32> = buf[..n_freqs]
                .iter()
                .map(|c| (c.re * c.re + c.im * c.im + 1e-9).sqrt())
                .collect();
            for m in 0..self.cfg.num_mels {
                let acc: f32 = self.filterbank[m]
                    .iter()
                    .zip(&mags)
                    .map(|(w, mag)| w * mag)
                    .sum();
                mel[m * frames + t] = acc.max(1e-5).ln();
            }
        }
        Tensor::from_vec(mel, (self.cfg.num_mels, frames), &self.device).map_err(Into::into)
    }

    /// `wav`: `[batch, samples]`. Returns `[batch, frames, num_mels]`
    /// (matches `MelSpectrogram.forward(...).transpose(1, 2)` in
    /// `vevo_utils.extract_mel_feature`, pre-normalization).
    pub fn forward_batch(&self, wav: &Tensor) -> Result<Tensor> {
        let (b, _) = wav.dims2()?;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let row: Vec<f32> = wav.get(i)?.to_dtype(DType::F32)?.to_vec1()?;
            let m = self.forward(&row)?; // [num_mels, frames]
            out.push(m.transpose(0, 1)?.unsqueeze(0)?); // [1, frames, num_mels]
        }
        Tensor::cat(&out, 0).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn shapes_are_sane() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(MelConfig::default(), &dev);
        let wav = vec![0f32; 24_000 * 3];
        let out = mel.forward(&wav).unwrap();
        assert_eq!(out.dims(), &[128, 150]);
    }

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_e2e_fixture.safetensors");
        if !path.exists() {
            return None;
        }
        Some(candle_core::safetensors::load(path, &Device::Cpu).unwrap())
    }

    #[test]
    fn matches_official_normalized() {
        let Some(fx) = fixture() else { return };
        let dev = Device::Cpu;
        let ref_24k = fx["ref_24k"].clone();
        let want = fx["ref_mel"].squeeze(0).unwrap().to_vec2::<f32>().unwrap();

        let mel = MelSpectrogram::new(MelConfig::default(), &dev);
        let raw = mel.forward_batch(&ref_24k).unwrap().squeeze(0).unwrap();
        let mean = -4.92f32;
        let var = 8.14f32;
        let norm = ((raw - mean as f64).unwrap() / (var.sqrt() as f64))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(norm.len(), want.len(), "frame count mismatch");
        let mut dmax = 0f32;
        for (gr, wr) in norm.iter().zip(&want) {
            for (g, w) in gr.iter().zip(wr) {
                dmax = dmax.max((g - w).abs());
            }
        }
        assert!(dmax < 1e-2, "max abs diff {dmax}");
    }
}

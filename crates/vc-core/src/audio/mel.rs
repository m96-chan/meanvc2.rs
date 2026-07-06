//! Log-mel-spectrogram extraction (STFT via `rustfft` + HTK mel filterbank).

use std::sync::Arc;

use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::config::MelConfig;
use crate::{Error, Result};

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Triangular HTK-style mel filterbank, `[n_mels][n_fft / 2 + 1]`.
fn mel_filterbank(cfg: &MelConfig) -> Vec<Vec<f32>> {
    let n_bins = cfg.n_fft / 2 + 1;
    let mel_min = hz_to_mel(cfg.f_min);
    let mel_max = hz_to_mel(cfg.f_max);
    let mel_points: Vec<f32> = (0..cfg.n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (cfg.n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_freq = |bin: usize| bin as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32;

    let mut banks = vec![vec![0f32; n_bins]; cfg.n_mels];
    for (m, bank) in banks.iter_mut().enumerate() {
        let (lo, center, hi) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        for (bin, w) in bank.iter_mut().enumerate() {
            let f = bin_freq(bin);
            if f > lo && f < hi {
                *w = if f <= center {
                    (f - lo) / (center - lo)
                } else {
                    (hi - f) / (hi - center)
                };
            }
        }
    }
    banks
}

/// Log-mel-spectrogram extractor.
///
/// Frames are centered (reflect padding), windowed with a Hann window of
/// `win_length` samples zero-padded to `n_fft`, and mapped through an HTK
/// mel filterbank followed by `ln(max(mel, 1e-5))`.
pub struct MelSpectrogram {
    cfg: MelConfig,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
}

impl std::fmt::Debug for MelSpectrogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelSpectrogram").field("cfg", &self.cfg).finish()
    }
}

impl MelSpectrogram {
    pub fn new(cfg: MelConfig) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(cfg.n_fft);
        let window: Vec<f32> = (0..cfg.win_length)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / cfg.win_length as f32;
                x.sin().powi(2)
            })
            .collect();
        let filterbank = mel_filterbank(&cfg);
        Self {
            cfg,
            fft,
            window,
            filterbank,
        }
    }

    pub fn config(&self) -> &MelConfig {
        &self.cfg
    }

    /// Number of mel frames produced for `num_samples` input samples.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        num_samples / self.cfg.hop_length + 1
    }

    /// Computes the log-mel-spectrogram of a mono waveform.
    ///
    /// Returns a `[frames, n_mels]` tensor on `device`.
    pub fn compute(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        if samples.is_empty() {
            return Err(Error::Input("empty waveform".into()));
        }
        let cfg = &self.cfg;
        let half = cfg.n_fft / 2;
        let n_bins = half + 1;

        // Center (reflect) padding.
        let mut padded = Vec::with_capacity(samples.len() + cfg.n_fft);
        for i in (1..=half).rev() {
            padded.push(samples[i.min(samples.len() - 1)]);
        }
        padded.extend_from_slice(samples);
        for i in 0..half {
            let idx = samples.len().saturating_sub(2 + i);
            padded.push(samples[idx]);
        }

        let frames = self.num_frames(samples.len());
        let mut out = Vec::with_capacity(frames * cfg.n_mels);
        let mut buf = vec![Complex32::default(); cfg.n_fft];
        let mut power = vec![0f32; n_bins];
        let win_offset = (cfg.n_fft - cfg.win_length) / 2;

        for frame in 0..frames {
            let start = frame * cfg.hop_length;
            buf.fill(Complex32::default());
            for (i, &w) in self.window.iter().enumerate() {
                // torch.stft convention: the win_length window is centered
                // in the n_fft frame, and the signal segment shifts with it.
                let s = padded.get(start + win_offset + i).copied().unwrap_or(0.0);
                buf[win_offset + i] = Complex32::new(s * w, 0.0);
            }
            self.fft.process(&mut buf);
            for (bin, p) in power.iter_mut().enumerate() {
                *p = buf[bin].norm_sqr();
            }
            for bank in &self.filterbank {
                let mel: f32 = bank.iter().zip(&power).map(|(w, p)| w * p).sum();
                out.push(mel.max(1e-5).ln());
            }
        }
        Ok(Tensor::from_vec(out, (frames, cfg.n_mels), device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_finiteness() {
        let cfg = MelConfig::default();
        let mel = MelSpectrogram::new(cfg.clone());
        // 1 s of a 440 Hz tone at 16 kHz.
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let t = mel.compute(&samples, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[16_000 / cfg.hop_length + 1, cfg.n_mels]);
        let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
        assert!(v.iter().flatten().all(|x| x.is_finite()));
    }
}

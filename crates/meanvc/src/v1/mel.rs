//! The official v1 mel-spectrogram domain (`MelSpectrogramFeatures` in
//! `infer_ref.py`): magnitude STFT → librosa (slaney) filterbank →
//! dB compression → linear normalization into `[-1, 1]`.
//!
//! ```text
//! S   = sqrt(re² + im² + 1e-6)                    (16 kHz, n_fft 1024,
//! M   = slaney_mel @ S                             win 640, hop 160, 80 mels)
//! dB  = 20·log10(max(10^(min_db/20), M)) − 20      min_db = −115
//! out = clamp(2·(dB − min_db)/(−min_db) − 1, −1, 1)
//! ```
//!
//! Both the DiT decoder and the official Vocos checkpoint operate in this
//! domain, which is disjoint from the crate's ln-based [`crate::audio`]
//! front-end.

use std::sync::Arc;

use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::{Error, Result};

const MIN_DB: f32 = -115.0;
const REF_DB: f32 = 20.0;

fn hz_to_slaney_mel(f: f32) -> f32 {
    if f < 1000.0 {
        3.0 * f / 200.0
    } else {
        15.0 + 27.0 * (f / 1000.0).ln() / 6.4f32.ln()
    }
}

fn slaney_mel_to_hz(m: f32) -> f32 {
    if m < 15.0 {
        200.0 * m / 3.0
    } else {
        1000.0 * (6.4f32.ln() * (m - 15.0) / 27.0).exp()
    }
}

/// Slaney-scale, slaney-normalized triangular filterbank
/// (librosa `mel` defaults), `[n_mels][n_fft / 2 + 1]`.
fn slaney_filterbank(sample_rate: usize, n_fft: usize, n_mels: usize, f_min: f32, f_max: f32) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let m_min = hz_to_slaney_mel(f_min);
    let m_max = hz_to_slaney_mel(f_max);
    let hz: Vec<f32> = (0..n_mels + 2)
        .map(|i| slaney_mel_to_hz(m_min + (m_max - m_min) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let bin_freq = |bin: usize| bin as f32 * sample_rate as f32 / n_fft as f32;

    let mut banks = vec![vec![0f32; n_bins]; n_mels];
    for (m, bank) in banks.iter_mut().enumerate() {
        let (lo, center, hi) = (hz[m], hz[m + 1], hz[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (bin, w) in bank.iter_mut().enumerate() {
            let f = bin_freq(bin);
            let lower = (f - lo) / (center - lo);
            let upper = (hi - f) / (hi - center);
            *w = lower.min(upper).max(0.0) * enorm;
        }
    }
    banks
}

/// Official v1 mel extractor (fixed 16 kHz / 1024 / 640 / 160 / 80 / 0–8 kHz).
pub struct MelV1 {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
}

impl std::fmt::Debug for MelV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelV1").finish()
    }
}

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 1024;
pub const WIN_LENGTH: usize = 640;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;

impl Default for MelV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl MelV1 {
    pub fn new() -> Self {
        // torch.hann_window(periodic): sin²(π i / N).
        let window: Vec<f32> = (0..WIN_LENGTH)
            .map(|i| (std::f32::consts::PI * i as f32 / WIN_LENGTH as f32).sin().powi(2))
            .collect();
        Self {
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window,
            filterbank: slaney_filterbank(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 8_000.0),
        }
    }

    /// `[frames, 80]` mel in `[-1, 1]`, matching the official extractor.
    pub fn compute(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        if samples.is_empty() {
            return Err(Error::Input("empty waveform".into()));
        }
        let half = N_FFT / 2;
        // Center reflect padding (torch.stft pad_mode="reflect").
        let mut padded = Vec::with_capacity(samples.len() + N_FFT);
        for i in (1..=half).rev() {
            padded.push(samples[i.min(samples.len() - 1)]);
        }
        padded.extend_from_slice(samples);
        for i in 0..half {
            let idx = samples.len().saturating_sub(2 + i);
            padded.push(samples[idx]);
        }

        let frames = samples.len() / HOP_LENGTH + 1;
        let min_level = 10f32.powf(MIN_DB / 20.0);
        let mut out = Vec::with_capacity(frames * N_MELS);
        let mut buf = vec![Complex32::default(); N_FFT];
        let mut mag = vec![0f32; half + 1];
        let win_offset = (N_FFT - WIN_LENGTH) / 2;

        for frame in 0..frames {
            let start = frame * HOP_LENGTH;
            buf.fill(Complex32::default());
            for (i, &w) in self.window.iter().enumerate() {
                // torch.stft places the zero-padded window at the center of
                // the n_fft frame, so the signal segment is offset too.
                let s = padded.get(start + win_offset + i).copied().unwrap_or(0.0);
                buf[win_offset + i] = Complex32::new(s * w, 0.0);
            }
            self.fft.process(&mut buf);
            for (bin, m) in mag.iter_mut().enumerate() {
                *m = (buf[bin].norm_sqr() + 1e-6).sqrt();
            }
            for bank in &self.filterbank {
                let mel: f32 = bank.iter().zip(&mag).map(|(w, s)| w * s).sum();
                let db = 20.0 * mel.max(min_level).log10() - REF_DB;
                let norm = (2.0 * (db - MIN_DB) / -MIN_DB - 1.0).clamp(-1.0, 1.0);
                out.push(norm);
            }
        }
        Ok(Tensor::from_vec(out, (frames, N_MELS), device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_in_unit_range() {
        let mel = MelV1::new();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let t = mel.compute(&samples, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[16_000 / HOP_LENGTH + 1, N_MELS]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| (-1.0..=1.0).contains(x)));
        // Silence sits near the floor (not exactly -1: the official chain
        // adds 1e-6 inside the magnitude sqrt).
        let silent = mel.compute(&[0.0; 3200], &Device::Cpu).unwrap();
        let sv: Vec<f32> = silent.flatten_all().unwrap().to_vec1().unwrap();
        assert!(sv.iter().all(|x| *x < -0.5));
    }
}

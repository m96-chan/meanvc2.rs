//! Kaldi-compatible log-mel filterbank features (torchaudio
//! `compliance.kaldi.fbank` with the official extraction settings:
//! 25 ms / 10 ms, dither 0, `snip_edges = true`, int16-scaled input,
//! 80 bins) — the front end of the released `fastu2++.pt`.

use std::sync::Arc;

use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::{Error, Result};

const SAMPLE_RATE: usize = 16_000;
const FRAME_LEN: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const N_FFT: usize = 512;
const N_MELS: usize = 80;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
const EPS: f32 = f32::EPSILON;

fn hz_to_kaldi_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Kaldi mel banks: triangles in mel space over FFT bins, no normalization.
fn kaldi_filterbank() -> Vec<Vec<f32>> {
    let n_bins = N_FFT / 2 + 1;
    let high = SAMPLE_RATE as f32 / 2.0;
    let m_lo = hz_to_kaldi_mel(LOW_FREQ);
    let m_hi = hz_to_kaldi_mel(high);
    let delta = (m_hi - m_lo) / (N_MELS + 1) as f32;
    let mut banks = vec![vec![0f32; n_bins]; N_MELS];
    for (m, bank) in banks.iter_mut().enumerate() {
        let left = m_lo + m as f32 * delta;
        let center = left + delta;
        let right = center + delta;
        for (bin, w) in bank.iter_mut().enumerate() {
            let mel = hz_to_kaldi_mel(bin as f32 * SAMPLE_RATE as f32 / N_FFT as f32);
            if mel > left && mel < right {
                *w = if mel <= center {
                    (mel - left) / delta
                } else {
                    (right - mel) / delta
                };
            }
        }
    }
    banks
}

/// Kaldi fbank extractor for the Fast-U2++ front end.
pub struct KaldiFbank {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
}

impl std::fmt::Debug for KaldiFbank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KaldiFbank").finish()
    }
}

impl Default for KaldiFbank {
    fn default() -> Self {
        Self::new()
    }
}

impl KaldiFbank {
    pub fn new() -> Self {
        // Povey window: hann^0.85.
        let window: Vec<f32> = (0..FRAME_LEN)
            .map(|i| {
                let hann = 0.5
                    - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_LEN - 1) as f32).cos();
                hann.powf(0.85)
            })
            .collect();
        Self {
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window,
            filterbank: kaldi_filterbank(),
        }
    }

    /// `samples` in `[-1, 1]` at 16 kHz → `[frames, 80]` log-mel features
    /// (kaldi convention: input scaled to int16 range, snip_edges framing).
    pub fn compute(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        if samples.len() < FRAME_LEN {
            return Err(Error::Input(format!(
                "need at least {FRAME_LEN} samples, got {}",
                samples.len()
            )));
        }
        let frames = 1 + (samples.len() - FRAME_LEN) / FRAME_SHIFT;
        let mut out = Vec::with_capacity(frames * N_MELS);
        let mut frame = vec![0f32; FRAME_LEN];
        let mut buf = vec![Complex32::default(); N_FFT];
        let mut power = vec![0f32; N_FFT / 2 + 1];

        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            for (i, s) in frame.iter_mut().enumerate() {
                *s = samples[start + i] * 32_768.0; // int16 scaling
            }
            // Remove DC offset.
            let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }
            // Pre-emphasis (kaldi: x[0] -= p * x[0]).
            for i in (1..FRAME_LEN).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];

            buf.fill(Complex32::default());
            for i in 0..FRAME_LEN {
                buf[i] = Complex32::new(frame[i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (bin, p) in power.iter_mut().enumerate() {
                *p = buf[bin].norm_sqr();
            }
            for bank in &self.filterbank {
                let mel: f32 = bank.iter().zip(&power).map(|(w, p)| w * p).sum();
                out.push(mel.max(EPS).ln());
            }
        }
        Ok(Tensor::from_vec(out, (frames, N_MELS), device)?)
    }
}

/// Linear ×`factor` upsampling along the time axis with
/// `align_corners = true` (`F.interpolate(..., mode="linear")`), used to
/// bring 40 ms BNFs to the 10 ms mel frame rate.
pub fn interpolate_linear(x: &Tensor, factor: usize) -> Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    let out_t = t * factor;
    let data: Vec<Vec<Vec<f32>>> = x.to_vec3()?;
    let mut out = Vec::with_capacity(b * out_t * d);
    for batch in data.iter().take(b) {
        for j in 0..out_t {
            // align_corners: src = j * (t - 1) / (out_t - 1).
            let src = if out_t > 1 {
                j as f32 * (t - 1) as f32 / (out_t - 1) as f32
            } else {
                0.0
            };
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(t - 1);
            let w = src - i0 as f32;
            for (a, b) in batch[i0].iter().zip(&batch[i1]) {
                out.push(a * (1.0 - w) + b * w);
            }
        }
    }
    Ok(Tensor::from_vec(out, (b, out_t, d), x.device())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbank_shapes_and_finiteness() {
        let fb = KaldiFbank::new();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect();
        let t = fb.compute(&samples, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1 + (16_000 - 400) / 160, 80]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn interpolation_endpoints_align() {
        let x = Tensor::from_vec(vec![0f32, 1., 2., 3.], (1, 2, 2), &Device::Cpu).unwrap();
        let up = interpolate_linear(&x, 3).unwrap();
        let v: Vec<Vec<f32>> = up.squeeze(0).unwrap().to_vec2().unwrap();
        assert_eq!(v[0], vec![0., 1.]); // first frame preserved
        assert_eq!(v[5], vec![2., 3.]); // last frame preserved (align_corners)
        assert!(v[2][0] > v[1][0] && v[3][0] > v[2][0]); // monotone ramp
    }
}

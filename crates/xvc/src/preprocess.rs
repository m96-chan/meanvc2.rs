//! X-VC audio preprocessing and the Whisper 128-mel front-end.
//!
//! Mirrors the official inference chain
//! (`models/codec/sac/utils.py::process_audio` +
//! `transformers.WhisperFeatureExtractor`, X-VC arXiv:2604.12456 / the
//! GLM-4-Voice tokenizer front-end):
//!
//! 1. [`preprocess`] — percentile **volume normalization** (target
//!    coefficient 0.2), **40 Hz high-pass biquad** (RBJ, Q = 0.707, the
//!    `torchaudio.functional.highpass_biquad` design) and zero-padding to a
//!    `latent_hop_length` (1280-sample) multiple. The official code runs
//!    this in float64 (`soundfile` output) and casts to f32 at the end;
//!    this module does the same, so parity is < 1e-5 abs.
//! 2. [`WhisperFeatureExtractor`] — the 128-bin Whisper log-mel
//!    spectrogram (n_fft 400, hop 160, Hann window, Slaney-scale /
//!    Slaney-normalized filterbank, `log10`, per-utterance dynamic-range
//!    clamp to `max - 8`, then `(x + 4) / 4`), with the input padded to the
//!    tokenizer stride (2 · 4 · 160 = 1280 samples) and the matching
//!    frame-level attention mask.

use std::sync::Arc;

use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use vc_core::{Error, Result};

/// Configuration of [`preprocess`] (`configs/xvc.yaml` values).
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Input sample rate in Hz (the caller is responsible for resampling).
    pub sample_rate: usize,
    /// Apply percentile volume normalization (`volume_normalize`).
    pub volume_normalize: bool,
    /// High-pass cutoff in Hz (`highpass_cutoff_freq`); 0 disables.
    pub highpass_cutoff_hz: f64,
    /// Zero-pad the output to a multiple of this many samples
    /// (`latent_hop_length`).
    pub pad_multiple: usize,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            volume_normalize: true,
            highpass_cutoff_hz: 40.0,
            pad_multiple: 1280,
        }
    }
}

/// Percentile volume normalization (`utils/audio.py::audio_volume_normalize`,
/// `coeff = 0.2`).
///
/// Rescales so that the mean of the 90th–99th percentile of significant
/// (> 0.01) absolute sample values reaches `coeff`, with the scale factor
/// clamped to `[0.1, 10]` and a final peak clamp at 1.0. Operates on (and
/// returns) float64 samples like the official NumPy code.
pub fn volume_normalize(audio: &[f64], coeff: f64) -> Vec<f64> {
    let mut temp: Vec<f64> = audio.iter().map(|s| s.abs()).collect();
    temp.sort_by(|a, b| a.total_cmp(b));
    let mut audio = audio.to_vec();
    if let Some(&max) = temp.last() {
        if max < 0.1 {
            let scale = max.max(1e-3);
            for s in &mut audio {
                *s = *s / scale * 0.1;
            }
        }
    }
    // NOTE: like the official code, the percentile statistics below use the
    // *original* amplitudes even when the small-signal rescale above fired.
    let significant: Vec<f64> = temp.into_iter().filter(|&s| s > 0.01).collect();
    let len = significant.len();
    if len <= 10 {
        return audio;
    }
    let (lo, hi) = ((0.9 * len as f64) as usize, (0.99 * len as f64) as usize);
    let volume = significant[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
    let scale = (coeff / volume).clamp(0.1, 10.0);
    for s in &mut audio {
        *s *= scale;
    }
    let peak = audio.iter().fold(0f64, |m, s| m.max(s.abs()));
    if peak > 1.0 {
        for s in &mut audio {
            *s /= peak;
        }
    }
    audio
}

/// RBJ high-pass biquad (`torchaudio.functional.highpass_biquad`,
/// Q = 0.707), zero initial conditions, output clamped to `[-1, 1]` like
/// `torchaudio.functional.lfilter(..., clamp=True)`.
pub fn highpass_biquad(audio: &[f64], sample_rate: f64, cutoff_hz: f64) -> Vec<f64> {
    const Q: f64 = 0.707;
    let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate;
    let alpha = w0.sin() / 2.0 / Q;
    let cos_w0 = w0.cos();

    let a0 = 1.0 + alpha;
    let (b0, b1, b2) = (
        (1.0 + cos_w0) / 2.0 / a0,
        (-1.0 - cos_w0) / a0,
        (1.0 + cos_w0) / 2.0 / a0,
    );
    let (a1, a2) = ((-2.0 * cos_w0) / a0, (1.0 - alpha) / a0);

    let mut out = Vec::with_capacity(audio.len());
    let (mut x1, mut x2, mut y1, mut y2) = (0f64, 0f64, 0f64, 0f64);
    for &x0 in audio {
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        out.push(y0);
        (x2, x1) = (x1, x0);
        (y2, y1) = (y1, y0);
    }
    for y in &mut out {
        *y = y.clamp(-1.0, 1.0);
    }
    out
}

/// The official preprocessing chain: volume normalization + high-pass +
/// zero-pad to a [`PreprocessConfig::pad_multiple`] boundary, float64 in
/// (`soundfile`-style samples in `[-1, 1]`), float32 out.
pub fn preprocess(audio: &[f64], cfg: &PreprocessConfig) -> Vec<f32> {
    let audio = if cfg.volume_normalize {
        volume_normalize(audio, 0.2)
    } else {
        audio.to_vec()
    };
    let audio = if cfg.highpass_cutoff_hz != 0.0 {
        highpass_biquad(&audio, cfg.sample_rate as f64, cfg.highpass_cutoff_hz)
    } else {
        audio
    };
    let mut out: Vec<f32> = audio.into_iter().map(|s| s as f32).collect();
    let rem = out.len() % cfg.pad_multiple;
    if rem != 0 {
        out.resize(out.len() + cfg.pad_multiple - rem, 0.0);
    }
    out
}

/// Slaney-scale Hz → mel (`transformers.audio_utils.hertz_to_mel`).
fn hertz_to_mel(freq: f64) -> f64 {
    if freq >= 1000.0 {
        15.0 + (freq / 1000.0).ln() * (27.0 / 6.4f64.ln())
    } else {
        3.0 * freq / 200.0
    }
}

/// Slaney-scale mel → Hz (`transformers.audio_utils.mel_to_hertz`).
fn mel_to_hertz(mel: f64) -> f64 {
    if mel >= 15.0 {
        1000.0 * (6.4f64.ln() / 27.0 * (mel - 15.0)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// Slaney-normalized triangular mel filterbank,
/// `[n_mels][n_fft / 2 + 1]` (`transformers.audio_utils.mel_filter_bank`
/// with `norm="slaney", mel_scale="slaney"`).
fn slaney_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize, f_max: f64) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let mel_max = hertz_to_mel(f_max);
    let filter_freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hertz(mel_max * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freq = |bin: usize| (sample_rate / 2) as f64 * bin as f64 / (n_bins - 1) as f64;

    let mut banks = vec![vec![0f64; n_bins]; n_mels];
    for (m, bank) in banks.iter_mut().enumerate() {
        let (lo, center, hi) = (filter_freqs[m], filter_freqs[m + 1], filter_freqs[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (bin, w) in bank.iter_mut().enumerate() {
            let f = fft_freq(bin);
            let down = (f - lo) / (center - lo);
            let up = (hi - f) / (hi - center);
            *w = down.min(up).max(0.0) * enorm;
        }
    }
    banks
}

/// Log-mel features + frame-level attention mask, the
/// `WhisperFeatureExtractor(..., padding="longest", pad_to_multiple_of=1280,
/// return_attention_mask=True)` output pair.
#[derive(Debug)]
pub struct MelFeatures {
    /// `[1, n_mels, frames]` log-mel spectrogram (`input_features`).
    pub input_features: Tensor,
    /// Frame validity (`attention_mask`), 1 = real audio, 0 = padding;
    /// length `frames`.
    pub attention_mask: Vec<u32>,
}

/// The Whisper log-mel front-end of the GLM-4-Voice tokenizer
/// (128 mel bins, n_fft 400, hop 160), including the pad-to-stride
/// behaviour of `WhisperVQEncoderWrapper.extract_and_encode`.
pub struct WhisperFeatureExtractor {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    /// Input padding stride in samples (conv stride 2 · pool 4 · hop 160).
    pad_multiple: usize,
    window: Vec<f64>,
    filterbank: Vec<Vec<f64>>,
    fft: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for WhisperFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperFeatureExtractor")
            .field("n_fft", &self.n_fft)
            .field("hop_length", &self.hop_length)
            .field("n_mels", &self.n_mels)
            .field("pad_multiple", &self.pad_multiple)
            .finish()
    }
}

impl Default for WhisperFeatureExtractor {
    /// The `zai-org/glm-4-voice-tokenizer` preprocessor: 128 mels,
    /// n_fft 400, hop 160, 16 kHz, stride padding 1280.
    fn default() -> Self {
        Self::new(400, 160, 128, 16_000, 1280)
    }
}

impl WhisperFeatureExtractor {
    pub fn new(
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        sample_rate: usize,
        pad_multiple: usize,
    ) -> Self {
        // Periodic Hann window (`torch.hann_window(n_fft)`).
        let window: Vec<f64> = (0..n_fft)
            .map(|i| {
                let x = std::f64::consts::PI * i as f64 / n_fft as f64;
                x.sin().powi(2)
            })
            .collect();
        Self {
            n_fft,
            hop_length,
            n_mels,
            pad_multiple,
            window,
            filterbank: slaney_filterbank(n_fft, n_mels, sample_rate, 8000.0),
            fft: FftPlanner::new().plan_fft_forward(n_fft),
        }
    }

    /// Computes `input_features` (`[1, n_mels, frames]`) and the frame
    /// attention mask for a mono 16 kHz waveform, zero-padding the input to
    /// a [`Self::pad_multiple`] boundary first.
    pub fn extract(&self, samples: &[f32], device: &Device) -> Result<MelFeatures> {
        if samples.is_empty() {
            return Err(Error::Input("empty waveform".into()));
        }
        let n = samples.len();
        let n_padded = n.div_ceil(self.pad_multiple) * self.pad_multiple;
        let mut wav: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        wav.resize(n_padded, 0.0);

        // Centered STFT: reflect-pad n_fft / 2 on both sides, then drop the
        // final frame like the Whisper reference.
        let half = self.n_fft / 2;
        let mut padded = Vec::with_capacity(n_padded + self.n_fft);
        padded.extend((1..=half).rev().map(|i| wav[i]));
        padded.extend_from_slice(&wav);
        padded.extend((0..half).map(|i| wav[n_padded - 2 - i]));

        let frames = n_padded / self.hop_length;
        let n_bins = half + 1;
        let mut power = vec![0f64; n_bins * frames]; // [bin][frame]
        let mut buf = vec![Complex64::default(); self.n_fft];
        for frame in 0..frames {
            let start = frame * self.hop_length;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex64::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (bin, b) in buf[..n_bins].iter().enumerate() {
                power[bin * frames + frame] = b.norm_sqr();
            }
        }

        // Mel projection + log10 + per-utterance dynamic-range clamp.
        let mut mel = vec![0f32; self.n_mels * frames];
        let mut max = f64::NEG_INFINITY;
        let mut log_mel = vec![0f64; self.n_mels * frames];
        for (m, bank) in self.filterbank.iter().enumerate() {
            for frame in 0..frames {
                let mut acc = 0f64;
                for (bin, &w) in bank.iter().enumerate() {
                    if w != 0.0 {
                        acc += w * power[bin * frames + frame];
                    }
                }
                let v = acc.max(1e-10).log10();
                log_mel[m * frames + frame] = v;
                max = max.max(v);
            }
        }
        for (out, &v) in mel.iter_mut().zip(log_mel.iter()) {
            *out = ((v.max(max - 8.0) + 4.0) / 4.0) as f32;
        }

        let input_features = Tensor::from_vec(mel, (1, self.n_mels, frames), device)?;
        let attention_mask: Vec<u32> = (0..frames)
            .map(|f| u32::from(f * self.hop_length < n))
            .collect();
        Ok(MelFeatures {
            input_features,
            attention_mask,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_pads_to_multiple() {
        let audio = vec![0.05f64; 2000];
        let out = preprocess(&audio, &PreprocessConfig::default());
        assert_eq!(out.len(), 2560);
        assert_eq!(out[2559], 0.0);
    }

    #[test]
    fn volume_normalize_small_signal_passthrough() {
        // Fewer than 10 significant samples: only the small-signal rescale.
        let mut audio = vec![0.0f64; 100];
        audio[0] = 0.05;
        let out = volume_normalize(&audio, 0.2);
        assert!((out[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn highpass_kills_dc() {
        let audio = vec![0.5f64; 4000];
        let out = highpass_biquad(&audio, 16_000.0, 40.0);
        assert!(out.last().unwrap().abs() < 1e-3);
    }

    #[test]
    fn extract_shapes_and_mask() {
        let ex = WhisperFeatureExtractor::default();
        // 1000 samples -> padded to 1280 -> 8 frames, mask 1 while < 1000.
        let feats = ex.extract(&vec![0.1f32; 1000], &Device::Cpu).unwrap();
        assert_eq!(feats.input_features.dims(), &[1, 128, 8]);
        assert_eq!(feats.attention_mask, vec![1, 1, 1, 1, 1, 1, 1, 0]);
    }
}

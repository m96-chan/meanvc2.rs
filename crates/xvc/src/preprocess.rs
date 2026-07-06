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
//! 3. [`FrameMelExtractor`] — the dB-scale 128-mel spectrogram
//!    (`mel_extractor.py`, torchaudio `MelSpectrogram` n_fft 1024 /
//!    win 640 / hop 320 + `AmplitudeToDB`) that conditions the acoustic
//!    converter (`frame_condition`).

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

/// Streaming RBJ high-pass biquad
/// (`torchaudio.functional.highpass_biquad`, Q = 0.707) with persistent
/// filter state, so live input can be filtered chunk by chunk with
/// results identical to the one-shot [`highpass_biquad`].
#[derive(Debug, Clone)]
pub struct HighpassBiquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl HighpassBiquad {
    pub fn new(sample_rate: f64, cutoff_hz: f64) -> Self {
        const Q: f64 = 0.707;
        let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate;
        let alpha = w0.sin() / 2.0 / Q;
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos_w0) / 2.0 / a0,
            b1: (-1.0 - cos_w0) / a0,
            b2: (1.0 + cos_w0) / 2.0 / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Filters one chunk in place (output clamped to `[-1, 1]` like
    /// `torchaudio.functional.lfilter(..., clamp=True)`), carrying the
    /// filter state to the next chunk.
    pub fn process(&mut self, chunk: &mut [f64]) {
        for x in chunk {
            let x0 = *x;
            let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            (self.x2, self.x1) = (self.x1, x0);
            (self.y2, self.y1) = (self.y1, y0);
            *x = y0.clamp(-1.0, 1.0);
        }
    }
}

/// RBJ high-pass biquad (`torchaudio.functional.highpass_biquad`,
/// Q = 0.707), zero initial conditions, output clamped to `[-1, 1]` like
/// `torchaudio.functional.lfilter(..., clamp=True)`.
pub fn highpass_biquad(audio: &[f64], sample_rate: f64, cutoff_hz: f64) -> Vec<f64> {
    let mut out = audio.to_vec();
    HighpassBiquad::new(sample_rate, cutoff_hz).process(&mut out);
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
/// `[n_mels][n_fft / 2 + 1]` (`transformers.audio_utils.mel_filter_bank` /
/// `torchaudio.functional.melscale_fbanks` with
/// `norm="slaney", mel_scale="slaney"` — identical constructions).
fn slaney_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: usize,
    f_min: f64,
    f_max: f64,
) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hertz_to_mel(f_min);
    let mel_max = hertz_to_mel(f_max);
    let filter_freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hertz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
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
            filterbank: slaney_filterbank(n_fft, n_mels, sample_rate, 0.0, 8000.0),
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

/// The frame-level condition mel extractor
/// (`models/codec/sac/modules/mel_extractor.py::MelExtractor`,
/// `configs/xvc.yaml` `mel_extractor`): a torchaudio-style 128-bin log-mel
/// spectrogram in dB — `MelSpectrogram(n_fft=1024, win_length=640,
/// hop_length=320, f_min=10, power=1, norm/mel_scale="slaney",
/// center=True/reflect)` + `1e-9` + `AmplitudeToDB(stype="magnitude",
/// top_db=80)` (20·log₁₀, per-utterance clamp at `max − 80` dB).
///
/// This is the mel that conditions the acoustic converter
/// (`frame_condition = mel_extractor(target_wav_cond)`).
pub struct FrameMelExtractor {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    top_db: f64,
    /// Hann(win_length) centre-padded to `n_fft` like `torch.stft`.
    window: Vec<f64>,
    filterbank: Vec<Vec<f64>>,
    fft: Arc<dyn Fft<f64>>,
}

impl std::fmt::Debug for FrameMelExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameMelExtractor")
            .field("n_fft", &self.n_fft)
            .field("hop_length", &self.hop_length)
            .field("n_mels", &self.n_mels)
            .finish()
    }
}

impl Default for FrameMelExtractor {
    /// The `configs/xvc.yaml` `mel_extractor`: 128 mels, n_fft 1024,
    /// win 640, hop 320, f_min 10 Hz, f_max Nyquist, 16 kHz.
    fn default() -> Self {
        Self::new(1024, 640, 320, 128, 16_000, 10.0, 80.0)
    }
}

impl FrameMelExtractor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_fft: usize,
        win_length: usize,
        hop_length: usize,
        n_mels: usize,
        sample_rate: usize,
        f_min: f64,
        top_db: f64,
    ) -> Self {
        // Periodic Hann window of `win_length`, centre-padded to `n_fft`
        // (`torch.stft` pads the window symmetrically).
        let mut window = vec![0f64; n_fft];
        let left = (n_fft - win_length) / 2;
        for i in 0..win_length {
            let x = std::f64::consts::PI * i as f64 / win_length as f64;
            window[left + i] = x.sin().powi(2);
        }
        Self {
            n_fft,
            hop_length,
            n_mels,
            top_db,
            window,
            filterbank: slaney_filterbank(
                n_fft,
                n_mels,
                sample_rate,
                f_min,
                (sample_rate / 2) as f64,
            ),
            fft: FftPlanner::new().plan_fft_forward(n_fft),
        }
    }

    /// Mono 16 kHz waveform → `[1, n_mels, samples / hop + 1]` dB-scaled
    /// log-mel (the centered STFT emits one frame per hop plus one).
    pub fn extract(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        if samples.len() < 2 {
            return Err(Error::Input("waveform too short for reflect pad".into()));
        }
        let n = samples.len();
        let wav: Vec<f64> = samples.iter().map(|&s| s as f64).collect();

        // Centered STFT, reflect padding of n_fft / 2 on both sides.
        let half = self.n_fft / 2;
        let reflect = |i: isize| -> f64 {
            let n = n as isize;
            let mut i = i;
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * (n - 1) - i;
            }
            wav[i as usize]
        };
        let frames = n / self.hop_length + 1;
        let n_bins = half + 1;
        let mut mag = vec![0f64; n_bins * frames]; // [bin][frame]
        let mut buf = vec![Complex64::default(); self.n_fft];
        for frame in 0..frames {
            let start = frame as isize * self.hop_length as isize - half as isize;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex64::new(reflect(start + i as isize) * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (bin, b) in buf[..n_bins].iter().enumerate() {
                mag[bin * frames + frame] = b.norm();
            }
        }

        // Mel projection (+1e-9) → 20·log10 → per-utterance top_db clamp.
        let mut db = vec![0f64; self.n_mels * frames];
        let mut max = f64::NEG_INFINITY;
        for (m, bank) in self.filterbank.iter().enumerate() {
            for frame in 0..frames {
                let mut acc = 0f64;
                for (bin, &w) in bank.iter().enumerate() {
                    if w != 0.0 {
                        acc += w * mag[bin * frames + frame];
                    }
                }
                // `mel + eps` then `AmplitudeToDB` (amin 1e-10).
                let v = 20.0 * (acc + 1e-9).max(1e-10).log10();
                db[m * frames + frame] = v;
                max = max.max(v);
            }
        }
        let floor = max - self.top_db;
        let mel: Vec<f32> = db.into_iter().map(|v| v.max(floor) as f32).collect();
        Ok(Tensor::from_vec(mel, (1, self.n_mels, frames), device)?)
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
    fn streaming_highpass_matches_one_shot() {
        let audio: Vec<f64> = (0..4000)
            .map(|i| ((i * 31 % 97) as f64 / 97.0 - 0.5) * 0.4)
            .collect();
        let want = highpass_biquad(&audio, 16_000.0, 40.0);
        let mut hp = HighpassBiquad::new(16_000.0, 40.0);
        let mut got = Vec::new();
        for chunk in audio.chunks(333) {
            let mut c = chunk.to_vec();
            hp.process(&mut c);
            got.extend(c);
        }
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn highpass_kills_dc() {
        let audio = vec![0.5f64; 4000];
        let out = highpass_biquad(&audio, 16_000.0, 40.0);
        assert!(out.last().unwrap().abs() < 1e-3);
    }

    #[test]
    fn frame_mel_shape_and_clamp() {
        let ex = FrameMelExtractor::default();
        let samples: Vec<f32> = (0..3200)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect();
        let mel = ex.extract(&samples, &Device::Cpu).unwrap();
        // Centered STFT: samples / hop + 1 frames.
        assert_eq!(mel.dims(), &[1, 128, 11]);
        let v: Vec<f32> = mel.flatten_all().unwrap().to_vec1().unwrap();
        let max = v.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
        let min = v.iter().fold(f32::INFINITY, |m, &x| m.min(x));
        assert!(v.iter().all(|x| x.is_finite()));
        // AmplitudeToDB top_db: dynamic range clamped to 80 dB.
        assert!(max - min <= 80.0 + 1e-4, "range {} > top_db", max - min);
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

//! Vocos vocoder (Siuzdak, 2024) — candle port.
//!
//! Vocos generates waveforms from mel-spectrograms without transposed
//! convolutions: a ConvNeXt backbone predicts the magnitude and phase of
//! the STFT, and an inverse STFT reconstructs the waveform. Layout and
//! parameter names follow the reference implementation
//! (<https://github.com/gemelo-ai/vocos>) so converted checkpoints map
//! 1:1: `backbone.embed`, `backbone.norm`, `backbone.convnext.{i}.*`,
//! `backbone.final_layer_norm`, `head.out`.

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv1d, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear, Module,
    VarBuilder,
};
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::encoders::Vocoder;
use crate::{Error, Result};

/// Configuration of the Vocos vocoder.
///
/// Defaults follow the published `vocos-mel-24khz` architecture but sized
/// for the 16 kHz / 80-mel setup of this crate; `n_fft` and `hop_length`
/// must match the mel analysis of the checkpoint you load.
#[derive(Debug, Clone)]
pub struct VocosConfig {
    /// Number of input mel bins.
    pub input_channels: usize,
    /// Backbone hidden dimension.
    pub dim: usize,
    /// ConvNeXt block expansion dimension.
    pub intermediate_dim: usize,
    /// Number of ConvNeXt blocks.
    pub num_layers: usize,
    /// FFT size of the ISTFT head.
    pub n_fft: usize,
    /// Hop length of the ISTFT head, in samples.
    pub hop_length: usize,
    /// Output sample rate in Hz.
    pub sample_rate: usize,
}

impl Default for VocosConfig {
    fn default() -> Self {
        Self {
            input_channels: 80,
            dim: 512,
            intermediate_dim: 1536,
            num_layers: 8,
            n_fft: 1024,
            hop_length: 160,
            sample_rate: 16_000,
        }
    }
}

impl VocosConfig {
    /// The official MeanVC v1 vocoder (`vocos.pt` on ASLP-lab/MeanVC):
    /// 16 kHz, dim 320, n_fft 640, hop 160, mel input in the
    /// [`crate::v1::MelV1`] domain.
    pub fn official_meanvc1() -> Self {
        Self {
            input_channels: 80,
            dim: 320,
            intermediate_dim: 1536,
            num_layers: 8,
            n_fft: 640,
            hop_length: 160,
            sample_rate: 16_000,
        }
    }

    /// A config whose ISTFT head matches the given mel analysis settings.
    pub fn for_mel(mel: &crate::config::MelConfig) -> Self {
        Self {
            input_channels: mel.n_mels,
            n_fft: mel.n_fft,
            hop_length: mel.hop_length,
            sample_rate: mel.sample_rate,
            ..Self::default()
        }
    }
}

/// ConvNeXt block: depthwise conv → LayerNorm → pointwise MLP → layer scale.
#[derive(Debug)]
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, intermediate_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: 3,
            groups: dim,
            ..Default::default()
        };
        Ok(Self {
            dwconv: conv1d(dim, dim, 7, conv_cfg, vb.pp("dwconv"))?,
            norm: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm"))?,
            pwconv1: linear(dim, intermediate_dim, vb.pp("pwconv1"))?,
            pwconv2: linear(intermediate_dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get_with_hints((dim,), "gamma", candle_nn::Init::Const(1e-2))?,
        })
    }

    /// `x`: `[batch, dim, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.dwconv.forward(x)?.transpose(1, 2)?; // [b, t, d]
        let x = self.norm.forward(&x)?;
        let x = self.pwconv2.forward(&self.pwconv1.forward(&x)?.gelu_erf()?)?;
        let x = x.broadcast_mul(&self.gamma)?.transpose(1, 2)?;
        residual + x
    }
}

/// The Vocos model: ConvNeXt backbone + ISTFT head.
pub struct Vocos {
    embed: Conv1d,
    norm: LayerNorm,
    convnext: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
    /// Head projection to `n_fft + 2` channels (magnitude ‖ phase).
    out: Linear,
    window: Vec<f32>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    cfg: VocosConfig,
}

impl std::fmt::Debug for Vocos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocos").field("cfg", &self.cfg).finish()
    }
}

impl Vocos {
    pub fn new(cfg: VocosConfig, vb: VarBuilder) -> Result<Self> {
        let embed_cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let bb = vb.pp("backbone");
        let mut convnext = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            convnext.push(ConvNeXtBlock::new(
                cfg.dim,
                cfg.intermediate_dim,
                bb.pp(format!("convnext.{i}")),
            )?);
        }
        let window: Vec<f32> = (0..cfg.n_fft)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / cfg.n_fft as f32;
                x.sin().powi(2)
            })
            .collect();
        Ok(Self {
            ifft: FftPlanner::new().plan_fft_inverse(cfg.n_fft),
            embed: conv1d(cfg.input_channels, cfg.dim, 7, embed_cfg, bb.pp("embed"))?,
            norm: layer_norm(cfg.dim, LayerNormConfig::default(), bb.pp("norm"))?,
            convnext,
            final_layer_norm: layer_norm(
                cfg.dim,
                LayerNormConfig::default(),
                bb.pp("final_layer_norm"),
            )?,
            out: linear(cfg.dim, cfg.n_fft + 2, vb.pp("head.out"))?,
            window,
            cfg,
        })
    }

    /// Loads the model from a safetensors checkpoint (converted from the
    /// upstream PyTorch weights with matching parameter names).
    pub fn load<P: AsRef<std::path::Path>>(
        cfg: VocosConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb)
    }

    pub fn config(&self) -> &VocosConfig {
        &self.cfg
    }

    /// Predicts the complex STFT of the waveform.
    ///
    /// `mel`: `[batch, n_mels, frames]` →
    /// `(real, imag)`: each `[batch, n_fft / 2 + 1, frames]`.
    fn spectrogram(&self, mel: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let x = self.embed.forward(mel)?;
        let x = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let mut x = x;
        for block in &self.convnext {
            x = block.forward(&x)?;
        }
        let x = self.final_layer_norm.forward(&x.transpose(1, 2)?)?;
        let x = self.out.forward(&x)?.transpose(1, 2)?; // [b, n_fft + 2, t]
        let half = self.cfg.n_fft / 2 + 1;
        let mag = x.narrow(1, 0, half)?;
        let phase = x.narrow(1, half, half)?;
        // exp with a safety clip, as in the reference implementation.
        let mag = mag.clamp(-1e2f32, 2.0 + 100f32.ln())?.exp()?.clamp(0f32, 1e2f32)?;
        let real = mag.mul(&phase.cos()?)?;
        let imag = mag.mul(&phase.sin()?)?;
        Ok((real, imag))
    }

    /// Inverse STFT with centered Hann-windowed overlap-add.
    fn istft(&self, real: &[Vec<f32>], imag: &[Vec<f32>]) -> Vec<f32> {
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.hop_length;
        let half = n_fft / 2 + 1;
        let frames = real.len();
        let padded_len = (frames - 1) * hop + n_fft;
        let mut y = vec![0f32; padded_len];
        let mut wsum = vec![0f32; padded_len];

        let mut buf = vec![Complex32::default(); n_fft];
        for (frame, (re, im)) in real.iter().zip(imag.iter()).enumerate() {
            // Hermitian-symmetric full spectrum.
            for bin in 0..half {
                buf[bin] = Complex32::new(re[bin], im[bin]);
            }
            for bin in half..n_fft {
                let src = buf[n_fft - bin];
                buf[bin] = Complex32::new(src.re, -src.im);
            }
            self.ifft.process(&mut buf);
            let offset = frame * hop;
            for i in 0..n_fft {
                let s = buf[i].re / n_fft as f32;
                y[offset + i] += s * self.window[i];
                wsum[offset + i] += self.window[i] * self.window[i];
            }
        }
        for (s, w) in y.iter_mut().zip(&wsum) {
            if *w > 1e-11 {
                *s /= w;
            }
        }
        // Trim the centering padding.
        let start = n_fft / 2;
        let len = frames.saturating_sub(1) * hop;
        y[start..start + len].to_vec()
    }
}

impl Vocoder for Vocos {
    fn sample_rate(&self) -> usize {
        self.cfg.sample_rate
    }

    /// `mel`: `[frames, n_mels]` (or `[1, frames, n_mels]`) log-mel
    /// spectrogram, as produced by the MeanVC 2 decoder.
    fn synthesize(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let mel = match mel.dims().len() {
            2 => mel.clone(),
            3 => mel.squeeze(0)?,
            _ => {
                return Err(Error::Input(
                    "mel must be [frames, n_mels] or [1, frames, n_mels]".into(),
                ))
            }
        };
        let (frames, n_mels) = mel.dims2()?;
        if n_mels != self.cfg.input_channels {
            return Err(Error::Input(format!(
                "expected {} mel bins, got {n_mels}",
                self.cfg.input_channels
            )));
        }
        if frames < 2 {
            return Err(Error::Input("need at least 2 mel frames".into()));
        }
        let mel = mel.transpose(0, 1)?.unsqueeze(0)?; // [1, n_mels, t]
        let (real, imag) = self.spectrogram(&mel)?;
        let real: Vec<Vec<f32>> = real.squeeze(0)?.transpose(0, 1)?.to_vec2()?;
        let imag: Vec<Vec<f32>> = imag.squeeze(0)?.transpose(0, 1)?.to_vec2()?;
        Ok(self.istft(&real, &imag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny() -> (Vocos, Device) {
        let dev = Device::Cpu;
        let cfg = VocosConfig {
            input_channels: 20,
            dim: 32,
            intermediate_dim: 64,
            num_layers: 2,
            n_fft: 64,
            hop_length: 16,
            sample_rate: 16_000,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        (Vocos::new(cfg, vb).unwrap(), dev)
    }

    #[test]
    fn synthesize_has_expected_length_and_is_finite() {
        let (vocos, dev) = tiny();
        let frames = 12;
        let mel = Tensor::randn(0f32, 1f32, (frames, 20), &dev).unwrap();
        let wav = vocos.synthesize(&mel).unwrap();
        assert_eq!(wav.len(), (frames - 1) * vocos.config().hop_length);
        assert!(wav.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn istft_reconstructs_a_windowed_sine() {
        // Analysis with the same window/hop followed by our ISTFT must
        // reconstruct the interior of the signal (COLA condition).
        let (vocos, _dev) = tiny();
        let n_fft = vocos.config().n_fft;
        let hop = vocos.config().hop_length;
        let frames = 20;
        let padded_len = (frames - 1) * hop + n_fft;
        let signal: Vec<f32> = (0..padded_len)
            .map(|i| (2.0 * std::f32::consts::PI * 5.0 * i as f32 / n_fft as f32).sin())
            .collect();

        // Forward STFT (centered layout: frame f covers f*hop..f*hop+n_fft).
        let fft = FftPlanner::new().plan_fft_forward(n_fft);
        let mut real = Vec::new();
        let mut imag = Vec::new();
        for f in 0..frames {
            let mut buf: Vec<Complex32> = (0..n_fft)
                .map(|i| Complex32::new(signal[f * hop + i] * vocos.window[i], 0.0))
                .collect();
            fft.process(&mut buf);
            real.push(buf[..n_fft / 2 + 1].iter().map(|c| c.re).collect());
            imag.push(buf[..n_fft / 2 + 1].iter().map(|c| c.im).collect());
        }
        let out = vocos.istft(&real, &imag);
        // Compare against the original, skipping the trimmed padding.
        let start = n_fft / 2;
        for (i, s) in out.iter().enumerate() {
            let expected = signal[start + i];
            assert!(
                (s - expected).abs() < 1e-3,
                "sample {i}: {s} vs {expected}"
            );
        }
    }

    #[test]
    fn rejects_wrong_mel_bins() {
        let (vocos, dev) = tiny();
        let mel = Tensor::randn(0f32, 1f32, (10, 21), &dev).unwrap();
        assert!(vocos.synthesize(&mel).is_err());
    }
}

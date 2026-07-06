//! ECAPA-TDNN speaker encoder (Desplanques et al., 2020) — candle port.
//!
//! Extracts a global speaker embedding from a waveform: log-mel filterbank
//! features → TDNN stem → 3 SE-Res2Net blocks with increasing dilation →
//! multi-layer feature aggregation → attentive statistics pooling (with
//! global context) → 192-dim embedding.
//!
//! The module tree mirrors SpeechBrain's `ECAPA_TDNN`
//! (<https://github.com/speechbrain/speechbrain>, `lobes/models/ECAPA_TDNN.py>`):
//! `blocks.{0..3}`, `mfa`, `asp`, `asp_bn`, `fc` — so converted checkpoints
//! map structurally. Exact numeric parity of the filterbank front-end with
//! SpeechBrain's `Fbank` is tracked as a golden-test follow-up in issue #4.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Module, ModuleT,
    VarBuilder,
};

use crate::audio::MelSpectrogram;
use crate::config::MelConfig;
use crate::encoders::SpeakerEncoder;
use crate::{Error, Result};

/// Configuration of the ECAPA-TDNN encoder.
///
/// Defaults follow `speechbrain/spkrec-ecapa-voxceleb`.
#[derive(Debug, Clone)]
pub struct EcapaConfig {
    /// Number of input filterbank bins.
    pub n_mels: usize,
    /// Channels of the four TDNN/SE-Res2Net stages.
    pub channels: usize,
    /// Channels after multi-layer feature aggregation.
    pub mfa_channels: usize,
    /// Res2Net split cardinality.
    pub res2net_scale: usize,
    /// Squeeze-excitation bottleneck channels.
    pub se_channels: usize,
    /// Attention bottleneck channels of the statistics pooling.
    pub attention_channels: usize,
    /// Output embedding dimension.
    pub embedding_dim: usize,
    /// Expected input sample rate.
    pub sample_rate: usize,
}

impl Default for EcapaConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            channels: 1024,
            mfa_channels: 3072,
            res2net_scale: 8,
            se_channels: 128,
            attention_channels: 128,
            embedding_dim: 192,
            sample_rate: 16_000,
        }
    }
}

impl EcapaConfig {
    /// Filterbank settings for the front-end (25 ms window / 10 ms hop).
    pub fn feature_config(&self) -> MelConfig {
        MelConfig {
            sample_rate: self.sample_rate,
            n_fft: 400,
            hop_length: self.sample_rate / 100,
            win_length: self.sample_rate / 40,
            n_mels: self.n_mels,
            f_min: 0.0,
            f_max: self.sample_rate as f32 / 2.0,
        }
    }
}

/// Conv1d ("same" padding) + ReLU + BatchNorm1d.
#[derive(Debug)]
struct TdnnBlock {
    conv: Conv1d,
    norm: BatchNorm,
}

impl TdnnBlock {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv1dConfig {
            padding: dilation * (kernel - 1) / 2,
            dilation,
            ..Default::default()
        };
        Ok(Self {
            conv: conv1d(in_c, out_c, kernel, cfg, vb.pp("conv"))?,
            norm: batch_norm(out_c, BatchNormConfig::default(), vb.pp("norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.norm.forward_t(&self.conv.forward(x)?.relu()?, false)
    }
}

/// Res2Net: hierarchical multi-scale convolution over channel splits.
#[derive(Debug)]
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        channels: usize,
        scale: usize,
        kernel: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        assert_eq!(channels % scale, 0);
        let hidden = channels / scale;
        let blocks = (0..scale - 1)
            .map(|i| TdnnBlock::new(hidden, hidden, kernel, dilation, vb.pp(format!("blocks.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self { blocks, scale })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let chunks = x.chunk(self.scale, 1)?;
        let mut ys: Vec<Tensor> = vec![chunks[0].clone()];
        let mut prev: Option<Tensor> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let input = match &prev {
                None => chunks[i + 1].clone(),
                Some(p) => (&chunks[i + 1] + p)?,
            };
            let y = block.forward(&input)?;
            prev = Some(y.clone());
            ys.push(y);
        }
        Tensor::cat(&ys, 1)
    }
}

/// Squeeze-excitation over the time-pooled channel descriptor.
#[derive(Debug)]
struct SeBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SeBlock {
    fn new(channels: usize, se_channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            conv1: conv1d(channels, se_channels, 1, Default::default(), vb.pp("conv1"))?,
            conv2: conv1d(se_channels, channels, 1, Default::default(), vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let s = x.mean_keepdim(D::Minus1)?; // [b, c, 1]
        let s = self.conv1.forward(&s)?.relu()?;
        let s = candle_nn::ops::sigmoid(&self.conv2.forward(&s)?)?;
        x.broadcast_mul(&s)
    }
}

/// SE-Res2Net block: 1x1 TDNN → Res2Net → 1x1 TDNN → SE, with residual.
#[derive(Debug)]
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SeBlock,
}

impl SeRes2NetBlock {
    fn new(cfg: &EcapaConfig, kernel: usize, dilation: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            tdnn1: TdnnBlock::new(cfg.channels, cfg.channels, 1, 1, vb.pp("tdnn1"))?,
            res2net: Res2NetBlock::new(
                cfg.channels,
                cfg.res2net_scale,
                kernel,
                dilation,
                vb.pp("res2net_block"),
            )?,
            tdnn2: TdnnBlock::new(cfg.channels, cfg.channels, 1, 1, vb.pp("tdnn2"))?,
            se: SeBlock::new(cfg.channels, cfg.se_channels, vb.pp("se_block"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.tdnn1.forward(x)?;
        let h = self.res2net.forward(&h)?;
        let h = self.tdnn2.forward(&h)?;
        let h = self.se.forward(&h)?;
        x + h
    }
}

/// Attentive statistics pooling with global context.
#[derive(Debug)]
struct AttentiveStatsPooling {
    tdnn: TdnnBlock,
    conv: Conv1d,
}

impl AttentiveStatsPooling {
    fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            tdnn: TdnnBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?,
            conv: conv1d(attention_channels, channels, 1, Default::default(), vb.pp("conv"))?,
        })
    }

    /// `x`: `[batch, channels, time]` → `[batch, 2 * channels, 1]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(D::Minus1)?;
        let std = (var + 1e-4)?.sqrt()?;
        let global = Tensor::cat(
            &[
                x.clone(),
                mean.broadcast_as(x.shape())?,
                std.broadcast_as(x.shape())?,
            ],
            1,
        )?;
        let attn = self.tdnn.forward(&global)?.tanh()?;
        let attn = self.conv.forward(&attn)?;
        let weights = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let mu = x.mul(&weights)?.sum_keepdim(D::Minus1)?;
        let sigma = (x.sqr()?.mul(&weights)?.sum_keepdim(D::Minus1)?
            - mu.sqr()?)?
        .clamp(1e-4f32, f32::MAX)?
        .sqrt()?;
        Tensor::cat(&[mu, sigma], 1)
    }
}

/// The ECAPA-TDNN speaker encoder.
#[derive(Debug)]
pub struct Ecapa {
    stem: TdnnBlock,
    layers: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatsPooling,
    asp_bn: BatchNorm,
    fc: Conv1d,
    features: MelSpectrogram,
    cfg: EcapaConfig,
}

impl Ecapa {
    pub fn new(cfg: EcapaConfig, vb: VarBuilder) -> Result<Self> {
        let stem = TdnnBlock::new(cfg.n_mels, cfg.channels, 5, 1, vb.pp("blocks.0"))?;
        // Dilations 2, 3, 4 as in the paper.
        let layers = (0..3)
            .map(|i| SeRes2NetBlock::new(&cfg, 3, i + 2, vb.pp(format!("blocks.{}", i + 1))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            stem,
            layers,
            mfa: TdnnBlock::new(cfg.channels * 3, cfg.mfa_channels, 1, 1, vb.pp("mfa"))?,
            asp: AttentiveStatsPooling::new(cfg.mfa_channels, cfg.attention_channels, vb.pp("asp"))?,
            asp_bn: batch_norm(
                cfg.mfa_channels * 2,
                BatchNormConfig::default(),
                vb.pp("asp_bn"),
            )?,
            fc: conv1d(
                cfg.mfa_channels * 2,
                cfg.embedding_dim,
                1,
                Default::default(),
                vb.pp("fc"),
            )?,
            features: MelSpectrogram::new(cfg.feature_config()),
            cfg,
        })
    }

    /// Loads the model from a safetensors checkpoint (converted from the
    /// SpeechBrain weights with matching module paths).
    pub fn load<P: AsRef<std::path::Path>>(
        cfg: EcapaConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb)
    }

    pub fn config(&self) -> &EcapaConfig {
        &self.cfg
    }

    /// Embeds filterbank features `[batch, time, n_mels]` into
    /// `[batch, embedding_dim]`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = features.transpose(1, 2)?.contiguous()?; // [b, mel, t]
        let f1 = self.stem.forward(&x)?;
        let f2 = self.layers[0].forward(&f1)?;
        let f3 = self.layers[1].forward(&f2)?;
        let f4 = self.layers[2].forward(&f3)?;
        let cat = Tensor::cat(&[f2, f3, f4], 1)?;
        let x = self.mfa.forward(&cat)?.relu()?;
        let pooled = self.asp.forward(&x)?;
        let pooled = self.asp_bn.forward_t(&pooled, false)?;
        Ok(self.fc.forward(&pooled)?.squeeze(D::Minus1)?)
    }
}

impl SpeakerEncoder for Ecapa {
    fn embedding_dim(&self) -> usize {
        self.cfg.embedding_dim
    }

    fn embed(&self, samples: &[f32], sample_rate: usize) -> Result<Tensor> {
        if sample_rate != self.cfg.sample_rate {
            return Err(Error::Input(format!(
                "expected {} Hz input, got {sample_rate} Hz (resample first)",
                self.cfg.sample_rate
            )));
        }
        let device = Device::Cpu;
        let features = self.features.compute(samples, &device)?;
        // Sentence-level mean normalization, as in SpeechBrain's
        // InputNormalization(norm_type="sentence", std_norm=False).
        let mean = features.mean_keepdim(0)?;
        let features = features.broadcast_sub(&mean)?.unsqueeze(0)?;
        Ok(self.forward(&features)?.squeeze(0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny() -> Ecapa {
        let cfg = EcapaConfig {
            n_mels: 20,
            channels: 16,
            mfa_channels: 48,
            res2net_scale: 4,
            se_channels: 8,
            attention_channels: 8,
            embedding_dim: 12,
            sample_rate: 16_000,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        Ecapa::new(cfg, vb).unwrap()
    }

    #[test]
    fn forward_shapes() {
        let ecapa = tiny();
        let feats = Tensor::randn(0f32, 1f32, (2, 50, 20), &Device::Cpu).unwrap();
        let emb = ecapa.forward(&feats).unwrap();
        assert_eq!(emb.dims(), &[2, 12]);
    }

    #[test]
    fn embed_wav_is_finite_and_deterministic() {
        let ecapa = tiny();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let a = ecapa.embed(&samples, 16_000).unwrap();
        let b = ecapa.embed(&samples, 16_000).unwrap();
        assert_eq!(a.dims(), &[12]);
        let av: Vec<f32> = a.to_vec1().unwrap();
        let bv: Vec<f32> = b.to_vec1().unwrap();
        assert!(av.iter().all(|x| x.is_finite()));
        assert_eq!(av, bv);
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let ecapa = tiny();
        assert!(ecapa.embed(&[0.0; 8000], 8_000).is_err());
    }
}

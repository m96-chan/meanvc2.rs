//! Fast-U2++ streaming ASR encoder (Liang et al., 2023) — candle port of
//! the WeNet U2++ conformer encoder, used as the bottleneck-feature (BNF)
//! extractor of the MeanVC 2 pipeline.
//!
//! Pipeline: 80-dim log-mel filterbank (25 ms / 10 ms) → optional global
//! CMVN → Conv2d subsampling ×4 (→ 40 ms frames, matching the paper's BNF
//! frame length) → conformer blocks with chunk-based causal attention
//! masks. The final encoder output (after `after_norm`) is the BNF
//! sequence.
//!
//! The module tree mirrors WeNet (<https://github.com/wenet-e2e/wenet>):
//! `embed.conv.{0,2}` / `embed.out.0` / `encoders.{i}.{self_attn,
//! feed_forward, feed_forward_macaron, conv_module, norm_*}` /
//! `after_norm` / `global_cmvn`, so converted checkpoints map 1:1.
//! Following WeNet, the relative-position attention omits `rel_shift`.
//!
//! Chunked processing is implemented with attention masks over the full
//! utterance (identical outputs to step-by-step streaming); an incremental
//! per-chunk cache for true streaming inference is a follow-up in issue #4.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv2d, layer_norm, linear, BatchNorm, BatchNormConfig, Conv1d,
    Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, Module, ModuleT,
    VarBuilder,
};

use crate::audio::MelSpectrogram;
use crate::config::MelConfig;
use crate::encoders::SemanticEncoder;
use crate::frc;
use crate::{Error, Result};

/// Configuration of the Fast-U2++ encoder.
///
/// Defaults follow WeNet's streaming U2++ conformer recipes
/// (d_model 256, 4 heads, 12 layers, kernel 15, causal conv with
/// layer-norm) with an 80 ms attention chunk (2 subsampled frames), the
/// setting used by the MeanVC 2 paper.
#[derive(Debug, Clone)]
pub struct FastU2ppConfig {
    /// Number of input filterbank bins.
    pub n_mels: usize,
    /// Encoder hidden size (= BNF dimension).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feed-forward inner dimension.
    pub ff_dim: usize,
    /// Number of conformer blocks.
    pub num_layers: usize,
    /// Depthwise convolution kernel size.
    pub cnn_kernel: usize,
    /// Whether the conv modules have dual streaming/non-streaming paths
    /// (Fast-U2++) and which one to run. `None` = single-path (plain U2++).
    pub dual_conv_streaming: Option<bool>,
    /// Attention chunk size in subsampled (40 ms) frames; 2 = 80 ms.
    pub chunk_frames: usize,
    /// Number of past chunks each chunk attends to (`usize::MAX` = all).
    pub left_chunks: usize,
    /// Whether the checkpoint contains `global_cmvn` statistics.
    pub cmvn: bool,
    /// Expected input sample rate.
    pub sample_rate: usize,
}

impl Default for FastU2ppConfig {
    fn default() -> Self {
        Self {
            n_mels: 80,
            d_model: 256,
            num_heads: 4,
            ff_dim: 2048,
            num_layers: 7,
            cnn_kernel: 9,
            dual_conv_streaming: Some(true),
            chunk_frames: 2,
            left_chunks: usize::MAX,
            cmvn: false,
            sample_rate: 16_000,
        }
    }
}

impl FastU2ppConfig {
    /// Filterbank settings for the front-end (25 ms window / 10 ms hop).
    pub fn feature_config(&self) -> MelConfig {
        MelConfig {
            sample_rate: self.sample_rate,
            n_fft: 512,
            hop_length: self.sample_rate / 100,
            win_length: self.sample_rate / 40,
            n_mels: self.n_mels,
            f_min: 0.0,
            f_max: self.sample_rate as f32 / 2.0,
        }
    }
}

/// Sinusoidal positional encoding table `[1, len, d_model]`.
fn sinusoidal_pe(len: usize, d_model: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mut data = vec![0f32; len * d_model];
    for pos in 0..len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10_000f32.powf(2.0 * i as f32 / d_model as f32);
            data[pos * d_model + 2 * i] = angle.sin();
            data[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }
    Tensor::from_vec(data, (1, len, d_model), device)
}

/// Conv2d subsampling ×4: `[batch, time, n_mels]` → `[batch, time / 4, d]`.
#[derive(Debug)]
struct Conv2dSubsampling4 {
    conv0: Conv2d,
    conv1: Conv2d,
    out: Linear,
}

impl Conv2dSubsampling4 {
    fn new(n_mels: usize, d_model: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let freq = (n_mels - 1) / 2;
        let freq = (freq - 1) / 2;
        Ok(Self {
            conv0: conv2d(1, d_model, 3, cfg, vb.pp("conv.0"))?,
            conv1: conv2d(d_model, d_model, 3, cfg, vb.pp("conv.2"))?,
            out: linear(d_model * freq, d_model, vb.pp("out.0"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.unsqueeze(1)?; // [b, 1, t, mel]
        let x = self.conv0.forward(&x)?.relu()?;
        let x = self.conv1.forward(&x)?.relu()?;
        let (b, c, t, f) = x.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, t, c * f))?;
        self.out.forward(&x)
    }
}

/// WeNet-style relative-position multi-head attention (no rel-shift).
#[derive(Debug)]
struct RelPositionAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl RelPositionAttention {
    fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = d_model / num_heads;
        let bias_init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        };
        Ok(Self {
            linear_q: linear(d_model, d_model, vb.pp("linear_q"))?,
            linear_k: linear(d_model, d_model, vb.pp("linear_k"))?,
            linear_v: linear(d_model, d_model, vb.pp("linear_v"))?,
            linear_out: linear(d_model, d_model, vb.pp("linear_out"))?,
            linear_pos: candle_nn::linear_no_bias(d_model, d_model, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get_with_hints((num_heads, head_dim), "pos_bias_u", bias_init)?,
            pos_bias_v: vb.get_with_hints((num_heads, head_dim), "pos_bias_v", bias_init)?,
            num_heads,
            head_dim,
        })
    }

    /// `x`: `[batch, time, d]`, `pos_emb`: `[1, time, d]`,
    /// `mask`: additive `[time, time]`.
    fn forward(&self, x: &Tensor, pos_emb: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |x: Tensor, b: usize, t: usize| -> candle_core::Result<Tensor> {
            x.reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.linear_q.forward(x)?, b, t)?; // [b, h, t, d_k]
        let k = split(self.linear_k.forward(x)?, b, t)?;
        let v = split(self.linear_v.forward(x)?, b, t)?;
        let p = split(self.linear_pos.forward(pos_emb)?, 1, t)?; // [1, h, t, d_k]

        let bias = |bias: &Tensor| bias.reshape((1, self.num_heads, 1, self.head_dim));
        let q_u = q.broadcast_add(&bias(&self.pos_bias_u)?)?;
        let q_v = q.broadcast_add(&bias(&self.pos_bias_v)?)?;

        let ac = q_u.matmul(&k.transpose(2, 3)?)?; // [b, h, t, t]
        let bd = q_v.broadcast_matmul(&p.transpose(2, 3)?)?; // [b, h, t, t]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = ((ac + bd)? * scale)?.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.linear_out.forward(&out)
    }
}

/// Position-wise feed-forward with Swish activation.
#[derive(Debug)]
struct FeedForward {
    w_1: Linear,
    w_2: Linear,
}

impl FeedForward {
    fn new(d_model: usize, ff_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            w_1: linear(d_model, ff_dim, vb.pp("w_1"))?,
            w_2: linear(ff_dim, d_model, vb.pp("w_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.w_2.forward(&self.w_1.forward(x)?.silu()?)
    }
}

/// Norm applied inside the convolution module (layer norm for causal
/// streaming models, batch norm otherwise).
#[derive(Debug)]
enum ConvNorm {
    Layer(LayerNorm),
    Batch(BatchNorm),
}

/// Conformer convolution module (causal depthwise convolution).
#[derive(Debug)]
struct ConvolutionModule {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    pointwise_conv2: Conv1d,
    norm: ConvNorm,
    kernel: usize,
    causal: bool,
}

impl ConvolutionModule {
    fn new(
        d_model: usize,
        kernel: usize,
        use_layer_norm: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let dw_cfg = Conv1dConfig {
            groups: d_model,
            ..Default::default()
        };
        let norm = if use_layer_norm {
            ConvNorm::Layer(layer_norm(
                d_model,
                LayerNormConfig::default(),
                vb.pp("norm"),
            )?)
        } else {
            ConvNorm::Batch(batch_norm(d_model, BatchNormConfig::default(), vb.pp("norm"))?)
        };
        Ok(Self {
            pointwise_conv1: conv1d(
                d_model,
                2 * d_model,
                1,
                Default::default(),
                vb.pp("pointwise_conv1"),
            )?,
            depthwise_conv: conv1d(d_model, d_model, kernel, dw_cfg, vb.pp("depthwise_conv"))?,
            pointwise_conv2: conv1d(
                d_model,
                d_model,
                1,
                Default::default(),
                vb.pp("pointwise_conv2"),
            )?,
            norm,
            kernel,
            causal,
        })
    }

    /// `x`: `[batch, time, d]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.transpose(1, 2)?; // [b, d, t]
        let x = self.pointwise_conv1.forward(&x)?;
        // GLU over the channel dim.
        let halves = x.chunk(2, 1)?;
        let x = halves[0].mul(&candle_nn::ops::sigmoid(&halves[1])?)?;
        let (b, d, _) = x.dims3()?;
        let x = if self.causal {
            // Causal left padding keeps the module streamable.
            let pad = Tensor::zeros((b, d, self.kernel - 1), x.dtype(), x.device())?;
            Tensor::cat(&[pad, x], 2)?
        } else {
            let pad = Tensor::zeros((b, d, (self.kernel - 1) / 2), x.dtype(), x.device())?;
            Tensor::cat(&[pad.clone(), x, pad], 2)?
        };
        let x = self.depthwise_conv.forward(&x)?;
        let x = match &self.norm {
            ConvNorm::Layer(ln) => ln.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?,
            ConvNorm::Batch(bn) => bn.forward_t(&x, false)?,
        };
        let x = self.pointwise_conv2.forward(&x.silu()?)?;
        x.transpose(1, 2)
    }
}

/// One conformer block (WeNet `ConformerEncoderLayer`).
#[derive(Debug)]
struct ConformerBlock {
    feed_forward_macaron: FeedForward,
    self_attn: RelPositionAttention,
    conv_module: ConvolutionModule,
    /// Fast-U2++ dual-mode convolution: the alternate (non-streaming) path.
    conv_module_alt: Option<ConvolutionModule>,
    use_alt_conv: bool,
    feed_forward: FeedForward,
    norm_ff_macaron: LayerNorm,
    norm_mha: LayerNorm,
    norm_conv: LayerNorm,
    norm_ff: LayerNorm,
    norm_final: LayerNorm,
}

impl ConformerBlock {
    fn new(cfg: &FastU2ppConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let ln = LayerNormConfig::default();
        Ok(Self {
            feed_forward_macaron: FeedForward::new(
                cfg.d_model,
                cfg.ff_dim,
                vb.pp("feed_forward_macaron"),
            )?,
            self_attn: RelPositionAttention::new(cfg.d_model, cfg.num_heads, vb.pp("self_attn"))?,
            conv_module: match cfg.dual_conv_streaming {
                None => ConvolutionModule::new(
                    cfg.d_model,
                    cfg.cnn_kernel,
                    true,
                    true,
                    vb.pp("conv_module"),
                )?,
                Some(_) => ConvolutionModule::new(
                    cfg.d_model,
                    cfg.cnn_kernel,
                    true,
                    true,
                    vb.pp("conv_module.streaming_conv"),
                )?,
            },
            conv_module_alt: match cfg.dual_conv_streaming {
                None => None,
                Some(_) => Some(ConvolutionModule::new(
                    cfg.d_model,
                    cfg.cnn_kernel,
                    true,
                    false,
                    vb.pp("conv_module.non_streaming_conv"),
                )?),
            },
            use_alt_conv: cfg.dual_conv_streaming == Some(false),
            feed_forward: FeedForward::new(cfg.d_model, cfg.ff_dim, vb.pp("feed_forward"))?,
            norm_ff_macaron: layer_norm(cfg.d_model, ln, vb.pp("norm_ff_macaron"))?,
            norm_mha: layer_norm(cfg.d_model, ln, vb.pp("norm_mha"))?,
            norm_conv: layer_norm(cfg.d_model, ln, vb.pp("norm_conv"))?,
            norm_ff: layer_norm(cfg.d_model, ln, vb.pp("norm_ff"))?,
            norm_final: layer_norm(cfg.d_model, ln, vb.pp("norm_final"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let x = (x + self
            .feed_forward_macaron
            .forward(&self.norm_ff_macaron.forward(x)?)?
            .affine(0.5, 0.0)?)?;
        let x = (&x
            + self
                .self_attn
                .forward(&self.norm_mha.forward(&x)?, pos_emb, mask)?)?;
        let conv = match (&self.conv_module_alt, self.use_alt_conv) {
            (Some(alt), true) => alt,
            _ => &self.conv_module,
        };
        let x = (&x + conv.forward(&self.norm_conv.forward(&x)?)?)?;
        let x = (&x
            + self
                .feed_forward
                .forward(&self.norm_ff.forward(&x)?)?
                .affine(0.5, 0.0)?)?;
        self.norm_final.forward(&x)
    }
}

/// The Fast-U2++ conformer encoder.
#[derive(Debug)]
pub struct FastU2pp {
    cmvn: Option<(Tensor, Tensor)>,
    embed: Conv2dSubsampling4,
    encoders: Vec<ConformerBlock>,
    after_norm: LayerNorm,
    features: MelSpectrogram,
    cfg: FastU2ppConfig,
}

impl FastU2pp {
    pub fn new(cfg: FastU2ppConfig, vb: VarBuilder) -> Result<Self> {
        let cmvn = if cfg.cmvn {
            Some((
                vb.get((cfg.n_mels,), "global_cmvn.mean")?,
                vb.get((cfg.n_mels,), "global_cmvn.istd")?,
            ))
        } else {
            None
        };
        let mut encoders = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            encoders.push(ConformerBlock::new(&cfg, vb.pp(format!("encoders.{i}")))?);
        }
        Ok(Self {
            cmvn,
            embed: Conv2dSubsampling4::new(cfg.n_mels, cfg.d_model, vb.pp("embed"))?,
            encoders,
            after_norm: layer_norm(cfg.d_model, LayerNormConfig::default(), vb.pp("after_norm"))?,
            features: MelSpectrogram::new(cfg.feature_config()),
            cfg,
        })
    }

    /// Loads the encoder from a safetensors checkpoint (WeNet `encoder.*`
    /// subtree, converted with matching parameter names).
    pub fn load<P: AsRef<std::path::Path>>(
        cfg: FastU2ppConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb)
    }

    pub fn config(&self) -> &FastU2ppConfig {
        &self.cfg
    }

    /// Encodes filterbank features `[batch, time, n_mels]` into BNFs
    /// `[batch, time / 4, d_model]` under the chunked-causal attention mask.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = match &self.cmvn {
            Some((mean, istd)) => features.broadcast_sub(mean)?.broadcast_mul(istd)?,
            None => features.clone(),
        };
        let x = self.embed.forward(&x)?;
        let (_, t, _) = x.dims3()?;
        if t == 0 {
            return Err(Error::Input("input too short after subsampling".into()));
        }
        let x = x.affine((self.cfg.d_model as f64).sqrt(), 0.0)?;
        let pos_emb = sinusoidal_pe(t, self.cfg.d_model, x.device())?;
        // Chunk-causal mask: each chunk sees `left_chunks` past chunks and
        // no future chunks.
        let past = self.cfg.left_chunks.min(t.div_ceil(self.cfg.chunk_frames));
        let mask = frc::layer_mask(t, self.cfg.chunk_frames, past, 0, x.device())?;
        let mut x = x;
        for block in &self.encoders {
            x = block.forward(&x, &pos_emb, &mask)?;
        }
        Ok(self.after_norm.forward(&x)?)
    }
}

impl SemanticEncoder for FastU2pp {
    fn bnf_dim(&self) -> usize {
        self.cfg.d_model
    }

    fn frame_shift_ms(&self) -> f32 {
        40.0
    }

    fn extract(&self, samples: &[f32], sample_rate: usize) -> Result<Tensor> {
        if sample_rate != self.cfg.sample_rate {
            return Err(Error::Input(format!(
                "expected {} Hz input, got {sample_rate} Hz (resample first)",
                self.cfg.sample_rate
            )));
        }
        let device = Device::Cpu;
        let features = self.features.compute(samples, &device)?.unsqueeze(0)?;
        self.forward(&features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny() -> FastU2pp {
        let cfg = FastU2ppConfig {
            n_mels: 20,
            d_model: 32,
            num_heads: 2,
            ff_dim: 48,
            num_layers: 2,
            cnn_kernel: 7,
            chunk_frames: 2,
            left_chunks: usize::MAX,
            dual_conv_streaming: None,
            cmvn: false,
            sample_rate: 16_000,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        FastU2pp::new(cfg, vb).unwrap()
    }

    #[test]
    fn forward_subsamples_by_4() {
        let model = tiny();
        let feats = Tensor::randn(0f32, 1f32, (1, 43, 20), &Device::Cpu).unwrap();
        let bnf = model.forward(&feats).unwrap();
        // ((43 - 1) / 2 - 1) / 2 = 10 subsampled frames.
        assert_eq!(bnf.dims(), &[1, 10, 32]);
    }

    #[test]
    fn extract_produces_40ms_frames() {
        let model = tiny();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 330.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect();
        let bnf = model.extract(&samples, 16_000).unwrap();
        let (b, t, d) = bnf.dims3().unwrap();
        assert_eq!((b, d), (1, 32));
        // 1 s of audio -> ~25 BNF frames at 40 ms.
        assert!((23..=26).contains(&t), "unexpected frame count {t}");
        let v: Vec<f32> = bnf.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn chunk_mask_blocks_future_leakage() {
        // Prefix property: with a causal chunk mask, feeding a longer input
        // must not change the BNFs of the already-complete leading chunks
        // (up to conv look-ahead inside the subsampling front-end).
        let model = tiny();
        let dev = Device::Cpu;
        let full = Tensor::randn(0f32, 1f32, (1, 83, 20), &dev).unwrap();
        // 83 frames -> 20 subsampled; truncate raw frames so the prefix
        // yields 12 subsampled frames (6 complete chunks).
        let prefix = full.narrow(1, 0, 51).unwrap();
        let bnf_full = model.forward(&full).unwrap();
        let bnf_prefix = model.forward(&prefix).unwrap();
        let t = bnf_prefix.dim(1).unwrap();
        // The conv2d subsampling has a small intrinsic look-ahead (~3 raw
        // frames); drop the last chunk of the prefix and compare the rest.
        let cmp = t - model.config().chunk_frames;
        let a = bnf_full.narrow(1, 0, cmp).unwrap();
        let b = bnf_prefix.narrow(1, 0, cmp).unwrap();
        let diff: f32 = (&a - &b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-4, "future frames leaked into past chunks: {diff}");
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let model = tiny();
        assert!(model.extract(&[0.0; 8000], 44_100).is_err());
    }
}

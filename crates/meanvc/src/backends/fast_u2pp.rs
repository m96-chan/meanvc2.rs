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
//! Two chunked-decoding paths produce identical outputs: [`FastU2pp::forward`]
//! runs attention masks over the full utterance, while
//! [`FastU2pp::forward_chunk`] streams incrementally with per-layer
//! attention-K/V and depthwise-conv caches (WeNet's `forward_chunk`
//! equivalent, issue #9) at O(chunk) cost per call.

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
    /// The released MeanVC v1 checkpoint (`fastu2++.pt`): global CMVN and
    /// the official decode chunking (5-frame chunks, 2 left chunks).
    pub fn official_meanvc1() -> Self {
        Self {
            chunk_frames: 5,
            left_chunks: 2,
            cmvn: true,
            ..Self::default()
        }
    }

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
pub(crate) fn sinusoidal_pe(
    len: usize,
    d_model: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    sinusoidal_pe_range(0, len, d_model, device)
}

/// Sinusoidal positional encoding for absolute positions
/// `start .. start + len`, shape `[1, len, d_model]` (WeNet
/// `position_encoding(offset, size)`).
fn sinusoidal_pe_range(
    start: usize,
    len: usize,
    d_model: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut data = vec![0f32; len * d_model];
    for (row, pos) in (start..start + len).enumerate() {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10_000f32.powf(2.0 * i as f32 / d_model as f32);
            data[row * d_model + 2 * i] = angle.sin();
            data[row * d_model + 2 * i + 1] = angle.cos();
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
        let (out, _, _) = self.forward_cached(x, pos_emb, Some(mask), None)?;
        Ok(out)
    }

    /// Cache-aware attention (WeNet streaming `forward` with `cache`).
    ///
    /// `x`: current chunk `[batch, time, d]`; `cache`: previous keys/values
    /// `[batch, heads, cache_t, head_dim]` each, prepended to this chunk's
    /// K/V; `pos_emb`: `[1, cache_t + time, d]` over the absolute positions
    /// of the concatenated keys; `mask`: optional additive
    /// `[time, cache_t + time]` (streaming passes `None`: the cache is
    /// already trimmed to the allowed left context).
    ///
    /// Returns `(out, k, v)` where `k`/`v` are the *untrimmed* concatenated
    /// keys/values `[batch, heads, cache_t + time, head_dim]` for the next
    /// cache.
    fn forward_cached(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        mask: Option<&Tensor>,
        cache: Option<(&Tensor, &Tensor)>,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;
        let split = |x: Tensor, b: usize, t: usize| -> candle_core::Result<Tensor> {
            x.reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.linear_q.forward(x)?, b, t)?; // [b, h, t, d_k]
        let mut k = split(self.linear_k.forward(x)?, b, t)?;
        let mut v = split(self.linear_v.forward(x)?, b, t)?;
        if let Some((ck, cv)) = cache {
            k = Tensor::cat(&[ck, &k], 2)?;
            v = Tensor::cat(&[cv, &v], 2)?;
        }
        let t_key = k.dim(2)?;
        let p = split(self.linear_pos.forward(pos_emb)?, 1, t_key)?; // [1, h, t_key, d_k]

        let bias = |bias: &Tensor| bias.reshape((1, self.num_heads, 1, self.head_dim));
        let q_u = q.broadcast_add(&bias(&self.pos_bias_u)?)?;
        let q_v = q.broadcast_add(&bias(&self.pos_bias_v)?)?;

        let ac = q_u.matmul(&k.transpose(2, 3)?)?; // [b, h, t, t_key]
        let bd = q_v.broadcast_matmul(&p.transpose(2, 3)?)?; // [b, h, t, t_key]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = ((ac + bd)? * scale)?;
        let scores = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out =
            weights
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((b, t, self.num_heads * self.head_dim))?;
        Ok((self.linear_out.forward(&out)?, k, v))
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
            ConvNorm::Batch(batch_norm(
                d_model,
                BatchNormConfig::default(),
                vb.pp("norm"),
            )?)
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
        if self.causal {
            let (out, _) = self.forward_cached(x, None)?;
            return Ok(out);
        }
        // WeNet pads the module INPUT, before pointwise_conv1/GLU. Padding
        // after the GLU is not equivalent: pointwise_conv1 has a bias, so
        // the padded zeros become non-zero before the depthwise conv.
        let x = x.transpose(1, 2)?; // [b, d, t]
        let (b, d, _) = x.dims3()?;
        let pad = Tensor::zeros((b, d, (self.kernel - 1) / 2), x.dtype(), x.device())?;
        let x = Tensor::cat(&[pad.clone(), x, pad], 2)?;
        self.conv_stack(&x)?.transpose(1, 2)
    }

    /// Streaming causal path (WeNet `ConvolutionModule.forward` with
    /// `cache`): `x` `[batch, time, d]`, `cache` the previous module input
    /// `[batch, d, kernel - 1]` (zeros on the first chunk).
    ///
    /// Returns `(out [batch, time, d], new_cache [batch, d, kernel - 1])` —
    /// the cache is the raw input to `pointwise_conv1`, exactly as WeNet
    /// stores it.
    fn forward_cached(
        &self,
        x: &Tensor,
        cache: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        debug_assert!(self.causal, "conv cache requires the causal path");
        let x = x.transpose(1, 2)?; // [b, d, t]
        let (b, d, _) = x.dims3()?;
        let lorder = self.kernel - 1;
        let x = match cache {
            Some(cache) => Tensor::cat(&[cache, &x], 2)?,
            None => {
                let pad = Tensor::zeros((b, d, lorder), x.dtype(), x.device())?;
                Tensor::cat(&[pad, x], 2)?
            }
        };
        let padded = x.dim(2)?;
        let new_cache = x.narrow(2, padded - lorder, lorder)?;
        let out = self.conv_stack(&x)?.transpose(1, 2)?;
        Ok((out, new_cache))
    }

    /// pointwise_conv1 → GLU → depthwise → norm → SiLU → pointwise_conv2 on
    /// an already-padded `[batch, d, time]` input.
    fn conv_stack(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.pointwise_conv1.forward(x)?;
        // GLU over the channel dim.
        let halves = x.chunk(2, 1)?;
        let x = halves[0].mul(&candle_nn::ops::sigmoid(&halves[1])?)?;
        let x = self.depthwise_conv.forward(&x)?;
        let x = match &self.norm {
            ConvNorm::Layer(ln) => ln.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?,
            ConvNorm::Batch(bn) => bn.forward_t(&x, false)?,
        };
        self.pointwise_conv2.forward(&x.silu()?)
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

    /// Streaming forward with per-layer caches (WeNet
    /// `ConformerEncoderLayer.forward` with `att_cache` / `cnn_cache`).
    ///
    /// `x`: current chunk `[batch, time, d]`; `pos_emb`:
    /// `[1, cache_t + time, d]`; caches as in
    /// [`RelPositionAttention::forward_cached`] /
    /// [`ConvolutionModule::forward_cached`]. Always uses the causal
    /// (streaming) convolution path.
    ///
    /// Returns `(out, (k, v), cnn_cache)` — the untrimmed attention K/V and
    /// the next conv cache.
    #[allow(clippy::type_complexity)]
    fn forward_cached(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        att_cache: Option<(&Tensor, &Tensor)>,
        cnn_cache: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, (Tensor, Tensor), Tensor)> {
        let x = (x + self
            .feed_forward_macaron
            .forward(&self.norm_ff_macaron.forward(x)?)?
            .affine(0.5, 0.0)?)?;
        let (att, k, v) =
            self.self_attn
                .forward_cached(&self.norm_mha.forward(&x)?, pos_emb, None, att_cache)?;
        let x = (&x + att)?;
        let (conv, new_cnn) = self
            .conv_module
            .forward_cached(&self.norm_conv.forward(&x)?, cnn_cache)?;
        let x = (&x + conv)?;
        let x = (&x
            + self
                .feed_forward
                .forward(&self.norm_ff.forward(&x)?)?
                .affine(0.5, 0.0)?)?;
        Ok((self.norm_final.forward(&x)?, (k, v), new_cnn))
    }
}

/// Incremental state for [`FastU2pp::forward_chunk`] — the Rust equivalent
/// of WeNet's `forward_chunk_by_chunk` loop state (`att_cache`, `cnn_cache`,
/// `offset`) plus the raw-fbank window buffering.
///
/// Create with [`FastU2pp::stream`]; feed fbank frames with
/// [`FastU2pp::forward_chunk`].
#[derive(Debug)]
pub struct FastU2ppStream {
    /// Per-layer attention cache `(K, V)`, each
    /// `[batch, heads, cache_t, head_dim]` with `cache_t <=
    /// left_chunks * chunk_frames`.
    att_cache: Vec<Option<(Tensor, Tensor)>>,
    /// Per-layer depthwise-conv left context `[batch, d_model, kernel - 1]`.
    cnn_cache: Vec<Option<Tensor>>,
    /// Raw fbank frames not yet consumed by a full window
    /// `[batch, n, n_mels]`.
    fbank_buf: Option<Tensor>,
    /// Absolute subsampled-frame position of the next output frame.
    offset: usize,
}

impl FastU2ppStream {
    fn new(num_layers: usize) -> Self {
        Self {
            att_cache: vec![None; num_layers],
            cnn_cache: vec![None; num_layers],
            fbank_buf: None,
            offset: 0,
        }
    }

    /// Resets the stream to its initial (empty) state.
    pub fn reset(&mut self) {
        for kv in &mut self.att_cache {
            *kv = None;
        }
        for c in &mut self.cnn_cache {
            *c = None;
        }
        self.fbank_buf = None;
        self.offset = 0;
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

    /// Subsampled embedding before the conformer stack (×`xscale`),
    /// exposed for parity debugging against the official `encoder.embed`.
    pub fn subsample(&self, features: &Tensor) -> Result<Tensor> {
        Ok(self
            .embed
            .forward(features)?
            .affine((self.cfg.d_model as f64).sqrt(), 0.0)?)
    }

    /// Runs only the first conformer block on a subsampled embedding
    /// (parity debugging).
    #[doc(hidden)]
    pub fn debug_layer0(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let mask = frc::layer_mask(
            t,
            self.cfg.chunk_frames,
            self.cfg.left_chunks.min(t),
            0,
            x.device(),
        )?;
        Ok(self.encoders[0].forward(x, pos_emb, &mask)?)
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

    /// Raw fbank frames consumed per streaming window
    /// (`(chunk_frames - 1) * subsampling + context`, WeNet's
    /// `decoding_window`).
    fn window_frames(&self) -> usize {
        (self.cfg.chunk_frames - 1) * 4 + 7
    }

    /// Creates an empty incremental-decoding state for [`Self::forward_chunk`].
    pub fn stream(&self) -> FastU2ppStream {
        FastU2ppStream::new(self.cfg.num_layers)
    }

    /// Incremental streaming forward (WeNet `forward_chunk_by_chunk`,
    /// issue #9): pushes new fbank frames `[batch, n, n_mels]` into the
    /// stream and returns any BNF frames that became ready,
    /// `[batch, chunk_frames * windows, d_model]` (or `None` when fewer
    /// than one full window is buffered).
    ///
    /// Each window consumes `(chunk_frames - 1) * 4 + 7` raw frames with a
    /// stride of `4 * chunk_frames` and yields `chunk_frames` subsampled
    /// frames; attention runs over the per-layer K/V cache of the last
    /// `left_chunks * chunk_frames` frames plus the current chunk, and the
    /// depthwise convs carry a `kernel - 1` frame left context — output is
    /// identical to the mask-based [`Self::forward`] but with O(chunk)
    /// per-call cost. Always uses the causal (streaming) conv path.
    pub fn forward_chunk(
        &self,
        features: &Tensor,
        state: &mut FastU2ppStream,
    ) -> Result<Option<Tensor>> {
        if self.cfg.dual_conv_streaming == Some(false) {
            return Err(Error::Input(
                "forward_chunk streams with the causal conv path; \
                 dual_conv_streaming = Some(false) selects the non-causal one"
                    .into(),
            ));
        }
        let mut buf = match state.fbank_buf.take() {
            Some(buf) => Tensor::cat(&[&buf, features], 1)?,
            None => features.clone(),
        };
        let window = self.window_frames();
        let stride = 4 * self.cfg.chunk_frames;
        let mut outs = Vec::new();
        while buf.dim(1)? >= window {
            let xs = buf.narrow(1, 0, window)?;
            outs.push(self.forward_window(&xs, state)?);
            let remaining = buf.dim(1)? - stride;
            buf = buf.narrow(1, stride, remaining)?;
        }
        state.fbank_buf = Some(buf);
        if outs.is_empty() {
            return Ok(None);
        }
        Ok(Some(Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1)?))
    }

    /// Runs one full window (WeNet `forward_chunk`): `xs`
    /// `[batch, window_frames, n_mels]` → `[batch, chunk_frames, d_model]`,
    /// updating the per-layer caches and the position offset.
    fn forward_window(&self, xs: &Tensor, state: &mut FastU2ppStream) -> Result<Tensor> {
        let xs = match &self.cmvn {
            Some((mean, istd)) => xs.broadcast_sub(mean)?.broadcast_mul(istd)?,
            None => xs.clone(),
        };
        let x = self.embed.forward(&xs)?;
        let chunk = x.dim(1)?;
        let mut x = x.affine((self.cfg.d_model as f64).sqrt(), 0.0)?;

        let cache_t1 = match &state.att_cache[0] {
            Some((k, _)) => k.dim(2)?,
            None => 0,
        };
        let key_len = cache_t1 + chunk;
        // Absolute positions of the concatenated keys:
        // offset - cache_t1 .. offset + chunk.
        let pos_emb = sinusoidal_pe_range(
            state.offset - cache_t1,
            key_len,
            self.cfg.d_model,
            x.device(),
        )?;
        // Trim the next attention cache to the allowed left context
        // (WeNet `required_cache_size`); unlimited left context keeps all.
        let next_cache_start = match self.cfg.left_chunks {
            usize::MAX => 0,
            left => key_len.saturating_sub(left * self.cfg.chunk_frames),
        };
        for (i, block) in self.encoders.iter().enumerate() {
            let att = state.att_cache[i].as_ref().map(|(k, v)| (k, v));
            let cnn = state.cnn_cache[i].as_ref();
            let (out, (k, v), new_cnn) = block.forward_cached(&x, &pos_emb, att, cnn)?;
            let keep = key_len - next_cache_start;
            state.att_cache[i] = Some((
                k.narrow(2, next_cache_start, keep)?,
                v.narrow(2, next_cache_start, keep)?,
            ));
            state.cnn_cache[i] = Some(new_cnn);
            x = out;
        }
        state.offset += chunk;
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

    fn tiny_with(left_chunks: usize, dual: Option<bool>) -> FastU2pp {
        let cfg = FastU2ppConfig {
            n_mels: 20,
            d_model: 32,
            num_heads: 2,
            ff_dim: 48,
            num_layers: 2,
            cnn_kernel: 7,
            chunk_frames: 2,
            left_chunks,
            dual_conv_streaming: dual,
            cmvn: false,
            sample_rate: 16_000,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        FastU2pp::new(cfg, vb).unwrap()
    }

    /// Feeds `feats` to `forward_chunk` in pieces of `feed` raw frames and
    /// concatenates everything the stream emits.
    fn stream_all(model: &FastU2pp, feats: &Tensor, feed: usize) -> Tensor {
        let mut state = model.stream();
        let mut outs = Vec::new();
        let n = feats.dim(1).unwrap();
        let mut cur = 0;
        while cur < n {
            let take = feed.min(n - cur);
            let chunk = feats.narrow(1, cur, take).unwrap();
            cur += take;
            if let Some(bn) = model.forward_chunk(&chunk, &mut state).unwrap() {
                outs.push(bn);
            }
        }
        Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    #[test]
    fn forward_chunk_matches_masked_forward() {
        // Incremental streaming with att K/V + conv caches must reproduce
        // the mask-based full-utterance forward exactly (same attention
        // pattern, same causal convs, same absolute positions).
        let dev = Device::Cpu;
        for (left_chunks, dual) in [(2, None), (2, Some(true)), (usize::MAX, None)] {
            let model = tiny_with(left_chunks, dual);
            let feats = Tensor::randn(0f32, 1f32, (1, 83, 20), &dev).unwrap();
            let offline = model.forward(&feats).unwrap(); // 20 subsampled frames
                                                          // chunk_frames = 2 -> window 11 raw frames, stride 8.
            let streamed = stream_all(&model, &feats, 8);
            let t = streamed.dim(1).unwrap().min(offline.dim(1).unwrap());
            assert!(t >= 18, "stream emitted too few frames: {t}");
            let a = offline.narrow(1, 0, t).unwrap();
            let b = streamed.narrow(1, 0, t).unwrap();
            let diff = max_abs_diff(&a, &b);
            assert!(
                diff < 1e-4,
                "streaming/offline mismatch {diff} (left_chunks={left_chunks}, dual={dual:?})"
            );
        }
    }

    #[test]
    fn forward_chunk_is_invariant_to_feed_size() {
        // The internal fbank buffering must make the output independent of
        // how the caller slices the feature stream.
        let dev = Device::Cpu;
        let model = tiny_with(2, None);
        let feats = Tensor::randn(0f32, 1f32, (1, 60, 20), &dev).unwrap();
        let a = stream_all(&model, &feats, 5);
        let b = stream_all(&model, &feats, 32);
        assert_eq!(a.dims(), b.dims());
        assert!(max_abs_diff(&a, &b) < 1e-6);
    }
}

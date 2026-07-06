//! GLM-4-Voice speech tokenizer (X-VC `semantic_encoder`) + the semantic
//! adapter — candle port of the official modules
//! (`models/codec/sac/modules/semantic_encoder.py::WhisperVQEncoder` and
//! `models/codec/sac/modules/decoder.py::Decoder_with_upsample`,
//! X-VC arXiv:2604.12456 §3 / GLM-4-Voice).
//!
//! [`WhisperVqEncoder`] is a frozen Whisper-large-v3 encoder truncated to
//! 16 layers (`quantize_encoder_only`, no final LayerNorm) with four
//! modifications:
//!
//! * **causal convolutions** — `conv1`/`conv2` are left-padded by
//!   `kernel_size - 1` instead of symmetric padding;
//! * **block-causal self-attention** — every 100 ms frame attends to all
//!   past frames plus its own 200-frame (4 s) block, `tril ∪ block-diag`
//!   as an additive mask;
//! * **average pooling** k=4 after layer 16, 50 Hz → 12.5 Hz;
//! * **vector quantization** — nearest-neighbour lookup in a 16384×1280
//!   codebook, plus a learned post-VQ positional embedding
//!   (`embed_positions2`).
//!
//! [`SemanticAdapter`] upsamples the 12.5 Hz code embeddings back to the
//! 50 Hz / 1024-dim acoustic latent rate (Vocos-ConvNeXt backbone with two
//! ×2 `SamplingBlock` stages).
//!
//! Weights load from `ckpt/xvc_tokenizer.safetensors`
//! (`tools/convert_xvc_tokenizer.py`, official parameter names 1:1); see
//! [`load`].

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    conv1d, conv_transpose1d, layer_norm, linear, linear_no_bias, ops::softmax, Conv1d,
    Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, LayerNormConfig, Linear,
    Module, VarBuilder,
};
use vc_core::{Error, Result};

/// Architecture of the GLM-4-Voice tokenizer
/// (`zai-org/glm-4-voice-tokenizer` `config.json`).
#[derive(Debug, Clone)]
pub struct GlmTokenizerConfig {
    /// Mel input bins (`num_mel_bins`).
    pub n_mels: usize,
    /// Hidden width (`d_model`).
    pub d_model: usize,
    /// Encoder layers kept (`quantize_position`, = `pooling_position`).
    pub num_layers: usize,
    /// Attention heads (`encoder_attention_heads`).
    pub num_heads: usize,
    /// Feed-forward width (`encoder_ffn_dim`).
    pub ffn_dim: usize,
    /// Learned positional-embedding capacity (`max_source_positions`).
    pub max_source_positions: usize,
    /// Block-causal attention block, in pre-pool frames
    /// (`quantize_causal_block_size`).
    pub block_size: usize,
    /// Average-pool kernel after the last layer (`pooling_kernel_size`).
    pub pooling_kernel_size: usize,
    /// VQ codebook entries (`quantize_vocab_size`).
    pub vocab_size: usize,
}

impl Default for GlmTokenizerConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            d_model: 1280,
            num_layers: 16,
            num_heads: 20,
            ffn_dim: 5120,
            max_source_positions: 1500,
            block_size: 200,
            pooling_kernel_size: 4,
            vocab_size: 16384,
        }
    }
}

/// Pre-LayerNorm Whisper encoder self-attention (`WhisperAttention`,
/// SDPA semantics: additive mask, softmax in f32).
#[derive(Debug)]
struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
}

impl SelfAttention {
    fn new(d_model: usize, num_heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            q_proj: linear(d_model, d_model, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d_model, d_model, vb.pp("k_proj"))?,
            v_proj: linear(d_model, d_model, vb.pp("v_proj"))?,
            out_proj: linear(d_model, d_model, vb.pp("out_proj"))?,
            num_heads,
        })
    }

    /// `x`: `[batch, time, d_model]`, `mask`: additive `[1, 1, time, time]`.
    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let head_dim = c / self.num_heads;
        let shape = (b, t, self.num_heads, head_dim);
        let q = self
            .q_proj
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = (head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let probs = softmax(&scores.broadcast_add(mask)?, D::Minus1)?;
        let ctx = probs.matmul(&v)?.transpose(1, 2)?.reshape((b, t, c))?;
        self.out_proj.forward(&ctx)
    }
}

/// `WhisperVQEncoderLayer`: pre-LN attention + pre-LN GELU MLP.
#[derive(Debug)]
struct EncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: SelfAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn new(cfg: &GlmTokenizerConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn_layer_norm: layer_norm(
                cfg.d_model,
                LayerNormConfig::default(),
                vb.pp("self_attn_layer_norm"),
            )?,
            self_attn: SelfAttention::new(cfg.d_model, cfg.num_heads, vb.pp("self_attn"))?,
            final_layer_norm: layer_norm(
                cfg.d_model,
                LayerNormConfig::default(),
                vb.pp("final_layer_norm"),
            )?,
            fc1: linear(cfg.d_model, cfg.ffn_dim, vb.pp("fc1"))?,
            fc2: linear(cfg.ffn_dim, cfg.d_model, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.self_attn_layer_norm.forward(x)?;
        let x = (x + self.self_attn.forward(&h, mask)?)?;
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu_erf()?)?;
        x + h
    }
}

/// Output of [`WhisperVqEncoder::forward`].
#[derive(Debug)]
pub struct TokenizerOutput {
    /// Layer-16 hidden states before pooling, `[batch, d_model, frames]`
    /// (50 Hz, padded to a pool-kernel multiple) —
    /// `whisper_hidden_states_50hz`.
    pub hidden_50hz: Tensor,
    /// Pooled 12.5 Hz hidden states, the VQ input,
    /// `[batch, tokens, d_model]`.
    pub hidden_prevq: Tensor,
    /// VQ code indices, `[batch, tokens]` i64 — `quantized_token_ids`.
    pub token_ids: Tensor,
    /// Quantized embeddings + `embed_positions2`,
    /// `[batch, tokens, d_model]` — `last_hidden_state`
    /// (`quantize_encoder_only`: no final LayerNorm).
    pub hidden_postvq: Tensor,
}

/// The GLM-4-Voice tokenizer (`WhisperVQEncoder`), ~343.6M params fp32.
pub struct WhisperVqEncoder {
    cfg: GlmTokenizerConfig,
    conv1: Conv1d,
    conv2: Conv1d,
    /// `[max_source_positions, d_model]`.
    embed_positions: Tensor,
    layers: Vec<EncoderLayer>,
    /// `[vocab_size, d_model]`.
    codebook: Tensor,
    /// `codebook.t()`, contiguous `[d_model, vocab_size]`.
    codebook_t: Tensor,
    /// Per-code squared norms, `[vocab_size]`.
    codebook_sqr: Tensor,
    /// `[ceil(max_source_positions / pool), d_model]`.
    embed_positions2: Tensor,
}

impl std::fmt::Debug for WhisperVqEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperVqEncoder")
            .field("cfg", &self.cfg)
            .finish()
    }
}

impl WhisperVqEncoder {
    pub fn new(cfg: GlmTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        // Causal convs: no framework padding, the forward left-pads by
        // `kernel - 1` (CausalConv1d).
        let conv_cfg = Conv1dConfig::default();
        let conv2_cfg = Conv1dConfig {
            stride: 2,
            ..Default::default()
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(EncoderLayer::new(&cfg, vb.pp(format!("layers.{i}")))?);
        }
        let codebook = vb.get((cfg.vocab_size, cfg.d_model), "codebook.weight")?;
        let codebook_t = codebook.t()?.contiguous()?;
        let codebook_sqr = codebook.sqr()?.sum(D::Minus1)?;
        Ok(Self {
            conv1: conv1d(cfg.n_mels, cfg.d_model, 3, conv_cfg, vb.pp("conv1"))?,
            conv2: conv1d(cfg.d_model, cfg.d_model, 3, conv2_cfg, vb.pp("conv2"))?,
            embed_positions: vb.get(
                (cfg.max_source_positions, cfg.d_model),
                "embed_positions.weight",
            )?,
            layers,
            codebook,
            codebook_t,
            codebook_sqr,
            embed_positions2: vb.get(
                (
                    cfg.max_source_positions.div_ceil(cfg.pooling_kernel_size),
                    cfg.d_model,
                ),
                "embed_positions2.weight",
            )?,
            cfg,
        })
    }

    pub fn config(&self) -> &GlmTokenizerConfig {
        &self.cfg
    }

    /// Additive block-causal attention mask
    /// (`get_block_causal_attention_mask`): position `i` attends to `j`
    /// iff (`j <= i` or `i`, `j` share a `block_size` block) and key `j`
    /// is valid. `[1, 1, seq, seq]` f32, 0 = attend, `f32::MIN` = masked.
    fn block_causal_mask(
        &self,
        seq: usize,
        valid: &[bool],
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let block = self.cfg.block_size;
        let mut data = vec![f32::MIN; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if (j <= i || j / block == i / block) && valid[j] {
                    data[i * seq + j] = 0.0;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, seq, seq), device)
    }

    /// Nearest-neighbour codebook lookup (`vector_quantize`): distances
    /// via `‖c‖² + ‖x‖² − 2·x·cᵀ`, argmin over the 16384 codes.
    /// `flat`: `[n, d_model]` → (codes `[n, d_model]`, indices `[n]` u32).
    fn vector_quantize(&self, flat: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let x_sqr = flat.sqr()?.sum_keepdim(D::Minus1)?; // [n, 1]
        let dist = (flat.matmul(&self.codebook_t)? * (-2.0))?
            .broadcast_add(&x_sqr)?
            .broadcast_add(&self.codebook_sqr)?; // [n, vocab]
        let indices = dist.argmin(D::Minus1)?; // [n] u32
        let codes = self.codebook.index_select(&indices, 0)?;
        Ok((codes, indices))
    }

    /// Runs the tokenizer on Whisper log-mel features.
    ///
    /// * `mel` — `[batch, n_mels, frames]`
    ///   ([`crate::preprocess::WhisperFeatureExtractor`] output; `frames`
    ///   must be even, i.e. the input was padded to the 1280-sample
    ///   stride).
    /// * `attention_mask` — per-mel-frame validity (1/0), length `frames`.
    pub fn forward(&self, mel: &Tensor, attention_mask: &[u32]) -> Result<TokenizerOutput> {
        let (batch, _, t_mel) = mel.dims3()?;
        if attention_mask.len() != t_mel {
            return Err(Error::Input(format!(
                "attention_mask length {} != mel frames {t_mel}",
                attention_mask.len()
            )));
        }
        let device = mel.device();
        let pool = self.cfg.pooling_kernel_size;

        // Causal convs (left pad 2) + GELU; conv2 has stride 2 -> 50 Hz.
        let x = self
            .conv1
            .forward(&mel.pad_with_zeros(D::Minus1, 2, 0)?)?
            .gelu_erf()?;
        let x = self
            .conv2
            .forward(&x.pad_with_zeros(D::Minus1, 2, 0)?)?
            .gelu_erf()?;
        let x = x.transpose(1, 2)?.contiguous()?; // [B, T, C]
        let t = x.dim(1)?;

        // Sinusoid-initialized (frozen) absolute positions.
        let mut h = x.broadcast_add(&self.embed_positions.i(..t)?)?;

        // Frame validity at the post-conv rate (`attention_mask[:, ::2]`).
        let valid: Vec<bool> = (0..t).map(|i| attention_mask[2 * i] != 0).collect();
        let mask = self.block_causal_mask(t, &valid, device)?;

        for layer in &self.layers {
            h = layer.forward(&h, &mask)?;
        }

        // Pooling (after layer `pooling_position` = the last one): pad the
        // 50 Hz sequence to a pool multiple, then average-pool k=4.
        let t_pad = t.div_ceil(pool) * pool;
        if t_pad > t {
            h = h.pad_with_zeros(1, 0, t_pad - t)?;
        }
        let hidden_50hz = h.transpose(1, 2)?.contiguous()?; // [B, C, Tpad]
        let tokens = t_pad / pool;
        let pooled = h
            .reshape((batch, tokens, pool, self.cfg.d_model))?
            .mean(2)?; // [B, tokens, C]

        // VQ + learned post-VQ positions (`embed_positions2`).
        let flat = pooled.reshape((batch * tokens, self.cfg.d_model))?;
        let (codes, indices) = self.vector_quantize(&flat)?;
        let hidden_postvq = codes
            .reshape((batch, tokens, self.cfg.d_model))?
            .broadcast_add(&self.embed_positions2.i(..tokens)?)?;
        let token_ids = indices.reshape((batch, tokens))?.to_dtype(DType::I64)?;

        Ok(TokenizerOutput {
            hidden_50hz,
            hidden_prevq: pooled,
            token_ids,
            hidden_postvq,
        })
    }

    /// Codebook lookup (`embed_ids`): `[batch, tokens]` i64 →
    /// `[batch, tokens, d_model]`.
    pub fn embed_ids(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let flat = ids.flatten_all()?.to_dtype(DType::U32)?;
        Ok(self
            .codebook
            .index_select(&flat, 0)?
            .reshape((b, t, self.cfg.d_model))?)
    }
}

/// Configuration of [`SemanticAdapter`] (`configs/xvc.yaml`
/// `semantic_adapter`).
#[derive(Debug, Clone)]
pub struct SemanticAdapterConfig {
    /// Input width (`input_channels`, the tokenizer `d_model`).
    pub input_channels: usize,
    /// Backbone width (`vocos_dim`).
    pub vocos_dim: usize,
    /// ConvNeXt expansion width (`vocos_intermediate_dim`).
    pub vocos_intermediate_dim: usize,
    /// Main backbone depth (`vocos_num_layers`).
    pub vocos_num_layers: usize,
    /// Output width (`out_channels`, the acoustic latent dim).
    pub out_channels: usize,
    /// Upsample stage ratios (`sample_ratios`).
    pub sample_ratios: Vec<usize>,
}

impl Default for SemanticAdapterConfig {
    fn default() -> Self {
        Self {
            input_channels: 1280,
            vocos_dim: 384,
            vocos_intermediate_dim: 2048,
            vocos_num_layers: 12,
            out_channels: 1024,
            sample_ratios: vec![2, 2],
        }
    }
}

/// Depthwise conv1d (stride/dilation 1) expressed as `k` shifted
/// broadcast-mul-adds:
/// `y[b, c, t] = bias[c] + Σᵢ w[c, 0, i] · x_pad[b, c, t + i]`.
/// Mathematically identical to the grouped conv; on CUDA candle's grouped
/// conv1d dispatches **one kernel per group** (= `dim` launches per call),
/// which made the two ConvNeXt stacks (semantic adapter + prenet) dominate
/// the whole GPU pipeline (issue #38: 90 + 167 ms of the ~275 ms window
/// forward) — this formulation needs ~2·k launches.
fn depthwise_conv1d_shifted(conv: &Conv1d, x: &Tensor) -> candle_core::Result<Tensor> {
    let w = conv.weight(); // [dim, 1, k]
    let (dim, _, k) = w.dims3()?;
    let pad = conv.config().padding;
    let xp = if pad > 0 {
        x.pad_with_zeros(D::Minus1, pad, pad)?
    } else {
        x.clone()
    };
    let t_out = xp.dim(D::Minus1)? - (k - 1);
    let mut acc: Option<Tensor> = None;
    for i in 0..k {
        let wi = w.narrow(D::Minus1, i, 1)?.reshape((1, dim, 1))?;
        let term = xp.narrow(D::Minus1, i, t_out)?.broadcast_mul(&wi)?;
        acc = Some(match acc {
            Some(a) => (a + term)?,
            None => term,
        });
    }
    let y = acc.expect("kernel size > 0");
    match conv.bias() {
        Some(b) => y.broadcast_add(&b.reshape((1, dim, 1))?),
        None => Ok(y),
    }
}

/// Depthwise conv1d on the fast path for the device: the stock grouped
/// conv1d on CPU (bit-exact with the golden fixtures), the shifted-add
/// formulation on accelerators (see [`depthwise_conv1d_shifted`]).
fn depthwise_conv1d(conv: &Conv1d, x: &Tensor) -> candle_core::Result<Tensor> {
    if matches!(x.device(), Device::Cpu) {
        conv.forward(x)
    } else {
        depthwise_conv1d_shifted(conv, x)
    }
}

/// ConvNeXt block (`models/codec/sac/modules/vocos.py::ConvNeXtBlock`):
/// depthwise conv → LayerNorm (eps 1e-6) → pointwise GELU MLP → layer
/// scale, residual. `[batch, dim, time]` in/out.
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
        let ln = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            dwconv: conv1d(dim, dim, 7, conv_cfg, vb.pp("dwconv"))?,
            norm: layer_norm(dim, ln, vb.pp("norm"))?,
            pwconv1: linear(dim, intermediate_dim, vb.pp("pwconv1"))?,
            pwconv2: linear(intermediate_dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let h = depthwise_conv1d(&self.dwconv, x)?.transpose(1, 2)?; // [B, T, C]
        let h = self.norm.forward(&h.contiguous()?)?;
        let h = self
            .pwconv2
            .forward(&self.pwconv1.forward(&h)?.gelu_erf()?)?;
        let h = h.broadcast_mul(&self.gamma)?.transpose(1, 2)?;
        residual + h
    }
}

/// Unconditional `VocosBackbone`: embed conv → LayerNorm → ConvNeXt stack
/// → final LayerNorm. `[batch, dim, time]` → `[batch, time, dim]`.
#[derive(Debug)]
struct VocosBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    convnext: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
}

impl VocosBackbone {
    fn new(
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let ln = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let mut convnext = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            convnext.push(ConvNeXtBlock::new(
                dim,
                intermediate_dim,
                vb.pp(format!("convnext.{i}")),
            )?);
        }
        Ok(Self {
            embed: conv1d(input_channels, dim, 7, conv_cfg, vb.pp("embed"))?,
            norm: layer_norm(dim, ln, vb.pp("norm"))?,
            convnext,
            final_layer_norm: layer_norm(dim, ln, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.embed.forward(x)?;
        let mut x = self
            .norm
            .forward(&x.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        for block in &self.convnext {
            x = block.forward(&x)?;
        }
        self.final_layer_norm
            .forward(&x.transpose(1, 2)?.contiguous()?)
    }
}

/// ×`scale` upsampler (`models/codec/sac/modules/sampler.py::SamplingBlock`):
/// `repeat_interleave` skip + depthwise transposed conv (LeakyReLU 0.2).
/// With no downsampler the official skip wiring degenerates to
/// `(upmerge + repeat) + upmerge` where `upmerge = repeat + deconv`
/// (i.e. 3·repeat + 2·deconv). `[batch, time, dim]` → `[batch, dim, scale·time]`.
///
/// With `scale == 1` (the prenet's `sample_ratios = [1, 1]`) the official
/// block has no upsampler at all and every skip is the identity, so it
/// degenerates to a transpose plus `3·x`.
#[derive(Debug)]
struct SamplingBlock {
    de_conv_upsampler: Option<ConvTranspose1d>,
    scale: usize,
}

impl SamplingBlock {
    fn new(dim: usize, scale: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let de_conv_upsampler = if scale > 1 {
            let cfg = ConvTranspose1dConfig {
                padding: scale / 2 + scale % 2,
                output_padding: scale % 2,
                stride: scale,
                dilation: 1,
                groups: dim,
            };
            Some(conv_transpose1d(
                dim,
                dim,
                scale * 2,
                cfg,
                vb.pp("de_conv_upsampler.1"),
            )?)
        } else {
            None
        };
        Ok(Self {
            de_conv_upsampler,
            scale,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // [B, C, T]
        let Some(de_conv_upsampler) = &self.de_conv_upsampler else {
            // scale == 1: conv_res = skip1 = skip2 = x.
            return x * 3.0;
        };
        let (b, c, t) = x.dims3()?;
        let repeat =
            x.unsqueeze(3)?
                .expand((b, c, t, self.scale))?
                .reshape((b, c, t * self.scale))?;
        let deconv = de_conv_upsampler.forward(&candle_nn::ops::leaky_relu(&x, 0.2)?)?;
        // Official skip sum, in evaluation order: conv_res + skip1 + skip2
        // with conv_res = skip2 = upmerge (= repeat + deconv), skip1 = repeat.
        let upmerge = (&repeat + deconv)?;
        (upmerge.clone() + repeat)? + upmerge
    }
}

/// The semantic adapter (`semantic_adapter`, `Decoder_with_upsample`):
/// 12.5 Hz code embeddings → 50 Hz / 1024-dim latents, ~29.3M params.
pub struct SemanticAdapter {
    linear_pre: Linear,
    upsample: Vec<(SamplingBlock, VocosBackbone)>,
    vocos_backbone: VocosBackbone,
    linear: Linear,
    cfg: SemanticAdapterConfig,
}

impl std::fmt::Debug for SemanticAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticAdapter")
            .field("cfg", &self.cfg)
            .finish()
    }
}

impl SemanticAdapter {
    pub fn new(cfg: SemanticAdapterConfig, vb: VarBuilder) -> Result<Self> {
        let mut upsample = Vec::with_capacity(cfg.sample_ratios.len());
        for (i, &ratio) in cfg.sample_ratios.iter().enumerate() {
            let stage = vb.pp(format!("upsample.{i}"));
            upsample.push((
                SamplingBlock::new(cfg.vocos_dim, ratio, stage.pp("0"))?,
                VocosBackbone::new(
                    cfg.vocos_dim,
                    cfg.vocos_dim,
                    cfg.vocos_intermediate_dim,
                    2,
                    stage.pp("1"),
                )?,
            ));
        }
        Ok(Self {
            linear_pre: linear(cfg.input_channels, cfg.vocos_dim, vb.pp("linear_pre"))?,
            upsample,
            vocos_backbone: VocosBackbone::new(
                cfg.vocos_dim,
                cfg.vocos_dim,
                cfg.vocos_intermediate_dim,
                cfg.vocos_num_layers,
                vb.pp("vocos_backbone"),
            )?,
            linear: linear(cfg.vocos_dim, cfg.out_channels, vb.pp("linear"))?,
            cfg,
        })
    }

    pub fn config(&self) -> &SemanticAdapterConfig {
        &self.cfg
    }

    /// `x`: `[batch, input_channels, tokens]` (12.5 Hz code embeddings,
    /// i.e. `embed_ids(...).transpose(1, 2)`) →
    /// `[batch, out_channels, tokens · ∏ratios]` (50 Hz latents).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.linear_pre.forward(&x.transpose(1, 2)?.contiguous()?)?; // [B, T, vocos_dim]
        for (sampler, backbone) in &self.upsample {
            h = backbone.forward(&sampler.forward(&h)?)?; // [B, r·T, vocos_dim]
        }
        let h = self
            .vocos_backbone
            .forward(&h.transpose(1, 2)?.contiguous()?)?;
        Ok(self.linear.forward(&h)?.transpose(1, 2)?.contiguous()?)
    }
}

/// Loads the tokenizer + semantic adapter from
/// `ckpt/xvc_tokenizer.safetensors` (see `tools/convert_xvc_tokenizer.py`)
/// with the default GLM-4-Voice / X-VC configuration.
pub fn load(
    path: impl AsRef<std::path::Path>,
    device: &Device,
) -> Result<(WhisperVqEncoder, SemanticAdapter)> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], DType::F32, device)? };
    let encoder = WhisperVqEncoder::new(GlmTokenizerConfig::default(), vb.clone())?;
    let adapter =
        SemanticAdapter::new(SemanticAdapterConfig::default(), vb.pp("semantic_adapter"))?;
    Ok((encoder, adapter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// The shifted-add depthwise conv (the CUDA fast path) must match the
    /// stock grouped conv1d (the CPU / golden-fixture path).
    #[test]
    fn depthwise_shifted_matches_grouped_conv() {
        let dev = Device::Cpu;
        let dim = 24;
        let t = 37;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let conv = conv1d(
            dim,
            dim,
            7,
            Conv1dConfig {
                padding: 3,
                groups: dim,
                ..Default::default()
            },
            vb.pp("dw"),
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, dim, t), &dev).unwrap();
        let want = conv.forward(&x).unwrap();
        let got = depthwise_conv1d_shifted(&conv, &x).unwrap();
        assert_eq!(got.dims(), want.dims());
        let diff = (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "shifted depthwise conv diff {diff}");
    }

    fn tiny_cfg() -> GlmTokenizerConfig {
        GlmTokenizerConfig {
            n_mels: 8,
            d_model: 16,
            num_layers: 2,
            num_heads: 2,
            ffn_dim: 32,
            max_source_positions: 64,
            block_size: 4,
            pooling_kernel_size: 4,
            vocab_size: 32,
        }
    }

    #[test]
    fn forward_shapes() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = tiny_cfg();
        let enc = WhisperVqEncoder::new(cfg, vb).unwrap();
        // 16 mel frames -> 8 post-conv -> 2 tokens.
        let mel = Tensor::randn(0f32, 1f32, (1, 8, 16), &dev).unwrap();
        let out = enc.forward(&mel, &[1u32; 16]).unwrap();
        assert_eq!(out.hidden_50hz.dims(), &[1, 16, 8]);
        assert_eq!(out.hidden_prevq.dims(), &[1, 2, 16]);
        assert_eq!(out.token_ids.dims(), &[1, 2]);
        assert_eq!(out.hidden_postvq.dims(), &[1, 2, 16]);

        // embed_ids inverts to codebook rows.
        let emb = enc.embed_ids(&out.token_ids).unwrap();
        assert_eq!(emb.dims(), &[1, 2, 16]);
    }

    #[test]
    fn block_causal_mask_geometry() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let enc = WhisperVqEncoder::new(tiny_cfg(), vb).unwrap();
        // block 4, seq 6: [0..4) is block 0, [4..6) block 1.
        let mask = enc
            .block_causal_mask(6, &[true; 6], &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let attend = |i: usize, j: usize| mask[i * 6 + j] == 0.0;
        assert!(attend(0, 3), "future within own block");
        assert!(!attend(0, 4), "no future across blocks");
        assert!(attend(4, 0), "full past");
        assert!(attend(4, 5), "future within second block");
    }

    #[test]
    fn masked_keys_are_excluded() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let enc = WhisperVqEncoder::new(tiny_cfg(), vb).unwrap();
        let mut valid = [true; 6];
        valid[5] = false;
        let mask = enc
            .block_causal_mask(6, &valid, &Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(mask[4 * 6 + 5] == f32::MIN, "padded key masked");
    }

    #[test]
    fn sampling_block_scale_one_is_triple_identity() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let block = SamplingBlock::new(4, 1, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 5, 4), &dev).unwrap();
        let y = block.forward(&x).unwrap(); // [B, C, T]
        let expect = (x.transpose(1, 2).unwrap() * 3.0).unwrap();
        let diff = (y - expect)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-6,
            "scale-1 SamplingBlock must be 3·x, diff {diff}"
        );
    }

    #[test]
    fn adapter_with_unit_ratios_preserves_time() {
        // The prenet configuration: `sample_ratios = [1, 1]`.
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = SemanticAdapterConfig {
            input_channels: 24,
            vocos_dim: 8,
            vocos_intermediate_dim: 16,
            vocos_num_layers: 2,
            out_channels: 12,
            sample_ratios: vec![1, 1],
        };
        let prenet = SemanticAdapter::new(cfg, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 24, 5), &dev).unwrap();
        let y = prenet.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 12, 5]);
    }

    #[test]
    fn adapter_upsamples_4x() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = SemanticAdapterConfig {
            input_channels: 16,
            vocos_dim: 8,
            vocos_intermediate_dim: 16,
            vocos_num_layers: 2,
            out_channels: 12,
            sample_ratios: vec![2, 2],
        };
        let adapter = SemanticAdapter::new(cfg, vb).unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 16, 5), &dev).unwrap();
        let y = adapter.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 12, 20]);
    }
}

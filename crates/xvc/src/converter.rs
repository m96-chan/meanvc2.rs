//! MMDiT acoustic converter — candle port of X-VC's dual-conditioning
//! one-step converter (arXiv:2604.12456 §2.2, `AcousticConverter`).
//!
//! A 6-block MMDiT (42 M params, dim 512 / 8 heads): every block runs
//! **joint attention** over the concatenation of the acoustic-latent
//! sequence (`x`, 50 Hz, RoPE) and the frame-level condition sequence
//! (target mel, 128-dim, its own RoPE restarting at position 0), while
//! the 192-dim utterance-level speaker embedding modulates the `x` stream
//! through AdaLN-Zero. The condition stream uses plain affine LayerNorms.
//! The last block is `context_pre_only`: the condition stream is attended
//! to but produces no output. One forward pass converts the latents —
//! no diffusion loop.
//!
//! Layout and parameter names mirror the official implementation
//! (`models/codec/sac/modules/acoustic_converter.py` of
//! [Jerrister/X-VC](https://github.com/Jerrister/X-VC)):
//! `acoustic_converter.input_embed.*`, `.transformer_blocks.{i}.*`,
//! `.norm_out.*`, `.proj_out.*`, `.rotary_embed.inv_freq`.
//!
//! Faithful quirks of the official code, kept for weight parity:
//!
//! * RoPE (`x_transformers`, interleaved pairs) is applied to the packed
//!   `[batch, time, heads·head_dim]` projections **before** the head
//!   split with `rot_dim = head_dim = 64`, so only the first head is
//!   rotated;
//! * the condition stream additionally receives an absolute sinusoidal
//!   position embedding (`freqs_cis`: 256 cos ‖ 256 sin) on top of its
//!   input projection, before a grouped-conv position embedding.

use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{
    conv1d, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear, Module,
    VarBuilder,
};

use vc_core::Result;

/// Configuration of the MMDiT converter. Defaults are the released X-VC
/// converter (`configs/xvc.yaml`: dim 512, depth 6, heads 8×64).
#[derive(Debug, Clone)]
pub struct AcousticConverterConfig {
    /// Channels of the acoustic-latent stream `x` (in and out).
    pub in_channels_x: usize,
    /// Channels of the frame-level condition (mel bins).
    pub in_channels_c: usize,
    /// Dimension of the utterance-level (speaker) condition.
    pub condition_dim: usize,
    /// Transformer width.
    pub dim: usize,
    /// Number of MMDiT blocks.
    pub depth: usize,
    /// Attention heads.
    pub heads: usize,
    /// Per-head dimension (also the RoPE rotation width).
    pub dim_head: usize,
    /// Feed-forward expansion factor.
    pub ff_mult: usize,
    /// Maximum length of the precomputed sinusoidal position table.
    pub max_pos: usize,
}

impl Default for AcousticConverterConfig {
    fn default() -> Self {
        Self {
            in_channels_x: 1024,
            in_channels_c: 128,
            condition_dim: 192,
            dim: 512,
            depth: 6,
            heads: 8,
            dim_head: 64,
            ff_mult: 4,
            max_pos: 1024,
        }
    }
}

const LN_EPS: f64 = 1e-6;

/// LayerNorm without affine parameters (`elementwise_affine=False`).
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(var + eps)?.sqrt()?)
}

/// The sinusoidal table of `precompute_freqs_cis`: row `t` is
/// `[cos(t·ω₀) … cos(t·ω₂₅₅) ‖ sin(t·ω₀) … sin(t·ω₂₅₅)]` with
/// `ωᵢ = 10000^(−2i/dim)`. Arithmetic in f32, matching torch.
fn precompute_freqs_cis(dim: usize, end: usize, device: &Device) -> candle_core::Result<Tensor> {
    let half = dim / 2;
    let theta = 10000f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let mut data = Vec::with_capacity(end * dim);
    for t in 0..end {
        let row: Vec<f32> = freqs.iter().map(|f| t as f32 * f).collect();
        data.extend(row.iter().map(|a| a.cos()));
        data.extend(row.iter().map(|a| a.sin()));
    }
    Tensor::from_vec(data, (end, dim), device)
}

/// Rotary embedding (`x_transformers::RotaryEmbedding`, interleaved
/// pairs). `inv_freq` (`[dim_head / 2]`) comes from the checkpoint.
#[derive(Debug)]
struct RotaryEmbedding {
    inv_freq: Vec<f32>,
    device: Device,
}

/// `cos`/`sin` tables of shape `[1, time, dim_head / 2]`.
type Rope = (Tensor, Tensor);

impl RotaryEmbedding {
    fn new(dim_head: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let inv_freq = vb.get(dim_head / 2, "inv_freq")?.to_vec1::<f32>()?;
        Ok(Self {
            inv_freq,
            device: vb.device().clone(),
        })
    }

    fn forward_from_seq_len(&self, seq_len: usize) -> candle_core::Result<Rope> {
        let half = self.inv_freq.len();
        let mut cos = Vec::with_capacity(seq_len * half);
        let mut sin = Vec::with_capacity(seq_len * half);
        for t in 0..seq_len {
            for &f in &self.inv_freq {
                let a = t as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (1, seq_len, half), &self.device)?,
            Tensor::from_vec(sin, (1, seq_len, half), &self.device)?,
        ))
    }
}

/// `x_transformers::apply_rotary_pos_emb` on packed heads: rotates only
/// the first `rot_dim = 2·(cos width)` channels of `[batch, time, dim]`
/// as interleaved pairs, passing the rest through.
fn apply_rotary_pos_emb(x: &Tensor, (cos, sin): &Rope) -> candle_core::Result<Tensor> {
    let (b, t, d) = x.dims3()?;
    let half = cos.dim(D::Minus1)?;
    let rot_dim = 2 * half;
    let xr = x.narrow(D::Minus1, 0, rot_dim)?.reshape((b, t, half, 2))?;
    let x1 = xr.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let x2 = xr.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    // (x1, x2) → (x1·cos − x2·sin, x2·cos + x1·sin)
    let o1 = x1.broadcast_mul(cos)?.sub(&x2.broadcast_mul(sin)?)?;
    let o2 = x2.broadcast_mul(cos)?.add(&x1.broadcast_mul(sin)?)?;
    let rotated = Tensor::stack(&[o1, o2], 3)?.reshape((b, t, rot_dim))?;
    let rest = x.narrow(D::Minus1, rot_dim, d - rot_dim)?;
    Tensor::cat(&[rotated, rest], D::Minus1)
}

/// Two grouped k31 convolutions with Mish, added residually to the
/// embedding (`ConvPositionEmbedding`). `x`: `[batch, time, dim]`.
#[derive(Debug)]
struct ConvPositionEmbedding {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ConvPositionEmbedding {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv1dConfig {
            padding: 15,
            groups: 16,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv1d(dim, dim, 31, cfg, vb.pp("conv1d.0"))?,
            conv2: conv1d(dim, dim, 31, cfg, vb.pp("conv1d.2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = candle_nn::ops::mish(&self.conv1.forward(&x)?)?;
        let x = candle_nn::ops::mish(&self.conv2.forward(&x)?)?;
        x.transpose(1, 2)
    }
}

/// Input embedding: linear projections of `x` and the condition into the
/// transformer width, each followed by a residual conv position
/// embedding; the condition additionally gets the absolute sinusoidal
/// table added first.
#[derive(Debug)]
struct InputEmbedding {
    linear_x: Linear,
    linear_cond: Linear,
    conv_pos_embed_x: ConvPositionEmbedding,
    conv_pos_embed_cond: ConvPositionEmbedding,
    /// `[max_pos, dim]` sinusoidal table (non-persistent in the official
    /// checkpoint; recomputed here).
    freqs_cis: Tensor,
}

impl InputEmbedding {
    fn new(cfg: &AcousticConverterConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            linear_x: linear(cfg.in_channels_x, cfg.dim, vb.pp("linear_x"))?,
            linear_cond: linear(cfg.in_channels_c, cfg.dim, vb.pp("linear_cond"))?,
            conv_pos_embed_x: ConvPositionEmbedding::new(cfg.dim, vb.pp("conv_pos_embed_x"))?,
            conv_pos_embed_cond: ConvPositionEmbedding::new(cfg.dim, vb.pp("conv_pos_embed_cond"))?,
            freqs_cis: precompute_freqs_cis(cfg.dim, cfg.max_pos, vb.device())?,
        })
    }

    /// `x`: `[batch, in_x, time]`, `cond`: `[batch, in_c, time_c]` →
    /// (`[batch, time, dim]`, `[batch, time_c, dim]`).
    fn forward(&self, x: &Tensor, cond: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let x = x.transpose(1, 2)?.contiguous()?;
        let x_embed = self.linear_x.forward(&x)?;
        let x_embed = (self.conv_pos_embed_x.forward(&x_embed)? + &x_embed)?;

        let cond = cond.transpose(1, 2)?.contiguous()?;
        let cond_embed = self.linear_cond.forward(&cond)?;
        let seq_len = cond_embed.dim(1)?;
        // get_pos_embed_indices with start = 0 and scale = 1 is just
        // 0..seq_len, clamped to the table (positions past it repeat the
        // last row, as in the official code).
        let max_pos = self.freqs_cis.dim(0)?;
        let pos_embed = if seq_len <= max_pos {
            self.freqs_cis.narrow(0, 0, seq_len)?
        } else {
            let last = self.freqs_cis.narrow(0, max_pos - 1, 1)?;
            let tail = last.expand((seq_len - max_pos, self.freqs_cis.dim(1)?))?;
            Tensor::cat(&[self.freqs_cis.clone(), tail], 0)?
        };
        let cond_embed = cond_embed.broadcast_add(&pos_embed.unsqueeze(0)?)?;
        let cond_embed = (self.conv_pos_embed_cond.forward(&cond_embed)? + &cond_embed)?;

        Ok((x_embed, cond_embed))
    }
}

/// AdaLN-Zero: the global condition is projected (SiLU → linear, zero
/// initialized upstream) to shift/scale/gate for both the attention and
/// the feed-forward of the `x` stream.
#[derive(Debug)]
struct AdaLayerNormZero {
    linear: Linear,
}

impl AdaLayerNormZero {
    fn new(dim: usize, cond_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            linear: linear(cond_dim, dim * 6, vb.pp("linear"))?,
        })
    }

    /// Returns `(norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)`;
    /// modulation tensors are `[batch, dim]`.
    fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dim = x.dim(D::Minus1)?;
        let m = self.linear.forward(&cond.silu()?)?;
        let chunk = |i: usize| m.narrow(D::Minus1, i * dim, dim);
        let (shift_msa, scale_msa) = (chunk(0)?, chunk(1)?);
        let x = layer_norm_no_affine(x, LN_EPS)?
            .broadcast_mul(&(scale_msa + 1.0)?.unsqueeze(1)?)?
            .broadcast_add(&shift_msa.unsqueeze(1)?)?;
        Ok((x, chunk(2)?, chunk(3)?, chunk(4)?, chunk(5)?))
    }
}

/// Final AdaLN (scale/shift only).
#[derive(Debug)]
struct AdaLayerNormZeroFinal {
    linear: Linear,
}

impl AdaLayerNormZeroFinal {
    fn new(dim: usize, cond_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            linear: linear(cond_dim, dim * 2, vb.pp("linear"))?,
        })
    }

    fn forward(&self, x: &Tensor, cond: &Tensor) -> candle_core::Result<Tensor> {
        let dim = x.dim(D::Minus1)?;
        let m = self.linear.forward(&cond.silu()?)?;
        let scale = m.narrow(D::Minus1, 0, dim)?;
        let shift = m.narrow(D::Minus1, dim, dim)?;
        layer_norm_no_affine(x, LN_EPS)?
            .broadcast_mul(&(scale + 1.0)?.unsqueeze(1)?)?
            .broadcast_add(&shift.unsqueeze(1)?)
    }
}

/// `linear → GELU (tanh) → linear`.
#[derive(Debug)]
struct FeedForward {
    lin1: Linear,
    lin2: Linear,
}

impl FeedForward {
    fn new(dim: usize, mult: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            lin1: linear(dim, dim * mult, vb.pp("ff.0.0"))?,
            lin2: linear(dim * mult, dim, vb.pp("ff.1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.lin2.forward(&self.lin1.forward(x)?.gelu()?)
    }
}

/// Joint multi-head attention over `[x ‖ c]` with per-stream q/k/v
/// projections and (except in the last block) per-stream out
/// projections.
#[derive(Debug)]
struct JointAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_q_c: Linear,
    to_k_c: Linear,
    to_v_c: Linear,
    to_out: Linear,
    /// `None` in the `context_pre_only` (last) block.
    to_out_c: Option<Linear>,
    heads: usize,
}

impl JointAttention {
    fn new(
        cfg: &AcousticConverterConfig,
        context_pre_only: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let inner = cfg.heads * cfg.dim_head;
        let proj = |name: &str| linear(cfg.dim, inner, vb.pp(name));
        Ok(Self {
            to_q: proj("to_q")?,
            to_k: proj("to_k")?,
            to_v: proj("to_v")?,
            to_q_c: proj("to_q_c")?,
            to_k_c: proj("to_k_c")?,
            to_v_c: proj("to_v_c")?,
            to_out: linear(inner, cfg.dim, vb.pp("to_out.0"))?,
            to_out_c: if context_pre_only {
                None
            } else {
                Some(linear(inner, cfg.dim, vb.pp("to_out_c"))?)
            },
            heads: cfg.heads,
        })
    }

    /// `x`: `[batch, t_x, dim]`, `c`: `[batch, t_c, dim]` →
    /// (`x_attn [batch, t_x, dim]`, `c_attn [batch, t_c, dim]` unless
    /// `context_pre_only`).
    fn forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        rope: &Rope,
        c_rope: &Rope,
    ) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let (b, t_x, _) = x.dims3()?;
        let t_c = c.dim(1)?;

        // Packed-head RoPE before the head split (official quirk: only
        // the first head is rotated).
        let q = apply_rotary_pos_emb(&self.to_q.forward(x)?, rope)?;
        let k = apply_rotary_pos_emb(&self.to_k.forward(x)?, rope)?;
        let v = self.to_v.forward(x)?;
        let qc = apply_rotary_pos_emb(&self.to_q_c.forward(c)?, c_rope)?;
        let kc = apply_rotary_pos_emb(&self.to_k_c.forward(c)?, c_rope)?;
        let vc = self.to_v_c.forward(c)?;

        let q = Tensor::cat(&[q, qc], 1)?;
        let k = Tensor::cat(&[k, kc], 1)?;
        let v = Tensor::cat(&[v, vc], 1)?;

        let t = t_x + t_c;
        let heads = self.heads;
        let head_dim = q.dim(D::Minus1)? / heads;
        let split = |z: &Tensor| -> candle_core::Result<Tensor> {
            z.reshape((b, t, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(&q)?, split(&k)?, split(&v)?);

        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att = softmax_last_dim(&att)?;
        let out = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, heads * head_dim))?;

        let x_out = self.to_out.forward(&out.narrow(1, 0, t_x)?)?;
        let c_out = match &self.to_out_c {
            Some(to_out_c) => Some(to_out_c.forward(&out.narrow(1, t_x, t_c)?)?),
            None => None,
        };
        Ok((x_out, c_out))
    }
}

/// One MMDiT block: joint attention with AdaLN-Zero modulation of the
/// `x` stream and plain LayerNorms on the condition stream.
#[derive(Debug)]
struct ConverterBlock {
    attn_norm_c: LayerNorm,
    attn_norm_x: AdaLayerNormZero,
    attn: JointAttention,
    /// `None` in the last (`context_pre_only`) block.
    ff_norm_c: Option<LayerNorm>,
    ff_c: Option<FeedForward>,
    ff_x: FeedForward,
}

impl ConverterBlock {
    fn new(
        cfg: &AcousticConverterConfig,
        context_pre_only: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let ln_cfg = LayerNormConfig {
            eps: LN_EPS,
            ..Default::default()
        };
        let (ff_norm_c, ff_c) = if context_pre_only {
            (None, None)
        } else {
            (
                Some(layer_norm(cfg.dim, ln_cfg, vb.pp("ff_norm_c"))?),
                Some(FeedForward::new(cfg.dim, cfg.ff_mult, vb.pp("ff_c"))?),
            )
        };
        Ok(Self {
            attn_norm_c: layer_norm(cfg.dim, ln_cfg, vb.pp("attn_norm_c"))?,
            attn_norm_x: AdaLayerNormZero::new(cfg.dim, cfg.condition_dim, vb.pp("attn_norm_x"))?,
            attn: JointAttention::new(cfg, context_pre_only, vb.pp("attn"))?,
            ff_norm_c,
            ff_c,
            ff_x: FeedForward::new(cfg.dim, cfg.ff_mult, vb.pp("ff_x"))?,
        })
    }

    /// Returns `(c, x)`; `c` is `None` after the last block.
    fn forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        spk: &Tensor,
        rope: &Rope,
        c_rope: &Rope,
    ) -> candle_core::Result<(Option<Tensor>, Tensor)> {
        let norm_c = self.attn_norm_c.forward(c)?;
        let (norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.attn_norm_x.forward(x, spk)?;

        let (x_attn, c_attn) = self.attn.forward(&norm_x, &norm_c, rope, c_rope)?;

        let c = match (&self.ff_norm_c, &self.ff_c, c_attn) {
            (Some(ff_norm_c), Some(ff_c), Some(c_attn)) => {
                let c = (c + c_attn)?;
                Some((&c + ff_c.forward(&ff_norm_c.forward(&c)?)?)?)
            }
            _ => None,
        };

        let x = x.add(&x_attn.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;
        let norm_x = layer_norm_no_affine(&x, LN_EPS)?
            .broadcast_mul(&(scale_mlp + 1.0)?.unsqueeze(1)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?)?;
        let x = x.add(
            &self
                .ff_x
                .forward(&norm_x)?
                .broadcast_mul(&gate_mlp.unsqueeze(1)?)?,
        )?;

        Ok((c, x))
    }
}

/// The X-VC dual-conditioning converter: one non-iterative MMDiT pass
/// rewriting source codec latents toward the target speaker.
#[derive(Debug)]
pub struct AcousticConverter {
    input_embed: InputEmbedding,
    rotary_embed: RotaryEmbedding,
    blocks: Vec<ConverterBlock>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
    config: AcousticConverterConfig,
}

impl AcousticConverter {
    /// Loads the converter from `xvc_converter.safetensors` (produced by
    /// `tools/convert_xvc_generator.py`, official tensor names) with the
    /// default [`AcousticConverterConfig`].
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(
            AcousticConverterConfig::default(),
            vb.pp("acoustic_converter"),
        )
    }

    /// Builds the converter from a [`VarBuilder`] rooted at
    /// `acoustic_converter`.
    pub fn new(config: AcousticConverterConfig, vb: VarBuilder) -> Result<Self> {
        let blocks = (0..config.depth)
            .map(|i| {
                ConverterBlock::new(
                    &config,
                    i == config.depth - 1,
                    vb.pp(format!("transformer_blocks.{i}")),
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            input_embed: InputEmbedding::new(&config, vb.pp("input_embed"))?,
            rotary_embed: RotaryEmbedding::new(config.dim_head, vb.pp("rotary_embed"))?,
            blocks,
            norm_out: AdaLayerNormZeroFinal::new(
                config.dim,
                config.condition_dim,
                vb.pp("norm_out"),
            )?,
            proj_out: linear(config.dim, config.in_channels_x, vb.pp("proj_out"))?,
            config,
        })
    }

    pub fn config(&self) -> &AcousticConverterConfig {
        &self.config
    }

    /// One conversion step.
    ///
    /// * `acoustic_latent` — `[batch, in_channels_x, time]` (the prenet
    ///   output, 50 Hz);
    /// * `frame_condition` — `[batch, in_channels_c, time_c]` (target
    ///   mel; may have a different length);
    /// * `speaker_condition` — `[batch, condition_dim]`.
    ///
    /// Returns the converted latent `[batch, in_channels_x, time]`.
    pub fn forward(
        &self,
        acoustic_latent: &Tensor,
        frame_condition: &Tensor,
        speaker_condition: &Tensor,
    ) -> Result<Tensor> {
        let (x_embed, cond_embed) = self.input_embed.forward(acoustic_latent, frame_condition)?;

        let rope = self.rotary_embed.forward_from_seq_len(x_embed.dim(1)?)?;
        let c_rope = self.rotary_embed.forward_from_seq_len(cond_embed.dim(1)?)?;

        let mut x = x_embed;
        let mut c = cond_embed;
        for block in &self.blocks {
            let (new_c, new_x) = block.forward(&x, &c, speaker_condition, &rope, &c_rope)?;
            x = new_x;
            if let Some(new_c) = new_c {
                c = new_c;
            }
        }

        let x = self.norm_out.forward(&x, speaker_condition)?;
        Ok(self.proj_out.forward(&x)?.transpose(1, 2)?.contiguous()?)
    }
}

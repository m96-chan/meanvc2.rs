//! UpsampleConformerEncoder — the flow's token encoder (CosyVoice 2 §2.3).
//!
//! WeNet-style conformer stack without macaron/conv modules (so effectively
//! a transformer with ESPnet relative-position attention), preceded by a
//! 3-token pre-lookahead conv and followed by a nearest ×2 upsample
//! (25 Hz tokens → 50 Hz mel frames) plus 4 more layers:
//!
//! ```text
//! [1,T,512] ─ Linear+LN (embed, ×√512) ─ pre-lookahead(conv k4 → conv k3)
//!   ─ 6 × [LN → rel-pos MHA → LN → FFN(2048, SiLU)]
//!   ─ nearest ×2 + causal conv k5 ─ Linear+LN (up_embed)
//!   ─ 4 × layer ─ after-norm → [1,2T,512]
//! ```
//!
//! `streaming=true` applies the training-time static chunk masks
//! (25 tokens / 50 frames, full left context).

use candle_core::{Device, Tensor, D};
use candle_nn::ops::{leaky_relu, softmax};
use candle_nn::{
    conv1d, layer_norm, linear, linear_no_bias, Conv1d, LayerNorm, Linear, Module, VarBuilder,
};
use vc_core::Result;

const DIM: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = DIM / HEADS;
const FF: usize = 2048;
/// Streaming chunk size in tokens (must match training `static_chunk_size`).
pub const CHUNK_TOKENS: usize = 25;
/// Pre-lookahead length in tokens.
pub const PRE_LOOKAHEAD: usize = 3;

/// ESPnet relative positional encoding table for length `t`:
/// `[1, 2t-1, 512]`, index `j` ↦ relative position `t-1-j`.
fn rel_pos_table(t: usize, device: &Device) -> Result<Tensor> {
    let mut pe = vec![0f32; (2 * t - 1) * DIM];
    for j in 0..2 * t - 1 {
        let pos = (t as i64 - 1 - j as i64) as f64;
        for i in 0..DIM / 2 {
            let div = (-(2.0 * i as f64) * (10000f64).ln() / DIM as f64).exp();
            pe[j * DIM + 2 * i] = (pos * div).sin() as f32;
            pe[j * DIM + 2 * i + 1] = (pos * div).cos() as f32;
        }
    }
    Ok(Tensor::from_vec(pe, (1, 2 * t - 1, DIM), device)?)
}

/// Chunked additive attention bias `[1, 1, t, t]` (0 = keep, -1e10 = drop):
/// position `i` may attend to `j < (i/chunk + 1)*chunk` (full left context).
pub(crate) fn chunk_bias(t: usize, chunk: usize, device: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        let hi = (i / chunk + 1) * chunk;
        for j in hi.min(t)..t {
            m[i * t + j] = -1e10;
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, t, t), device)?)
}

struct RelPosAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPosAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_q: linear(DIM, DIM, vb.pp("linear_q"))?,
            linear_k: linear(DIM, DIM, vb.pp("linear_k"))?,
            linear_v: linear(DIM, DIM, vb.pp("linear_v"))?,
            linear_out: linear(DIM, DIM, vb.pp("linear_out"))?,
            linear_pos: linear_no_bias(DIM, DIM, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((HEADS, HEAD_DIM), "pos_bias_u")?,
            pos_bias_v: vb.get((HEADS, HEAD_DIM), "pos_bias_v")?,
        })
    }

    /// Transformer-XL rel-shift: `[b,h,t,2t-1]` → `[b,h,t,t]`.
    fn rel_shift(x: &Tensor) -> Result<Tensor> {
        let (b, h, t, w) = x.dims4()?;
        let zero = Tensor::zeros((b, h, t, 1), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&zero, x], D::Minus1)?; // [b,h,t,2t]
        let x = x.reshape((b, h, w + 1, t))?;
        let x = x.narrow(2, 1, w)?; // [b,h,2t-1,t]
        let x = x.contiguous()?.reshape((b, h, t, w))?;
        Ok(x.narrow(D::Minus1, 0, w / 2 + 1)?)
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.linear_q.forward(x)?.reshape((b, t, HEADS, HEAD_DIM))?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, HEADS, HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, HEADS, HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, HEADS, HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // [1,h,2t-1,d]

        let q_u = q
            .broadcast_add(&self.pos_bias_u.reshape((1, 1, HEADS, HEAD_DIM))?)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let q_v = q
            .broadcast_add(&self.pos_bias_v.reshape((1, 1, HEADS, HEAD_DIM))?)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        let ac = q_u.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let bd = q_v.broadcast_matmul(&p.transpose(D::Minus2, D::Minus1)?)?;
        let bd = Self::rel_shift(&bd)?;
        let mut scores = ((ac + bd)? / (HEAD_DIM as f64).sqrt())?;
        if let Some(bias) = bias {
            scores = scores.broadcast_add(bias)?;
        }
        let w = softmax(&scores, D::Minus1)?;
        let out = w.matmul(&v)?; // [b,h,t,d]
        let out = out.permute((0, 2, 1, 3))?.reshape((b, t, DIM))?;
        Ok(self.linear_out.forward(&out)?)
    }
}

struct Layer {
    attn: RelPosAttention,
    norm_mha: LayerNorm,
    w1: Linear,
    w2: Linear,
    norm_ff: LayerNorm,
}

impl Layer {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: RelPosAttention::load(vb.pp("self_attn"))?,
            norm_mha: layer_norm(DIM, 1e-12, vb.pp("norm_mha"))?,
            w1: linear(DIM, FF, vb.pp("feed_forward.w_1"))?,
            w2: linear(FF, DIM, vb.pp("feed_forward.w_2"))?,
            norm_ff: layer_norm(DIM, 1e-12, vb.pp("norm_ff"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let x = x.add(
            &self
                .attn
                .forward(&self.norm_mha.forward(x)?, pos_emb, bias)?,
        )?;
        let h = self.w1.forward(&self.norm_ff.forward(&x)?)?.silu()?;
        Ok(x.add(&self.w2.forward(&h)?)?)
    }
}

struct Embed {
    linear: Linear,
    norm: LayerNorm,
}

impl Embed {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear(DIM, DIM, vb.pp("out.0"))?,
            norm: layer_norm(DIM, 1e-5, vb.pp("out.1"))?,
        })
    }

    /// Linear + LN + ×√dim scaling (ESPnet rel-pos convention).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok((self.norm.forward(&self.linear.forward(x)?)? * (DIM as f64).sqrt())?)
    }
}

pub struct ConformerEncoder {
    embed: Embed,
    pre_conv1: Conv1d,
    pre_conv2: Conv1d,
    layers: Vec<Layer>,
    up_conv: Conv1d,
    up_embed: Embed,
    up_layers: Vec<Layer>,
    after_norm: LayerNorm,
}

impl ConformerEncoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..6 {
            layers.push(Layer::load(vb.pp(format!("encoders.{i}")))?);
        }
        let mut up_layers = Vec::new();
        for i in 0..4 {
            up_layers.push(Layer::load(vb.pp(format!("up_encoders.{i}")))?);
        }
        let pl = vb.pp("pre_lookahead_layer");
        Ok(Self {
            embed: Embed::load(vb.pp("embed"))?,
            pre_conv1: conv1d(
                DIM,
                DIM,
                PRE_LOOKAHEAD + 1,
                Default::default(),
                pl.pp("conv1"),
            )?,
            pre_conv2: conv1d(DIM, DIM, 3, Default::default(), pl.pp("conv2"))?,
            layers,
            up_conv: conv1d(DIM, DIM, 5, Default::default(), vb.pp("up_layer.conv"))?,
            up_embed: Embed::load(vb.pp("up_embed"))?,
            up_layers,
            after_norm: layer_norm(DIM, 1e-5, vb.pp("after_norm"))?,
        })
    }

    /// Pre-lookahead layer: right-pad with `context` (already embedded)
    /// or zeros, conv k4 → leaky-relu → causal conv k3 → residual.
    fn pre_lookahead(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let inputs = x.transpose(1, 2)?.contiguous()?; // [1,512,T]
        let padded = match context {
            Some(c) => {
                let c = c.transpose(1, 2)?.contiguous()?;
                Tensor::cat(&[&inputs, &c], D::Minus1)?
            }
            None => inputs.pad_with_zeros(D::Minus1, 0, PRE_LOOKAHEAD)?,
        };
        let h = leaky_relu(&self.pre_conv1.forward(&padded)?, 0.01)?;
        let h = h.pad_with_zeros(D::Minus1, 2, 0)?;
        let h = self.pre_conv2.forward(&h)?;
        Ok(h.transpose(1, 2)?.add(x)?)
    }

    /// `xs`: `[1, T, 512]` raw token embeddings (before scaling);
    /// `context`: optional `[1, 3, 512]` lookahead token embeddings.
    /// Returns `[1, 2·T, 512]`.
    pub fn forward(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        streaming: bool,
    ) -> Result<Tensor> {
        let dev = xs.device();
        let x = self.embed.forward(xs)?;
        let ctx = match context {
            Some(c) => Some(self.embed.forward(c)?),
            None => None,
        };
        let t = x.dim(1)?;
        let pos = rel_pos_table(t, dev)?;
        let bias = if streaming {
            Some(chunk_bias(t, CHUNK_TOKENS, dev)?)
        } else {
            None
        };
        let mut x = self.pre_lookahead(&x, ctx.as_ref())?;
        for l in &self.layers {
            x = l.forward(&x, &pos, bias.as_ref())?;
        }

        // nearest ×2 upsample + left-pad 4 + conv k5
        let xt = x.transpose(1, 2)?.contiguous()?; // [1,512,T]
        let (b, c, tt) = xt.dims3()?;
        let up = xt
            .unsqueeze(D::Minus1)?
            .broadcast_as((b, c, tt, 2))?
            .reshape((b, c, tt * 2))?;
        let up = up.pad_with_zeros(D::Minus1, 4, 0)?;
        let up = self.up_conv.forward(&up)?;
        let x = up.transpose(1, 2)?.contiguous()?;

        let x = self.up_embed.forward(&x)?;
        let t2 = x.dim(1)?;
        let pos2 = rel_pos_table(t2, dev)?;
        let bias2 = if streaming {
            Some(chunk_bias(t2, CHUNK_TOKENS * 2, dev)?)
        } else {
            None
        };
        let mut x = x;
        for l in &self.up_layers {
            x = l.forward(&x, &pos2, bias2.as_ref())?;
        }
        Ok(self.after_norm.forward(&x)?)
    }
}

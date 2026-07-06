//! DiT block with adaLN-Zero conditioning (Peebles & Xie, 2023), as used by
//! the MeanVC 2 decoder. Each block receives its own FRC attention mask so
//! the receptive field can expand layer by layer.

use candle_core::Tensor;
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

use super::attention::MultiHeadAttention;

/// `x * (1 + scale) + shift`, with `shift`/`scale` of shape `[batch, dim]`
/// broadcast over the time axis of `x` (`[batch, time, dim]`).
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    x.broadcast_mul(&(scale + 1.0)?.unsqueeze(1)?)?
        .broadcast_add(&shift.unsqueeze(1)?)
}

/// Position-wise feed-forward network (GELU, 4x expansion).
#[derive(Debug)]
struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(dim, dim * 4, vb.pp("fc1"))?,
            fc2: linear(dim * 4, dim, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

/// A single DiT block: adaLN-Zero-modulated self-attention + MLP.
#[derive(Debug)]
pub struct DitBlock {
    norm1: LayerNorm,
    attn: MultiHeadAttention,
    norm2: LayerNorm,
    mlp: FeedForward,
    ada_ln: Linear,
}

impl DitBlock {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        cond_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let ln_cfg = LayerNormConfig {
            affine: false,
            ..Default::default()
        };
        Ok(Self {
            norm1: layer_norm(hidden_dim, ln_cfg, vb.pp("norm1"))?,
            attn: MultiHeadAttention::new(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                num_heads,
                vb.pp("attn"),
            )?,
            norm2: layer_norm(hidden_dim, ln_cfg, vb.pp("norm2"))?,
            mlp: FeedForward::new(hidden_dim, vb.pp("mlp"))?,
            ada_ln: linear(cond_dim, 6 * hidden_dim, vb.pp("ada_ln"))?,
        })
    }

    /// `x`: `[batch, time, hidden]`, `cond`: `[batch, cond_dim]`,
    /// `mask`: this layer's FRC mask (`[time, time]`, additive).
    pub fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let params = self.ada_ln.forward(&cond.silu()?)?;
        let chunks = params.chunk(6, 1)?;
        let (shift_msa, scale_msa, gate_msa) = (&chunks[0], &chunks[1], &chunks[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&chunks[3], &chunks[4], &chunks[5]);

        let h = modulate(&self.norm1.forward(x)?, shift_msa, scale_msa)?;
        let h = self.attn.forward(&h, &h, mask)?;
        let x = x.broadcast_add(&h.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;

        let h = modulate(&self.norm2.forward(&x)?, shift_mlp, scale_mlp)?;
        let h = self.mlp.forward(&h)?;
        x.broadcast_add(&h.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)
    }
}

/// Final adaLN-modulated projection from the hidden space to mel bins.
#[derive(Debug)]
pub struct FinalLayer {
    norm: LayerNorm,
    ada_ln: Linear,
    proj: Linear,
}

impl FinalLayer {
    pub fn new(
        hidden_dim: usize,
        cond_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let ln_cfg = LayerNormConfig {
            affine: false,
            ..Default::default()
        };
        Ok(Self {
            norm: layer_norm(hidden_dim, ln_cfg, vb.pp("norm"))?,
            ada_ln: linear(cond_dim, 2 * hidden_dim, vb.pp("ada_ln"))?,
            proj: linear(hidden_dim, out_dim, vb.pp("proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, cond: &Tensor) -> candle_core::Result<Tensor> {
        let params = self.ada_ln.forward(&cond.silu()?)?;
        let chunks = params.chunk(2, 1)?;
        let h = modulate(&self.norm.forward(x)?, &chunks[0], &chunks[1])?;
        self.proj.forward(&h)
    }
}

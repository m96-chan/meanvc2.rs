//! Multi-reference timbre encoder (MRTE) — v1's timbre conditioning.
//!
//! BNFs query the reference mel-spectrogram via cross-attention; the keys
//! are the reference mel frames concatenated with a projected global voice
//! print (`vp_proj`). Post-norm residual layers, ReLU feed-forward —
//! matching `prompt_vp.py::MRTE` (`prompt_vp_encoders.{i}.self_attn.*`).

use candle_core::{Tensor, D};
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

use super::MeanVc1Config;

/// Cross-attention with distinct q/k/v input dims (`MultiHeadedAttention`).
#[derive(Debug)]
struct CrossAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl CrossAttention {
    fn new(
        q_in: usize,
        k_in: usize,
        v_in: usize,
        n_feat: usize,
        heads: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            linear_q: linear(q_in, n_feat, vb.pp("linear_q"))?,
            linear_k: linear(k_in, n_feat, vb.pp("linear_k"))?,
            linear_v: linear(v_in, n_feat, vb.pp("linear_v"))?,
            linear_out: linear(n_feat, n_feat, vb.pp("linear_out"))?,
            heads,
            head_dim: n_feat / heads,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
        let (b, tq, _) = q.dims3()?;
        let tk = k.dim(1)?;
        let split = |x: Tensor, t: usize| -> candle_core::Result<Tensor> {
            x.reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.linear_q.forward(q)?, tq)?;
        let k = split(self.linear_k.forward(k)?, tk)?;
        let v = split(self.linear_v.forward(v)?, tk)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, tq, self.heads * self.head_dim))?;
        self.linear_out.forward(&out)
    }
}

/// One MRTE layer: post-norm cross-attention + post-norm ReLU FFN.
#[derive(Debug)]
struct MrteLayer {
    attn: CrossAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ffn_in: Linear,
    ffn_out: Linear,
}

impl MrteLayer {
    fn new(cfg: &MeanVc1Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let n = cfg.bn_dim;
        let ln = LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        Ok(Self {
            attn: CrossAttention::new(
                n,
                cfg.n_mels + n,
                cfg.n_mels,
                n,
                cfg.mrte_heads,
                vb.pp("self_attn"),
            )?,
            norm1: layer_norm(n, ln, vb.pp("norm1"))?,
            norm2: layer_norm(n, ln, vb.pp("norm2"))?,
            ffn_in: linear(n, n, vb.pp("ffn.w_1"))?,
            ffn_out: linear(n, n, vb.pp("ffn.w_2"))?,
        })
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.norm1.forward(&(q + self.attn.forward(q, k, v)?)?)?;
        let h = self.ffn_out.forward(&self.ffn_in.forward(&x)?.relu()?)?;
        self.norm2.forward(&(&x + h)?)
    }
}

/// The multi-reference timbre encoder.
#[derive(Debug)]
pub struct Mrte {
    vp_proj: Linear,
    layers: Vec<MrteLayer>,
}

impl Mrte {
    pub fn new(cfg: &MeanVc1Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let layers = (0..cfg.mrte_blocks)
            .map(|i| MrteLayer::new(cfg, vb.pp(format!("prompt_vp_encoders.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            vp_proj: linear_no_bias(cfg.bn_dim, cfg.bn_dim, vb.pp("vp_proj"))?,
            layers,
        })
    }

    /// `cond` (BNFs): `[b, t, bn_dim]`, `prompts` (reference mel):
    /// `[b, t_ref, n_mels]`, `spks` (voice print): `[b, bn_dim]` →
    /// timbre-aware BNFs `[b, t, bn_dim]`.
    pub fn forward(&self, cond: &Tensor, prompts: &Tensor, spks: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t_ref, _) = prompts.dims3()?;
        let ge = self
            .vp_proj
            .forward(spks)?
            .unsqueeze(1)?
            .expand((b, t_ref, spks.dim(1)?))?;
        let key = Tensor::cat(&[prompts.clone(), ge], 2)?;
        let mut q = cond.clone();
        for layer in &self.layers {
            q = layer.forward(&q, &key, prompts)?;
        }
        Ok(q)
    }
}

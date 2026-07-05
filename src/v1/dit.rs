//! v1 ChunkDiT building blocks: timestep embedding, rotary positions,
//! rms-qk-norm attention, and the adaLN-Zero chunk block — matching the
//! official `modules.py` parameter tree.

use candle_core::{DType, Device, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

use super::MeanVc1Config;

/// Sinusoidal timestep embedding + 2-layer MLP (`time_mlp.{0,2}`).
///
/// Note the official `SinusPositionEmbedding` divides by `half_dim - 1`
/// and scales the timestep by 1000.
#[derive(Debug)]
pub struct TimestepEmbedding {
    mlp_in: Linear,
    mlp_out: Linear,
    freq_dim: usize,
}

impl TimestepEmbedding {
    pub fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let freq_dim = 256;
        Ok(Self {
            mlp_in: linear(freq_dim, dim, vb.pp("time_mlp.0"))?,
            mlp_out: linear(dim, dim, vb.pp("time_mlp.2"))?,
            freq_dim,
        })
    }

    /// `t`: `[batch]` -> `[batch, dim]`.
    pub fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let half = self.freq_dim / 2;
        let log_max = 10_000f32.ln();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-log_max * i as f32 / (half - 1) as f32).exp())
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
        let args = t
            .to_dtype(DType::F32)?
            .reshape(((), 1))?
            .broadcast_mul(&(freqs * 1_000.0)?)?;
        let emb = Tensor::cat(&[args.sin()?, args.cos()?], 1)?;
        self.mlp_out.forward(&self.mlp_in.forward(&emb)?.silu()?)
    }
}

/// Rotary position embedding (x_transformers convention: interleaved
/// pairs, no xpos scaling).
#[derive(Debug)]
pub struct RotaryEmbedding {
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    /// `(cos, sin)` tables of shape `[len, head_dim]` for the given
    /// absolute positions.
    pub fn freqs(&self, positions: &[usize], device: &Device) -> candle_core::Result<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / 10_000f32.powf(2.0 * i as f32 / self.head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(positions.len() * self.head_dim);
        let mut sin = Vec::with_capacity(positions.len() * self.head_dim);
        for &p in positions {
            for &f in &inv {
                let theta = p as f32 * f;
                // Interleaved duplication: [θ0, θ0, θ1, θ1, ...].
                cos.push(theta.cos());
                cos.push(theta.cos());
                sin.push(theta.sin());
                sin.push(theta.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (positions.len(), self.head_dim), device)?,
            Tensor::from_vec(sin, (positions.len(), self.head_dim), device)?,
        ))
    }

    /// Applies rotary embedding to `x` `[b, h, t, head_dim]`.
    pub fn apply(&self, x: &Tensor, rope: &(Tensor, Tensor)) -> candle_core::Result<Tensor> {
        let (b, h, t, d) = x.dims4()?;
        // rotate_half over interleaved pairs: (x0, x1) -> (-x1, x0).
        let pairs = x.reshape((b, h, t, d / 2, 2))?;
        let x0 = pairs.narrow(4, 0, 1)?;
        let x1 = pairs.narrow(4, 1, 1)?;
        let rotated = Tensor::cat(&[x1.neg()?, x0], 4)?.reshape((b, h, t, d))?;
        x.broadcast_mul(&rope.0)?
            .broadcast_add(&rotated.broadcast_mul(&rope.1)?)
    }
}

/// RMS norm over the head dimension (`torch.nn.RMSNorm(dim_head, 1e-6)`).
#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
}

impl RmsNorm {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get_with_hints((dim,), "weight", candle_nn::Init::Const(1.0))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let ms = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        x.broadcast_div(&(ms + 1e-6)?.sqrt()?)?
            .broadcast_mul(&self.weight)
    }
}

/// Self-attention with rms qk-norm and rotary positions
/// (`Attention` + `ChunkAttnProcessor`).
#[derive(Debug)]
struct ChunkAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: RotaryEmbedding,
    heads: usize,
    head_dim: usize,
}

impl ChunkAttention {
    fn new(cfg: &MeanVc1Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let inner = cfg.heads * cfg.head_dim;
        Ok(Self {
            to_q: linear(cfg.dim, inner, vb.pp("to_q"))?,
            to_k: linear(cfg.dim, inner, vb.pp("to_k"))?,
            to_v: linear(cfg.dim, inner, vb.pp("to_v"))?,
            to_out: linear(inner, cfg.dim, vb.pp("to_out.0"))?,
            q_norm: RmsNorm::new(cfg.head_dim, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(cfg.head_dim, vb.pp("k_norm"))?,
            rotary: RotaryEmbedding::new(cfg.head_dim),
            heads: cfg.heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &(Tensor, Tensor),
    ) -> candle_core::Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let split = |x: Tensor| -> candle_core::Result<Tensor> {
            x.reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = self.q_norm.forward(&split(self.to_q.forward(x)?)?)?;
        let k = self.k_norm.forward(&split(self.to_k.forward(x)?)?)?;
        let v = split(self.to_v.forward(x)?)?;
        let q = self.rotary.apply(&q, rope)?;
        let k = self.rotary.apply(&k, rope)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

/// adaLN-Zero DiT block with chunked attention (`ChunkDiTBlock`).
#[derive(Debug)]
pub struct ChunkDiTBlock {
    ada_ln: Linear,
    attn_norm: LayerNorm,
    attn: ChunkAttention,
    ff_norm: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl ChunkDiTBlock {
    pub fn new(cfg: &MeanVc1Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let ln = LayerNormConfig {
            affine: false,
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            ada_ln: linear(cfg.dim, cfg.dim * 6, vb.pp("attn_norm.linear"))?,
            attn_norm: layer_norm(cfg.dim, ln, vb.pp("attn_norm.norm"))?,
            attn: ChunkAttention::new(cfg, vb.pp("attn"))?,
            ff_norm: layer_norm(cfg.dim, ln, vb.pp("ff_norm"))?,
            ff_in: linear(cfg.dim, cfg.dim * cfg.ff_mult, vb.pp("ff.ff.0.0"))?,
            ff_out: linear(cfg.dim * cfg.ff_mult, cfg.dim, vb.pp("ff.ff.2"))?,
        })
    }

    /// `x`: `[b, seq, dim]`, `time`: `[b, dim]`, `mask`: additive
    /// `[seq, seq]`, `rope`: `(cos, sin)` `[seq, head_dim]`.
    pub fn forward(
        &self,
        x: &Tensor,
        time: &Tensor,
        mask: Option<&Tensor>,
        rope: &(Tensor, Tensor),
    ) -> candle_core::Result<Tensor> {
        let params = self.ada_ln.forward(&time.silu()?)?;
        let c = params.chunk(6, 1)?;
        let (shift_msa, scale_msa, gate_msa) = (&c[0], &c[1], &c[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&c[3], &c[4], &c[5]);

        let modulate = |x: &Tensor, shift: &Tensor, scale: &Tensor| -> candle_core::Result<Tensor> {
            x.broadcast_mul(&(scale + 1.0)?.unsqueeze(1)?)?
                .broadcast_add(&shift.unsqueeze(1)?)
        };
        let h = modulate(&self.attn_norm.forward(x)?, shift_msa, scale_msa)?;
        let h = self.attn.forward(&h, mask, rope)?;
        let x = x.broadcast_add(&h.broadcast_mul(&gate_msa.unsqueeze(1)?)?)?;

        let h = modulate(&self.ff_norm.forward(&x)?, shift_mlp, scale_mlp)?;
        // GELU (tanh approximation), as in the official FeedForward.
        let h = self.ff_out.forward(&self.ff_in.forward(&h)?.gelu()?)?;
        x.broadcast_add(&h.broadcast_mul(&gate_mlp.unsqueeze(1)?)?)
    }
}

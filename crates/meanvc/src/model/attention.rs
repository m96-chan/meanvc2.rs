//! Multi-head attention primitives shared by the DiT decoder (self-attention
//! with FRC masks) and the UTTE (cross-attention over universal timbre
//! tokens).

use candle_core::{Tensor, D};
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

/// Multi-head attention with separate query and key/value inputs.
///
/// Used as self-attention when `q_input == kv_input` and as cross-attention
/// otherwise (UTTE queries the 32 universal timbre tokens with BNFs).
#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(
        q_dim: usize,
        kv_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        assert_eq!(hidden_dim % num_heads, 0);
        Ok(Self {
            q_proj: linear(q_dim, hidden_dim, vb.pp("q_proj"))?,
            k_proj: linear(kv_dim, hidden_dim, vb.pp("k_proj"))?,
            v_proj: linear(kv_dim, hidden_dim, vb.pp("v_proj"))?,
            out_proj: linear(hidden_dim, hidden_dim, vb.pp("out_proj"))?,
            num_heads,
            head_dim: hidden_dim / num_heads,
        })
    }

    /// `queries`: `[batch, t_q, q_dim]`, `keys_values`: `[batch, t_kv, kv_dim]`,
    /// `mask`: optional additive mask broadcastable to `[batch, heads, t_q, t_kv]`
    /// (e.g. an FRC mask of shape `[t_q, t_kv]`).
    pub fn forward(
        &self,
        queries: &Tensor,
        keys_values: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        self.forward_kv(queries, keys_values, keys_values, mask)
    }

    /// Like [`Self::forward`] but with distinct key and value inputs, as
    /// required by the UTTE where keys and values are separate sets of
    /// universal timbre tokens.
    pub fn forward_kv(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, t_q, _) = queries.dims3()?;
        let t_kv = keys.dim(1)?;

        let split = |x: Tensor, t: usize| -> candle_core::Result<Tensor> {
            // [b, t, h*d] -> [b, h, t, d]
            x.reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.q_proj.forward(queries)?, t_q)?;
        let k = split(self.k_proj.forward(keys)?, t_kv)?;
        let v = split(self.v_proj.forward(values)?, t_kv)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut logits = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = mask {
            logits = logits.broadcast_add(mask)?;
        }
        // `ops::softmax` (composed of basic ops) rather than the fused
        // `softmax_last_dim`: the latter is registered with no backward op,
        // which would silently sever both gradients and forward-mode
        // tangents through the attention weights.
        let weights = softmax(&logits, D::Minus1)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t_q, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

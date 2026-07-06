//! Universal timbre token encoder (UTTE).
//!
//! UTTE decouples fine-grained timbre extraction from direct reliance on
//! reference mel-spectrograms. The global speaker embedding `s` is mapped by
//! two independent two-layer MLPs into `K` key/value pairs, fused with
//! learnable speaker-agnostic priors:
//!
//! ```text
//! k_i = MLP_k(s)_i + tanh(k_i_prior)
//! v_i = MLP_v(s)_i + tanh(v_i_prior)
//! ```
//!
//! Bottleneck features (BNFs) then act as queries in a cross-attention over
//! the resulting universal timbre tokens, retrieving pronunciation-aware
//! timbre cues and producing *timbre-aware BNFs* for the decoder.

use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::attention::MultiHeadAttention;
use crate::config::UtteConfig;

/// Two-layer MLP mapping the speaker embedding to `K` tokens of size
/// `hidden_dim`.
#[derive(Debug)]
struct TokenMlp {
    fc1: Linear,
    fc2: Linear,
    num_tokens: usize,
    hidden_dim: usize,
}

impl TokenMlp {
    fn new(cfg: &UtteConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(cfg.speaker_dim, cfg.hidden_dim, vb.pp("fc1"))?,
            fc2: linear(
                cfg.hidden_dim,
                cfg.num_tokens * cfg.hidden_dim,
                vb.pp("fc2"),
            )?,
            num_tokens: cfg.num_tokens,
            hidden_dim: cfg.hidden_dim,
        })
    }

    /// `[batch, speaker_dim]` -> `[batch, num_tokens, hidden_dim]`.
    fn forward(&self, speaker: &Tensor) -> candle_core::Result<Tensor> {
        let b = speaker.dim(0)?;
        self.fc2
            .forward(&self.fc1.forward(speaker)?.silu()?)?
            .reshape((b, self.num_tokens, self.hidden_dim))
    }
}

/// The universal timbre token encoder.
#[derive(Debug)]
pub struct Utte {
    mlp_k: TokenMlp,
    mlp_v: TokenMlp,
    k_prior: Tensor,
    v_prior: Tensor,
    q_in: Linear,
    cross_attn: MultiHeadAttention,
    /// Applying `tanh` to the priors before additive fusion empirically
    /// improves token diversity (paper, Sec. 3.3); disabled in the
    /// "w/o tanh" ablation.
    use_tanh_prior: bool,
}

impl Utte {
    pub fn new(cfg: &UtteConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            mlp_k: TokenMlp::new(cfg, vb.pp("mlp_k"))?,
            mlp_v: TokenMlp::new(cfg, vb.pp("mlp_v"))?,
            k_prior: vb.get_with_hints(
                (cfg.num_tokens, cfg.hidden_dim),
                "k_prior",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?,
            v_prior: vb.get_with_hints(
                (cfg.num_tokens, cfg.hidden_dim),
                "v_prior",
                candle_nn::Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            )?,
            q_in: linear(cfg.bnf_dim, cfg.hidden_dim, vb.pp("q_in"))?,
            cross_attn: MultiHeadAttention::new(
                cfg.hidden_dim,
                cfg.hidden_dim,
                cfg.hidden_dim,
                cfg.num_heads,
                vb.pp("cross_attn"),
            )?,
            use_tanh_prior: true,
        })
    }

    /// Builds the universal timbre tokens for a batch of speaker embeddings.
    ///
    /// `speaker`: `[batch, speaker_dim]` ->
    /// `(keys, values)`: each `[batch, num_tokens, hidden_dim]`.
    pub fn timbre_tokens(&self, speaker: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let prior = |p: &Tensor| -> candle_core::Result<Tensor> {
            if self.use_tanh_prior {
                p.tanh()
            } else {
                Ok(p.clone())
            }
        };
        let k = self.mlp_k.forward(speaker)?.broadcast_add(&prior(&self.k_prior)?)?;
        let v = self.mlp_v.forward(speaker)?.broadcast_add(&prior(&self.v_prior)?)?;
        Ok((k, v))
    }

    /// Produces timbre-aware BNFs.
    ///
    /// `bnf`: `[batch, time, bnf_dim]`, `speaker`: `[batch, speaker_dim]` ->
    /// `[batch, time, hidden_dim]` (residual connection with the projected
    /// BNF queries preserves content information).
    pub fn forward(&self, bnf: &Tensor, speaker: &Tensor) -> candle_core::Result<Tensor> {
        let (keys, values) = self.timbre_tokens(speaker)?;
        let queries = self.q_in.forward(bnf)?;
        let cues = self.cross_attn.forward_kv(&queries, &keys, &values, None)?;
        queries + cues
    }
}

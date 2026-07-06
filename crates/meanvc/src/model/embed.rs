//! Conditioning embeddings for the DiT decoder.
//!
//! Mean flows condition the network on *two* timesteps `(r, t)` — the
//! endpoints of the interval over which the average velocity is predicted —
//! instead of the single timestep of standard diffusion/flow models. Both
//! are embedded with sinusoidal features and fused by a small MLP, and the
//! global speaker embedding is projected and added to form the final
//! conditioning vector used by adaLN modulation.

use candle_core::{DType, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Dimension of the raw sinusoidal features per timestep.
const FREQ_DIM: usize = 256;

/// Sinusoidal embedding of a scalar timestep in `[0, 1]`.
///
/// `t` has shape `[batch]`; the result has shape `[batch, dim]`.
pub fn sinusoidal_embedding(t: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    let device = t.device();
    let half = dim / 2;
    let log_max_period = 10_000f32.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-log_max_period * i as f32 / half as f32).exp())
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), device)?;
    // Scale to the usual diffusion timestep range before embedding.
    let args = t
        .to_dtype(DType::F32)?
        .reshape(((), 1))?
        .broadcast_mul(&(freqs * 1_000.0)?)?;
    Tensor::cat(&[args.cos()?, args.sin()?], 1)
}

/// Fuses `(r, t)` timestep embeddings and the speaker embedding into a
/// single conditioning vector for adaLN.
#[derive(Debug)]
pub struct ConditionEmbedder {
    time_in: Linear,
    time_out: Linear,
    spk_proj: Linear,
}

impl ConditionEmbedder {
    pub fn new(speaker_dim: usize, cond_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            time_in: linear(2 * FREQ_DIM, cond_dim, vb.pp("time_in"))?,
            time_out: linear(cond_dim, cond_dim, vb.pp("time_out"))?,
            spk_proj: linear(speaker_dim, cond_dim, vb.pp("spk_proj"))?,
        })
    }

    /// `r`, `t`: `[batch]` timesteps; `speaker`: `[batch, speaker_dim]`.
    /// Returns `[batch, cond_dim]`.
    pub fn forward(&self, r: &Tensor, t: &Tensor, speaker: &Tensor) -> candle_core::Result<Tensor> {
        let r_emb = sinusoidal_embedding(r, FREQ_DIM)?;
        let t_emb = sinusoidal_embedding(t, FREQ_DIM)?;
        let time = Tensor::cat(&[t_emb, r_emb], 1)?;
        let time = self.time_out.forward(&self.time_in.forward(&time)?.silu()?)?;
        let spk = self.spk_proj.forward(speaker)?;
        time + spk
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn sinusoidal_shape() {
        let t = Tensor::from_vec(vec![0f32, 0.5, 1.0], (3,), &Device::Cpu).unwrap();
        let e = sinusoidal_embedding(&t, 256).unwrap();
        assert_eq!(e.dims(), &[3, 256]);
    }
}

//! Causal masked-diff flow: tokens → mel (CosyVoice 2 §2.3–2.4,
//! `CausalMaskedDiffWithXvec` + `CausalConditionalCFM`).
//!
//! Token embeddings (vocab 6561 → 512) run through the
//! [`ConformerEncoder`](crate::encoder::ConformerEncoder), project to 80-d
//! `mu`, and a 10-step Euler CFM solve with classifier-free guidance 0.7
//! turns fixed noise into mel. Determinism: the official model draws its
//! ODE start noise **once at construction with torch seed 0** — that exact
//! tensor ships in the checkpoint as `rand_noise`, so outputs are
//! bit-comparable with the official implementation.
//!
//! Prompt conditioning: prompt tokens are concatenated before the source
//! tokens, the prompt's real mel is placed in the `cond` channel, and the
//! prompt span is cut from the output.

use candle_core::{DType, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use vc_core::Result;

use crate::encoder::{ConformerEncoder, PRE_LOOKAHEAD};
use crate::unet::Estimator;

pub const VOCAB: usize = 6561;
const DIM: usize = 512;
const N_MEL: usize = 80;
const N_TIMESTEPS: usize = 10;
const CFG_RATE: f64 = 0.7;

pub struct Flow {
    input_embedding: Embedding,
    spk_affine: Linear,
    encoder: ConformerEncoder,
    encoder_proj: Linear,
    estimator: Estimator,
    rand_noise: Tensor, // [1, 80, 15000]
}

impl Flow {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_embedding: embedding(VOCAB, DIM, vb.pp("input_embedding"))?,
            spk_affine: linear(192, N_MEL, vb.pp("spk_embed_affine_layer"))?,
            encoder: ConformerEncoder::load(vb.pp("encoder"))?,
            encoder_proj: linear(DIM, N_MEL, vb.pp("encoder_proj"))?,
            estimator: Estimator::load(vb.pp("decoder.estimator"))?,
            rand_noise: vb.get((1, N_MEL, 50 * 300), "rand_noise")?,
        })
    }

    /// L2-normalize + project the 192-d x-vector to the 80-d conditioning.
    fn project_spk(&self, embedding: &Tensor) -> Result<Tensor> {
        let norm = embedding.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let e = embedding.broadcast_div(&norm)?;
        Ok(self.spk_affine.forward(&e)?)
    }

    /// Encoder half: `tokens` `[1, T]` (u32, prompt ⧺ source) → `mu`
    /// `[1, 2T, 80]`. `finalize=false` treats the trailing
    /// [`PRE_LOOKAHEAD`] tokens as lookahead context (streaming chunk).
    pub fn mu(&self, tokens: &Tensor, streaming: bool, finalize: bool) -> Result<Tensor> {
        let emb = self
            .input_embedding
            .forward(&tokens.to_dtype(DType::U32)?)?;
        let h = if finalize {
            self.encoder.forward(&emb, None, streaming)?
        } else {
            let t = emb.dim(1)?;
            let main = emb.narrow(1, 0, t - PRE_LOOKAHEAD)?;
            let ctx = emb.narrow(1, t - PRE_LOOKAHEAD, PRE_LOOKAHEAD)?;
            self.encoder.forward(&main, Some(&ctx), streaming)?
        };
        Ok(self.encoder_proj.forward(&h)?)
    }

    /// 10-step Euler CFM solve with CFG (batch-2 estimator calls).
    ///
    /// `mu`: `[1, T2, 80]`; `embedding`: raw `[1, 192]` x-vector;
    /// `prompt_feat`: `[1, P, 80]` (P ≤ T2). Returns mel `[1, 80, T2 - P]`
    /// (the prompt span removed).
    pub fn cfm(
        &self,
        mu: &Tensor,
        embedding: &Tensor,
        prompt_feat: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let dev = mu.device();
        let t2 = mu.dim(1)?;
        let p = prompt_feat.dim(1)?;
        let spks = self.project_spk(embedding)?; // [1, 80]

        let mu = mu.transpose(1, 2)?.contiguous()?; // [1, 80, T2]
        let cond = Tensor::cat(
            &[
                &prompt_feat.transpose(1, 2)?.contiguous()?,
                &Tensor::zeros((1, N_MEL, t2 - p), DType::F32, dev)?,
            ],
            D::Minus1,
        )?;

        let mut x = self.rand_noise.narrow(D::Minus1, 0, t2)?.to_device(dev)?;
        // cosine t-schedule: t = 1 - cos(π/2 · linspace(0, 1, 11))
        let ts: Vec<f64> = (0..=N_TIMESTEPS)
            .map(|i| {
                let u = i as f64 / N_TIMESTEPS as f64;
                1.0 - (u * std::f64::consts::FRAC_PI_2).cos()
            })
            .collect();

        let zero_mu = mu.zeros_like()?;
        let zero_spk = spks.zeros_like()?;
        let zero_cond = cond.zeros_like()?;
        let mu2 = Tensor::cat(&[&mu, &zero_mu], 0)?;
        let spk2 = Tensor::cat(&[&spks, &zero_spk], 0)?;
        let cond2 = Tensor::cat(&[&cond, &zero_cond], 0)?;

        for step in 0..N_TIMESTEPS {
            let t = ts[step];
            let dt = ts[step + 1] - t;
            let x2 = Tensor::cat(&[&x, &x], 0)?;
            let tt = Tensor::from_vec(vec![t as f32, t as f32], 2, dev)?;
            let d = self
                .estimator
                .forward(&x2, &mu2, &tt, &spk2, &cond2, streaming)?;
            let d_cond = d.narrow(0, 0, 1)?;
            let d_uncond = d.narrow(0, 1, 1)?;
            let dphi = ((d_cond * (1.0 + CFG_RATE))? - (d_uncond * CFG_RATE)?)?;
            x = (x + (dphi * dt)?)?;
        }
        Ok(x.narrow(D::Minus1, p, t2 - p)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn ckpt(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt")
            .join(name);
        p.exists().then_some(p)
    }

    fn fixture() -> Option<HashMap<String, Tensor>> {
        candle_core::safetensors::load(ckpt("cosyvoice_e2e_fixture.safetensors")?, &Device::Cpu)
            .ok()
    }

    fn load_flow() -> Option<Flow> {
        let w = ckpt("cosyvoice_flow.safetensors")?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &Device::Cpu).unwrap() };
        Some(Flow::load(vb).unwrap())
    }

    fn stats(a: &Tensor, b: &Tensor) -> (f32, f32) {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        (dot / (na * nb), max)
    }

    #[test]
    fn mu_matches_official() {
        let (Some(fx), Some(flow)) = (fixture(), load_flow()) else {
            return;
        };
        let tokens = Tensor::cat(&[&fx["prompt_tokens"], &fx["source_tokens"]], 1).unwrap();
        let mu = flow.mu(&tokens, false, true).unwrap();
        let (cos, max_d) = stats(&mu, &fx["flow_mu"]);
        assert!(
            cos > 0.9999 && max_d < 2e-2,
            "mu: cos {cos}, max abs diff {max_d}"
        );
    }

    #[test]
    fn cfm_mel_matches_official() {
        let (Some(fx), Some(flow)) = (fixture(), load_flow()) else {
            return;
        };
        let tokens = Tensor::cat(&[&fx["prompt_tokens"], &fx["source_tokens"]], 1).unwrap();
        let mu = flow.mu(&tokens, false, true).unwrap();
        let mel = flow
            .cfm(&mu, &fx["embedding"], &fx["prompt_feat"], false)
            .unwrap();
        let (cos, max_d) = stats(&mel, &fx["cfm_mel"]);
        assert!(
            cos > 0.999 && max_d < 5e-2,
            "cfm mel: cos {cos}, max abs diff {max_d}"
        );
    }

    #[test]
    fn streaming_chunk_matches_official() {
        let (Some(fx), Some(flow)) = (fixture(), load_flow()) else {
            return;
        };
        let n0 = 25 + PRE_LOOKAHEAD;
        let src0 = fx["source_tokens"].narrow(1, 0, n0).unwrap();
        let tokens = Tensor::cat(&[&fx["prompt_tokens"], &src0], 1).unwrap();
        let mu = flow.mu(&tokens, true, false).unwrap();
        let mel = flow
            .cfm(&mu, &fx["embedding"], &fx["prompt_feat"], true)
            .unwrap();
        let (cos, max_d) = stats(&mel, &fx["stream_mel_chunk0"]);
        assert!(
            cos > 0.999 && max_d < 5e-2,
            "stream chunk: cos {cos}, max abs diff {max_d}"
        );
    }
}

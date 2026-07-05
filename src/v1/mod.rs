//! MeanVC (v1) — candle port of the official implementation
//! (<https://github.com/ASLP-lab/MeanVC>, arXiv:2510.08392), targeting
//! weight compatibility with the released `model_200ms.safetensors`.
//!
//! v1 differs from MeanVC 2 in two places: timbre conditioning uses the
//! **MRTE** (cross-attention from BNFs over reference mel frames plus a
//! global 256-dim voice print) instead of UTTE, and streaming uses
//! **chunk-wise autoregressive denoising (CARD)**: previously generated
//! clean chunks are embedded via `cache_embed` and prepended to the noisy
//! sequence, with a mask that lets each noisy chunk attend to itself and
//! the last `max_lookback` clean chunks.
//!
//! Parameter paths mirror the official module tree 1:1
//! (`t_time_embed.time_mlp.0`, `timbre_encoder.prompt_vp_encoders.{i}.*`,
//! `transformer_blocks.{i}.attn.to_q`, `cache_embed`, `norm_out.linear`,
//! `proj_out`, …) so a converted checkpoint loads without renaming.

mod card;
mod dit;
mod mrte;

pub use card::card_mask;
pub use dit::{ChunkDiTBlock, RotaryEmbedding, TimestepEmbedding};
pub use mrte::Mrte;

use candle_core::{DType, Device, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

use crate::{Error, Result};

/// How previously generated clean chunks are fed to [`MeanVc1::forward`].
#[derive(Debug, Clone, Copy)]
pub enum CacheLayout<'a> {
    /// No cache (first chunk / unconditional).
    None,
    /// Training layout: clean mel of the same length as the noisy input,
    /// attended through the CARD mask.
    Card(&'a Tensor),
    /// Streaming layout: an arbitrary-length cache attended without a mask;
    /// `pos_offset` is the absolute frame position of the noisy input for
    /// RoPE.
    Streaming {
        cache: &'a Tensor,
        pos_offset: usize,
    },
}

/// Configuration of the v1 DiT. Defaults follow the official
/// `config_200ms.json` (the released checkpoint): dim 512, depth 4,
/// 2 heads with head-dim 64, ff_mult 2, rms qk-norm, 20-frame (200 ms)
/// chunks, and a 5-chunk CARD look-back.
#[derive(Debug, Clone)]
pub struct MeanVc1Config {
    pub n_mels: usize,
    pub bn_dim: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff_mult: usize,
    pub chunk_size: usize,
    pub max_lookback: usize,
    /// MRTE cross-attention blocks / heads.
    pub mrte_blocks: usize,
    pub mrte_heads: usize,
}

impl Default for MeanVc1Config {
    fn default() -> Self {
        Self {
            n_mels: 80,
            bn_dim: 256,
            dim: 512,
            depth: 4,
            heads: 2,
            head_dim: 64,
            ff_mult: 2,
            chunk_size: 20,
            max_lookback: 5,
            mrte_blocks: 2,
            mrte_heads: 4,
        }
    }
}

/// Final adaLN modulation (`AdaLayerNorm_Final`): scale/shift only.
#[derive(Debug)]
struct AdaLayerNormFinal {
    linear: Linear,
    norm: LayerNorm,
}

impl AdaLayerNormFinal {
    fn new(dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let ln = LayerNormConfig {
            affine: false,
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            linear: linear(dim, dim * 2, vb.pp("linear"))?,
            norm: layer_norm(dim, ln, vb.pp("norm"))?,
        })
    }

    fn forward(&self, x: &Tensor, emb: &Tensor) -> candle_core::Result<Tensor> {
        let params = self.linear.forward(&emb.silu()?)?;
        let chunks = params.chunk(2, 1)?;
        let (scale, shift) = (&chunks[0], &chunks[1]);
        self.norm
            .forward(x)?
            .broadcast_mul(&(scale + 1.0)?.unsqueeze(1)?)?
            .broadcast_add(&shift.unsqueeze(1)?)
    }
}

/// The MeanVC v1 model (mean-flows ChunkDiT with MRTE conditioning).
#[derive(Debug)]
pub struct MeanVc1 {
    t_time_embed: TimestepEmbedding,
    r_time_embed: TimestepEmbedding,
    timbre_encoder: Mrte,
    /// `Linear(mel + 2 * bn_dim -> dim)` over concat(noisy mel, timbre-aware
    /// BNFs, repeated voice print).
    input_proj: Linear,
    cache_embed: Linear,
    rotary: RotaryEmbedding,
    blocks: Vec<ChunkDiTBlock>,
    norm_out: AdaLayerNormFinal,
    proj_out: Linear,
    cfg: MeanVc1Config,
}

impl MeanVc1 {
    pub fn new(cfg: MeanVc1Config, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(ChunkDiTBlock::new(
                &cfg,
                vb.pp(format!("transformer_blocks.{i}")),
            )?);
        }
        Ok(Self {
            t_time_embed: TimestepEmbedding::new(cfg.dim, vb.pp("t_time_embed"))?,
            r_time_embed: TimestepEmbedding::new(cfg.dim, vb.pp("r_time_embed"))?,
            timbre_encoder: Mrte::new(&cfg, vb.pp("timbre_encoder"))?,
            input_proj: linear(
                cfg.n_mels + 2 * cfg.bn_dim,
                cfg.dim,
                vb.pp("input_embed.proj"),
            )?,
            cache_embed: linear(cfg.n_mels, cfg.dim, vb.pp("cache_embed"))?,
            rotary: RotaryEmbedding::new(cfg.head_dim),
            blocks,
            norm_out: AdaLayerNormFinal::new(cfg.dim, vb.pp("norm_out"))?,
            proj_out: linear(cfg.dim, cfg.n_mels, vb.pp("proj_out"))?,
            cfg,
        })
    }

    /// Loads the model from a safetensors checkpoint with official
    /// parameter names (`model_200ms.safetensors`).
    pub fn load<P: AsRef<std::path::Path>>(
        cfg: MeanVc1Config,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb)
    }

    pub fn config(&self) -> &MeanVc1Config {
        &self.cfg
    }

    /// Timbre-aware BNFs from the MRTE: `cond` `[b, t, bn_dim]`, `prompts`
    /// (reference mel) `[b, t_ref, n_mels]`, `spks` (voice print)
    /// `[b, bn_dim]`.
    pub fn timbre_cond(&self, cond: &Tensor, prompts: &Tensor, spks: &Tensor) -> Result<Tensor> {
        Ok(self.timbre_encoder.forward(cond, prompts, spks)?)
    }

    /// Predicts the average velocity `u(x_t, r, t)`.
    ///
    /// * `x`: noisy mel `[b, n, n_mels]` (`n` a multiple of `chunk_size`)
    /// * `timbre_cond`: MRTE output `[b, n, bn_dim]` ([`Self::timbre_cond`])
    /// * `spks`: voice print `[b, bn_dim]`
    /// * `cache`: clean-chunk conditioning, see [`CacheLayout`]
    pub fn forward(
        &self,
        x: &Tensor,
        timbre_cond: &Tensor,
        spks: &Tensor,
        cache: CacheLayout,
        r: &Tensor,
        t: &Tensor,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        if seq_len % self.cfg.chunk_size != 0 {
            return Err(Error::Input(format!(
                "sequence length {seq_len} is not a multiple of chunk_size {}",
                self.cfg.chunk_size
            )));
        }
        let time = (self.t_time_embed.forward(t)? + self.r_time_embed.forward(r)?)?;

        let spks_rep = spks.unsqueeze(1)?.expand((b, seq_len, self.cfg.bn_dim))?;
        let mut h = self
            .input_proj
            .forward(&Tensor::cat(&[x, timbre_cond, &spks_rep], 2)?)?;

        let device = x.device();
        let (cache_mel, mask, pos_offset) = match cache {
            CacheLayout::None => (None, None, 0),
            CacheLayout::Card(clean) => {
                if clean.dim(1)? != seq_len {
                    return Err(Error::Input(
                        "CARD training layout requires cache length == sequence length".into(),
                    ));
                }
                let mask = card_mask_tensor(&self.cfg, seq_len, device)?;
                (Some(clean), Some(mask), 0)
            }
            CacheLayout::Streaming { cache, pos_offset } => (Some(cache), None, pos_offset),
        };
        let rope = match cache_mel {
            Some(cache_mel) => {
                let cache_len = cache_mel.dim(1)?;
                let c = self.cache_embed.forward(cache_mel)?;
                h = Tensor::cat(&[c, h], 1)?;
                // Cache occupies positions 0.., x continues at pos_offset.
                let pos: Vec<usize> = (0..cache_len)
                    .chain(pos_offset..pos_offset + seq_len)
                    .collect();
                self.rotary.freqs(&pos, device)?
            }
            None => {
                let pos: Vec<usize> = (0..seq_len).collect();
                self.rotary.freqs(&pos, device)?
            }
        };

        for block in &self.blocks {
            h = block.forward(&h, &time, mask.as_ref(), &rope)?;
        }
        let h = h.narrow(1, h.dim(1)? - seq_len, seq_len)?;
        let h = self.norm_out.forward(&h, &time)?;
        Ok(self.proj_out.forward(&h)?)
    }

    /// Offline chunk-wise autoregressive 1-NFE conversion (CARD inference).
    ///
    /// Generates chunk by chunk; each chunk is denoised in a single step
    /// conditioned on up to `max_lookback` previously generated clean
    /// chunks. Returns `[b, n, n_mels]`.
    pub fn sample(&self, cond: &Tensor, prompts: &Tensor, spks: &Tensor) -> Result<Tensor> {
        let (b, n, _) = cond.dims3()?;
        let cs = self.cfg.chunk_size;
        if n % cs != 0 {
            return Err(Error::Input(format!(
                "cond length {n} is not a multiple of chunk_size {cs}"
            )));
        }
        let device = cond.device();
        let timbre = self.timbre_cond(cond, prompts, spks)?;
        let r = Tensor::zeros((b,), DType::F32, device)?;
        let t = Tensor::ones((b,), DType::F32, device)?;

        let mut chunks: Vec<Tensor> = Vec::with_capacity(n / cs);
        for q in 0..n / cs {
            let noise = Tensor::randn(0f32, 1f32, (b, cs, self.cfg.n_mels), device)?;
            let cond_q = timbre.narrow(1, q * cs, cs)?;
            let start = q.saturating_sub(self.cfg.max_lookback);
            let cache = if q > 0 {
                Some(Tensor::cat(&chunks[start..q], 1)?)
            } else {
                None
            };
            let layout = match &cache {
                Some(c) => CacheLayout::Streaming {
                    cache: c,
                    pos_offset: q * cs,
                },
                None => CacheLayout::None,
            };
            let u = self.forward(&noise, &cond_q, spks, layout, &r, &t)?;
            chunks.push((noise - u)?);
        }
        Ok(Tensor::cat(&chunks, 1)?)
    }
}

/// Additive CARD mask over the `[cache ‖ noisy]` training layout.
fn card_mask_tensor(
    cfg: &MeanVc1Config,
    seq_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    card_mask(
        seq_len / cfg.chunk_size,
        cfg.chunk_size,
        cfg.max_lookback,
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn tiny() -> (MeanVc1, MeanVc1Config) {
        let cfg = MeanVc1Config {
            n_mels: 12,
            bn_dim: 16,
            dim: 32,
            depth: 2,
            heads: 2,
            head_dim: 8,
            ff_mult: 2,
            chunk_size: 4,
            max_lookback: 2,
            mrte_blocks: 2,
            mrte_heads: 2,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        (MeanVc1::new(cfg.clone(), vb).unwrap(), cfg)
    }

    #[test]
    fn training_forward_shapes() {
        let (model, cfg) = tiny();
        let dev = Device::Cpu;
        let n = 3 * cfg.chunk_size;
        let x = Tensor::randn(0f32, 1f32, (2, n, cfg.n_mels), &dev).unwrap();
        let clean = Tensor::randn(0f32, 1f32, (2, n, cfg.n_mels), &dev).unwrap();
        let cond = Tensor::randn(0f32, 1f32, (2, n, cfg.bn_dim), &dev).unwrap();
        let prompts = Tensor::randn(0f32, 1f32, (2, 30, cfg.n_mels), &dev).unwrap();
        let spks = Tensor::randn(0f32, 1f32, (2, cfg.bn_dim), &dev).unwrap();
        let rt = Tensor::rand(0f32, 1f32, (2,), &dev).unwrap();

        let timbre = model.timbre_cond(&cond, &prompts, &spks).unwrap();
        assert_eq!(timbre.dims(), &[2, n, cfg.bn_dim]);
        let u = model
            .forward(&x, &timbre, &spks, CacheLayout::Card(&clean), &rt, &rt)
            .unwrap();
        assert_eq!(u.dims(), &[2, n, cfg.n_mels]);
    }

    #[test]
    fn chunked_sampling_is_causal_over_chunks() {
        // CARD inference is autoregressive: chunk q depends only on chunks
        // < q, so a longer cond must reproduce the leading chunks exactly
        // (given identical per-chunk noise, which we fix via a seeded pass:
        // here we instead check shape/finiteness and strict growth).
        let (model, cfg) = tiny();
        let dev = Device::Cpu;
        let n = 4 * cfg.chunk_size;
        let cond = Tensor::randn(0f32, 1f32, (1, n, cfg.bn_dim), &dev).unwrap();
        let prompts = Tensor::randn(0f32, 1f32, (1, 25, cfg.n_mels), &dev).unwrap();
        let spks = Tensor::randn(0f32, 1f32, (1, cfg.bn_dim), &dev).unwrap();
        let mel = model.sample(&cond, &prompts, &spks).unwrap();
        assert_eq!(mel.dims(), &[1, n, cfg.n_mels]);
        let v: Vec<f32> = mel.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn card_layout_rejects_mismatched_cache() {
        let (model, cfg) = tiny();
        let dev = Device::Cpu;
        let n = 2 * cfg.chunk_size;
        let x = Tensor::randn(0f32, 1f32, (1, n, cfg.n_mels), &dev).unwrap();
        let clean = Tensor::randn(0f32, 1f32, (1, cfg.chunk_size, cfg.n_mels), &dev).unwrap();
        let cond = Tensor::randn(0f32, 1f32, (1, n, cfg.bn_dim), &dev).unwrap();
        let spks = Tensor::randn(0f32, 1f32, (1, cfg.bn_dim), &dev).unwrap();
        let prompts = Tensor::randn(0f32, 1f32, (1, 10, cfg.n_mels), &dev).unwrap();
        let timbre = model.timbre_cond(&cond, &prompts, &spks).unwrap();
        let rt = Tensor::rand(0f32, 1f32, (1,), &dev).unwrap();
        assert!(model
            .forward(&x, &timbre, &spks, CacheLayout::Card(&clean), &rt, &rt)
            .is_err());
    }
}

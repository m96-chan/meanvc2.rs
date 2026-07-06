//! The FRC-scheduled DiT decoder.
//!
//! The decoder receives the noisy mel-spectrogram `z_t` concatenated with
//! the timbre-aware BNFs, projects them into the hidden space (`L_in` in
//! Fig. 1 of the paper), runs a stack of DiT blocks — each with its own FRC
//! attention mask — and projects back to mel bins, predicting the *average
//! velocity* `u(z_t, r, t)` of the mean-flows formulation.

use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::dit::{DitBlock, FinalLayer};
use super::embed::ConditionEmbedder;
use crate::config::DecoderConfig;
use crate::frc;

#[derive(Debug)]
pub struct DitDecoder {
    l_in: Linear,
    blocks: Vec<DitBlock>,
    final_layer: FinalLayer,
    cond_embed: ConditionEmbedder,
    cfg: DecoderConfig,
}

impl DitDecoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let l_in = linear(cfg.n_mels + cfg.bnf_dim, cfg.hidden_dim, vb.pp("l_in"))?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(DitBlock::new(
                cfg.hidden_dim,
                cfg.num_heads,
                cfg.time_embed_dim,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self {
            l_in,
            blocks,
            final_layer: FinalLayer::new(
                cfg.hidden_dim,
                cfg.time_embed_dim,
                cfg.n_mels,
                vb.pp("final"),
            )?,
            cond_embed: ConditionEmbedder::new(
                cfg.speaker_dim,
                cfg.time_embed_dim,
                vb.pp("cond"),
            )?,
            cfg: cfg.clone(),
        })
    }

    pub fn config(&self) -> &DecoderConfig {
        &self.cfg
    }

    /// Builds the per-layer FRC masks for a sequence of `seq_len` frames.
    pub fn frc_masks(
        &self,
        seq_len: usize,
        device: &candle_core::Device,
    ) -> candle_core::Result<Vec<Tensor>> {
        frc::decoder_masks(
            seq_len,
            self.cfg.chunk_frames,
            &self.cfg.past_receptive,
            &self.cfg.future_receptive,
            device,
        )
    }

    /// Predicts the average velocity `u(z_t, r, t)`.
    ///
    /// * `z_t`: noisy mel-spectrogram `[batch, time, n_mels]`
    /// * `cond_bnf`: timbre-aware BNFs `[batch, time, bnf_dim]` (already
    ///   upsampled to the mel frame rate)
    /// * `speaker`: global speaker embedding `[batch, speaker_dim]`
    /// * `r`, `t`: interval endpoints, `[batch]` each
    /// * `masks`: per-layer FRC masks from [`Self::frc_masks`], or `None`
    ///   for full (offline) attention
    pub fn forward(
        &self,
        z_t: &Tensor,
        cond_bnf: &Tensor,
        speaker: &Tensor,
        r: &Tensor,
        t: &Tensor,
        masks: Option<&[Tensor]>,
    ) -> candle_core::Result<Tensor> {
        if let Some(masks) = masks {
            assert_eq!(
                masks.len(),
                self.blocks.len(),
                "one FRC mask per DiT block is required"
            );
        }
        let cond = self.cond_embed.forward(r, t, speaker)?;
        let mut x = self.l_in.forward(&Tensor::cat(&[z_t, cond_bnf], 2)?)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &cond, masks.map(|m| &m[i]))?;
        }
        self.final_layer.forward(&x, &cond)
    }
}

//! Neural network modules: UTTE, DiT blocks, and the full MeanVC 2 model.

mod attention;
mod decoder;
mod dit;
mod embed;
mod utte;

pub use attention::MultiHeadAttention;
pub use decoder::DitDecoder;
pub use dit::{DitBlock, FinalLayer};
pub use embed::{sinusoidal_embedding, ConditionEmbedder};
pub use utte::Utte;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::MeanVc2Config;
use crate::{Error, Result};

/// The trainable part of MeanVC 2: UTTE + DiT decoder.
///
/// The streaming ASR (BNF extractor), speaker encoder, and vocoder are
/// pretrained external models — see [`crate::encoders`] for the integration
/// traits.
#[derive(Debug)]
pub struct MeanVc2 {
    pub utte: Utte,
    pub decoder: DitDecoder,
    cfg: MeanVc2Config,
}

impl MeanVc2 {
    /// Builds the model from a [`VarBuilder`] (random init via
    /// `candle_nn::VarMap`, or pretrained weights via
    /// `VarBuilder::from_mmaped_safetensors`).
    pub fn new(cfg: MeanVc2Config, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            utte: Utte::new(&cfg.utte, vb.pp("utte"))?,
            decoder: DitDecoder::new(&cfg.decoder, vb.pp("decoder"))?,
            cfg,
        })
    }

    /// Loads the model from a safetensors checkpoint.
    pub fn load<P: AsRef<std::path::Path>>(
        cfg: MeanVc2Config,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)?
        };
        Self::new(cfg, vb)
    }

    pub fn config(&self) -> &MeanVc2Config {
        &self.cfg
    }

    /// Fuses BNFs with the speaker identity into timbre-aware BNFs.
    ///
    /// `bnf`: `[batch, time, bnf_dim]` at the mel frame rate,
    /// `speaker`: `[batch, speaker_dim]`.
    pub fn timbre_aware_bnf(&self, bnf: &Tensor, speaker: &Tensor) -> Result<Tensor> {
        Ok(self.utte.forward(bnf, speaker)?)
    }

    /// Predicts the average velocity `u(z_t, r, t)` — the quantity the
    /// mean-flows objective regresses.
    pub fn forward(
        &self,
        z_t: &Tensor,
        cond_bnf: &Tensor,
        speaker: &Tensor,
        r: &Tensor,
        t: &Tensor,
        masks: Option<&[Tensor]>,
    ) -> Result<Tensor> {
        Ok(self.decoder.forward(z_t, cond_bnf, speaker, r, t, masks)?)
    }

    /// Offline (non-streaming) 1-NFE conversion of a full utterance.
    ///
    /// `bnf`: `[batch, time, bnf_dim]` at the mel frame rate,
    /// `speaker`: `[batch, speaker_dim]`. Returns the converted
    /// mel-spectrogram `[batch, time, n_mels]`.
    ///
    /// FRC masks are applied so that offline and streaming outputs match.
    pub fn convert(&self, bnf: &Tensor, speaker: &Tensor) -> Result<Tensor> {
        let (b, time, _) = bnf.dims3()?;
        if time % self.cfg.decoder.chunk_frames != 0 {
            return Err(Error::Input(format!(
                "sequence length {time} is not a multiple of chunk_frames {}",
                self.cfg.decoder.chunk_frames
            )));
        }
        let device = bnf.device();
        let cond_bnf = self.timbre_aware_bnf(bnf, speaker)?;
        let masks = self.decoder.frc_masks(time, device)?;
        let noise = Tensor::randn(0f32, 1f32, (b, time, self.cfg.decoder.n_mels), device)?;
        crate::meanflow::sample_1nfe(self, &noise, &cond_bnf, speaker, Some(&masks))
    }
}

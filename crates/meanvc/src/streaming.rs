//! Chunk-by-chunk streaming conversion.
//!
//! FRC gives the stacked decoder a bounded receptive field — with the
//! paper's defaults, 6 past chunks, the current chunk, and **1 future
//! chunk** (40 ms look-ahead). [`StreamingConverter`] exploits this: it
//! buffers incoming BNF chunks and, as soon as the look-ahead requirement is
//! met, denoises a sliding window covering the full receptive field and
//! emits the mel frames of the oldest pending chunk.
//!
//! Per-chunk noise is sampled once and cached so that overlapping windows
//! see identical noise, keeping consecutive chunk outputs consistent.

use candle_core::Tensor;

use crate::frc;
use crate::meanflow;
use crate::model::MeanVc2;
use crate::{Error, Result};

/// Streaming 1-NFE mel-spectrogram generator.
pub struct StreamingConverter<'a> {
    model: &'a MeanVc2,
    /// Global speaker embedding, `[1, speaker_dim]`.
    speaker: Tensor,
    /// Timbre-aware BNF chunks, each `[1, chunk_frames, hidden]`.
    cond_chunks: Vec<Tensor>,
    /// Cached per-chunk noise, each `[1, chunk_frames, n_mels]`.
    noise_chunks: Vec<Tensor>,
    /// Index of the next chunk to emit.
    emitted: usize,
    /// Total past/future receptive field of the decoder, in chunks.
    past: usize,
    future: usize,
}

impl<'a> StreamingConverter<'a> {
    /// Starts a new streaming session for one target speaker.
    ///
    /// `speaker` is the global speaker embedding of the reference audio,
    /// shape `[1, speaker_dim]` (or `[speaker_dim]`).
    pub fn new(model: &'a MeanVc2, speaker: &Tensor) -> Result<Self> {
        let speaker = match speaker.dims().len() {
            1 => speaker.unsqueeze(0)?,
            2 => speaker.clone(),
            _ => {
                return Err(Error::Input(
                    "speaker embedding must be [speaker_dim] or [1, speaker_dim]".into(),
                ))
            }
        };
        let cfg = &model.config().decoder;
        let (past, future) =
            frc::total_receptive_field(&cfg.past_receptive, &cfg.future_receptive);
        Ok(Self {
            model,
            speaker,
            cond_chunks: Vec::new(),
            noise_chunks: Vec::new(),
            emitted: 0,
            past,
            future,
        })
    }

    /// Look-ahead required before a chunk can be emitted, in chunks.
    pub fn lookahead_chunks(&self) -> usize {
        self.future
    }

    /// Feeds one BNF chunk (`[1, chunk_frames, bnf_dim]`, already at the mel
    /// frame rate) and returns any mel chunks that became ready, each
    /// `[1, chunk_frames, n_mels]`.
    pub fn push(&mut self, bnf_chunk: &Tensor) -> Result<Vec<Tensor>> {
        let cfg = &self.model.config().decoder;
        let bnf_dim = self.model.config().utte.bnf_dim;
        let (_, frames, dim) = bnf_chunk.dims3()?;
        if frames != cfg.chunk_frames || dim != bnf_dim {
            return Err(Error::Input(format!(
                "expected BNF chunk of shape [1, {}, {bnf_dim}], got [1, {frames}, {dim}]",
                cfg.chunk_frames
            )));
        }
        let cond = self.model.timbre_aware_bnf(bnf_chunk, &self.speaker)?;
        let noise = Tensor::randn(
            0f32,
            1f32,
            (1, cfg.chunk_frames, cfg.n_mels),
            bnf_chunk.device(),
        )?;
        self.cond_chunks.push(cond);
        self.noise_chunks.push(noise);
        self.drain(false)
    }

    /// Flushes the trailing chunks that never received full look-ahead.
    /// Call once after the last [`Self::push`].
    pub fn finish(&mut self) -> Result<Vec<Tensor>> {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> Result<Vec<Tensor>> {
        let cfg = &self.model.config().decoder;
        let mut ready = Vec::new();
        while self.emitted < self.cond_chunks.len() {
            let available_future = self.cond_chunks.len() - 1 - self.emitted;
            if !flush && available_future < self.future {
                break;
            }
            let start = self.emitted.saturating_sub(self.past);
            let end = (self.emitted + self.future).min(self.cond_chunks.len() - 1);

            let cond = Tensor::cat(&self.cond_chunks[start..=end], 1)?;
            let noise = Tensor::cat(&self.noise_chunks[start..=end], 1)?;
            let seq_len = cond.dim(1)?;
            let masks = self.model.decoder.frc_masks(seq_len, cond.device())?;
            let mel = meanflow::sample_1nfe(self.model, &noise, &cond, &self.speaker, Some(&masks))?;

            let offset = (self.emitted - start) * cfg.chunk_frames;
            ready.push(mel.narrow(1, offset, cfg.chunk_frames)?);
            self.emitted += 1;

            // Windows never reach further back than `past` chunks, so older
            // buffers can be dropped to keep memory bounded on long streams.
            let keep_from = self.emitted.saturating_sub(self.past);
            if keep_from > 64 {
                self.cond_chunks.drain(..keep_from);
                self.noise_chunks.drain(..keep_from);
                self.emitted -= keep_from;
            }
        }
        Ok(ready)
    }
}

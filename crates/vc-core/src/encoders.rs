//! Integration traits for the pretrained, frozen components of the MeanVC 2
//! pipeline.
//!
//! MeanVC 2 trains only the UTTE and the DiT decoder; the remaining pieces
//! are external pretrained models (frozen during training):
//!
//! * **Semantic encoder** — a streaming ASR bottleneck-feature extractor.
//!   The paper uses Fast-U2++ (WeNet) with an 80 ms chunk size, producing
//!   BNFs from 16 kHz waveforms at a 40 ms frame length.
//! * **Speaker encoder** — a speaker-verification embedding model. The
//!   paper uses ECAPA-TDNN (192-dim embeddings).
//! * **Vocoder** — a mel-to-waveform synthesizer. The paper uses Vocos.
//!
//! Implement these traits against your runtime of choice (e.g. ONNX Runtime
//! via `ort`, or candle ports of the respective models) to assemble the full
//! conversion pipeline.

use candle_core::Tensor;

use crate::Result;

/// Streaming ASR bottleneck-feature (BNF) extractor.
pub trait SemanticEncoder {
    /// Dimension of the produced BNFs.
    fn bnf_dim(&self) -> usize;

    /// BNF frame shift in milliseconds (40 ms for Fast-U2++).
    fn frame_shift_ms(&self) -> f32;

    /// Extracts BNFs from a mono waveform.
    ///
    /// `samples` are `sample_rate`-Hz PCM in `[-1, 1]`. Returns
    /// `[time, bnf_dim]`.
    fn extract(&self, samples: &[f32], sample_rate: usize) -> Result<Tensor>;
}

/// Global speaker-embedding extractor (e.g. ECAPA-TDNN).
pub trait SpeakerEncoder {
    /// Dimension of the speaker embedding.
    fn embedding_dim(&self) -> usize;

    /// Extracts a single utterance-level embedding, `[embedding_dim]`.
    fn embed(&self, samples: &[f32], sample_rate: usize) -> Result<Tensor>;
}

/// Mel-to-waveform vocoder (e.g. Vocos).
pub trait Vocoder {
    /// Output sample rate in Hz.
    fn sample_rate(&self) -> usize;

    /// Synthesizes a waveform from a `[time, n_mels]` mel-spectrogram.
    fn synthesize(&self, mel: &Tensor) -> Result<Vec<f32>>;
}

/// Repeats each BNF frame `factor` times along the time axis so that BNFs
/// (e.g. 40 ms frames) match the mel frame rate (e.g. 10 ms → `factor = 4`).
///
/// `bnf`: `[batch, time, bnf_dim]` -> `[batch, time * factor, bnf_dim]`.
pub fn upsample_bnf(bnf: &Tensor, factor: usize) -> Result<Tensor> {
    let (b, t, d) = bnf.dims3()?;
    Ok(bnf
        .unsqueeze(2)?
        .expand((b, t, factor, d))?
        .reshape((b, t * factor, d))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn upsample_repeats_frames() {
        let bnf = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &Device::Cpu).unwrap();
        let up = upsample_bnf(&bnf, 3).unwrap();
        assert_eq!(up.dims(), &[1, 6, 2]);
        let v: Vec<Vec<f32>> = up.squeeze(0).unwrap().to_vec2().unwrap();
        assert_eq!(v[0], v[1]);
        assert_eq!(v[1], v[2]);
        assert_eq!(v[3], vec![3., 4.]);
    }
}

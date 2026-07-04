//! # meanvc2
//!
//! Unofficial Rust implementation of **MeanVC 2: Robust Low-Latency Streaming
//! Zero-Shot Voice Conversion** ([arXiv:2606.09050]) built on
//! [`candle`](https://github.com/huggingface/candle).
//!
//! MeanVC 2 follows a recognition–synthesis framework:
//!
//! 1. A pretrained **streaming ASR** extracts bottleneck features (BNFs) from
//!    the source waveform (external, see [`encoders::SemanticEncoder`]).
//! 2. A pretrained **speaker encoder** extracts a global speaker embedding
//!    from the reference waveform (external, see [`encoders::SpeakerEncoder`]).
//! 3. The **universal timbre token encoder** ([`model::Utte`]) retrieves
//!    fine-grained timbre cues from universal timbre tokens via
//!    cross-attention, producing timbre-aware BNFs.
//! 4. A **DiT decoder** ([`model::DitDecoder`]) with **future-receptive
//!    chunking** ([`frc`]) generates the target mel-spectrogram in a
//!    streaming manner with a single function evaluation (1-NFE) thanks to
//!    the **mean flows** formulation ([`meanflow`]).
//! 5. A **vocoder** (e.g. Vocos, external, see [`encoders::Vocoder`])
//!    converts the mel-spectrogram to a waveform.
//!
//! [arXiv:2606.09050]: https://arxiv.org/abs/2606.09050

pub mod audio;
pub mod backends;
pub mod config;
pub mod encoders;
pub mod frc;
pub mod meanflow;
pub mod model;
pub mod streaming;

pub use config::{DecoderConfig, MeanVc2Config, MelConfig, UtteConfig};
pub use model::MeanVc2;
pub use streaming::StreamingConverter;

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("invalid input: {0}")]
    Input(String),
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

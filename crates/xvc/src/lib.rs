//! X-VC: zero-shot streaming voice conversion in codec space
//! (Jerrister/X-VC, arXiv:2604.12456) — candle port, weight-compatible
//! with the official checkpoints.
//!
//! See [docs/xvc.md](https://github.com/m96-chan/babiniku.rs/blob/main/docs/xvc.md)
//! for the evaluation notes and
//! [issue #30](https://github.com/m96-chan/babiniku.rs/issues/30) for the
//! port plan (Phase 1: stage-by-stage weight-compatible port on top of
//! [`vc_core`]).
//!
//! Implemented stages (each golden-tested against the official
//! implementation, see `tests/`):
//! * [`preprocess`] — volume norm / 40 Hz high-pass / Whisper 128-mel
//! * [`tokenizer`] — GLM-4-Voice tokenizer (truncated Whisper-large-v3
//!   encoder + VQ) and the 12.5→50 Hz semantic adapter
//! * [`speaker`] — frozen ERes2Net speaker encoder (Kaldi fbank-80 →
//!   192-d utterance embedding)
//! * [`codec`] — SAC acoustic codec (DAC-style encoder/decoder, FVQ)
//! * [`converter`] — 6-block MMDiT acoustic converter (one-step)
//! * [`pipeline`] — the end-to-end engine ([`XvcEngine`]): prenet fusion,
//!   offline conversion and the official chunk-streaming driver
//!   ([`XvcStream`])

pub mod codec;
pub mod converter;
pub mod pipeline;
pub mod preprocess;
pub mod speaker;
pub mod tokenizer;

pub use codec::{SacCodec, SacCodecConfig, SacEncodeOutput};
pub use converter::{AcousticConverter, AcousticConverterConfig};
pub use pipeline::{
    Reference, StageTimings, StreamConfig, StreamStep, XvcEngine, XvcPipelinedStream, XvcStream,
};
pub use speaker::SpeakerEncoder;

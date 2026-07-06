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
//! * [`speaker`] — frozen ERes2Net speaker encoder (Kaldi fbank-80 →
//!   192-d utterance embedding)
//! * [`codec`] — SAC acoustic codec (DAC-style encoder/decoder, FVQ)
//! * [`converter`] — 6-block MMDiT acoustic converter (one-step)

pub mod codec;
pub mod converter;
pub mod speaker;

pub use codec::{SacCodec, SacCodecConfig, SacEncodeOutput};
pub use converter::{AcousticConverter, AcousticConverterConfig};
pub use speaker::SpeakerEncoder;

//! # vc-core
//!
//! Shared, engine-agnostic foundation of the babiniku.rs voice-conversion
//! workspace. Every engine crate (e.g. `meanvc`, `xvc`) builds on:
//!
//! * [`encoders`] — integration traits for the frozen external models of a
//!   recognition–synthesis VC pipeline (semantic encoder, speaker encoder,
//!   vocoder) plus small helpers such as [`encoders::upsample_bnf`].
//! * [`audio`] — the audio front-end ([`audio::MelSpectrogram`], log-mel
//!   extraction via `rustfft`).
//! * [`bwe`] — bandwidth-extension post-processing for 16 kHz engine
//!   output ([`bwe::Upsampler3x`] 16→48 kHz, [`bwe::Exciter`] harmonic
//!   high-band synthesis; issue #42).
//! * [`config::MelConfig`] — the mel front-end configuration.
//! * [`Error`] / [`Result`] — the common error type.
//!
//! Everything model-specific lives in the engine crates.

pub mod audio;
pub mod bwe;
pub mod config;
pub mod declick;
pub mod encoders;

pub use config::MelConfig;

/// Common error type of the voice-conversion crates.
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

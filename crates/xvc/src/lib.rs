//! # xvc
//!
//! Scaffold for the **X-VC** engine — *Zero-shot Streaming Voice Conversion
//! in Codec Space* ([arXiv:2604.12456](https://arxiv.org/abs/2604.12456)).
//!
//! X-VC is the workspace's language-agnostic engine candidate: its semantic
//! side is the GLM-4-Voice tokenizer (Whisper-encoder based, multilingual
//! incl. Japanese), removing the Mandarin lock of MeanVC.
//!
//! See [`docs/xvc.md`](https://github.com/m96-chan/babiniku.rs/blob/main/docs/xvc.md)
//! for the evaluation notes and
//! [issue #30](https://github.com/m96-chan/babiniku.rs/issues/30) for the
//! port plan (Phase 1: stage-by-stage weight-compatible port on top of
//! [`vc_core`]).

pub mod preprocess;
pub mod tokenizer;

pub use preprocess::{preprocess, PreprocessConfig, WhisperFeatureExtractor};
pub use tokenizer::{SemanticAdapter, WhisperVqEncoder};

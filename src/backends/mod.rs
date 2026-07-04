//! Concrete inference backends for the frozen external models of the
//! MeanVC 2 pipeline (see [`crate::encoders`] for the traits and issue #4
//! for status).
//!
//! All backends are pure-candle ports — no ONNX runtime or other extra
//! dependencies — so they are not feature-gated. Pretrained weights are
//! loaded from safetensors files supplied by the user; converting the
//! upstream PyTorch checkpoints and validating against golden outputs is
//! tracked in issue #4.

mod vocos;

pub use vocos::{Vocos, VocosConfig};

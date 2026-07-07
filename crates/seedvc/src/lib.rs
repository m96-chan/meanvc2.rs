//! Seed-VC engine: a weight-compatible pure-candle port of the
//! [Seed-VC](https://github.com/Plachtaa/seed-vc) offline voice
//! conversion model `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan`
//! (the issue-#49 A/B winner): Whisper-small content features →
//! length regulator → DiT (U-ViT, 384 dim × 9 layers) + WaveNet
//! refiner sampled with 10-step conditional flow matching →
//! 80-bin mel @ 22 050 Hz → BigVGAN vocoder, with a CAM++ speaker
//! encoder for the timbre condition.
//!
//! ## License
//!
//! This crate is **GPL-3.0** (the upstream implementation and released
//! weights are GPL-3.0). It is feature-gated in `vc-demo`; the rest of
//! the workspace remains MIT OR Apache-2.0, and binaries built without
//! the `seedvc` feature carry no GPL obligations.
//!
//! ## Status (issue #50)
//!
//! Phase 1 scaffolding: module layout and the weight-conversion tooling
//! land first; each inference stage is ported against per-stage golden
//! fixtures generated from the official implementation (skip-if-absent,
//! like `crates/xvc`).

pub use vc_core::{Error, Result};

pub mod bigvgan;
pub mod campplus;
pub mod dit;
pub mod mel;
pub mod pipeline;
pub mod regulator;
pub mod stream;
pub mod whisper;

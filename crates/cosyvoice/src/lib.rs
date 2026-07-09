//! CosyVoice2 voice-conversion engine (unofficial Rust port).
//!
//! Weight-compatible port of the **VC path** of
//! [CosyVoice 2](https://arxiv.org/abs/2412.10117) (FunAudioLLM/CosyVoice,
//! Apache-2.0): the text LLM is bypassed entirely (issue #71 recon) —
//! `inference_vc` feeds source speech tokens straight into the flow decoder:
//!
//! ```text
//! source 16 kHz ─ whisper 128-mel ─ FSQ tokenizer (25 Hz, vocab 6561) ┐
//! prompt 16 kHz ─ (same) ─ prompt tokens ───────────────────────────┤
//! prompt 16 kHz ─ kaldi fbank 80 ─ CAM++ ─ x-vector 192 ────────────┤
//! prompt 24 kHz ─ mel 80 (hop 480) ─ prompt feats ──────────────────┴─▶
//!   UpsampleConformer (×2 → 50 Hz) ─ causal CFM U-Net (10 Euler, CFG 0.7)
//!   ─ mel 80 @ 24 kHz ─ HiFT (NSF + iSTFT) ─ 24 kHz audio
//! ```
//!
//! Every stage is verified against the official implementation with golden
//! fixtures (`tools/gen_cosyvoice_fixtures.py`, skip-if-absent). Checkpoint
//! conversion: `tools/convert_cosyvoice.py`. Tracked in issue #75.

pub mod campplus;
pub mod encoder;
pub mod flow;
pub mod hift;
pub mod mel;
pub mod pipeline;
pub mod stream;
pub mod tokenizer;
pub mod unet;

pub use pipeline::CosyVoiceEngine;
pub use stream::{CosyVoiceStream, StreamConfig};
pub use vc_core::{Error, Result};

/// Engine-native sample rates.
pub const TOKEN_SR: u32 = 16_000;
pub const MEL_SR: u32 = 24_000;
/// Speech-token frame rate (Hz) and tokens-per-second bookkeeping.
pub const TOKEN_RATE: usize = 25;
/// Mel frames per token after the ×2 upsampling encoder.
pub const TOKEN_MEL_RATIO: usize = 2;

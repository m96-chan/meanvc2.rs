//! Vevo-Timbre engine: a weight-compatible pure-candle port of
//! **Vevo-Timbre** ([Amphion](https://github.com/open-mmlab/Amphion),
//! ICLR 2025) — style-preserved zero-shot voice conversion via
//! HuBERT-large content-style tokens, a flow-matching DiffLlama
//! converter, and a Vocos vocoder.
//!
//! ## License
//!
//! The **code** in this crate is original and MIT OR Apache-2.0 like
//! the rest of the workspace (Amphion's own code is MIT — no GPL-style
//! poisoning here). The **released weights** are **CC-BY-NC-4.0**:
//! `babiniku-fetch vevo` prompts for that before downloading, and
//! weights are never bundled into a distributed binary.
//!
//! See issue [#74](https://github.com/m96-chan/babiniku.rs/issues/74)
//! (Phase 1 port) and [#72](https://github.com/m96-chan/babiniku.rs/issues/72)
//! (Phase 0 recon).

pub use vc_core::{Error, Result};

pub mod fmt;
pub mod hubert;
pub mod mel;
pub mod pipeline;
pub mod repcodec;
pub mod stream;
pub mod vocos;

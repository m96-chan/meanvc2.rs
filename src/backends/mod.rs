//! Concrete inference backends for the frozen external models of the
//! MeanVC 2 pipeline (see [`crate::encoders`] for the traits and issue #4
//! for status).
//!
//! All backends are pure-candle ports — no ONNX runtime or other extra
//! dependencies — so they are not feature-gated. Pretrained weights are
//! loaded from safetensors files supplied by the user; converting the
//! upstream PyTorch checkpoints and validating against golden outputs is
//! tracked in issue #4.

mod ecapa;
mod fast_u2pp;
mod vocos;
#[cfg(feature = "wavlm")]
mod wavlm_sv;

pub use ecapa::{Ecapa, EcapaConfig};
pub use fast_u2pp::{FastU2pp, FastU2ppConfig, FastU2ppStream};

/// Exposed for parity debugging.
#[doc(hidden)]
pub fn debug_sinusoidal_pe(
    len: usize,
    d: usize,
    dev: &candle_core::Device,
) -> crate::Result<candle_core::Tensor> {
    Ok(fast_u2pp::sinusoidal_pe(len, d, dev)?)
}
pub use vocos::{Vocos, VocosConfig};
#[cfg(feature = "wavlm")]
pub use wavlm_sv::WavLmSv;

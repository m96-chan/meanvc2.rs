//! WavLM-Large speaker-verification voice print (ONNX Runtime backend,
//! feature `wavlm`) — the `spks` conditioning input of MeanVC v1.
//!
//! The official pipeline computes the 256-dim voice print with
//! `ECAPA_TDNN_SMALL(feat_type="wavlm_large")` from microsoft/UniSpeech
//! (checkpoint `wavlm_large_finetune.pth`). That model is exported to ONNX
//! with a **fixed 5 s input** (80 000 samples; the s3prl wrapper is not
//! traceable with dynamic shapes) by `tools/export_wavlm_onnx.py`; this
//! backend tiles/crops the reference audio to that length. Tiling a 3.8 s
//! utterance changes the embedding by < 0.006 cosine vs the full-length
//! original, well within speaker-identity tolerance.

use candle_core::{Device, Tensor};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use std::sync::Mutex;

use crate::encoders::SpeakerEncoder;
use crate::{Error, Result};

/// Fixed input length of the exported graph (5 s at 16 kHz).
pub const INPUT_SAMPLES: usize = 80_000;
const EMB_DIM: usize = 256;
const SAMPLE_RATE: usize = 16_000;

/// WavLM-Large + ECAPA speaker-verification model via ONNX Runtime.
pub struct WavLmSv {
    session: Mutex<Session>,
}

impl std::fmt::Debug for WavLmSv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WavLmSv").finish()
    }
}

impl WavLmSv {
    /// Loads the exported ONNX graph (`wavlm_sv.onnx`, ~1.3 GB).
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| Error::Input(format!("onnxruntime: {e}")))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Tiles/crops a waveform to the fixed graph length.
    fn fix_length(samples: &[f32]) -> Result<Vec<f32>> {
        if samples.is_empty() {
            return Err(Error::Input("empty reference waveform".into()));
        }
        Ok(samples
            .iter()
            .copied()
            .cycle()
            .take(INPUT_SAMPLES)
            .collect())
    }
}

impl SpeakerEncoder for WavLmSv {
    fn embedding_dim(&self) -> usize {
        EMB_DIM
    }

    fn embed(&self, samples: &[f32], sample_rate: usize) -> Result<Tensor> {
        if sample_rate != SAMPLE_RATE {
            return Err(Error::Input(format!(
                "expected {SAMPLE_RATE} Hz input, got {sample_rate} Hz (resample first)"
            )));
        }
        let wav = Self::fix_length(samples)?;
        let input = OrtTensor::from_array(([1usize, INPUT_SAMPLES], wav))
            .map_err(|e| Error::Input(format!("onnx input: {e}")))?;
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["wav" => input])
            .map_err(|e| Error::Input(format!("onnx run: {e}")))?;
        let (_, data) = outputs["embedding"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Input(format!("onnx output: {e}")))?;
        Ok(Tensor::from_vec(data.to_vec(), (EMB_DIM,), &Device::Cpu)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ckpt(name: &str) -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("ckpt")
            .join(name);
        if !p.exists() {
            eprintln!("skipping: {name} not found (see tools/export_wavlm_onnx.py)");
            return None;
        }
        Some(p)
    }

    #[test]
    fn matches_onnxruntime_golden() {
        let (Some(model), Some(golden)) = (ckpt("wavlm_sv.onnx"), ckpt("wavlm_golden.safetensors"))
        else {
            return;
        };
        let sv = WavLmSv::load(model).unwrap();
        let fx = candle_core::safetensors::load(golden, &Device::Cpu).unwrap();
        let wav: Vec<f32> = fx["wav_tiled"].to_vec1().unwrap();
        // The fixture wav is already tiled to INPUT_SAMPLES; embed() re-tiles
        // idempotently.
        let emb = sv.embed(&wav, 16_000).unwrap();
        let d = (&emb - &fx["embedding_ref"])
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-3, "embedding diverges from onnxruntime golden: {d}");
    }
}

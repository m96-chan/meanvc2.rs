//! Offline wav-to-wav CosyVoice2 VC pipeline.
//!
//! Wires the ported stages into the same graph as the official
//! `inference_vc` (issue #71 recon): tokenizer (source + prompt) → CAM++
//! embedding → flow (encoder + CFM) → HiFT vocoder. No LLM — the VC path
//! never touches it.
//!
//! **Note on golden-testing this module:** `HiftGenerator`'s F0 predictor
//! feeds a `cumsum`-integrated sine phase (`SineGen2`), so a single
//! misjudged voiced/unvoiced frame permanently shifts the harmonic phase
//! for the rest of the clip — a sub-0.01% mel deviation (e.g. from a
//! different, still-correct resampler) can collapse whole-clip audio
//! correlation to noise even though every stage is individually correct.
//! That's why the tests below check pipeline *wiring* against exact
//! official tensors (§`wiring_matches_official`) separately from
//! feature-extraction *fidelity* via cosine similarity
//! (§`prepare_reference_extracts_reasonable_features`), rather than
//! asserting whole-clip audio correlation end to end.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use vc_core::profile::resample_analysis;
use vc_core::Result;

use crate::campplus::CamPlusPlus;
use crate::flow::Flow;
use crate::hift::{HiftGenerator, NoiseMode};
use crate::mel::{kaldi_fbank80, MelFrontend};
use crate::tokenizer::SpeechTokenizer;
use crate::{MEL_SR, TOKEN_SR};

/// A prepared reference speaker: tokens, x-vector, and 24 kHz mel feats,
/// computed once and reused across conversions.
pub struct Reference {
    pub(crate) tokens: Tensor,
    pub(crate) embedding: Tensor,
    pub(crate) feat: Tensor,
}

impl Reference {
    /// Number of FSQ speech tokens (25 Hz).
    pub fn tokens_len(&self) -> usize {
        self.tokens.dim(1).unwrap()
    }

    /// Number of 24 kHz mel frames in the prompt feat (should equal
    /// `tokens_len() * 2` — the flow's `token_mel_ratio` — for the
    /// encoder's prompt/source split inside [`Flow::cfm`] to land on the
    /// intended boundary; see `crate::pipeline` module docs for what
    /// happens when it doesn't).
    pub fn feat_len(&self) -> usize {
        self.feat.dim(1).unwrap()
    }
}

/// The full CosyVoice2 VC engine: tokenizer + CAM++ + flow + HiFT.
pub struct CosyVoiceEngine {
    mel: MelFrontend,
    tokenizer: SpeechTokenizer,
    campplus: CamPlusPlus,
    flow: Flow,
    hift: HiftGenerator,
    device: Device,
}

impl CosyVoiceEngine {
    /// Load all stages from `<ckpt>/cosyvoice_*.safetensors`
    /// (see `tools/convert_cosyvoice.py`).
    pub fn load(ckpt: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let d = ckpt.as_ref();
        let vb =
            |name: &str| -> Result<VarBuilder<'static>> {
                Ok(unsafe {
                    VarBuilder::from_mmaped_safetensors(&[d.join(name)], DType::F32, device)?
                })
            };
        Ok(Self {
            mel: MelFrontend::load(d.join("cosyvoice_mel.safetensors"))?,
            tokenizer: SpeechTokenizer::load(vb("cosyvoice_tokenizer.safetensors")?, device)?,
            campplus: CamPlusPlus::load(vb("cosyvoice_campplus.safetensors")?)?,
            flow: Flow::load(vb("cosyvoice_flow.safetensors")?)?,
            hift: HiftGenerator::load(vb("cosyvoice_hift.safetensors")?)?,
            device: device.clone(),
        })
    }

    /// Tokenize 16 kHz mono audio (whisper 128-mel → FSQ tokens).
    pub(crate) fn tokenize(&self, audio_16k: &[f32]) -> Result<Tensor> {
        let mel = self.mel.whisper_mel128(audio_16k, &self.device)?;
        self.tokenizer.tokenize(&mel)
    }

    pub(crate) fn flow_ref(&self) -> &Flow {
        &self.flow
    }

    pub(crate) fn hift_ref(&self) -> &HiftGenerator {
        &self.hift
    }

    /// Raw CAM++ speaker embedding for arbitrary 16 kHz audio (192-d,
    /// pre-projection) — a debugging/diagnostic hook for checking
    /// speaker similarity independent of the flow/HiFT pipeline; not
    /// used by [`Self::prepare_reference`] or [`Self::convert_offline`]
    /// (those call the same underlying stages directly).
    pub fn embed_for_debug(&self, audio_16k: &[f32]) -> Result<Vec<f32>> {
        let fbank = kaldi_fbank80(audio_16k, &self.device)?;
        let emb = self.campplus.embed(&fbank)?;
        Ok(emb.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Precompute everything the flow needs from a reference speaker's
    /// audio (any sample rate; resampled internally to 16 kHz / 24 kHz).
    pub fn prepare_reference(&self, audio: &[f32], sr: u32) -> Result<Reference> {
        let audio_16k = resample_analysis(audio, sr as usize, TOKEN_SR as usize);
        let audio_24k = resample_analysis(audio, sr as usize, MEL_SR as usize);
        let tokens = self.tokenize(&audio_16k)?;
        let fbank = kaldi_fbank80(&audio_16k, &self.device)?;
        let embedding = self.campplus.embed(&fbank)?;
        let mut feat = self.mel.hifigan_mel80(&audio_24k, &self.device)?;

        // `tokens` (16 kHz path) and `feat` (24 kHz path) are computed by
        // two independently-resampled/STFT'd signals, so their frame
        // counts can drift by a frame or two from the `token_mel_ratio`
        // (2) the official model was trained under — `Flow::cfm` locates
        // the prompt/source boundary in `mu` at exactly
        // `tokens.len() * 2`, so any drift here shifts that boundary.
        // Snap `feat` to the exact expected length (pad by repeating the
        // last frame, or trim) rather than let it silently disagree.
        let want = tokens.dim(1)? * crate::TOKEN_MEL_RATIO;
        let have = feat.dim(1)?;
        if have < want {
            let last = feat.narrow(1, have - 1, 1)?;
            let pad = last
                .broadcast_as((1, want - have, feat.dim(2)?))?
                .contiguous()?;
            feat = Tensor::cat(&[&feat, &pad], 1)?;
        } else if have > want {
            feat = feat.narrow(1, 0, want)?;
        }

        Ok(Reference {
            tokens,
            embedding,
            feat,
        })
    }

    /// Offline conversion: `source` (any sample rate) → the reference
    /// speaker's voice, 24 kHz mono audio.
    pub fn convert_offline(
        &self,
        source: &[f32],
        source_sr: u32,
        reference: &Reference,
    ) -> Result<Vec<f32>> {
        self.convert_offline_with(source, source_sr, reference, NoiseMode::Random)
    }

    /// Same as [`Self::convert_offline`] but with an explicit
    /// [`NoiseMode`] — `Deterministic` reproduces the official golden
    /// fixtures bit-for-bit; real use should stick to `Random`.
    pub fn convert_offline_with(
        &self,
        source: &[f32],
        source_sr: u32,
        reference: &Reference,
        noise: NoiseMode,
    ) -> Result<Vec<f32>> {
        let source_16k = resample_analysis(source, source_sr as usize, TOKEN_SR as usize);
        let source_tokens = self.tokenize(&source_16k)?;
        let tokens = Tensor::cat(&[&reference.tokens, &source_tokens], 1)?;
        let mu = self.flow.mu(&tokens, false, true)?;
        let mel = self
            .flow
            .cfm(&mu, &reference.embedding, &reference.feat, false)?;
        let (audio, _source) = self.hift.vocode(&mel, noise)?;
        Ok(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn ckpt_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ckpt")
    }

    fn have_ckpts() -> bool {
        [
            "cosyvoice_mel.safetensors",
            "cosyvoice_tokenizer.safetensors",
            "cosyvoice_campplus.safetensors",
            "cosyvoice_flow.safetensors",
            "cosyvoice_hift.safetensors",
        ]
        .iter()
        .all(|f| ckpt_dir().join(f).exists())
    }

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let p = ckpt_dir().join("cosyvoice_e2e_fixture.safetensors");
        p.exists()
            .then(|| candle_core::safetensors::load(p, &Device::Cpu).ok())
            .flatten()
    }

    fn cos_max(a: &Tensor, b: &Tensor) -> (f32, f32) {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        (dot / (na * nb), max)
    }

    fn corr(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    /// Validates the pipeline's *wiring* (token concatenation order,
    /// which tensor feeds which argument) using exact official
    /// intermediate tensors throughout — isolated from any feature-
    /// extraction (resampling / CAM++) noise, since §module docs.
    #[test]
    fn wiring_matches_official() {
        if !have_ckpts() {
            return;
        }
        let Some(fx) = fixture() else { return };
        let engine = CosyVoiceEngine::load(ckpt_dir(), &Device::Cpu).unwrap();

        let reference = Reference {
            tokens: fx["prompt_tokens"].to_dtype(DType::U32).unwrap(),
            embedding: fx["embedding"].clone(),
            feat: fx["prompt_feat"].clone(),
        };
        let source_16k = fx["source_16k"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let audio = engine
            .convert_offline_with(&source_16k, TOKEN_SR, &reference, NoiseMode::Deterministic)
            .unwrap();

        let want = fx["e2e_audio"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(audio.len(), want.len(), "output length");
        let c = corr(&audio, &want);
        assert!(c > 0.999, "wiring e2e correlation {c}");
    }

    /// Validates `prepare_reference`'s audio → (tokens, embedding, feat)
    /// extraction against the official values, via cosine similarity
    /// (not downstream audio — see module docs on F0 phase sensitivity).
    #[test]
    fn prepare_reference_extracts_reasonable_features() {
        if !have_ckpts() {
            return;
        }
        let Some(fx) = fixture() else { return };
        let engine = CosyVoiceEngine::load(ckpt_dir(), &Device::Cpu).unwrap();
        let prompt_16k = fx["prompt_16k"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let reference = engine.prepare_reference(&prompt_16k, TOKEN_SR).unwrap();

        assert_eq!(reference.tokens.dims(), fx["prompt_tokens"].dims());
        let got_tok = reference
            .tokens
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let want_tok = fx["prompt_tokens"]
            .flatten_all()
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let miss = got_tok
            .iter()
            .zip(&want_tok)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            miss * 100 <= want_tok.len(),
            "{miss}/{} token mismatches",
            want_tok.len()
        );

        // CAM++ full-length embedding: the official ONNX drifts from the
        // true dynamic model at this length (documented in campplus.rs);
        // a loose bound is the meaningful check here.
        let (cos, _) = cos_max(&reference.embedding, &fx["embedding"]);
        assert!(cos > 0.995, "embedding cosine {cos}");

        // Our resampler (16k->24k) differs slightly from the official
        // torchaudio path; cosine similarity should still be excellent.
        let plen = fx["prompt_feat"].dim(1).unwrap();
        let got_feat = reference.feat.narrow(1, 0, plen).unwrap();
        let (cos, _) = cos_max(&got_feat, &fx["prompt_feat"]);
        assert!(cos > 0.999, "prompt_feat cosine {cos}");
    }

    /// End-to-end smoke test through the public API exactly as a caller
    /// would use it (own audio → own features → conversion) — checks the
    /// pipeline runs and produces finite, correctly-shaped, non-degenerate
    /// audio. Not a fidelity check (see module docs).
    #[test]
    fn offline_conversion_produces_valid_audio() {
        if !have_ckpts() {
            return;
        }
        let Some(fx) = fixture() else { return };
        let engine = CosyVoiceEngine::load(ckpt_dir(), &Device::Cpu).unwrap();
        let prompt_16k = fx["prompt_16k"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let source_16k = fx["source_16k"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let reference = engine.prepare_reference(&prompt_16k, TOKEN_SR).unwrap();
        let audio = engine
            .convert_offline(&source_16k, TOKEN_SR, &reference)
            .unwrap();

        assert!(!audio.is_empty());
        assert!(audio.iter().all(|s| s.is_finite()));
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        assert!(rms > 1e-4 && rms < 1.0, "implausible RMS {rms}");
    }
}

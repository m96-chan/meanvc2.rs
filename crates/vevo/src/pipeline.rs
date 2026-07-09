//! Offline `VevoEngine`: the official `inference_fm` flow wired from
//! the ported stages — 24 kHz in/out, HuBERT-large content-style
//! tokens at 16 kHz, timbre-preserved conversion via a 32-step CFM
//! prompted by the reference's own mel, Vocos out.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::fmt::{FlowMatchingTransformer, FmtConfig};
use crate::hubert::{HubertConfig, HubertLarge};
use crate::mel::{MelConfig, MelSpectrogram};
use crate::repcodec::{RepCodec, RepCodecConfig};
use crate::vocos::{Vocos, VocosConfig};
use vc_core::Result;

/// `torchaudio.functional.resample` (sinc_interp_hann, width 6,
/// rolloff 0.99) — duplicated per-engine like `seedvc::pipeline`
/// (each engine crate is self-contained in this workspace).
pub fn resample(input: &[f32], orig: usize, new: usize) -> Vec<f32> {
    let g = gcd(orig, new);
    let (orig_f, new_f) = (orig / g, new / g);
    let width = 6usize;
    let rolloff = 0.99f64;
    let base = rolloff * (orig_f.min(new_f) as f64);
    let kw = (width as f64 * orig_f as f64 / base).ceil() as i64;
    let scale = base / orig_f as f64;
    let taps = (2 * kw + orig_f as i64) as usize;
    let mut kernels = vec![vec![0f64; taps]; new_f];
    for (i, row) in kernels.iter_mut().enumerate() {
        for (jj, k) in row.iter_mut().enumerate() {
            let j = jj as i64 - kw;
            let t = (-(i as f64) / new_f as f64 + j as f64 / orig_f as f64) * base;
            let t = t.clamp(-(width as f64), width as f64);
            let win = (t * std::f64::consts::PI / width as f64 / 2.0)
                .cos()
                .powi(2);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (t * std::f64::consts::PI).sin() / (t * std::f64::consts::PI)
            };
            *k = scale * sinc * win;
        }
    }
    let n_out = (input.len() * new_f).div_ceil(orig_f);
    let mut out = vec![0f32; n_out];
    for (n, o) in out.iter_mut().enumerate() {
        let block = n / new_f;
        let phase = n % new_f;
        let center = (block * orig_f) as i64;
        let mut acc = 0f64;
        for (jj, k) in kernels[phase].iter().enumerate() {
            let idx = center + jj as i64 - kw;
            if idx >= 0 && (idx as usize) < input.len() {
                acc += k * input[idx as usize] as f64;
            }
        }
        *o = acc as f32;
    }
    out
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

const MEL_MEAN: f32 = -4.92;
const MEL_VAR: f32 = 8.14;

pub struct VevoEngine {
    hubert: HubertLarge,
    hubert_mean: Tensor,
    hubert_std: Tensor,
    repcodec: RepCodec,
    mel: MelSpectrogram,
    pub(crate) fmt: FlowMatchingTransformer,
    pub(crate) vocos: Vocos,
    device: Device,
}

impl VevoEngine {
    pub fn load(ckpt: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let d = ckpt.as_ref();
        let stats = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[d.join("vevo_hubert_stats.safetensors")],
                DType::F32,
                device,
            )?
        };
        Ok(Self {
            hubert: HubertLarge::load(
                &HubertConfig::default(),
                d.join("vevo_hubert.safetensors"),
                device,
            )?,
            hubert_mean: stats.get(1024, "mean")?,
            hubert_std: stats.get(1024, "std")?,
            repcodec: RepCodec::load(
                &RepCodecConfig::default(),
                d.join("vevo_repcodec.safetensors"),
                device,
            )?,
            mel: MelSpectrogram::new(MelConfig::default(), device),
            fmt: FlowMatchingTransformer::load(
                &FmtConfig::default(),
                d.join("vevo_fmt.safetensors"),
                device,
            )?,
            vocos: Vocos::load(
                VocosConfig::default(),
                d.join("vevo_vocos.safetensors"),
                device,
            )?,
            device: device.clone(),
        })
    }

    /// z-normalized HuBERT-large layer-18 features → content-style
    /// codebook indices, `[1, T]` int64 (matches
    /// `extract_hubert_codec(..., token_type="hubert_codec")`).
    pub fn content_style_codes(&self, wave16k: &[f32]) -> Result<Tensor> {
        let wav = Tensor::from_vec(wave16k.to_vec(), (1, wave16k.len()), &self.device)?;
        let feats = self.hubert.extract_features(&wav)?;
        let normed = feats
            .broadcast_sub(&self.hubert_mean)?
            .broadcast_div(&self.hubert_std)?;
        self.repcodec.quantize(&normed)
    }

    /// z-normalized mel of a 24 kHz waveform, the CFM `prompt`.
    pub fn mel_feature(&self, wave24k: &[f32]) -> Result<Tensor> {
        let wav = Tensor::from_vec(wave24k.to_vec(), (1, wave24k.len()), &self.device)?;
        let raw = self.mel.forward_batch(&wav)?; // [1, t, 128]
        ((raw - MEL_MEAN as f64)? / MEL_VAR.sqrt() as f64).map_err(Into::into)
    }

    /// Vevo-Timbre `inference_fm`: style-preserved zero-shot VC. `src`/
    /// `timbre_ref` at 24 kHz. `noise`: the CFM's initial target-length
    /// noise (`Some` for golden-fixture replay; `None` samples fresh
    /// Gaussian noise). Returns 24 kHz samples.
    pub fn inference_fm(
        &self,
        src_24k: &[f32],
        timbre_ref_24k: &[f32],
        steps: usize,
        noise: Option<Tensor>,
    ) -> Result<Vec<f32>> {
        let src_16k = resample(src_24k, 24_000, 16_000);
        let ref_16k = resample(timbre_ref_24k, 24_000, 16_000);

        let src_codes = self.content_style_codes(&src_16k)?;
        let ref_codes = self.content_style_codes(&ref_16k)?;
        let codes = Tensor::cat(&[&ref_codes, &src_codes], 1)?;
        let cond = self.fmt.cond_embed(&codes)?;

        let prompt = self.mel_feature(timbre_ref_24k)?;
        let target_len = codes.dim(1)? - prompt.dim(1)?;
        let noise = match noise {
            Some(n) => n,
            None => Tensor::randn(0f32, 1f32, (1, target_len, prompt.dim(2)?), &self.device)?,
        };

        let mel = self
            .fmt
            .reverse_diffusion(&cond, &prompt, noise, steps, 1.0, 0.75)?;
        let wave = self.vocos.synthesize(&mel)?;
        Ok(wave)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use std::collections::HashMap;

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_e2e_fixture.safetensors");
        if !path.exists() {
            return None;
        }
        Some(candle_core::safetensors::load(path, &Device::Cpu).unwrap())
    }

    fn ckpt_dir() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
        let need = [
            "vevo_hubert.safetensors",
            "vevo_hubert_stats.safetensors",
            "vevo_repcodec.safetensors",
            "vevo_fmt.safetensors",
            "vevo_vocos.safetensors",
        ];
        need.iter().all(|f| path.join(f).exists()).then_some(path)
    }

    fn corr(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn e2e_matches_official() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt_dir()) else {
            return;
        };
        let dev = Device::Cpu;
        let engine = VevoEngine::load(&ckpt, &dev).unwrap();

        let src_24k: Vec<f32> = fx["src_24k"].i(0).unwrap().to_vec1().unwrap();
        let ref_24k: Vec<f32> = fx["ref_24k"].i(0).unwrap().to_vec1().unwrap();
        let noise = fx["cfm_noise"].clone();
        let want: Vec<f32> = fx["wave_out"]
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();

        let got = engine
            .inference_fm(&src_24k, &ref_24k, 32, Some(noise))
            .unwrap();
        assert_eq!(got.len(), want.len(), "sample count mismatch");
        let c = corr(&got, &want);
        assert!(c > 0.999, "e2e correlation {c}");
    }
}

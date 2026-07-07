//! Offline Seed-VC engine: the `inference.py` single-pass flow wired
//! from the ported stages — 22 050 Hz in/out, whisper content at 16 k,
//! CAM++ style from the reference, length regulation to the mel grid,
//! 10-step CFM with cfg 0.7 prompted by the reference mel, BigVGAN out.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{bigvgan::BigVgan, campplus::CampPlus, campplus::FbankExtractor, dit::Cfm, mel::MelExtractor, regulator::InterpolateRegulator, whisper::WhisperEncoder, Result};

/// `torchaudio.functional.resample` (sinc_interp_hann, width 6,
/// rolloff 0.99) — the resampler used by the official pipeline for
/// 22 050 → 16 000.
pub fn resample(input: &[f32], orig: usize, new: usize) -> Vec<f32> {
    resample_width(input, orig, new, 6)
}

/// Like [`resample`] with an explicit low-pass filter width: 6 matches
/// torchaudio's default (used for parity on the input side); the
/// 22 050 → 48 000 output hop uses 16 for a cleaner image rejection.
pub fn resample_width(input: &[f32], orig: usize, new: usize, width: usize) -> Vec<f32> {
    let g = gcd(orig, new);
    let (orig_f, new_f) = (orig / g, new / g);
    let rolloff = 0.99f64;
    let base = rolloff * (orig_f.min(new_f) as f64);
    let kw = (width as f64 * orig_f as f64 / base).ceil() as i64;
    let scale = base / orig_f as f64;
    // kernels[phase][j], taps j over [-kw, kw + orig) — the tap range is
    // ASYMMETRIC (torchaudio: arange(-width, width + orig_freq)): later
    // phases centre at j = i·orig/new > 0, so the support slides right.
    let taps = (2 * kw + orig_f as i64) as usize;
    let mut kernels = vec![vec![0f64; taps]; new_f];
    for (i, row) in kernels.iter_mut().enumerate() {
        for (jj, k) in row.iter_mut().enumerate() {
            let j = jj as i64 - kw;
            // t in low-pass periods (torchaudio: (-i/new + j/orig) · base).
            let t = (-(i as f64) / new_f as f64 + j as f64 / orig_f as f64) * base;
            let t = t.clamp(-(width as f64), width as f64);
            let win = (t * std::f64::consts::PI / width as f64 / 2.0).cos().powi(2);
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

pub struct SeedVcEngine {
    whisper: WhisperEncoder,
    campplus: CampPlus,
    fbank: FbankExtractor,
    mel: MelExtractor,
    regulator: InterpolateRegulator,
    cfm: Cfm,
    vocoder: BigVgan,
    device: Device,
}

impl SeedVcEngine {
    pub fn load(ckpt: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let d = ckpt.as_ref();
        let vb = |name: &str| -> Result<VarBuilder> {
            Ok(unsafe {
                VarBuilder::from_mmaped_safetensors(&[d.join(name)], DType::F32, device)?
            })
        };
        Ok(Self {
            whisper: WhisperEncoder::load(vb("seedvc_whisper.safetensors")?)?,
            campplus: CampPlus::load(vb("seedvc_campplus.safetensors")?)?,
            fbank: FbankExtractor::new(),
            mel: MelExtractor::new(),
            regulator: InterpolateRegulator::load(vb("seedvc_regulator.safetensors")?)?,
            cfm: Cfm::load(vb("seedvc_dit.safetensors")?)?,
            vocoder: BigVgan::load(d.join("seedvc_bigvgan.safetensors"), device)?,
            device: device.clone(),
        })
    }

    /// Whisper content features from 16 kHz samples.
    pub fn whisper_features(&self, wave16k: &[f32]) -> Result<Tensor> {
        self.whisper.forward(wave16k, &self.device)
    }

    /// 22 050 Hz mel as a `[1, 80, T]` tensor.
    pub fn mel22(&self, wave22k: &[f32]) -> Result<Tensor> {
        self.mel_tensor(wave22k)
    }

    /// Length regulation of whisper features to `target_len` mel frames.
    pub fn regulate(&self, features: &Tensor, target_len: usize) -> Result<Tensor> {
        self.regulator.forward(features, target_len)
    }

    /// CAM++ fbank of a 16 kHz reference.
    pub fn ref_fbank(&self, wave16k: &[f32]) -> Vec<Vec<f32>> {
        self.fbank.extract(wave16k)
    }

    /// CAM++ speaker embedding from an fbank.
    pub fn campplus_embed(&self, fbank: &[Vec<f32>]) -> Result<Tensor> {
        self.campplus.embed(fbank, &self.device)
    }

    /// CFM sampling (see [`crate::dit::Cfm::inference`]).
    pub fn cfm_inference(
        &self,
        cat_condition: &Tensor,
        prompt_mel: &Tensor,
        style: &Tensor,
        noise: &Tensor,
        steps: usize,
        cfg_rate: f64,
    ) -> Result<Tensor> {
        self.cfm
            .inference(cat_condition, prompt_mel, style, noise, steps, cfg_rate)
    }

    /// BigVGAN synthesis: `[1, 80, T]` mel → samples at 22 050 Hz.
    pub fn vocode(&self, mel: &Tensor) -> Result<Vec<f32>> {
        Ok(self.vocoder.forward(mel)?.flatten_all()?.to_vec1()?)
    }

    fn mel_tensor(&self, wave22k: &[f32]) -> Result<Tensor> {
        let m = self.mel.extract(wave22k);
        let (bins, frames) = (m.len(), m[0].len());
        let flat: Vec<f32> = m.into_iter().flatten().collect();
        Ok(Tensor::from_vec(flat, (1, bins, frames), &self.device)?)
    }

    /// Offline conversion: source & reference at 22 050 Hz →
    /// converted waveform at 22 050 Hz. `noise` is the CFM initial
    /// noise `[1, 80, prompt+source mel frames]` (the fixture's for
    /// golden replay, fresh gaussian noise for normal use); `steps` /
    /// `cfg_rate` official defaults are 10 / 0.7.
    pub fn convert_offline(
        &self,
        source_22k: &[f32],
        ref_22k: &[f32],
        steps: usize,
        cfg_rate: f64,
        noise: &Tensor,
    ) -> Result<Vec<f32>> {
        let src16 = resample(source_22k, 22_050, 16_000);
        let ref16 = resample(ref_22k, 22_050, 16_000);

        let s_alt = self.whisper.forward(&src16, &self.device)?;
        let s_ori = self.whisper.forward(&ref16, &self.device)?;

        let mel = self.mel_tensor(source_22k)?;
        let mel2 = self.mel_tensor(ref_22k)?;
        let (t_src, t_ref) = (mel.dim(2)?, mel2.dim(2)?);

        let fb = self.fbank.extract(&ref16);
        let style = self.campplus.embed(&fb, &self.device)?;

        let cond = self.regulator.forward(&s_alt, t_src)?;
        let prompt = self.regulator.forward(&s_ori, t_ref)?;
        let cat = Tensor::cat(&[&prompt, &cond], 1)?;

        let vc_mel = self
            .cfm
            .inference(&cat, &mel2, &style, noise, steps, cfg_rate)?;
        let vc_mel = vc_mel.narrow(2, t_ref, vc_mel.dim(2)? - t_ref)?;
        let wave = self.vocoder.forward(&vc_mel)?;
        Ok(wave.flatten_all()?.to_vec1()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn engine_e2e_matches_official() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
        for f in [
            "seedvc_whisper.safetensors",
            "seedvc_campplus.safetensors",
            "seedvc_regulator.safetensors",
            "seedvc_dit.safetensors",
            "seedvc_bigvgan.safetensors",
            "seedvc_e2e_fixture.safetensors",
        ] {
            if !dir.join(f).exists() {
                return;
            }
        }
        let dev = Device::Cpu;
        let eng = SeedVcEngine::load(&dir, &dev).unwrap();
        let fx = candle_core::safetensors::load(dir.join("seedvc_e2e_fixture.safetensors"), &dev)
            .unwrap();
        let src: Vec<f32> = fx["source_22k"].i(0).unwrap().to_vec1().unwrap();
        let rf: Vec<f32> = fx["ref_22k"].i(0).unwrap().to_vec1().unwrap();
        // Stage-by-stage attribution against the fixture.
        let d = |a: &Tensor, b: &Tensor| -> f32 {
            (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap()
        };
        let src16 = resample(&src, 22_050, 16_000);
        let ref16 = resample(&rf, 22_050, 16_000);
        let s_alt = eng.whisper.forward(&src16, &dev).unwrap();
        println!("s_alt diff {:.2e}", d(&s_alt, &fx["s_alt"]));
        let mel2 = eng.mel_tensor(&rf).unwrap();
        println!("mel2 diff {:.2e}", d(&mel2, &fx["mel2"]));
        let fb = eng.fbank.extract(&ref16);
        let style = eng.campplus.embed(&fb, &dev).unwrap();
        println!("style diff {:.2e}", d(&style, &fx["style2"]));
        let cond = eng.regulator.forward(&s_alt, 516).unwrap();
        println!("cond diff {:.2e}", d(&cond, &fx["cond"]));
        let s_ori = eng.whisper.forward(&ref16, &dev).unwrap();
        let prompt = eng.regulator.forward(&s_ori, 516).unwrap();
        println!("prompt diff {:.2e}", d(&prompt, &fx["prompt_condition"]));
        let cat = Tensor::cat(&[&fx["prompt_condition"], &fx["cond"]], 1).unwrap();
        let vc_mel = eng.cfm.inference(&cat, &fx["mel2"], &fx["style2"], &fx["cfm_noise"], 10, 0.7).unwrap();
        let vc_mel_t = vc_mel.narrow(2, 516, vc_mel.dim(2).unwrap() - 516).unwrap();
        println!("vc_mel (fixture inputs) diff {:.2e}", d(&vc_mel_t, &fx["vc_mel"]));
        let got = eng
            .convert_offline(&src, &rf, 10, 0.7, &fx["cfm_noise"])
            .unwrap();
        let want: Vec<f32> = fx["vc_wave"].i(0).unwrap().to_vec1().unwrap();
        let n = got.len().min(want.len());
        let (mut dot, mut ga, mut wa, mut dmax) = (0f64, 0f64, 0f64, 0f32);
        for i in 0..n {
            dot += got[i] as f64 * want[i] as f64;
            ga += (got[i] as f64).powi(2);
            wa += (want[i] as f64).powi(2);
            dmax = dmax.max((got[i] - want[i]).abs());
        }
        let corr = dot / (ga.sqrt() * wa.sqrt() + 1e-12);
        println!("e2e: len {} vs {}, corr {corr:.6}, max abs {dmax:.2e}", got.len(), want.len());
        // The fixture chain carries CUDA TF32 noise at every stage and
        // the resampler is an independent implementation, so the e2e
        // bound is correlation-based (the per-stage goldens carry the
        // tight fp32 parity).
        assert!(corr > 0.99, "e2e correlation {corr}");
    }
}

//! HiFT vocoder — NSF source-filter + iSTFT-head HiFi-GAN @ 24 kHz
//! (HiFTNet, arXiv:2309.09493; CosyVoice2 config: upsample 8·5·3,
//! iSTFT n_fft 16 / hop 4 ⇒ 480 samples per mel frame).
//!
//! ```text
//! mel [1,80,T] ─ ConvRNN F0 predictor ─ f0 [T]
//!   ─ ×480 nearest ─ SineGen2 (9 harmonics) ─ tanh(linear) ─ source s
//! s ─ STFT16 ─ [1,18,F] ──────────────┐ (fused at every scale)
//! mel ─ conv k7 ─ [ConvT ×8,×5,×3 + source fusion + 3 Snake resblocks]
//!   ─ conv k7 → 9 log-magnitudes + 9 phases ─ iSTFT16 ─ audio (±0.99)
//! ```
//!
//! Randomness: the official `SineGen2` draws uniform initial phases for
//! harmonics 1–8 and Gaussian noise on both source branches.
//! [`HiftGenerator::vocode`] takes a [`NoiseMode`]: `Deterministic` zeroes
//! them (bit-matching the golden fixtures generated with the same patch);
//! `Random` reproduces the official behaviour with a local xorshift RNG
//! (device-independent).

use candle_core::{Tensor, D};
use candle_nn::ops::leaky_relu;
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use rustfft::{num_complex::Complex32, FftPlanner};
use vc_core::Result;

const SR: f32 = 24_000.0;
const HARMONICS: usize = 9; // fundamental + 8 overtones
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 10.0;
const UPSAMPLE: usize = 480; // 8·5·3·hop4
const N_FFT: usize = 16;
const HOP: usize = 4;
const AUDIO_LIMIT: f32 = 0.99;

/// How to fill the NSF's stochastic components.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NoiseMode {
    /// Zero phases/noise — matches the golden fixtures.
    Deterministic,
    /// Official behaviour (random harmonic phases + Gaussian noise).
    Random,
}

/// Small device-independent RNG (xorshift64*) for `NoiseMode::Random`.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    fn gaussian(&mut self) -> f32 {
        // Box-Muller
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

struct Snake {
    alpha: Tensor, // [C]
}

impl Snake {
    fn load(vb: VarBuilder, c: usize) -> Result<Self> {
        Ok(Self {
            alpha: vb.get(c, "alpha")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.alpha.reshape((1, (), 1))?;
        let s = x.broadcast_mul(&a)?.sin()?.sqr()?;
        Ok(x.add(&s.broadcast_div(&(a + 1e-9)?)?)?)
    }
}

struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    acts1: Vec<Snake>,
    acts2: Vec<Snake>,
}

impl ResBlock {
    fn load(vb: VarBuilder, ch: usize, k: usize, dilations: &[usize]) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut acts1 = Vec::new();
        let mut acts2 = Vec::new();
        for (i, d) in dilations.iter().enumerate() {
            let cfg1 = Conv1dConfig {
                padding: (k - 1) / 2 * d,
                dilation: *d,
                ..Default::default()
            };
            let cfg2 = Conv1dConfig {
                padding: (k - 1) / 2,
                ..Default::default()
            };
            convs1.push(conv1d(ch, ch, k, cfg1, vb.pp(format!("convs1.{i}")))?);
            convs2.push(conv1d(ch, ch, k, cfg2, vb.pp(format!("convs2.{i}")))?);
            acts1.push(Snake::load(vb.pp(format!("activations1.{i}")), ch)?);
            acts2.push(Snake::load(vb.pp(format!("activations2.{i}")), ch)?);
        }
        Ok(Self {
            convs1,
            convs2,
            acts1,
            acts2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.acts1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.acts2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = x.add(&xt)?;
        }
        Ok(x)
    }
}

struct F0Predictor {
    convs: Vec<Conv1d>,
    classifier: Linear,
}

impl F0Predictor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut convs = Vec::new();
        for (i, (ic, oc)) in [(80, 512), (512, 512), (512, 512), (512, 512), (512, 512)]
            .iter()
            .enumerate()
        {
            convs.push(conv1d(
                *ic,
                *oc,
                3,
                cfg,
                vb.pp(format!("condnet.{}", i * 2)),
            )?);
        }
        Ok(Self {
            convs,
            classifier: linear(512, 1, vb.pp("classifier"))?,
        })
    }

    /// `mel`: [1, 80, T] → f0 [1, T] (Hz, ≥ 0).
    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = mel.clone();
        for c in &self.convs {
            x = c.forward(&x)?.elu(1.0)?;
        }
        let x = x.transpose(1, 2)?.contiguous()?;
        Ok(self.classifier.forward(&x)?.squeeze(D::Minus1)?.abs()?)
    }
}

/// Linear interpolation matching `F.interpolate(mode='linear',
/// align_corners=False)` for an arbitrary output length.
fn interp_linear(input: &[f32], out_len: usize) -> Vec<f32> {
    let in_len = input.len();
    let scale = in_len as f64 / out_len as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(in_len - 1);
        let frac = (pos - i0 as f64) as f32;
        out.push(input[i0.min(in_len - 1)] * (1.0 - frac) + input[i1] * frac);
    }
    out
}

pub struct HiftGenerator {
    f0_predictor: F0Predictor,
    l_linear_w: Vec<f32>, // [9]
    l_linear_b: f32,
    conv_pre: Conv1d,
    ups: Vec<(Tensor, Tensor, usize, usize)>, // (weight, bias, stride, pad)
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    window: Vec<f32>, // hann 16 (periodic)
    device: candle_core::Device,
}

impl HiftGenerator {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let lw = vb.get((1, HARMONICS), "m_source.l_linear.weight")?;
        let lb = vb.get(1, "m_source.l_linear.bias")?;
        let k7 = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };

        let up_specs = [
            (512usize, 256usize, 16usize, 8usize),
            (256, 128, 11, 5),
            (128, 64, 7, 3),
        ];
        let mut ups = Vec::new();
        for (i, (ic, oc, k, s)) in up_specs.iter().enumerate() {
            ups.push((
                vb.get((*ic, *oc, *k), &format!("ups.{i}.weight"))?,
                vb.get(*oc, &format!("ups.{i}.bias"))?,
                *s,
                (*k - *s) / 2,
            ));
        }

        // source_downs: stride 15 (k30), stride 3 (k6), 1×1
        let sd_specs = [
            (18usize, 256usize, 30usize, 15usize, 7usize),
            (18, 128, 6, 3, 1),
            (18, 64, 1, 1, 0),
        ];
        let mut source_downs = Vec::new();
        for (i, (ic, oc, k, s, p)) in sd_specs.iter().enumerate() {
            let cfg = Conv1dConfig {
                padding: *p,
                stride: *s,
                ..Default::default()
            };
            source_downs.push(conv1d(
                *ic,
                *oc,
                *k,
                cfg,
                vb.pp(format!("source_downs.{i}")),
            )?);
        }

        let src_rb_specs = [(256usize, 7usize), (128, 7), (64, 11)];
        let mut source_resblocks = Vec::new();
        for (i, (ch, k)) in src_rb_specs.iter().enumerate() {
            source_resblocks.push(ResBlock::load(
                vb.pp(format!("source_resblocks.{i}")),
                *ch,
                *k,
                &[1, 3, 5],
            )?);
        }

        let mut resblocks = Vec::new();
        for i in 0..3usize {
            let ch = 512 >> (i + 1);
            for (j, k) in [3usize, 7, 11].iter().enumerate() {
                resblocks.push(ResBlock::load(
                    vb.pp(format!("resblocks.{}", i * 3 + j)),
                    ch,
                    *k,
                    &[1, 3, 5],
                )?);
            }
        }

        let window: Vec<f32> = (0..N_FFT)
            .map(|i| {
                let v = (std::f32::consts::PI * i as f32 / N_FFT as f32).sin();
                v * v
            })
            .collect();

        Ok(Self {
            f0_predictor: F0Predictor::load(vb.pp("f0_predictor"))?,
            l_linear_w: lw.flatten_all()?.to_vec1::<f32>()?,
            l_linear_b: lb.flatten_all()?.to_vec1::<f32>()?[0],
            conv_pre: conv1d(80, 512, 7, k7, vb.pp("conv_pre"))?,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post: conv1d(64, N_FFT + 2, 7, k7, vb.pp("conv_post"))?,
            window,
            device,
        })
    }

    /// NSF harmonic source from per-frame f0 (SineGen2 semantics).
    /// Returns the merged source `s`, length `T · 480`.
    fn source(&self, f0: &[f32], mode: NoiseMode) -> Vec<f32> {
        let t = f0.len();
        let l = t * UPSAMPLE;
        let mut rng = Rng(0x9E3779B97F4A7C15);
        // nearest ×480
        let mut f0_up = vec![0f32; l];
        for i in 0..l {
            f0_up[i] = f0[i / UPSAMPLE];
        }
        let mut harmonics = vec![0f32; l * HARMONICS];
        for h in 0..HARMONICS {
            // rad at 24 kHz
            let mut rad: Vec<f32> = f0_up
                .iter()
                .map(|f| (f * (h + 1) as f32 / SR).fract())
                .collect();
            let rand_ini = if h == 0 || mode == NoiseMode::Deterministic {
                0.0
            } else {
                rng.uniform()
            };
            rad[0] += rand_ini;
            // ↓ linear ×(1/480) → cumsum → ×480, linear ×480 (SineGen2)
            let rad_down = interp_linear(&rad, t);
            let mut phase = vec![0f32; t];
            let mut acc = 0f64;
            for (i, r) in rad_down.iter().enumerate() {
                acc += *r as f64;
                phase[i] = (acc * 2.0 * std::f64::consts::PI) as f32;
            }
            let scaled: Vec<f32> = phase.iter().map(|p| p * UPSAMPLE as f32).collect();
            let phase_up = interp_linear(&scaled, l);
            for i in 0..l {
                harmonics[i * HARMONICS + h] = phase_up[i].sin() * SINE_AMP;
            }
        }
        // uv gating + noise + harmonic merge (tanh(linear))
        let mut s = vec![0f32; l];
        for i in 0..l {
            let uv = if f0_up[i] > VOICED_THRESHOLD {
                1.0f32
            } else {
                0.0
            };
            let noise_amp = uv * NOISE_STD + (1.0 - uv) * SINE_AMP / 3.0;
            let mut acc = self.l_linear_b;
            for h in 0..HARMONICS {
                let noise = match mode {
                    NoiseMode::Deterministic => 0.0,
                    NoiseMode::Random => noise_amp * rng.gaussian(),
                };
                let v = harmonics[i * HARMONICS + h] * uv + noise;
                acc += self.l_linear_w[h] * v;
            }
            s[i] = acc.tanh();
        }
        s
    }

    /// STFT16 (center, reflect pad, hann) → `[1, 18, F]` (re ⧺ im).
    fn stft_source(&self, s: &[f32]) -> Result<Tensor> {
        let pad = N_FFT / 2;
        let mut x = Vec::with_capacity(s.len() + 2 * pad);
        for i in (1..=pad).rev() {
            x.push(s[i.min(s.len() - 1)]);
        }
        x.extend_from_slice(s);
        for i in 2..=pad + 1 {
            x.push(s[s.len().saturating_sub(i)]);
        }
        let frames = 1 + (x.len() - N_FFT) / HOP;
        let bins = N_FFT / 2 + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut out = vec![0f32; 2 * bins * frames];
        let mut buf = vec![Complex32::default(); N_FFT];
        for f in 0..frames {
            for i in 0..N_FFT {
                buf[i] = Complex32::new(x[f * HOP + i] * self.window[i], 0.0);
            }
            fft.process(&mut buf);
            for b in 0..bins {
                out[b * frames + f] = buf[b].re;
                out[(bins + b) * frames + f] = buf[b].im;
            }
        }
        Ok(Tensor::from_vec(out, (1, 2 * bins, frames), &self.device)?)
    }

    /// Inverse of the 18-channel head: 9 log-magnitudes + 9 phases → audio.
    fn istft(&self, mag: &[f32], phase: &[f32], frames: usize) -> Vec<f32> {
        let bins = N_FFT / 2 + 1;
        // precompute irfft tables
        let mut cos_t = vec![0f32; bins * N_FFT];
        let mut sin_t = vec![0f32; bins * N_FFT];
        for k in 0..bins {
            for n in 0..N_FFT {
                let ang = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / N_FFT as f32;
                cos_t[k * N_FFT + n] = ang.cos();
                sin_t[k * N_FFT + n] = ang.sin();
            }
        }
        let full = (frames - 1) * HOP + N_FFT;
        let mut y = vec![0f32; full];
        let mut norm = vec![0f32; full];
        let mut frame = [0f32; N_FFT];
        for f in 0..frames {
            for n in 0..N_FFT {
                let mut acc = 0f32;
                for k in 0..bins {
                    let m = mag[k * frames + f];
                    let p = phase[k * frames + f];
                    let re = m * p.cos();
                    let im = m * p.sin();
                    let w = if k == 0 || k == bins - 1 { 1.0 } else { 2.0 };
                    acc += w * (re * cos_t[k * N_FFT + n] - im * sin_t[k * N_FFT + n]);
                }
                frame[n] = acc / N_FFT as f32;
            }
            for n in 0..N_FFT {
                y[f * HOP + n] += frame[n] * self.window[n];
                norm[f * HOP + n] += self.window[n] * self.window[n];
            }
        }
        let pad = N_FFT / 2;
        let out_len = full - 2 * pad;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let n = norm[pad + i];
            out.push(if n > 1e-11 {
                y[pad + i] / n
            } else {
                y[pad + i]
            });
        }
        out
    }

    /// Compute f0 from mel (exposed for the streaming source cache).
    pub fn f0(&self, mel: &Tensor) -> Result<Vec<f32>> {
        Ok(self
            .f0_predictor
            .forward(mel)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    /// NSF source for a given f0 track (length `T·480`).
    pub fn nsf_source(&self, f0: &[f32], mode: NoiseMode) -> Vec<f32> {
        self.source(f0, mode)
    }

    /// Vocode `mel` `[1, 80, T]` with an explicit source `s` (length `T·480`)
    /// → audio `T·480` samples @ 24 kHz.
    pub fn decode(&self, mel: &Tensor, s: &[f32]) -> Result<Vec<f32>> {
        let s_stft = self.stft_source(s)?;
        let mut x = self.conv_pre.forward(mel)?;
        for i in 0..self.ups.len() {
            x = leaky_relu(&x, 0.1)?;
            let (w, b, stride, pad) = &self.ups[i];
            x = x.conv_transpose1d(w, *pad, 0, *stride, 1, 1)?;
            x = x.broadcast_add(&b.reshape((1, (), 1))?)?;
            if i == self.ups.len() - 1 {
                // ReflectionPad1d((1, 0))
                let first = x.narrow(D::Minus1, 1, 1)?;
                x = Tensor::cat(&[&first, &x], D::Minus1)?;
            }
            let si = self.source_downs[i].forward(&s_stft)?;
            let si = self.source_resblocks[i].forward(&si)?;
            x = x.add(&si)?;
            let mut xs: Option<Tensor> = None;
            for j in 0..3 {
                let r = self.resblocks[i * 3 + j].forward(&x)?;
                xs = Some(match xs {
                    Some(acc) => acc.add(&r)?,
                    None => r,
                });
            }
            x = (xs.unwrap() / 3.0)?;
        }
        let x = leaky_relu(&x, 0.01)?;
        let x = self.conv_post.forward(&x)?;
        let bins = N_FFT / 2 + 1;
        let frames = x.dim(D::Minus1)?;
        let flat = x.flatten_all()?.to_vec1::<f32>()?;
        let mut mag = vec![0f32; bins * frames];
        let mut phase = vec![0f32; bins * frames];
        for k in 0..bins {
            for f in 0..frames {
                mag[k * frames + f] = flat[k * frames + f].exp().min(1e2);
                phase[k * frames + f] = flat[(bins + k) * frames + f].sin();
            }
        }
        let audio = self.istft(&mag, &phase, frames);
        Ok(audio
            .into_iter()
            .map(|v| v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT))
            .collect())
    }

    /// Full vocode: mel `[1, 80, T]` → (audio `T·480`, source `s`).
    pub fn vocode(&self, mel: &Tensor, mode: NoiseMode) -> Result<(Vec<f32>, Vec<f32>)> {
        let f0 = self.f0(mel)?;
        let s = self.source(&f0, mode);
        let audio = self.decode(mel, &s)?;
        Ok((audio, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn ckpt(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt")
            .join(name);
        p.exists().then_some(p)
    }

    fn fixture() -> Option<HashMap<String, Tensor>> {
        candle_core::safetensors::load(ckpt("cosyvoice_e2e_fixture.safetensors")?, &Device::Cpu)
            .ok()
    }

    fn load_hift() -> Option<HiftGenerator> {
        let w = ckpt("cosyvoice_hift.safetensors")?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &Device::Cpu).unwrap() };
        Some(HiftGenerator::load(vb).unwrap())
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn f0_matches_official() {
        let (Some(fx), Some(h)) = (fixture(), load_hift()) else {
            return;
        };
        let f0 = h.f0(&fx["cfm_mel"]).unwrap();
        let want = fx["hift_f0"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let d = max_abs(&f0, &want);
        assert!(d < 1e-2, "f0 max abs diff {d}");
    }

    #[test]
    fn source_matches_official() {
        let (Some(fx), Some(h)) = (fixture(), load_hift()) else {
            return;
        };
        let want = fx["hift_source"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let f0 = fx["hift_f0"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let s = h.nsf_source(&f0, NoiseMode::Deterministic);
        let d = max_abs(&s, &want);
        assert!(d < 1e-3, "source max abs diff {d}");
    }

    #[test]
    fn audio_matches_official() {
        let (Some(fx), Some(h)) = (fixture(), load_hift()) else {
            return;
        };
        let s = fx["hift_source"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let audio = h.decode(&fx["cfm_mel"], &s).unwrap();
        let want = fx["hift_audio"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let d = max_abs(&audio, &want);
        let corr = {
            let dot: f32 = audio.iter().zip(&want).map(|(a, b)| a * b).sum();
            let na: f32 = audio.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nb: f32 = want.iter().map(|b| b * b).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        assert!(
            corr > 0.999 && d < 2e-2,
            "audio corr {corr}, max abs diff {d}"
        );
    }
}

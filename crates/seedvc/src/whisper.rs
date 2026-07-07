//! Whisper-small encoder as Seed-VC's content feature extractor
//! (`semantic_fn`): the HF `WhisperFeatureExtractor` log-mel front-end
//! (80 bins, n_fft 400, hop 160, pad/trim to 30 s) and the encoder
//! (conv1 → conv2 stride 2 → +sinusoidal positions → 12 pre-norm
//! transformer layers, 768 dim / 12 heads → final LayerNorm), trimmed
//! to `n_samples / 320 + 1` frames (50 Hz).

use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::{conv1d, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};
use rustfft::{num_complex::Complex64, FftPlanner};

use crate::Result;

const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;
const SR: usize = 16_000;
const CHUNK: usize = 30 * SR;
const DIM: usize = 768;
const HEADS: usize = 12;
const LAYERS: usize = 12;

fn hz_to_mel(f: f64) -> f64 {
    if f < 1_000.0 {
        f * 3.0 / 200.0
    } else {
        15.0 + (f / 1_000.0).ln() * (27.0 / (6.4f64).ln())
    }
}

fn mel_to_hz(m: f64) -> f64 {
    if m < 15.0 {
        m * 200.0 / 3.0
    } else {
        1_000.0 * ((m - 15.0) * (6.4f64).ln() / 27.0).exp()
    }
}

/// The 80-bin log-mel of `WhisperFeatureExtractor`: Slaney filters up
/// to 8 kHz, power spectrum, `log10(clamp(_, 1e-10))`, dynamic-range
/// clamp to `max − 8`, then `(x + 4) / 4`. Input is zero-padded (or
/// trimmed) to exactly 30 s → 3000 frames.
pub fn log_mel(wave16k: &[f32]) -> Vec<Vec<f32>> {
    let mut y = vec![0f64; CHUNK];
    for (d, s) in y.iter_mut().zip(wave16k) {
        *d = *s as f64;
    }
    // Reflect pad n_fft/2 both sides (torch.stft center=True).
    let pad = N_FFT / 2;
    let mut p = Vec::with_capacity(y.len() + 2 * pad);
    for i in (1..=pad).rev() {
        p.push(y[i]);
    }
    p.extend_from_slice(&y);
    for i in 2..=pad + 1 {
        p.push(y[y.len() - i]);
    }
    let frames = CHUNK / HOP; // 3000 (the extractor drops the last frame)
    let window: Vec<f64> = (0..N_FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
        .collect();
    // Slaney filterbank 0..8 kHz.
    let n_bins = N_FFT / 2 + 1;
    let (mlo, mhi) = (hz_to_mel(0.0), hz_to_mel(8_000.0));
    let pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mlo + (mhi - mlo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = vec![Complex64::new(0.0, 0.0); N_FFT];
    let mut mel = vec![vec![0f32; frames]; N_MELS];
    let mut power = vec![0f64; n_bins];
    for t in 0..frames {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex64::new(p[t * HOP + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for (k, pw) in power.iter_mut().enumerate() {
            *pw = buf[k].norm_sqr();
        }
        for m in 0..N_MELS {
            let (lo, ctr, hi) = (pts[m], pts[m + 1], pts[m + 2]);
            let enorm = 2.0 / (hi - lo);
            let mut e = 0f64;
            for (k, pw) in power.iter().enumerate() {
                let f = k as f64 * SR as f64 / N_FFT as f64;
                let w = ((f - lo) / (ctr - lo)).min((hi - f) / (hi - ctr)).max(0.0);
                e += w * enorm * pw;
            }
            mel[m][t] = (e.max(1e-10)).log10() as f32;
        }
    }
    let mx = mel
        .iter()
        .flat_map(|r| r.iter())
        .fold(f32::MIN, |a, &b| a.max(b));
    for r in mel.iter_mut() {
        for v in r.iter_mut() {
            *v = (v.max(mx - 8.0) + 4.0) / 4.0;
        }
    }
    mel
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let hd = DIM / HEADS;
        let shape = (b, t, HEADS, hd);
        let q = (self.q.forward(x)? * (hd as f64).powf(-0.5))?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.k.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let att = q.matmul(&k.transpose(2, 3)?)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, DIM))?;
        Ok(self.o.forward(&y)?)
    }
}

struct Layer {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    pos: Tensor,
    layers: Vec<Layer>,
    ln_post: LayerNorm,
}

impl WhisperEncoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model.encoder");
        let c1 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let c2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let conv1 = conv1d(N_MELS, DIM, 3, c1, vb.pp("conv1"))?;
        let conv2 = conv1d(DIM, DIM, 3, c2, vb.pp("conv2"))?;
        let pos = vb.pp("embed_positions").get((1_500, DIM), "weight")?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let lb = vb.pp(format!("layers.{i}"));
            layers.push(Layer {
                ln1: layer_norm(DIM, 1e-5, lb.pp("self_attn_layer_norm"))?,
                attn: Attention {
                    q: linear(DIM, DIM, lb.pp("self_attn.q_proj"))?,
                    k: linear_no_bias(DIM, DIM, lb.pp("self_attn.k_proj"))?,
                    v: linear(DIM, DIM, lb.pp("self_attn.v_proj"))?,
                    o: linear(DIM, DIM, lb.pp("self_attn.out_proj"))?,
                },
                ln2: layer_norm(DIM, 1e-5, lb.pp("final_layer_norm"))?,
                fc1: linear(DIM, 3_072, lb.pp("fc1"))?,
                fc2: linear(3_072, DIM, lb.pp("fc2"))?,
            });
        }
        let ln_post = layer_norm(DIM, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            pos,
            layers,
            ln_post,
        })
    }

    /// 16 kHz samples → `[1, n/320 + 1, 768]` content features.
    pub fn forward(&self, wave16k: &[f32], device: &candle_core::Device) -> Result<Tensor> {
        let mel = log_mel(wave16k);
        let frames = mel[0].len();
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let x = Tensor::from_vec(flat, (1, N_MELS, frames), device)?;
        let x = self.conv1.forward(&x)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let mut x = x.transpose(1, 2)?; // [1, 1500, 768]
        x = x.broadcast_add(&self.pos.unsqueeze(0)?)?;
        for l in &self.layers {
            let a = l.attn.forward(&l.ln1.forward(&x)?)?;
            x = (x + a)?;
            let f = l
                .fc2
                .forward(&l.fc1.forward(&l.ln2.forward(&x)?)?.gelu_erf()?)?;
            x = (x + f)?;
        }
        let x = self.ln_post.forward(&x)?;
        let keep = wave16k.len() / 320 + 1;
        Ok(x.i((.., ..keep, ..))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn whisper_features_match_official() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
        let (w, f) = (
            dir.join("seedvc_whisper.safetensors"),
            dir.join("seedvc_whisper_fixture.safetensors"),
        );
        if !w.exists() || !f.exists() {
            return;
        }
        let dev = Device::Cpu;
        let fx = candle_core::safetensors::load(f, &dev).unwrap();
        let wave: Vec<f32> = fx["wave16k"].i(0).unwrap().to_vec1().unwrap();

        // Front-end golden.
        let mel = log_mel(&wave);
        let want_feat = fx["input_features"].i(0).unwrap().to_vec2::<f32>().unwrap();
        let mut dfeat = 0f32;
        for (gr, wr) in mel.iter().zip(&want_feat) {
            for (g, v) in gr.iter().zip(wr) {
                dfeat = dfeat.max((g - v).abs());
            }
        }
        println!("whisper mel max abs {dfeat:.2e}");
        assert!(dfeat < 2e-4, "front-end mismatch: {dfeat}");

        // Encoder golden.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &dev).unwrap() };
        let enc = WhisperEncoder::load(vb).unwrap();
        let got = enc.forward(&wave, &dev).unwrap();
        let want = &fx["s_alt"];
        let d = (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("whisper encoder max abs {d:.2e}");
        assert!(d < 2e-3, "encoder mismatch: {d}");
    }
}

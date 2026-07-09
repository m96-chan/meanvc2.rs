//! Flow-matching converter: `FlowMatchingTransformer` + `DiffLlama`
//! (`models/vc/flow_matching_transformer/{fmt_model.py,llama_nar.py}`).
//!
//! `DiffLlama` is a non-causal Llama variant: RoPE + standard MHA/SwiGLU
//! like stock Llama, but every LayerNorm is replaced by
//! `LlamaAdaptiveRMSNorm` conditioned on the **diffusion timestep**
//! (not on the content/cond tokens — those are added once, at the
//! input, as a bias on the mel embedding). Full bidirectional
//! attention (the "NAR" in the name): no causal mask, only an
//! all-valid padding mask which is a no-op for our single-utterance,
//! unpadded, batch=1 inference.
//!
//! CFM sampling (`reverse_diffusion`) follows the official Euler
//! integrator with rescaled classifier-free guidance.

use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};

use vc_core::Result;

#[derive(Debug, Clone)]
pub struct FmtConfig {
    pub mel_dim: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub cond_codebook_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for FmtConfig {
    fn default() -> Self {
        Self {
            mel_dim: 128,
            hidden_size: 1024,
            num_heads: 16,
            num_layers: 16,
            cond_codebook_size: 8192,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        }
    }
}

/// `Linear -> SiLU -> Linear` (the `diff_step_mlp`/`cond_mlp`/`mel_mlp`/
/// `mel_out_mlp` pattern, nn.Sequential indices 0 and 2).
struct Mlp3 {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp3 {
    fn new(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(in_dim, hidden, vb.pp("0"))?,
            fc2: linear(hidden, out_dim, vb.pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.silu()?)
    }
}

/// `LlamaAdaptiveRMSNorm`: RMSNorm with a data-dependent scale (no
/// shift) predicted from a conditioning vector — here always the
/// diffusion-timestep embedding.
struct AdaRmsNorm {
    to_weight: Linear,
    eps: f64,
}

impl AdaRmsNorm {
    fn new(hidden: usize, dim_cond: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            to_weight: linear(dim_cond, hidden, vb.pp("to_weight"))?,
            eps,
        })
    }

    /// `x`: `[b, t, hidden]`, `cond`: `[b, hidden]`.
    fn forward(&self, x: &Tensor, cond: &Tensor) -> candle_core::Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(2)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight = self.to_weight.forward(cond)?.unsqueeze(1)?; // [b, 1, hidden]
        normed.broadcast_mul(&weight)
    }
}

/// Standard Llama SwiGLU MLP, no bias.
struct LlamaMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl LlamaMlp {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            gate: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        self.down.forward(&h)
    }
}

/// HF `rotate_half`-convention RoPE (NOT the interleaved/gpt-fast
/// convention used elsewhere in this workspace): `cos`/`sin` cover the
/// full head dim (each of the `d/2` frequencies duplicated into both
/// halves), and `rotate_half(x) = cat(-x[d/2:], x[:d/2])`.
fn rope_tables(
    t_len: usize,
    head_dim: usize,
    theta: f64,
    dev: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let mut cos = Vec::with_capacity(t_len * head_dim);
    let mut sin = Vec::with_capacity(t_len * head_dim);
    for p in 0..t_len {
        let angles: Vec<f32> = inv_freq.iter().map(|&f| p as f32 * f).collect();
        for &a in &angles {
            cos.push(a.cos());
        }
        for &a in &angles {
            cos.push(a.cos());
        }
        for &a in &angles {
            sin.push(a.sin());
        }
        for &a in &angles {
            sin.push(a.sin());
        }
    }
    let cos = Tensor::from_vec(cos, (t_len, head_dim), dev)?;
    let sin = Tensor::from_vec(sin, (t_len, head_dim), dev)?;
    Ok((cos, sin))
}

fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)?;
    let x1 = x.narrow(candle_core::D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(candle_core::D::Minus1, d / 2, d - d / 2)?;
    Tensor::cat(&[x2.neg()?, x1], candle_core::D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    // x: [b, h, t, d]; cos/sin: [t, d] -> broadcast as [1, 1, t, d].
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

struct SelfAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttn {
    fn new(hidden: usize, num_heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            q: linear_no_bias(hidden, hidden, vb.pp("q_proj"))?,
            k: linear_no_bias(hidden, hidden, vb.pp("k_proj"))?,
            v: linear_no_bias(hidden, hidden, vb.pp("v_proj"))?,
            o: linear_no_bias(hidden, hidden, vb.pp("o_proj"))?,
            num_heads,
            head_dim: hidden / num_heads,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let shape = (b, t, self.num_heads, self.head_dim);
        let q = self
            .q
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
        self.o.forward(&out)
    }
}

/// `LlamaNARDecoderLayer`: pre-AdaRMSNorm attention + pre-AdaRMSNorm
/// SwiGLU MLP, both conditioned on the diffusion-timestep embedding.
struct NarLayer {
    input_ln: AdaRmsNorm,
    attn: SelfAttn,
    post_attn_ln: AdaRmsNorm,
    mlp: LlamaMlp,
}

impl NarLayer {
    fn new(cfg: &FmtConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            input_ln: AdaRmsNorm::new(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            attn: SelfAttn::new(cfg.hidden_size, cfg.num_heads, vb.pp("self_attn"))?,
            post_attn_ln: AdaRmsNorm::new(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: LlamaMlp::new(cfg.hidden_size, cfg.hidden_size * 4, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        step: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let residual = x;
        let h = self.input_ln.forward(x, step)?;
        let h = self.attn.forward(&h, cos, sin)?;
        let x = (residual + h)?;

        let residual = &x;
        let h = self.post_attn_ln.forward(&x, step)?;
        let h = self.mlp.forward(&h)?;
        residual + h
    }
}

pub struct DiffLlama {
    layers: Vec<NarLayer>,
    final_norm: AdaRmsNorm,
    diff_step_mlp: Mlp3,
    cond_mlp: Mlp3,
    mel_mlp: Mlp3,
    mel_out_mlp: Mlp3,
    hidden_size: usize,
    num_heads: usize,
    rope_theta: f64,
}

impl DiffLlama {
    fn new(cfg: &FmtConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(NarLayer::new(cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            layers,
            final_norm: AdaRmsNorm::new(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("norm"),
            )?,
            diff_step_mlp: Mlp3::new(
                cfg.hidden_size,
                cfg.hidden_size * 4,
                cfg.hidden_size,
                vb.pp("diff_step_mlp"),
            )?,
            cond_mlp: Mlp3::new(
                cfg.hidden_size,
                cfg.hidden_size * 4,
                cfg.hidden_size,
                vb.pp("cond_mlp"),
            )?,
            mel_mlp: Mlp3::new(
                cfg.mel_dim,
                cfg.hidden_size * 4,
                cfg.hidden_size,
                vb.pp("mel_mlp"),
            )?,
            mel_out_mlp: Mlp3::new(
                cfg.hidden_size,
                cfg.hidden_size * 4,
                cfg.mel_dim,
                vb.pp("mel_out_mlp"),
            )?,
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_heads,
            rope_theta: cfg.rope_theta,
        })
    }

    /// `SinusoidalPosEmb`: note the official implementation concatenates
    /// `[sin, cos]` (sin first), not the usual `[cos, sin]`.
    fn diff_step_embedding(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let dim = self.hidden_size;
        let half = dim / 2;
        let dev = t.device();
        let scale = (10_000f64.ln() / (half as f64 - 1.0)) as f32;
        let freqs: Vec<f32> = (0..half).map(|i| (-(i as f32) * scale).exp()).collect();
        let freqs = Tensor::from_vec(freqs, half, dev)?;
        let t = t.unsqueeze(1)?; // [b, 1]
        let angles = t.broadcast_mul(&freqs.unsqueeze(0)?)?; // [b, half]
        Tensor::cat(&[angles.sin()?, angles.cos()?], 1)
    }

    /// `x`: `[b, t, mel_dim]`, `diffusion_step`: `[b]`, `cond`: `[b, t, hidden]`.
    pub fn forward(
        &self,
        x: &Tensor,
        diffusion_step: &Tensor,
        cond: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let cond_embedding = self.cond_mlp.forward(cond)?;
        let x = self.mel_mlp.forward(x)?;
        let step_emb = self.diff_step_embedding(diffusion_step)?;
        let step = self.diff_step_mlp.forward(&step_emb)?; // [b, hidden]
        let mut x = (x + cond_embedding)?;

        let t_len = x.dim(1)?;
        let head_dim = self.hidden_size / self.num_heads;
        let (cos, sin) = rope_tables(t_len, head_dim, self.rope_theta, x.device())?;

        for layer in &self.layers {
            x = layer.forward(&x, &step, &cos, &sin)?;
        }
        let x = self.final_norm.forward(&x, &step)?;
        self.mel_out_mlp.forward(&x)
    }
}

pub struct FlowMatchingTransformer {
    cond_emb: Embedding,
    pub diff_estimator: DiffLlama,
}

impl FlowMatchingTransformer {
    pub fn new(cfg: &FmtConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            cond_emb: embedding(cfg.cond_codebook_size, cfg.hidden_size, vb.pp("cond_emb"))?,
            diff_estimator: DiffLlama::new(cfg, vb.pp("diff_estimator"))?,
        })
    }

    pub fn load<P: AsRef<std::path::Path>>(
        cfg: &FmtConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb).map_err(Into::into)
    }

    /// `codes`: `[b, t]` int64 token indices. Returns `[b, t, hidden]`.
    pub fn cond_embed(&self, codes: &Tensor) -> Result<Tensor> {
        self.cond_emb.forward(codes).map_err(Into::into)
    }

    /// CFM Euler sampling with rescaled classifier-free guidance,
    /// matching `FlowMatchingTransformer.reverse_diffusion` exactly
    /// (`cfg=1.0`, `rescale_cfg=0.75` are the official inference
    /// defaults used by every `inference_fm`/`inference_ar_and_fm`
    /// call site).
    ///
    /// `cond`: `[1, prompt_len + target_len, hidden]` (already through
    /// `cond_embed`), `prompt`: `[1, prompt_len, mel_dim]`, `noise`:
    /// the pre-sampled initial `[1, target_len, mel_dim]` (pass a fixed
    /// tensor for golden-fixture reproducibility; `None` to sample
    /// fresh Gaussian noise).
    pub fn reverse_diffusion(
        &self,
        cond: &Tensor,
        prompt: &Tensor,
        noise: Tensor,
        n_timesteps: usize,
        cfg_scale: f64,
        rescale_cfg: f64,
    ) -> Result<Tensor> {
        let dev = cond.device();
        let prompt_len = prompt.dim(1)?;
        let target_len = cond.dim(1)? - prompt_len;
        let h = 1.0 / n_timesteps as f64;
        let mut xt = noise;

        for i in 0..n_timesteps {
            let xt_input = Tensor::cat(&[prompt, &xt], 1)?;
            let t_val = (i as f64 + 0.5) * h;
            let t = Tensor::full(t_val as f32, 1, dev)?;
            let full_out = self.diff_estimator.forward(&xt_input, &t, cond)?;
            let mut flow_pred = full_out.narrow(1, prompt_len, target_len)?;

            if cfg_scale > 0.0 {
                let cond_target = cond.narrow(1, prompt_len, target_len)?;
                let uncond = cond_target.zeros_like()?;
                let uncond_flow_pred = self.diff_estimator.forward(&xt, &t, &uncond)?;
                let pos_std = std_all(&flow_pred)?;
                let flow_pred_cfg =
                    (&flow_pred + ((&flow_pred - &uncond_flow_pred)? * cfg_scale)?)?;
                let cfg_std = std_all(&flow_pred_cfg)?;
                let rescaled = (&flow_pred_cfg * (pos_std / cfg_std))?;
                flow_pred = ((&rescaled * rescale_cfg)? + (&flow_pred_cfg * (1.0 - rescale_cfg))?)?;
            }

            let dxt = (flow_pred * h)?;
            xt = (&xt + dxt)?;
        }
        Ok(xt)
    }
}

fn std_all(x: &Tensor) -> candle_core::Result<f64> {
    let n = x.elem_count() as f64;
    let mean = (x.sum_all()?.to_scalar::<f32>()? as f64) / n;
    let var: f64 = x
        .to_dtype(DType::F64)?
        .broadcast_sub(&Tensor::new(mean, x.device())?)?
        .sqr()?
        .sum_all()?
        .to_scalar::<f64>()?
        / n;
    Ok(var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_e2e_fixture.safetensors");
        if !path.exists() {
            return None;
        }
        Some(candle_core::safetensors::load(path, &Device::Cpu).unwrap())
    }

    fn ckpt() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_fmt.safetensors");
        path.exists().then_some(path)
    }

    fn corr(a: &Tensor, b: &Tensor) -> f64 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    #[test]
    fn diff_estimator_cond_branch_matches_official() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let cfg = FmtConfig::default();
        let model = FlowMatchingTransformer::load(&cfg, ckpt, &dev).unwrap();

        let x = &fx["diff_estimator_cond_x"];
        let t = &fx["diff_estimator_cond_t"];
        let cond = &fx["diff_estimator_cond_cond"];
        let want = &fx["diff_estimator_cond_out"];

        let got = model.diff_estimator.forward(x, t, cond).unwrap();
        assert_eq!(got.dims(), want.dims());
        let c = corr(&got, want);
        assert!(c > 0.999, "correlation {c}");
    }

    #[test]
    fn diff_estimator_uncond_branch_matches_official() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let cfg = FmtConfig::default();
        let model = FlowMatchingTransformer::load(&cfg, ckpt, &dev).unwrap();

        let x = &fx["diff_estimator_uncond_x"];
        let t = &fx["diff_estimator_uncond_t"];
        let cond = &fx["diff_estimator_uncond_cond"];
        let want = &fx["diff_estimator_uncond_out"];

        let got = model.diff_estimator.forward(x, t, cond).unwrap();
        let c = corr(&got, want);
        assert!(c > 0.999, "correlation {c}");
    }

    #[test]
    fn reverse_diffusion_matches_official_trajectory() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let cfg = FmtConfig::default();
        let model = FlowMatchingTransformer::load(&cfg, ckpt, &dev).unwrap();

        let cond = &fx["cond_full"];
        let prompt = &fx["ref_mel"];
        let noise = fx["cfm_noise"].clone();
        let want = &fx["fm_mel"];

        let got = model
            .reverse_diffusion(cond, prompt, noise, 32, 1.0, 0.75)
            .unwrap();
        assert_eq!(got.dims(), want.dims());
        let c = corr(&got, want);
        assert!(c > 0.999, "correlation {c}");
    }
}

//! The CFM flow-matching sampler with its DiT + WaveNet estimator
//! (`modules/flow_matching.py::CFM` + `modules/diffusion_transformer.py::DiT`,
//! preset `config_dit_mel_seed_uvit_whisper_small_wavenet.yml`), 98.5 M params:
//!
//! - **Euler ODE** over `t_span = linspace(0, 1, n_steps + 1)` (the cosine
//!   schedule is commented out upstream) with classifier-free guidance as a
//!   stacked `B = 2` forward: row 0 conditional, row 1 with zeroed
//!   `prompt_x` / `style` / `mu`, combined as
//!   `(1 + rate)·cond − rate·null` (rate 0.7 at inference).
//! - **DiT estimator** (`hidden 512 × 13 layers × 8 heads`, gpt-fast style):
//!   inputs `cat([xᵀ, prompt_xᵀ, cond_projection(mu), style·1ᵀ]) →
//!   cond_x_merge_linear` (864 → 512); adaptive RMSNorm (`project_layer` on
//!   `t_embedder(t)`, `time_as_token`/`style_as_token` are **false** in this
//!   preset), interleaved RoPE (base 10⁴, head dim 64), SwiGLU FFN (1536),
//!   U-ViT skips: layers 0-5 push, layers 7-12 pop through `skip_in_linear`;
//!   final adaptive RMSNorm, then the `long_skip_connection`
//!   `skip_linear(cat([x_res, xᵀ]))` (592 → 512).
//! - **WaveNet refiner** (`final_layer_type: wavenet`): `conv1` (Linear
//!   512 → 512) → 8 × non-causal WN layers (k5, dilation 1, **reflect** pad —
//!   `SConv1d` ignores the `padding=` argument WN passes) gated by
//!   `t_embedder2(t)` through `cond_layer`, plus `res_projection(x_res)`;
//!   `final_layer` (adaLN-modulated LayerNorm + weight-norm Linear on
//!   `t_embedder(t)`) → `conv2` (1×1 conv to 80 mel bins).
//!
//! Weight-norm tensors (`x_embedder`, `final_layer.linear`, all WaveNet
//! convs) are stored raw (`weight_g`/`weight_v`) in
//! `ckpt/seedvc_dit.safetensors` and folded at load. `x_embedder`,
//! `cond_embedder`, `f0_embedder` and `content_mask_embedder` exist in the
//! checkpoint but are unused on this inference path.
//!
//! Everything is fp32; masks are all-ones for the full-length inputs used
//! here (single utterance, no padding), so attention/WN masking is a no-op
//! and omitted.

use candle_core::{Device, Module, Tensor, D};
use candle_nn::ops::{sigmoid, silu, softmax_last_dim};
use candle_nn::{linear, linear_no_bias, Conv1d, Conv1dConfig, Linear, VarBuilder};

use crate::Result;

const HIDDEN: usize = 512;
const DEPTH: usize = 13;
const N_HEADS: usize = 8;
const HEAD_DIM: usize = HIDDEN / N_HEADS;
const FFN_DIM: usize = 1536; // find_multiple(2*4*512/3, 256)
const IN_CHANNELS: usize = 80;
const CONTENT_DIM: usize = 512;
const STYLE_DIM: usize = 192;
const FREQ_EMB: usize = 256;
const ROPE_BASE: f64 = 10_000.0;
const RMS_EPS: f64 = 1e-5;
const LN_EPS: f64 = 1e-6;
const WN_LAYERS: usize = 8;
const WN_KERNEL: usize = 5;

/// Fold `weight_norm` for a Linear: `w = v · g / ‖v‖₂(dim 1)`.
fn wn_linear(vb: &VarBuilder, inp: usize, out: usize) -> Result<Linear> {
    let v = vb.get((out, inp), "weight_v")?;
    let g = vb.get((out, 1), "weight_g")?;
    let n = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    let w = v.broadcast_mul(&g.broadcast_div(&n)?)?;
    Ok(Linear::new(w, Some(vb.get(out, "bias")?)))
}

/// Fold `weight_norm` for a Conv1d (norm over dims 1..): plain conv, the
/// reflect padding of `SConv1d` is applied by the caller.
fn wn_conv1d(vb: &VarBuilder, inp: usize, out: usize, k: usize) -> Result<Conv1d> {
    let v = vb.get((out, inp, k), "weight_v")?;
    let g = vb.get((out, 1, 1), "weight_g")?;
    let n = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let w = v.broadcast_mul(&g.broadcast_div(&n)?)?;
    Ok(Conv1d::new(
        w,
        Some(vb.get(out, "bias")?),
        Conv1dConfig::default(),
    ))
}

/// `TimestepEmbedder`: sinusoidal embedding (scale 1000, max period 10⁴,
/// 256 dims, `[cos | sin]`) → Linear → SiLU → Linear.
struct TimestepEmbedder {
    l0: Linear,
    l2: Linear,
    freqs: Tensor, // [1, FREQ_EMB/2]
}

impl TimestepEmbedder {
    fn load(vb: &VarBuilder, dev: &Device) -> Result<Self> {
        let half = FREQ_EMB / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-(10_000f64).ln() * i as f64 / half as f64).exp() as f32)
            .collect();
        Ok(Self {
            l0: linear(FREQ_EMB, HIDDEN, vb.pp("mlp.0"))?,
            l2: linear(HIDDEN, HIDDEN, vb.pp("mlp.2"))?,
            freqs: Tensor::from_vec(freqs, (1, half), dev)?,
        })
    }

    /// `t` `[B]` → `[B, HIDDEN]`.
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let b = t.dim(0)?;
        let args = (t.reshape((b, 1))?.broadcast_mul(&self.freqs)? * 1000.0)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?;
        Ok(self.l2.forward(&silu(&self.l0.forward(&emb)?)?)?)
    }
}

/// `AdaptiveLayerNorm(RMSNorm)`: `w, b = project_layer(c).chunk(2)`;
/// `w · rmsnorm(x) + b` with the conditioning `c = t1` `[B, 1, D]`.
struct AdaRmsNorm {
    project: Linear,
    weight: Tensor,
}

impl AdaRmsNorm {
    fn load(vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            project: linear(HIDDEN, 2 * HIDDEN, vb.pp("project_layer"))?,
            weight: vb.get(HIDDEN, "norm.weight")?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let wb = self.project.forward(c)?; // [B, 1, 2D]
        let w = wb.narrow(D::Minus1, 0, HIDDEN)?;
        let b = wb.narrow(D::Minus1, HIDDEN, HIDDEN)?;
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = x
            .broadcast_div(&(ms + RMS_EPS)?.sqrt()?)?
            .broadcast_mul(&self.weight)?;
        Ok(xn.broadcast_mul(&w)?.broadcast_add(&b)?)
    }
}

/// Interleaved (gpt-fast) RoPE on `[B, T, H, D]` with `cos`/`sin` `[T, D/2]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    let xp = x.reshape((b, t, h, d / 2, 2))?;
    let x0 = xp.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = xp.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.reshape((1, t, 1, d / 2))?;
    let sin = sin.reshape((1, t, 1, d / 2))?;
    let o0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let o1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;
    Ok(Tensor::stack(&[o0, o1], 4)?.reshape((b, t, h, d))?)
}

/// One gpt-fast `TransformerBlock`: adaptive-RMSNorm pre-norm attention
/// (fused `wqkv`, RoPE, no mask — full-length inputs) and SwiGLU FFN, with
/// the U-ViT `skip_in_linear` on receiving layers.
struct Block {
    wqkv: Linear,
    wo: Linear,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    attn_norm: AdaRmsNorm,
    ffn_norm: AdaRmsNorm,
    skip_in: Linear,
}

impl Block {
    fn load(vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            wqkv: linear_no_bias(HIDDEN, 3 * HIDDEN, vb.pp("attention.wqkv"))?,
            wo: linear_no_bias(HIDDEN, HIDDEN, vb.pp("attention.wo"))?,
            w1: linear_no_bias(HIDDEN, FFN_DIM, vb.pp("feed_forward.w1"))?,
            w2: linear_no_bias(FFN_DIM, HIDDEN, vb.pp("feed_forward.w2"))?,
            w3: linear_no_bias(HIDDEN, FFN_DIM, vb.pp("feed_forward.w3"))?,
            attn_norm: AdaRmsNorm::load(&vb.pp("attention_norm"))?,
            ffn_norm: AdaRmsNorm::load(&vb.pp("ffn_norm"))?,
            skip_in: linear(2 * HIDDEN, HIDDEN, vb.pp("skip_in_linear"))?,
        })
    }

    fn attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self.wqkv.forward(x)?;
        let split = |off: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, off * HIDDEN, HIDDEN)?
                .reshape((b, t, N_HEADS, HEAD_DIM))?)
        };
        let q = apply_rope(&split(0)?, cos, sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_rope(&split(1)?, cos, sin)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = split(2)?.transpose(1, 2)?.contiguous()?;
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (HEAD_DIM as f64).sqrt())?;
        let att = softmax_last_dim(&att)?;
        let y = att.matmul(&v)?; // [B, H, T, Dh]
        let y = y.transpose(1, 2)?.reshape((b, t, HIDDEN))?;
        Ok(self.wo.forward(&y)?)
    }

    fn forward(
        &self,
        x: &Tensor,
        c: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        skip: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = match skip {
            Some(s) => self.skip_in.forward(&Tensor::cat(&[x, s], 2)?)?,
            None => x.clone(),
        };
        let h = (&x + self.attention(&self.attn_norm.forward(&x, c)?, cos, sin)?)?;
        let f = self.ffn_norm.forward(&h, c)?;
        let ff = self
            .w2
            .forward(&(silu(&self.w1.forward(&f)?)? * self.w3.forward(&f)?)?)?;
        Ok((h + ff)?)
    }
}

/// 13-layer non-causal transformer with U-ViT skips: layers `i < 6` push
/// their outputs, layers `i > 6` pop (LIFO: 7←5, 8←4, …, 12←0), then a final
/// adaptive RMSNorm.
struct Transformer {
    layers: Vec<Block>,
    norm: AdaRmsNorm,
}

impl Transformer {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            layers.push(Block::load(&vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            layers,
            norm: AdaRmsNorm::load(&vb.pp("norm"))?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let mut skips: Vec<Tensor> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let skip = if i > DEPTH / 2 { skips.pop() } else { None };
            x = layer.forward(&x, c, cos, sin, skip.as_ref())?;
            if i < DEPTH / 2 {
                skips.push(x.clone());
            }
        }
        self.norm.forward(&x, c)
    }
}

/// Reflect-pad 2 on the last dim (what `SConv1d` does for k5/s1/d1).
fn reflect_pad2(x: &Tensor) -> Result<Tensor> {
    let t = x.dim(2)?;
    Ok(Tensor::cat(
        &[
            x.narrow(2, 2, 1)?,
            x.narrow(2, 1, 1)?,
            x.clone(),
            x.narrow(2, t - 2, 1)?,
            x.narrow(2, t - 3, 1)?,
        ],
        2,
    )?)
}

/// Non-causal `WN` (k5, dilation 1, 8 layers, hidden 512) with the
/// timestep-conditioned gate `tanh ⊙ sigmoid` and res/skip splits.
struct WaveNet {
    cond_layer: Conv1d,
    in_layers: Vec<Conv1d>,
    res_skip: Vec<Conv1d>,
}

impl WaveNet {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let cond_layer = wn_conv1d(
            &vb.pp("cond_layer.conv.conv"),
            HIDDEN,
            2 * HIDDEN * WN_LAYERS,
            1,
        )?;
        let mut in_layers = Vec::new();
        let mut res_skip = Vec::new();
        for i in 0..WN_LAYERS {
            in_layers.push(wn_conv1d(
                &vb.pp(format!("in_layers.{i}.conv.conv")),
                HIDDEN,
                2 * HIDDEN,
                WN_KERNEL,
            )?);
            let out = if i < WN_LAYERS - 1 { 2 * HIDDEN } else { HIDDEN };
            res_skip.push(wn_conv1d(
                &vb.pp(format!("res_skip_layers.{i}.conv.conv")),
                HIDDEN,
                out,
                1,
            )?);
        }
        Ok(Self {
            cond_layer,
            in_layers,
            res_skip,
        })
    }

    /// `x` `[B, 512, T]`, `g` `[B, 512, 1]` → `[B, 512, T]`.
    fn forward(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let g = self.cond_layer.forward(g)?; // [B, 2·512·8, 1]
        let mut x = x.clone();
        let mut output = x.zeros_like()?;
        for i in 0..WN_LAYERS {
            let x_in = self.in_layers[i].forward(&reflect_pad2(&x)?)?;
            let g_l = g.narrow(1, i * 2 * HIDDEN, 2 * HIDDEN)?;
            let s = x_in.broadcast_add(&g_l)?;
            let acts = (s.narrow(1, 0, HIDDEN)?.tanh()? * sigmoid(&s.narrow(1, HIDDEN, HIDDEN)?)?)?;
            let rs = self.res_skip[i].forward(&acts)?;
            if i < WN_LAYERS - 1 {
                x = (x + rs.narrow(1, 0, HIDDEN)?)?;
                output = (output + rs.narrow(1, HIDDEN, HIDDEN)?)?;
            } else {
                output = (output + rs)?;
            }
        }
        Ok(output)
    }
}

/// `FinalLayer`: affine-free LayerNorm (eps 1e-6) modulated by
/// `adaLN_modulation(t1) = SiLU → Linear` shift/scale, then the weight-norm
/// Linear (512 → 512).
struct FinalLayer {
    linear: Linear,
    ada: Linear,
}

impl FinalLayer {
    fn load(vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: wn_linear(&vb.pp("linear"), HIDDEN, HIDDEN)?,
            ada: linear(HIDDEN, 2 * HIDDEN, vb.pp("adaLN_modulation.1"))?,
        })
    }

    /// `x` `[B, T, 512]`, `c` `[B, 512]`.
    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let sc = self.ada.forward(&silu(c)?)?;
        let shift = sc.narrow(1, 0, HIDDEN)?.unsqueeze(1)?;
        let scale = sc.narrow(1, HIDDEN, HIDDEN)?.unsqueeze(1)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xc.broadcast_div(&(var + LN_EPS)?.sqrt()?)?;
        let xm = xn
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        Ok(self.linear.forward(&xm)?)
    }
}

/// The `DiT` velocity estimator (transformer trunk + WaveNet refiner).
pub struct DitEstimator {
    cond_projection: Linear,
    cond_x_merge_linear: Linear,
    t_embedder: TimestepEmbedder,
    t_embedder2: TimestepEmbedder,
    transformer: Transformer,
    skip_linear: Linear,
    conv1: Linear,
    conv2: Conv1d,
    res_projection: Linear,
    wavenet: WaveNet,
    final_layer: FinalLayer,
    rope_freqs: Vec<f32>, // [HEAD_DIM/2]
}

impl DitEstimator {
    fn load(vb: &VarBuilder, dev: &Device) -> Result<Self> {
        let conv2 = Conv1d::new(
            vb.get((IN_CHANNELS, HIDDEN, 1), "conv2.weight")?,
            Some(vb.get(IN_CHANNELS, "conv2.bias")?),
            Conv1dConfig::default(),
        );
        // f32 like `precompute_freqs_cis` (fp32 weights → fp32 cache).
        let rope_freqs: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|i| (1.0 / ROPE_BASE.powf(2.0 * i as f64 / HEAD_DIM as f64)) as f32)
            .collect();
        Ok(Self {
            cond_projection: linear(CONTENT_DIM, HIDDEN, vb.pp("cond_projection"))?,
            cond_x_merge_linear: linear(
                HIDDEN + 2 * IN_CHANNELS + STYLE_DIM,
                HIDDEN,
                vb.pp("cond_x_merge_linear"),
            )?,
            t_embedder: TimestepEmbedder::load(&vb.pp("t_embedder"), dev)?,
            t_embedder2: TimestepEmbedder::load(&vb.pp("t_embedder2"), dev)?,
            transformer: Transformer::load(&vb.pp("transformer"))?,
            skip_linear: linear(HIDDEN + IN_CHANNELS, HIDDEN, vb.pp("skip_linear"))?,
            conv1: linear(HIDDEN, HIDDEN, vb.pp("conv1"))?,
            conv2,
            res_projection: linear(HIDDEN, HIDDEN, vb.pp("res_projection"))?,
            wavenet: WaveNet::load(&vb.pp("wavenet"))?,
            final_layer: FinalLayer::load(&vb.pp("final_layer"))?,
            rope_freqs,
        })
    }

    /// RoPE tables for positions `0..t_len` (angles accumulated in f32 like
    /// `torch.outer(arange, freqs)`).
    fn rope_tables(&self, t_len: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let half = HEAD_DIM / 2;
        let mut cos = Vec::with_capacity(t_len * half);
        let mut sin = Vec::with_capacity(t_len * half);
        for p in 0..t_len {
            for &f in &self.rope_freqs {
                let a = (p as f32 * f) as f64;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        Ok((
            Tensor::from_vec(cos, (t_len, half), dev)?,
            Tensor::from_vec(sin, (t_len, half), dev)?,
        ))
    }

    /// One velocity evaluation. `x`/`prompt_x` `[B, 80, T]`, `t` `[B]`,
    /// `style` `[B, 192]`, `cond` (mu) `[B, T, 512]` → `[B, 80, T]`.
    pub fn forward(
        &self,
        x: &Tensor,
        prompt_x: &Tensor,
        t: &Tensor,
        style: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        self.forward_impl(x, prompt_x, t, style, cond, None)
    }

    fn forward_impl(
        &self,
        x: &Tensor,
        prompt_x: &Tensor,
        t: &Tensor,
        style: &Tensor,
        cond: &Tensor,
        mut trace: Option<&mut Vec<(&'static str, Tensor)>>,
    ) -> Result<Tensor> {
        let (b, _, t_len) = x.dims3()?;
        let dev = x.device();
        let mut emit = |name: &'static str, t: &Tensor| {
            if let Some(v) = trace.as_deref_mut() {
                v.push((name, t.clone()));
            }
        };

        let t1 = self.t_embedder.forward(t)?; // [B, 512]
        emit("t1", &t1);
        let cond = self.cond_projection.forward(cond)?; // [B, T, 512]
        let xt = x.transpose(1, 2)?.contiguous()?; // [B, T, 80]
        let pt = prompt_x.transpose(1, 2)?.contiguous()?;
        let style_rep = style
            .reshape((b, 1, STYLE_DIM))?
            .broadcast_as((b, t_len, STYLE_DIM))?
            .contiguous()?;
        let x_in = Tensor::cat(&[&xt, &pt, &cond, &style_rep], 2)?;
        let x_in = self.cond_x_merge_linear.forward(&x_in)?; // [B, T, 512]
        emit("merged", &x_in);

        let c = t1.unsqueeze(1)?; // [B, 1, 512]
        let (cos, sin) = self.rope_tables(t_len, dev)?;
        let x_res = self.transformer.forward(&x_in, &c, &cos, &sin)?;
        emit("trans_out", &x_res);
        // long_skip_connection
        let x_res = self.skip_linear.forward(&Tensor::cat(&[&x_res, &xt], 2)?)?;
        emit("skip_out", &x_res);

        let h = self.conv1.forward(&x_res)?;
        emit("conv1_out", &h);
        let h = h.transpose(1, 2)?.contiguous()?; // [B, 512, T]
        let t2 = self.t_embedder2.forward(t)?; // [B, 512]
        let wn = self.wavenet.forward(&h, &t2.unsqueeze(2)?)?;
        emit("wavenet_out", &wn);
        let xw = (wn.transpose(1, 2)?.contiguous()? + self.res_projection.forward(&x_res)?)?;
        let fl = self.final_layer.forward(&xw, &t1)?;
        emit("final_out", &fl);
        let out = self.conv2.forward(&fl.transpose(1, 2)?.contiguous()?)?;
        Ok(out)
    }
}

/// The CFM wrapper: Euler ODE sampler with classifier-free guidance
/// (`BASECFM.inference` / `solve_euler`).
pub struct Cfm {
    pub estimator: DitEstimator,
}

impl Cfm {
    /// `vb` over `ckpt/seedvc_dit.safetensors` (keys `module.estimator.*`).
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let dev = vb.device().clone();
        Ok(Self {
            estimator: DitEstimator::load(&vb.pp("module.estimator"), &dev)?,
        })
    }

    /// `solve_euler` with the exact upstream schedule and prompt handling:
    /// `mu` (= `cat([prompt_condition, cond], 1)`) `[1, T, 512]`,
    /// `prompt_mel` `[1, 80, P]`, `style` `[1, 192]`, `noise` `[1, 80, T]`
    /// (the initial `z`) → the full `[1, 80, T]` mel; the caller drops the
    /// first `P` frames (they are zeroed like upstream).
    pub fn inference(
        &self,
        mu: &Tensor,
        prompt_mel: &Tensor,
        style: &Tensor,
        noise: &Tensor,
        n_timesteps: usize,
        inference_cfg_rate: f64,
    ) -> Result<Tensor> {
        let (b, mels, t_len) = noise.dims3()?;
        let p = prompt_mel.dim(2)?;
        let dev = noise.device();
        let zero_head = Tensor::zeros((b, mels, p), noise.dtype(), dev)?;
        // x[..., :P] = 0; prompt_x[..., :P] = prompt.
        let mut x = Tensor::cat(&[&zero_head, &noise.narrow(2, p, t_len - p)?], 2)?;
        let prompt_x = Tensor::cat(
            &[
                prompt_mel.clone(),
                Tensor::zeros((b, mels, t_len - p), noise.dtype(), dev)?,
            ],
            2,
        )?;
        // CFG rows (constant across steps): row 1 is the null condition.
        let prompt2 = Tensor::cat(&[&prompt_x, &prompt_x.zeros_like()?], 0)?;
        let style2 = Tensor::cat(&[style, &style.zeros_like()?], 0)?;
        let mu2 = Tensor::cat(&[mu, &mu.zeros_like()?], 0)?;

        // t_span = linspace(0, 1, n+1); t accumulates in f32 like upstream.
        let t_span: Vec<f32> = (0..=n_timesteps)
            .map(|i| (i as f64 / n_timesteps as f64) as f32)
            .collect();
        let mut t = t_span[0];
        for step in 1..=n_timesteps {
            let dt = t_span[step] - t_span[step - 1];
            let dphi = if inference_cfg_rate > 0.0 {
                let x2 = Tensor::cat(&[&x, &x], 0)?;
                let tt = Tensor::full(t, 2 * b, dev)?;
                let out = self
                    .estimator
                    .forward(&x2, &prompt2, &tt, &style2, &mu2)?;
                let o_cond = out.narrow(0, 0, b)?;
                let o_null = out.narrow(0, b, b)?;
                ((o_cond * (1.0 + inference_cfg_rate))? - (o_null * inference_cfg_rate)?)?
            } else {
                let tt = Tensor::full(t, b, dev)?;
                self.estimator.forward(&x, &prompt_x, &tt, style, mu)?
            };
            x = (x + (dphi * dt as f64)?)?;
            t += dt;
            // x[..., :P] = 0 after every step.
            x = Tensor::cat(&[&zero_head, &x.narrow(2, p, t_len - p)?], 2)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn load_fixture(name: &str) -> Option<HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt")
            .join(name);
        if !path.exists() {
            return None;
        }
        Some(candle_core::safetensors::load(path, &Device::Cpu).unwrap())
    }

    fn load_cfm() -> Option<Cfm> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/seedvc_dit.safetensors");
        if !path.exists() {
            return None;
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu).unwrap()
        };
        Some(Cfm::load(vb).unwrap())
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// Stacked CFG inputs exactly like `solve_euler` builds them.
    fn stacked_inputs(fx: &HashMap<String, Tensor>) -> (Tensor, Tensor, Tensor, Tensor) {
        let mu = Tensor::cat(&[&fx["prompt_condition"], &fx["cond"]], 1).unwrap();
        let mel2 = &fx["mel2"];
        let p = mel2.dim(2).unwrap();
        let noise = &fx["cfm_noise"];
        let (b, mels, t_len) = noise.dims3().unwrap();
        let zero_head = Tensor::zeros((b, mels, p), DType::F32, &Device::Cpu).unwrap();
        let x = Tensor::cat(&[&zero_head, &noise.narrow(2, p, t_len - p).unwrap()], 2).unwrap();
        let prompt_x = Tensor::cat(
            &[
                mel2.clone(),
                Tensor::zeros((b, mels, t_len - p), DType::F32, &Device::Cpu).unwrap(),
            ],
            2,
        )
        .unwrap();
        let x2 = Tensor::cat(&[&x, &x], 0).unwrap();
        let prompt2 = Tensor::cat(&[&prompt_x, &prompt_x.zeros_like().unwrap()], 0).unwrap();
        let style2 =
            Tensor::cat(&[&fx["style2"], &fx["style2"].zeros_like().unwrap()], 0).unwrap();
        let mu2 = Tensor::cat(&[&mu, &mu.zeros_like().unwrap()], 0).unwrap();
        (x2, prompt2, style2, mu2)
    }

    #[test]
    fn estimator_forward_matches_official() {
        let (Some(cfm), Some(fx), Some(sfx)) = (
            load_cfm(),
            load_fixture("seedvc_e2e_fixture.safetensors"),
            load_fixture("seedvc_dit_stage_fixture.safetensors"),
        ) else {
            return;
        };
        let (x2, prompt2, style2, mu2) = stacked_inputs(&fx);
        let mut trace = Vec::new();
        let got = cfm
            .estimator
            .forward_impl(&x2, &prompt2, &sfx["t"], &style2, &mu2, Some(&mut trace))
            .unwrap();
        for (name, t) in &trace {
            if let Some(want) = sfx.get(*name) {
                println!("{name} max abs diff {:.2e}", max_abs_diff(t, want));
            }
        }
        let d = max_abs_diff(&got, &sfx["out"]);
        println!("estimator out max abs diff {d:.2e}");
        assert!(d < 1e-3, "estimator mismatch: {d}");
    }

    #[test]
    fn cfm_trajectory_matches_official() {
        let (Some(cfm), Some(fx)) = (load_cfm(), load_fixture("seedvc_e2e_fixture.safetensors"))
        else {
            return;
        };
        let mu = Tensor::cat(&[&fx["prompt_condition"], &fx["cond"]], 1).unwrap();
        let got = cfm
            .inference(&mu, &fx["mel2"], &fx["style2"], &fx["cfm_noise"], 10, 0.7)
            .unwrap();
        let p = fx["mel2"].dim(2).unwrap();
        let t_len = got.dim(2).unwrap();
        let got = got.narrow(2, p, t_len - p).unwrap();
        let want = &fx["vc_mel"];
        let d = max_abs_diff(&got, want);
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
        let n = g.len() as f64;
        let (mg, mw) = (
            g.iter().map(|&v| v as f64).sum::<f64>() / n,
            w.iter().map(|&v| v as f64).sum::<f64>() / n,
        );
        let (mut num, mut dg, mut dw) = (0f64, 0f64, 0f64);
        for (&a, &b) in g.iter().zip(&w) {
            let (a, b) = (a as f64 - mg, b as f64 - mw);
            num += a * b;
            dg += a * a;
            dw += b * b;
        }
        let corr = num / (dg.sqrt() * dw.sqrt());
        // Achieved 4.70e-4 / corr 1.000000 (CUDA-generated golden vs this
        // CPU port, 10 chained CFG forwards); assert well below the 5e-2
        // budget but above cross-platform fp32 noise.
        println!("trajectory max abs diff {d:.2e}, correlation {corr:.6}");
        assert!(d < 5e-3, "trajectory mismatch: {d}");
        assert!(corr > 0.999, "trajectory decorrelated: {corr}");
    }
}

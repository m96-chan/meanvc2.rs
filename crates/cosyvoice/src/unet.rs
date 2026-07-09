//! CausalConditionalDecoder — the CFM velocity estimator (CosyVoice 2 §2.4).
//!
//! Matcha-TTS-style 1-D U-Net made causal for chunked streaming. With the
//! CosyVoice2-0.5B config (`channels=[256]`) there is **no temporal
//! down/upsampling** — it is a straight 256-wide net:
//!
//! ```text
//! in  [B, 320, T]  (pack of x, mu, spk, cond — 80 each)
//!   down: CausalResnet(320→256) + 4 × TransformerBlock + CausalConv k3
//!   mid : 12 × [CausalResnet(256→256) + 4 × TransformerBlock]
//!   up  : CausalResnet(512→256, skip cat) + 4 × TF + CausalConv k3
//!   final: CausalBlock(256) → Conv1×1 → [B, 80, T]
//! ```
//!
//! Causal pieces: `CausalConv1d` (left pad k-1) and `CausalBlock1D`
//! (causal conv + LayerNorm + Mish — note LN, not the GroupNorm of the
//! non-causal Matcha block). Transformer blocks are diffusers
//! `BasicTransformerBlock`s (LN → MHA → LN → GELU-FF) with an optional
//! chunked attention bias when streaming (50-frame chunks, full left
//! context). The timestep enters through resnet FiLM-style additions.

use candle_core::{Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm, Linear, Module, VarBuilder};
use vc_core::Result;

const CH: usize = 256;
const IN_CH: usize = 320;
const TIME_DIM: usize = 1024;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const INNER: usize = HEADS * HEAD_DIM; // 512
/// Streaming chunk size in mel frames.
pub const CHUNK_FRAMES: usize = 50;

fn mish(x: &Tensor) -> Result<Tensor> {
    // x * tanh(softplus(x))
    let sp = (x.exp()? + 1.0)?.log()?;
    Ok(x.mul(&sp.tanh()?)?)
}

/// Left-padded (causal) conv1d.
struct CausalConv1d {
    w: Tensor,
    b: Tensor,
    k: usize,
}

impl CausalConv1d {
    fn load(vb: VarBuilder, in_c: usize, out_c: usize, k: usize) -> Result<Self> {
        Ok(Self {
            w: vb.get((out_c, in_c, k), "weight")?,
            b: vb.get(out_c, "bias")?,
            k,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.pad_with_zeros(D::Minus1, self.k - 1, 0)?;
        let y = x.conv1d(&self.w, 0, 1, 1, 1)?;
        Ok(y.broadcast_add(&self.b.reshape((1, (), 1))?)?)
    }
}

/// CausalConv1d k3 + LayerNorm (over channels) + Mish.
struct CausalBlock1D {
    conv: CausalConv1d,
    norm: LayerNorm,
}

impl CausalBlock1D {
    fn load(vb: VarBuilder, in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            conv: CausalConv1d::load(vb.pp("block.0"), in_c, out_c, 3)?,
            norm: layer_norm(out_c, 1e-5, vb.pp("block.2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let h = self.norm.forward(&h.transpose(1, 2)?.contiguous()?)?;
        mish(&h.transpose(1, 2)?.contiguous()?)
    }
}

struct CausalResnetBlock1D {
    block1: CausalBlock1D,
    block2: CausalBlock1D,
    mlp: Linear,
    res_conv: CausalConv1d, // k1 (plain 1×1)
}

impl CausalResnetBlock1D {
    fn load(vb: VarBuilder, in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            block1: CausalBlock1D::load(vb.pp("block1"), in_c, out_c)?,
            block2: CausalBlock1D::load(vb.pp("block2"), out_c, out_c)?,
            mlp: linear(TIME_DIM, out_c, vb.pp("mlp.1"))?,
            res_conv: CausalConv1d::load(vb.pp("res_conv"), in_c, out_c, 1)?,
        })
    }

    /// `x`: [B, C, T]; `temb`: [B, 1024].
    fn forward(&self, x: &Tensor, temb: &Tensor) -> Result<Tensor> {
        let h = self.block1.forward(x)?;
        let t = self.mlp.forward(&mish(temb)?)?.unsqueeze(D::Minus1)?;
        let h = h.broadcast_add(&t)?;
        let h = self.block2.forward(&h)?;
        Ok(h.add(&self.res_conv.forward(x)?)?)
    }
}

/// diffusers BasicTransformerBlock (self-attn only, LN, GELU-FF).
struct TransformerBlock {
    norm1: LayerNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl TransformerBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(CH, 1e-5, vb.pp("norm1"))?,
            to_q: linear_no_bias(CH, INNER, vb.pp("attn1.to_q"))?,
            to_k: linear_no_bias(CH, INNER, vb.pp("attn1.to_k"))?,
            to_v: linear_no_bias(CH, INNER, vb.pp("attn1.to_v"))?,
            to_out: linear(INNER, CH, vb.pp("attn1.to_out.0"))?,
            norm3: layer_norm(CH, 1e-5, vb.pp("norm3"))?,
            ff_in: linear(CH, CH * 4, vb.pp("ff.net.0.proj"))?,
            ff_out: linear(CH * 4, CH, vb.pp("ff.net.2"))?,
        })
    }

    /// `x`: [B, T, 256]; `bias`: optional [1, 1, T, T] additive.
    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.norm1.forward(x)?;
        let q = self
            .to_q
            .forward(&h)?
            .reshape((b, t, HEADS, HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = self
            .to_k
            .forward(&h)?
            .reshape((b, t, HEADS, HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = self
            .to_v
            .forward(&h)?
            .reshape((b, t, HEADS, HEAD_DIM))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let mut scores =
            (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / (HEAD_DIM as f64).sqrt())?;
        if let Some(bias) = bias {
            scores = scores.broadcast_add(bias)?;
        }
        let att = softmax(&scores, D::Minus1)?.matmul(&v)?;
        let att = att.permute((0, 2, 1, 3))?.reshape((b, t, INNER))?;
        let x = x.add(&self.to_out.forward(&att)?)?;
        let h = self.ff_in.forward(&self.norm3.forward(&x)?)?.gelu_erf()?;
        Ok(x.add(&self.ff_out.forward(&h)?)?)
    }
}

struct Level {
    resnet: CausalResnetBlock1D,
    tfs: Vec<TransformerBlock>,
    conv: CausalConv1d,
}

struct MidLevel {
    resnet: CausalResnetBlock1D,
    tfs: Vec<TransformerBlock>,
}

/// The estimator network.
pub struct Estimator {
    time_l1: Linear,
    time_l2: Linear,
    down: Level,
    mid: Vec<MidLevel>,
    up: Level,
    final_block: CausalBlock1D,
    final_proj_w: Tensor,
    final_proj_b: Tensor,
}

fn load_tfs(vb: &VarBuilder, n: usize) -> Result<Vec<TransformerBlock>> {
    (0..n)
        .map(|i| TransformerBlock::load(vb.pp(format!("{i}"))))
        .collect()
}

impl Estimator {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let mut mid = Vec::with_capacity(12);
        for i in 0..12 {
            let mvb = vb.pp(format!("mid_blocks.{i}"));
            mid.push(MidLevel {
                resnet: CausalResnetBlock1D::load(mvb.pp("0"), CH, CH)?,
                tfs: load_tfs(&mvb.pp("1"), 4)?,
            });
        }
        Ok(Self {
            time_l1: linear(IN_CH, TIME_DIM, vb.pp("time_mlp.linear_1"))?,
            time_l2: linear(TIME_DIM, TIME_DIM, vb.pp("time_mlp.linear_2"))?,
            down: Level {
                resnet: CausalResnetBlock1D::load(vb.pp("down_blocks.0.0"), IN_CH, CH)?,
                tfs: load_tfs(&vb.pp("down_blocks.0.1"), 4)?,
                conv: CausalConv1d::load(vb.pp("down_blocks.0.2"), CH, CH, 3)?,
            },
            mid,
            up: Level {
                resnet: CausalResnetBlock1D::load(vb.pp("up_blocks.0.0"), CH * 2, CH)?,
                tfs: load_tfs(&vb.pp("up_blocks.0.1"), 4)?,
                conv: CausalConv1d::load(vb.pp("up_blocks.0.2"), CH, CH, 3)?,
            },
            final_block: CausalBlock1D::load(vb.pp("final_block"), CH, CH)?,
            final_proj_w: vb.get((80, CH, 1), "final_proj.weight")?,
            final_proj_b: vb.get(80, "final_proj.bias")?,
        })
    }

    /// Sinusoidal timestep embedding (Matcha `SinusoidalPosEmb`, scale 1000)
    /// followed by the 2-layer SiLU MLP. `t`: [B] → [B, 1024].
    fn time_embed(&self, t: &Tensor) -> Result<Tensor> {
        let b = t.dim(0)?;
        let tv = t.to_vec1::<f32>()?;
        let half = IN_CH / 2;
        let mut emb = vec![0f32; b * IN_CH];
        let log_base = (10000f64).ln() / (half - 1) as f64;
        for (bi, tb) in tv.iter().enumerate() {
            for i in 0..half {
                let f = (-(i as f64) * log_base).exp();
                let ang = 1000.0 * (*tb as f64) * f;
                emb[bi * IN_CH + i] = ang.sin() as f32;
                emb[bi * IN_CH + half + i] = ang.cos() as f32;
            }
        }
        let e = Tensor::from_vec(emb, (b, IN_CH), t.device())?;
        let h = self.time_l1.forward(&e)?.silu()?;
        Ok(self.time_l2.forward(&h)?)
    }

    /// One velocity evaluation.
    ///
    /// `x`/`mu`/`cond`: `[B, 80, T]`; `spks`: `[B, 80]`; `t`: `[B]`.
    /// Returns `[B, 80, T]`.
    pub fn forward(
        &self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        let (b, _, tt) = x.dims3()?;
        let temb = self.time_embed(t)?;
        let spk = spks.unsqueeze(D::Minus1)?.broadcast_as((b, 80, tt))?;
        let mut h = Tensor::cat(&[x, mu, &spk.contiguous()?, cond], 1)?;

        let bias = if streaming {
            Some(crate::encoder::chunk_bias(tt, CHUNK_FRAMES, x.device())?)
        } else {
            None
        };

        let run_tfs = |h: Tensor, tfs: &[TransformerBlock]| -> Result<Tensor> {
            let mut y = h.transpose(1, 2)?.contiguous()?;
            for tf in tfs {
                y = tf.forward(&y, bias.as_ref())?;
            }
            Ok(y.transpose(1, 2)?.contiguous()?)
        };

        h = self.down.resnet.forward(&h, &temb)?;
        h = run_tfs(h, &self.down.tfs)?;
        let skip = h.clone();
        h = self.down.conv.forward(&h)?;

        for m in &self.mid {
            h = m.resnet.forward(&h, &temb)?;
            h = run_tfs(h, &m.tfs)?;
        }

        h = Tensor::cat(&[&h, &skip], 1)?;
        h = self.up.resnet.forward(&h, &temb)?;
        h = run_tfs(h, &self.up.tfs)?;
        h = self.up.conv.forward(&h)?;

        let h = self.final_block.forward(&h)?;
        let y = h.conv1d(&self.final_proj_w, 0, 1, 1, 1)?;
        Ok(y.broadcast_add(&self.final_proj_b.reshape((1, 80, 1))?)?)
    }
}

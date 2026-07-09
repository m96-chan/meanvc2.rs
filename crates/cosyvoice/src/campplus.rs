//! CAM++ speaker encoder (192-d x-vector) — the prompt-conditioning branch.
//!
//! Fresh MIT/Apache port written from the Apache-2.0 reference in
//! alibaba-damo-academy/3D-Speaker (`speakerlab/models/campplus`): FCM 2-D
//! front-end → D-TDNN with CAM (context-aware masking) dense blocks →
//! stats pooling → 192-d embedding. Weights are the `campplus_cn_common`
//! release (numerically identical to the `campplus.onnx` shipped with
//! CosyVoice2 — verified in issue #71), converted by
//! `tools/convert_cosyvoice.py`. Input: [`crate::mel::kaldi_fbank80`]
//! features `[T, 80]`; output `[1, 192]`.
//!
//! Note: candle's `conv2d` has no asymmetric stride, so the FCM's
//! `(2, 1)`-strided convolutions run at stride 1 and slice every other
//! frequency row — numerically identical.
//!
//! **Known deviation (documented, not a bug here):** the official
//! `campplus.onnx` was traced at a fixed 200-frame input, baking trace-time
//! shapes into its CAM seg-pooling; at other lengths the ONNX output drifts
//! (cos ≈ 0.91–0.998 vs the true dynamic model). This port implements the
//! correct dynamic semantics of the 3D-Speaker reference: it matches the
//! ONNX **exactly at 200 frames** (the `embedding_200` golden) and matches
//! the torch reference at every length.

use candle_core::{Tensor, D};
use candle_nn::{conv1d_no_bias, ops::sigmoid, Conv1d, Conv1dConfig, Module, VarBuilder};
use vc_core::Result;

/// Eval-mode batch norm folded to `x * scale + shift`.
struct Bn {
    scale: Tensor,
    shift: Tensor,
}

impl Bn {
    fn load(vb: VarBuilder, c: usize, affine: bool) -> Result<Self> {
        let mean = vb.get(c, "running_mean")?;
        let var = vb.get(c, "running_var")?;
        let inv = ((var + 1e-5)?.sqrt()?.recip())?;
        let (scale, shift) = if affine {
            let w = vb.get(c, "weight")?;
            let b = vb.get(c, "bias")?;
            let scale = (w * &inv)?;
            let shift = (b - (mean * &scale)?)?;
            (scale, shift)
        } else {
            let shift = (mean.neg()? * &inv)?;
            (inv, shift)
        };
        Ok(Self { scale, shift })
    }

    /// Normalize channel dim 1 of a rank-3 or rank-4 tensor.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.scale.dim(0)?;
        let shape: Vec<usize> = match x.rank() {
            3 => vec![1, c, 1],
            4 => vec![1, c, 1, 1],
            r => return Err(candle_core::Error::Msg(format!("Bn: unsupported rank {r}")).into()),
        };
        Ok(x.broadcast_mul(&self.scale.reshape(shape.clone())?)?
            .broadcast_add(&self.shift.reshape(shape)?)?)
    }
}

/// 2-D conv (no bias) with torch semantics `stride=(row_stride, 1)`,
/// implemented as stride-1 conv + row slicing.
struct Conv2dRows {
    w: Tensor,
    row_stride: usize,
    padding: usize,
}

impl Conv2dRows {
    fn load(
        vb: VarBuilder,
        shape: (usize, usize, usize, usize),
        row_stride: usize,
        padding: usize,
    ) -> Result<Self> {
        Ok(Self {
            w: vb.get(shape, "weight")?,
            row_stride,
            padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d(&self.w, self.padding, 1, 1, 1)?;
        if self.row_stride == 1 {
            return Ok(y);
        }
        let rows = y.dim(2)?;
        let idx: Vec<u32> = (0..rows)
            .step_by(self.row_stride)
            .map(|i| i as u32)
            .collect();
        let n = idx.len();
        let idx = Tensor::from_vec(idx, n, x.device())?;
        Ok(y.index_select(&idx, 2)?)
    }
}

struct BasicResBlock {
    conv1: Conv2dRows,
    bn1: Bn,
    conv2: Conv2dRows,
    bn2: Bn,
    shortcut: Option<(Conv2dRows, Bn)>,
}

impl BasicResBlock {
    fn load(vb: VarBuilder, in_c: usize, c: usize, stride: usize) -> Result<Self> {
        let shortcut = if stride != 1 || in_c != c {
            Some((
                Conv2dRows::load(vb.pp("shortcut.0"), (c, in_c, 1, 1), stride, 0)?,
                Bn::load(vb.pp("shortcut.1"), c, true)?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1: Conv2dRows::load(vb.pp("conv1"), (c, in_c, 3, 3), stride, 1)?,
            bn1: Bn::load(vb.pp("bn1"), c, true)?,
            conv2: Conv2dRows::load(vb.pp("conv2"), (c, c, 3, 3), 1, 1)?,
            bn2: Bn::load(vb.pp("bn2"), c, true)?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.bn1.forward(&self.conv1.forward(x)?)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        let sc = match &self.shortcut {
            Some((c, b)) => b.forward(&c.forward(x)?)?,
            None => x.clone(),
        };
        Ok(out.add(&sc)?.relu()?)
    }
}

/// FCM 2-D front-end: (B, 80, T) → (B, 320, T).
struct Fcm {
    conv1: Conv2dRows,
    bn1: Bn,
    layer1: Vec<BasicResBlock>,
    layer2: Vec<BasicResBlock>,
    conv2: Conv2dRows,
    bn2: Bn,
}

impl Fcm {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: Conv2dRows::load(vb.pp("conv1"), (32, 1, 3, 3), 1, 1)?,
            bn1: Bn::load(vb.pp("bn1"), 32, true)?,
            layer1: vec![
                BasicResBlock::load(vb.pp("layer1.0"), 32, 32, 2)?,
                BasicResBlock::load(vb.pp("layer1.1"), 32, 32, 1)?,
            ],
            layer2: vec![
                BasicResBlock::load(vb.pp("layer2.0"), 32, 32, 2)?,
                BasicResBlock::load(vb.pp("layer2.1"), 32, 32, 1)?,
            ],
            conv2: Conv2dRows::load(vb.pp("conv2"), (32, 32, 3, 3), 2, 1)?,
            bn2: Bn::load(vb.pp("bn2"), 32, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?; // (B, 1, F, T)
        let mut out = self.bn1.forward(&self.conv1.forward(&x)?)?.relu()?;
        for b in self.layer1.iter().chain(&self.layer2) {
            out = b.forward(&out)?;
        }
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?.relu()?;
        let (b, c, f, t) = out.dims4()?;
        Ok(out.reshape((b, c * f, t))?)
    }
}

/// CAM layer: local conv gated by a context mask (mean + 100-frame pooling).
struct CamLayer {
    local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
}

impl CamLayer {
    fn load(vb: VarBuilder, bn_c: usize, out_c: usize, k: usize, dilation: usize) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: (k - 1) / 2 * dilation,
            dilation,
            ..Default::default()
        };
        Ok(Self {
            local: conv1d_no_bias(bn_c, out_c, k, cfg, vb.pp("linear_local"))?,
            linear1: candle_nn::conv1d(bn_c, bn_c / 2, 1, Default::default(), vb.pp("linear1"))?,
            linear2: candle_nn::conv1d(bn_c / 2, out_c, 1, Default::default(), vb.pp("linear2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.local.forward(x)?;
        let (b, c, t) = x.dims3()?;
        let mean = x.mean_keepdim(D::Minus1)?; // (B, C, 1)
                                               // 100-frame average pooling (ceil mode), expanded back to T
        let seg_len = 100usize;
        let n_seg = t.div_ceil(seg_len);
        let mut segs = Vec::with_capacity(n_seg);
        for s in 0..n_seg {
            let start = s * seg_len;
            let len = seg_len.min(t - start);
            let m = x.narrow(D::Minus1, start, len)?.mean_keepdim(D::Minus1)?;
            segs.push(m.broadcast_as((b, c, len))?);
        }
        let seg = Tensor::cat(&segs, D::Minus1)?;
        let context = seg.broadcast_add(&mean)?;
        let m = self.linear1.forward(&context)?.relu()?;
        let m = sigmoid(&self.linear2.forward(&m)?)?;
        Ok(y.mul(&m)?)
    }
}

struct DenseTdnnLayer {
    bn1: Bn,
    linear1: Conv1d,
    bn2: Bn,
    cam: CamLayer,
}

impl DenseTdnnLayer {
    fn load(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        bn_c: usize,
        k: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            bn1: Bn::load(vb.pp("nonlinear1.batchnorm"), in_c, true)?,
            linear1: conv1d_no_bias(in_c, bn_c, 1, Default::default(), vb.pp("linear1"))?,
            bn2: Bn::load(vb.pp("nonlinear2.batchnorm"), bn_c, true)?,
            cam: CamLayer::load(vb.pp("cam_layer"), bn_c, out_c, k, dilation)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(&self.bn1.forward(x)?.relu()?)?;
        self.cam.forward(&self.bn2.forward(&h)?.relu()?)
    }
}

struct Transit {
    bn: Bn,
    linear: Conv1d,
}

impl Transit {
    fn load(vb: VarBuilder, in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self {
            bn: Bn::load(vb.pp("nonlinear.batchnorm"), in_c, true)?,
            linear: conv1d_no_bias(in_c, out_c, 1, Default::default(), vb.pp("linear"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.linear.forward(&self.bn.forward(x)?.relu()?)?)
    }
}

/// The CAM++ model: `[T, 80]` fbank → `[1, 192]` x-vector.
pub struct CamPlusPlus {
    head: Fcm,
    tdnn_linear: Conv1d,
    tdnn_bn: Bn,
    blocks: Vec<Vec<DenseTdnnLayer>>,
    transits: Vec<Transit>,
    out_bn: Bn,
    dense_linear: Conv1d,
    dense_bn: Bn,
}

impl CamPlusPlus {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let xv = vb.pp("xvector");
        let growth = 32;
        let bn_size = 4;
        let mut channels = 128usize;
        let mut blocks = Vec::new();
        let mut transits = Vec::new();
        for (bi, (num_layers, dilation)) in [(12usize, 1usize), (24, 2), (16, 2)].iter().enumerate()
        {
            let bvb = xv.pp(format!("block{}", bi + 1));
            let mut layers = Vec::with_capacity(*num_layers);
            for i in 0..*num_layers {
                layers.push(DenseTdnnLayer::load(
                    bvb.pp(format!("tdnnd{}", i + 1)),
                    channels + i * growth,
                    growth,
                    bn_size * growth,
                    3,
                    *dilation,
                )?);
            }
            blocks.push(layers);
            channels += num_layers * growth;
            transits.push(Transit::load(
                xv.pp(format!("transit{}", bi + 1)),
                channels,
                channels / 2,
            )?);
            channels /= 2;
        }
        let tdnn_cfg = Conv1dConfig {
            padding: 2,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            head: Fcm::load(vb.pp("head"))?,
            tdnn_linear: conv1d_no_bias(320, 128, 5, tdnn_cfg, xv.pp("tdnn.linear"))?,
            tdnn_bn: Bn::load(xv.pp("tdnn.nonlinear.batchnorm"), 128, true)?,
            blocks,
            transits,
            out_bn: Bn::load(xv.pp("out_nonlinear.batchnorm"), channels, true)?,
            dense_linear: conv1d_no_bias(
                channels * 2,
                192,
                1,
                Default::default(),
                xv.pp("dense.linear"),
            )?,
            dense_bn: Bn::load(xv.pp("dense.nonlinear.batchnorm"), 192, false)?,
        })
    }

    /// `fbank`: `[T, 80]` mean-normalized kaldi fbank → `[1, 192]`.
    pub fn embed(&self, fbank: &Tensor) -> Result<Tensor> {
        let x = fbank.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // (1, 80, T)
        let x = self.head.forward(&x)?;
        let mut x = self
            .tdnn_bn
            .forward(&self.tdnn_linear.forward(&x)?)?
            .relu()?;
        for (layers, transit) in self.blocks.iter().zip(&self.transits) {
            for l in layers {
                let y = l.forward(&x)?;
                x = Tensor::cat(&[&x, &y], 1)?;
            }
            x = transit.forward(&x)?;
        }
        let x = self.out_bn.forward(&x)?.relu()?;
        // stats pooling: mean + unbiased std over time
        let t = x.dim(2)? as f64;
        let mean = x.mean(D::Minus1)?; // (1, C)
        let centered = x.broadcast_sub(&mean.unsqueeze(D::Minus1)?)?;
        let var = (centered.sqr()?.sum(D::Minus1)? / (t - 1.0))?;
        let std = var.sqrt()?;
        let stats = Tensor::cat(&[&mean, &std], 1)?.unsqueeze(D::Minus1)?; // (1, 2C, 1)
        let emb = self.dense_linear.forward(&stats)?;
        Ok(self.dense_bn.forward(&emb)?.squeeze(D::Minus1)?)
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

    #[test]
    fn embedding_matches_official_onnx() {
        let (Some(fx), Some(w)) = (fixture(), ckpt("cosyvoice_campplus.safetensors")) else {
            return;
        };
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &dev).unwrap() };
        let model = CamPlusPlus::load(vb).unwrap();

        fn cos_max(got: &Tensor, want: &Tensor) -> (f32, f32) {
            let a = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a.len(), b.len());
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

        // Exact parity at the ONNX trace length (200 frames).
        if let Some(want200) = fx.get("embedding_200") {
            let fb200 = fx["prompt_fbank"].narrow(0, 0, 200).unwrap();
            let (cos, max_d) = cos_max(&model.embed(&fb200).unwrap(), want200);
            assert!(
                cos > 0.99999 && max_d < 1e-3,
                "T=200: cosine {cos}, max abs diff {max_d}"
            );
        }
        // Full length: the official ONNX drifts from the true dynamic model
        // (trace-baked seg-pooling); only a loose agreement is expected.
        let (cos, _) = cos_max(&model.embed(&fx["prompt_fbank"]).unwrap(), &fx["embedding"]);
        assert!(cos > 0.995, "full length: cosine {cos}");
    }
}

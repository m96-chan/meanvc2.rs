//! RepCodec content-style tokenizer (`models/codec/kmeans/repcodec_model.py`
//! `RepCodec`, cfg `vq8192`): a `VocosBackbone` encoder over z-normalized
//! HuBERT-large layer-18 features, quantized by a single-codebook
//! `FactorizedVectorQuantize` (fvq, L2-normalized cosine lookup,
//! vocab 8192, codebook_dim 8). Only `quantize()` (encoder + VQ lookup,
//! no decoder) is needed for Vevo-Timbre.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{
    conv1d, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear,
    VarBuilder,
};

use vc_core::Result;

#[derive(Debug, Clone)]
pub struct RepCodecConfig {
    pub hidden_size: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vocos_dim: usize,
    pub vocos_intermediate_dim: usize,
    pub vocos_num_layers: usize,
}

impl Default for RepCodecConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            codebook_size: 8192,
            codebook_dim: 8,
            vocos_dim: 384,
            vocos_intermediate_dim: 2048,
            vocos_num_layers: 12,
        }
    }
}

/// ConvNeXt block (no AdaLayerNorm — `adanorm_num_embeddings=None` for
/// this tokenizer). Same shape as the Vocos vocoder's block
/// (`models/codec/amphion_codec/vocos.py::ConvNeXtBlock`).
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, intermediate_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: 3,
            groups: dim,
            ..Default::default()
        };
        Ok(Self {
            dwconv: conv1d(dim, dim, 7, conv_cfg, vb.pp("dwconv"))?,
            norm: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm"))?,
            pwconv1: linear(dim, intermediate_dim, vb.pp("pwconv1"))?,
            pwconv2: linear(intermediate_dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get((dim,), "gamma")?,
        })
    }

    /// `x`: `[batch, dim, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let x = self.dwconv.forward(x)?.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self
            .pwconv2
            .forward(&self.pwconv1.forward(&x)?.gelu_erf()?)?;
        let x = x.broadcast_mul(&self.gamma)?.transpose(1, 2)?;
        residual + x
    }
}

/// `models/codec/kmeans/vocos.py::VocosBackbone` — embed conv → LayerNorm
/// → N ConvNeXt blocks → final LayerNorm.
struct VocosBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
}

impl VocosBackbone {
    fn new(
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let embed_cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(ConvNeXtBlock::new(
                dim,
                intermediate_dim,
                vb.pp(format!("convnext.{i}")),
            )?);
        }
        Ok(Self {
            embed: conv1d(input_channels, dim, 7, embed_cfg, vb.pp("embed"))?,
            norm: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm"))?,
            blocks,
            final_norm: layer_norm(dim, LayerNormConfig::default(), vb.pp("final_layer_norm"))?,
        })
    }

    /// `x`: `[batch, input_channels, time]` -> `[batch, time, dim]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.embed.forward(x)?;
        let x = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.final_norm.forward(&x.transpose(1, 2)?)
    }
}

pub struct RepCodec {
    backbone: VocosBackbone,
    /// `encoder.1`: the `nn.Sequential`'s final `Linear(vocos_dim,
    /// hidden_size)`, projecting the backbone output back up.
    encoder_proj: Linear,
    /// `quantizer.in_project`: `FactorizedVectorQuantize`'s
    /// weight-normalized 1x1 conv (folded to a plain Linear-shaped
    /// weight at fetch time), `hidden_size -> codebook_dim`.
    vq_in_project: Linear,
    codebook: Tensor, // [codebook_size, codebook_dim], L2-normalized at load
}

impl RepCodec {
    pub fn new(cfg: &RepCodecConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let backbone = VocosBackbone::new(
            cfg.hidden_size,
            cfg.vocos_dim,
            cfg.vocos_intermediate_dim,
            cfg.vocos_num_layers,
            vb.pp("encoder.0"),
        )?;
        let encoder_proj = linear(cfg.vocos_dim, cfg.hidden_size, vb.pp("encoder.1"))?;
        let vq_in_project = linear(
            cfg.hidden_size,
            cfg.codebook_dim,
            vb.pp("quantizer.in_project"),
        )?;
        let codebook = vb.get(
            (cfg.codebook_size, cfg.codebook_dim),
            "quantizer.codebook.weight",
        )?;
        let codebook = l2_normalize_rows(&codebook)?;
        Ok(Self {
            backbone,
            encoder_proj,
            vq_in_project,
            codebook,
        })
    }

    pub fn load<P: AsRef<std::path::Path>>(
        cfg: &RepCodecConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb).map_err(Into::into)
    }

    /// `x`: `[batch, time, hidden_size]` (z-normalized HuBERT features).
    /// Returns the full encoder output (backbone + `encoder.1`
    /// projection), `[batch, time, hidden_size]` — matches
    /// `RepCodec.quantize`'s `self.encoder(x.transpose(1,2)).transpose(1,2)`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?; // [b, hidden, t]
        let x = self.backbone.forward(&x)?; // [b, t, vocos_dim]
        self.encoder_proj.forward(&x).map_err(Into::into)
    }

    /// `x`: `[batch, time, hidden_size]`. Returns int64 codebook indices
    /// `[batch, time]` (matches `RepCodec.quantize`, which returns
    /// `all_indices.squeeze(0)` for batch=1 — here we keep the batch
    /// dim for simplicity and squeeze at the call site).
    pub fn quantize(&self, x: &Tensor) -> Result<Tensor> {
        let enc = self.encode(x)?; // [b, t, hidden]
        let z_e = self.vq_in_project.forward(&enc)?; // [b, t, codebook_dim]
        let z_e = l2_normalize_rows(&z_e)?;
        // cosine distance argmax == dot-product argmax after L2 norm.
        let sims = z_e.broadcast_matmul(&self.codebook.t()?)?; // [b, t, codebook_size]
        sims.argmax(2)?.to_dtype(DType::I64).map_err(Into::into)
    }
}

/// L2-normalizes the last dimension of `x` (matches `F.normalize`, eps
/// handled the same way: divide by `max(norm, 1e-12)`).
fn l2_normalize_rows(x: &Tensor) -> candle_core::Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(x.rank() - 1)?.sqrt()?;
    let norm = norm.clamp(1e-12, f64::INFINITY)?;
    x.broadcast_div(&norm)
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

    fn ckpt() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_repcodec.safetensors");
        path.exists().then_some(path)
    }

    #[test]
    fn encoder_matches_official() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let cfg = RepCodecConfig::default();
        let model = RepCodec::load(&cfg, ckpt, &dev).unwrap();

        let got = model.encode(&fx["hubert_ref_norm"]).unwrap();
        let want = &fx["repcodec_ref_enc"]; // [b, hidden, t] layout from the fixture
        let want = want.transpose(1, 2).unwrap(); // -> [b, t, hidden]

        let corr = cosine_corr(&got, &want);
        assert!(corr > 0.999, "encoder correlation {corr}");
    }

    #[test]
    fn codes_match_official() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let cfg = RepCodecConfig::default();
        let model = RepCodec::load(&cfg, ckpt, &dev).unwrap();

        let got = model.quantize(&fx["hubert_ref_norm"]).unwrap();
        let got: Vec<i64> = got.i(0).unwrap().to_vec1().unwrap();
        let want: Vec<i64> = fx["repcodec_ref_codes"].i(0).unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), want.len());
        let matches = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        let ratio = matches as f64 / got.len() as f64;
        assert!(
            ratio > 0.95,
            "code match ratio {ratio} ({matches}/{})",
            got.len()
        );
    }

    fn cosine_corr(a: &Tensor, b: &Tensor) -> f64 {
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
}

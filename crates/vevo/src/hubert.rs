//! HuBERT-large (`facebook/hubert-large-ll60k`, as wrapped by
//! `torchaudio.pipelines.HUBERT_LARGE`): conv feature extractor +
//! pre-LN transformer encoder. Only layer 18 of 24 is ever needed by
//! Vevo-Timbre, so this port only builds/runs the first
//! [`HubertConfig::num_layers_used`] transformer layers.
//!
//! Naming mirrors torchaudio's `Wav2Vec2Model` state dict (minus the
//! `model.` prefix): `feature_extractor.conv_layers.{i}.*`,
//! `encoder.feature_projection.*`, `encoder.transformer.*`. The
//! positional-conv weight is weight-normalized upstream with an
//! unusual `dim=2` (per kernel-position magnitude); `babiniku-fetch`
//! folds it before saving, so this crate only ever loads a plain
//! `weight` tensor.

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv1d_no_bias, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear,
    Module, VarBuilder,
};

use vc_core::Result;

#[derive(Debug, Clone)]
pub struct HubertConfig {
    /// (out_channels, kernel_size, stride) for the 7 feature-extractor convs.
    pub conv_layers: Vec<(usize, usize, usize)>,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub ff_dim: usize,
    pub pos_conv_kernel: usize,
    pub pos_conv_groups: usize,
    /// Transformer layers to build; Vevo only needs layer 18 (1-indexed).
    pub num_layers_used: usize,
}

impl Default for HubertConfig {
    fn default() -> Self {
        Self {
            conv_layers: vec![
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ],
            embed_dim: 1024,
            num_heads: 16,
            ff_dim: 4096,
            pos_conv_kernel: 128,
            pos_conv_groups: 16,
            num_layers_used: 18,
        }
    }
}

struct ConvBlock {
    conv: Conv1d,
    ln: LayerNorm,
}

impl ConvBlock {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = Conv1dConfig {
            stride,
            ..Default::default()
        };
        Ok(Self {
            conv: conv1d_no_bias(in_ch, out_ch, kernel, cfg, vb.pp("conv"))?,
            ln: layer_norm(out_ch, LayerNormConfig::default(), vb.pp("layer_norm"))?,
        })
    }

    /// `x`: `[batch, in_ch, time]` -> `[batch, out_ch, time']`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.ln.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        x.gelu_erf()
    }
}

struct FeatureExtractor {
    blocks: Vec<ConvBlock>,
}

impl FeatureExtractor {
    fn new(cfg: &HubertConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.conv_layers.len());
        let mut in_ch = 1;
        for (i, &(out_ch, kernel, stride)) in cfg.conv_layers.iter().enumerate() {
            blocks.push(ConvBlock::new(
                in_ch,
                out_ch,
                kernel,
                stride,
                vb.pp(format!("conv_layers.{i}")),
            )?);
            in_ch = out_ch;
        }
        Ok(Self { blocks })
    }

    /// `wav`: `[batch, samples]` -> `[batch, frames, 512]`.
    fn forward(&self, wav: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = wav.unsqueeze(1)?; // [b, 1, t]
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x.transpose(1, 2) // [b, t, c]
    }
}

struct FeatureProjection {
    ln: LayerNorm,
    proj: Linear,
}

impl FeatureProjection {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            ln: layer_norm(in_dim, LayerNormConfig::default(), vb.pp("layer_norm"))?,
            proj: linear(in_dim, out_dim, vb.pp("projection"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.proj.forward(&self.ln.forward(x)?)
    }
}

struct EncoderLayer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    num_heads: usize,
}

impl EncoderLayer {
    fn new(cfg: &HubertConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let d = cfg.embed_dim;
        let attn = vb.pp("attention");
        Ok(Self {
            ln1: layer_norm(d, LayerNormConfig::default(), vb.pp("layer_norm"))?,
            q: linear(d, d, attn.pp("q_proj"))?,
            k: linear(d, d, attn.pp("k_proj"))?,
            v: linear(d, d, attn.pp("v_proj"))?,
            o: linear(d, d, attn.pp("out_proj"))?,
            ln2: layer_norm(d, LayerNormConfig::default(), vb.pp("final_layer_norm"))?,
            ff1: linear(d, cfg.ff_dim, vb.pp("feed_forward.intermediate_dense"))?,
            ff2: linear(cfg.ff_dim, d, vb.pp("feed_forward.output_dense"))?,
            num_heads: cfg.num_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x;
        let h = self.ln1.forward(x)?;
        let (b, t, d) = h.dims3()?;
        let hd = d / self.num_heads;
        let shape = (b, t, self.num_heads, hd);
        let q = self
            .q
            .forward(&h)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(&h)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(&h)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = (hd as f64).powf(-0.5);
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?; // [b, nh, t, hd]
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;
        let out = self.o.forward(&out)?;
        let x = (residual + out)?;

        let residual = &x;
        let h = self.ln2.forward(&x)?;
        let h = self.ff1.forward(&h)?.gelu_erf()?;
        let h = self.ff2.forward(&h)?;
        residual + h
    }
}

/// Weight-normalized (dim=2) positional conv embedding.
struct PosConvEmbed {
    conv: Conv1d,
    num_remove: usize,
}

impl PosConvEmbed {
    fn new(cfg: &HubertConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv_cfg = Conv1dConfig {
            padding: cfg.pos_conv_kernel / 2,
            groups: cfg.pos_conv_groups,
            ..Default::default()
        };
        Ok(Self {
            conv: candle_nn::conv1d(
                cfg.embed_dim,
                cfg.embed_dim,
                cfg.pos_conv_kernel,
                conv_cfg,
                vb,
            )?,
            num_remove: if cfg.pos_conv_kernel % 2 == 0 { 1 } else { 0 },
        })
    }

    /// `x`: `[batch, time, embed_dim]` -> same shape.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.conv.forward(&x.transpose(1, 2)?)?; // [b, c, t']
        let t = y.dim(2)? - self.num_remove;
        let y = y.narrow(2, 0, t)?;
        y.gelu_erf()?.transpose(1, 2)
    }
}

pub struct HubertLarge {
    feature_extractor: FeatureExtractor,
    feature_projection: FeatureProjection,
    pos_conv: PosConvEmbed,
    layers: Vec<EncoderLayer>,
}

impl HubertLarge {
    pub fn new(cfg: &HubertConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let feature_extractor = FeatureExtractor::new(cfg, vb.pp("feature_extractor"))?;
        let enc = vb.pp("encoder");
        let feature_projection =
            FeatureProjection::new(512, cfg.embed_dim, enc.pp("feature_projection"))?;
        let tvb = enc.pp("transformer");
        let pos_conv = PosConvEmbed::new(cfg, tvb.pp("pos_conv_embed.conv"))?;
        // NOTE: `encoder.transformer.layer_norm` is intentionally never
        // loaded/applied here. torchaudio's `Transformer` has its OWN
        // `layer_norm_first` flag (separate from each `EncoderLayer`'s),
        // which for HUBERT_LARGE is `False` — so `Transformer.layer_norm`
        // only fires at the very end of `forward()`, a path
        // `extract_features`/`get_intermediate_outputs` never takes. Each
        // `EncoderLayer` has its own pre-LN (`layer_norm_first=True`)
        // internally, which IS what normalizes the stream.
        let mut layers = Vec::with_capacity(cfg.num_layers_used);
        for i in 0..cfg.num_layers_used {
            layers.push(EncoderLayer::new(cfg, tvb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            feature_extractor,
            feature_projection,
            pos_conv,
            layers,
        })
    }

    pub fn load<P: AsRef<std::path::Path>>(
        cfg: &HubertConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(cfg, vb).map_err(Into::into)
    }

    /// `wav`: `[batch, samples]` (16 kHz). Returns the layer-`num_layers_used`
    /// hidden states, `[batch, frames, 1024]` at 50 Hz.
    pub fn extract_features(&self, wav: &Tensor) -> Result<Tensor> {
        // torchaudio's HUBERT_LARGE bundle normalizes the raw waveform to
        // zero-mean/unit-variance before the feature extractor
        // (`nn.functional.layer_norm(waveforms, waveforms.shape)`). Note:
        // every conv in the feature extractor is bias-free and immediately
        // followed by a per-position LayerNorm, so this step is
        // mathematically a pure scale-invariance no-op *only* if it were
        // scale-only — it also re-centers, which does NOT cancel (conv
        // weights don't sum to zero per channel), so it must be applied.
        let n = wav.elem_count() as f64;
        let mean = (wav.sum_all()? / n)?;
        let centered = wav.broadcast_sub(&mean)?;
        let var = ((&centered * &centered)?.sum_all()? / n)?;
        let denom = (var + 1e-5)?.sqrt()?;
        let wav = centered.broadcast_div(&denom)?;

        let x = self.feature_extractor.forward(&wav)?;
        let x = self.feature_projection.forward(&x)?;
        let pc = self.pos_conv.forward(&x)?;
        let mut x = (&x + pc)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
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
            .join("../../ckpt/vevo_hubert.safetensors");
        path.exists().then_some(path)
    }

    #[test]
    fn matches_official_layer18() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let cfg = HubertConfig::default();
        let model = HubertLarge::load(&cfg, ckpt, &dev).unwrap();

        let ref_16k = fx["ref_16k"].clone();
        let want = fx["hubert_ref_raw"]
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let got = model
            .extract_features(&ref_16k)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(got.len(), want.len(), "frame count mismatch");
        // The golden fixture was generated on CUDA; layer 18 is 18
        // pre-LN transformer blocks deep, so tiny GPU-vs-CPU float
        // differences accumulate into a double-digit max-abs-diff
        // (same class of residual as the seedvc/xvc golden tests) —
        // correlation is the meaningful check here.
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (gr, wr) in got.iter().zip(&want) {
            for (g, w) in gr.iter().zip(wr) {
                dot += (*g as f64) * (*w as f64);
                na += (*g as f64).powi(2);
                nb += (*w as f64).powi(2);
            }
        }
        let corr = dot / (na.sqrt() * nb.sqrt());
        assert!(corr > 0.999, "correlation {corr}");
    }
}

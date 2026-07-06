//! SAC acoustic codec — candle port of the X-VC acoustic branch
//! (arXiv:2604.12456 §2.1; SAC-16k-62.5Hz lineage).
//!
//! Three frozen modules operating directly on 16 kHz waveforms:
//!
//! * [`AcousticEncoder`] — DAC-style strided convolution stack
//!   (descript-audio-codec) with [`Snake1d`] activations, rates
//!   `[2, 4, 5, 8]` (320× downsampling) → 50 Hz / 1024-dim latents;
//! * [`FactorizedVectorQuantize`] — factorized VQ: 1024→8 projection,
//!   argmin over the L2-normalized 16384×8 codebook, 8→1024 out
//!   projection;
//! * [`AcousticDecoder`] — DAC/HiFiGAN-style transposed-convolution
//!   vocoder, rates `[8, 5, 4, 2]`, straight to the 16 kHz waveform
//!   (no mel/ISTFT stage).
//!
//! Layout and parameter names mirror the official implementation
//! (`models/codec/sac/modules/{acoustic_encoder,vocoder/wave_generator}.py`,
//! `models/codec/base/quantizer/factorized_vector_quantize.py` of
//! [Jerrister/X-VC](https://github.com/Jerrister/X-VC)) so the converted
//! checkpoint maps 1:1: `acoustic_encoder.block.{i}...`,
//! `acoustic_quantizer.{in_project,codebook,out_project}`,
//! `acoustic_decoder.model.{i}...`. The `weight_norm` parametrizations of
//! the official checkpoint are folded into plain conv weights by
//! `tools/convert_xvc_generator.py`.

use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    conv1d, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module,
    VarBuilder,
};

use vc_core::Result;

/// Configuration of the SAC acoustic codec.
///
/// Defaults are the released X-VC codec (`configs/xvc.yaml`): 16 kHz,
/// 320× down/upsampling (50 Hz latent rate), 1024-dim latents, 16384×8
/// factorized codebook.
#[derive(Debug, Clone)]
pub struct SacCodecConfig {
    /// First encoder convolution width (doubles at every strided block).
    pub encoder_dim: usize,
    /// Encoder downsampling rates; their product is the hop in samples.
    pub encoder_rates: Vec<usize>,
    /// Latent dimensionality (`z`, `zq`).
    pub latent_dim: usize,
    /// Number of codebook entries.
    pub codebook_size: usize,
    /// Factorized codebook dimensionality.
    pub codebook_dim: usize,
    /// First decoder convolution width (halves at every upsampling block).
    pub decoder_channels: usize,
    /// Decoder upsampling rates (the encoder rates reversed).
    pub decoder_rates: Vec<usize>,
    /// Transposed-convolution kernel sizes of the decoder blocks.
    pub decoder_kernel_sizes: Vec<usize>,
}

impl Default for SacCodecConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 5, 8],
            latent_dim: 1024,
            codebook_size: 16384,
            codebook_dim: 8,
            decoder_channels: 1536,
            decoder_rates: vec![8, 5, 4, 2],
            decoder_kernel_sizes: vec![16, 11, 8, 4],
        }
    }
}

impl SacCodecConfig {
    /// Samples per latent frame (the product of the encoder rates); 320
    /// for the released codec, i.e. 50 Hz latents at 16 kHz.
    pub fn hop_length(&self) -> usize {
        self.encoder_rates.iter().product()
    }
}

/// Snake activation (`x + sin²(αx) / (α + 1e-9)`) with a learned
/// per-channel `α` of shape `[1, channels, 1]`.
#[derive(Debug)]
struct Snake1d {
    alpha: Tensor,
    /// `1 / (α + 1e-9)`, precomputed at load time.
    alpha_recip: Tensor,
}

impl Snake1d {
    fn new(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        let alpha_recip = (&alpha + 1e-9)?.recip()?;
        Ok(Self { alpha, alpha_recip })
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let s = x.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        x + s.broadcast_mul(&self.alpha_recip)?
    }
}

/// `Snake → k7 dilated conv → Snake → k1 conv`, with a residual add.
/// The paddings (`3·dilation`) keep the length, so the official
/// centre-trim of the residual is a no-op here.
#[derive(Debug)]
struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let cfg = Conv1dConfig {
            padding: 3 * dilation,
            dilation,
            ..Default::default()
        };
        Ok(Self {
            snake1: Snake1d::new(dim, vb.pp("block.0"))?,
            conv1: conv1d(dim, dim, 7, cfg, vb.pp("block.1"))?,
            snake2: Snake1d::new(dim, vb.pp("block.2"))?,
            conv2: conv1d(dim, dim, 1, Default::default(), vb.pp("block.3"))?,
        })
    }

    /// `x`: `[batch, dim, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.snake1.forward(x)?;
        let y = self.conv1.forward(&y)?;
        let y = self.snake2.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        x + y
    }
}

/// Three dilated residual units followed by a strided downsampling conv
/// (`kernel = 2·stride`, `padding = ⌈stride/2⌉`) that doubles channels.
#[derive(Debug)]
struct EncoderBlock {
    res: Vec<ResidualUnit>,
    snake: Snake1d,
    conv: Conv1d,
}

impl EncoderBlock {
    /// `dim` is the output width; the units run at `dim / 2`.
    fn new(dim: usize, stride: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let res = [1, 3, 9]
            .iter()
            .enumerate()
            .map(|(i, &d)| ResidualUnit::new(dim / 2, d, vb.pp(format!("block.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let cfg = Conv1dConfig {
            padding: stride.div_ceil(2),
            stride,
            ..Default::default()
        };
        Ok(Self {
            res,
            snake: Snake1d::new(dim / 2, vb.pp("block.3"))?,
            conv: conv1d(dim / 2, dim, 2 * stride, cfg, vb.pp("block.4"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for unit in &self.res {
            x = unit.forward(&x)?;
        }
        self.conv.forward(&self.snake.forward(&x)?)
    }
}

/// DAC-style acoustic encoder: waveform `[batch, 1, samples]` → 50 Hz
/// latent `[batch, latent_dim, samples / 320]`.
#[derive(Debug)]
pub struct AcousticEncoder {
    conv_in: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake: Snake1d,
    conv_out: Conv1d,
}

impl AcousticEncoder {
    fn new(cfg: &SacCodecConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut dim = cfg.encoder_dim;
        let conv_in = conv1d(
            1,
            dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("block.0"),
        )?;
        let mut blocks = Vec::with_capacity(cfg.encoder_rates.len());
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            dim *= 2;
            blocks.push(EncoderBlock::new(
                dim,
                stride,
                vb.pp(format!("block.{}", i + 1)),
            )?);
        }
        let n = cfg.encoder_rates.len();
        Ok(Self {
            conv_in,
            blocks,
            snake: Snake1d::new(dim, vb.pp(format!("block.{}", n + 1)))?,
            conv_out: conv1d(
                dim,
                cfg.latent_dim,
                3,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(format!("block.{}", n + 2)),
            )?,
        })
    }

    /// `wav`: `[batch, 1, samples]` → `[batch, latent_dim, frames]`.
    pub fn forward(&self, wav: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.conv_in.forward(wav)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.conv_out.forward(&self.snake.forward(&x)?)
    }
}

/// Factorized vector quantizer (inference path): 1×1-conv projection into
/// the low-dimensional codebook space, nearest-neighbour lookup over the
/// L2-normalized codebook, and 1×1-conv out projection. The quantized
/// vectors are the **raw** (unnormalized) codebook entries; only the
/// distance computation normalizes.
#[derive(Debug)]
pub struct FactorizedVectorQuantize {
    in_project: Conv1d,
    out_project: Conv1d,
    /// Raw codebook, `[codebook_size, codebook_dim]`.
    codebook: Tensor,
    /// Row-normalized codebook (transposed, `[codebook_dim, size]`) and
    /// its per-entry squared norms `[1, size]`, precomputed at load time.
    codebook_norm_t: Tensor,
    codebook_norm_sq: Tensor,
}

impl FactorizedVectorQuantize {
    fn new(cfg: &SacCodecConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let codebook = vb.get((cfg.codebook_size, cfg.codebook_dim), "codebook.weight")?;
        // F.normalize: v / max(‖v‖, 1e-12).
        let norms = codebook
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        let normalized = codebook.broadcast_div(&norms)?;
        Ok(Self {
            in_project: conv1d(
                cfg.latent_dim,
                cfg.codebook_dim,
                1,
                Default::default(),
                vb.pp("in_project"),
            )?,
            out_project: conv1d(
                cfg.codebook_dim,
                cfg.latent_dim,
                1,
                Default::default(),
                vb.pp("out_project"),
            )?,
            codebook,
            codebook_norm_sq: normalized.sqr()?.sum_keepdim(D::Minus1)?.t()?,
            codebook_norm_t: normalized.t()?.contiguous()?,
        })
    }

    /// Projects the encoder latent into codebook space:
    /// `[batch, latent_dim, frames]` → `[batch, codebook_dim, frames]`.
    pub fn project(&self, z: &Tensor) -> candle_core::Result<Tensor> {
        self.in_project.forward(z)
    }

    /// Nearest codebook ids of projected latents `z_e`
    /// `[batch, codebook_dim, frames]` → `[batch, frames]` (u32).
    ///
    /// Distances follow the official formula on L2-normalized encodings
    /// and codebook: `‖e‖² − 2·e·cᵀ + ‖c‖²` (= cosine distance).
    pub fn codes(&self, z_e: &Tensor) -> candle_core::Result<Tensor> {
        let (b, d, t) = z_e.dims3()?;
        let enc = z_e.transpose(1, 2)?.reshape((b * t, d))?;
        let norms = enc
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        let enc = enc.broadcast_div(&norms)?;
        let dist = enc
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .broadcast_sub(&(enc.matmul(&self.codebook_norm_t)? * 2.0)?)?
            .broadcast_add(&self.codebook_norm_sq)?;
        dist.argmin(D::Minus1)?.reshape((b, t))
    }

    /// Codebook lookup + out projection: ids `[batch, frames]` →
    /// quantized latent `[batch, latent_dim, frames]`.
    pub fn decode_codes(&self, codes: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t) = codes.dims2()?;
        let flat = codes.reshape(b * t)?;
        let z_q = self
            .codebook
            .index_select(&flat, 0)?
            .reshape((b, t, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        self.out_project.forward(&z_q)
    }
}

/// `Snake → transposed conv (kernel k, stride s, padding (k−s)/2) → three
/// dilated residual units`, halving channels while upsampling by `s`.
#[derive(Debug)]
struct DecoderBlock {
    snake: Snake1d,
    convtr: ConvTranspose1d,
    /// The official `padding = (k − s) / 2` of the transposed conv. It is
    /// applied here as an output crop (mathematically identical), which
    /// keeps candle on its fast col2im path (only taken for `padding == 0`).
    crop: usize,
    res: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn new(
        input_dim: usize,
        output_dim: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let cfg = ConvTranspose1dConfig {
            stride,
            ..Default::default()
        };
        let res = [1, 3, 9]
            .iter()
            .enumerate()
            .map(|(i, &d)| ResidualUnit::new(output_dim, d, vb.pp(format!("block.{}", i + 2))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            snake: Snake1d::new(input_dim, vb.pp("block.0"))?,
            convtr: conv_transpose1d(input_dim, output_dim, kernel_size, cfg, vb.pp("block.1"))?,
            crop: (kernel_size - stride) / 2,
            res,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.convtr.forward(&self.snake.forward(x)?)?;
        x = x.narrow(D::Minus1, self.crop, x.dim(D::Minus1)? - 2 * self.crop)?;
        for unit in &self.res {
            x = unit.forward(&x)?;
        }
        Ok(x)
    }
}

/// DAC/HiFiGAN-style decoder (the vocoder): 50 Hz latent
/// `[batch, latent_dim, frames]` → waveform `[batch, 1, frames · 320]`
/// in `[-1, 1]` (final `tanh`).
#[derive(Debug)]
pub struct AcousticDecoder {
    conv_in: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake: Snake1d,
    conv_out: Conv1d,
}

impl AcousticDecoder {
    fn new(cfg: &SacCodecConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let conv_in = conv1d(
            cfg.latent_dim,
            cfg.decoder_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("model.0"),
        )?;
        let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
        let mut dim = cfg.decoder_channels;
        for (i, (&k, &s)) in cfg
            .decoder_kernel_sizes
            .iter()
            .zip(cfg.decoder_rates.iter())
            .enumerate()
        {
            blocks.push(DecoderBlock::new(
                dim,
                dim / 2,
                k,
                s,
                vb.pp(format!("model.{}", i + 1)),
            )?);
            dim /= 2;
        }
        let n = cfg.decoder_rates.len();
        Ok(Self {
            conv_in,
            blocks,
            snake: Snake1d::new(dim, vb.pp(format!("model.{}", n + 1)))?,
            conv_out: conv1d(
                dim,
                1,
                7,
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
                vb.pp(format!("model.{}", n + 2)),
            )?,
        })
    }

    /// `latent`: `[batch, latent_dim, frames]` → `[batch, 1, samples]`.
    pub fn forward(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.conv_in.forward(latent)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.conv_out.forward(&self.snake.forward(&x)?)?.tanh()
    }
}

/// Every stage of [`SacCodec::encode`].
#[derive(Debug)]
pub struct SacEncodeOutput {
    /// Continuous encoder latent, `[batch, latent_dim, frames]` (50 Hz).
    pub z: Tensor,
    /// Projected pre-quantization latent, `[batch, codebook_dim, frames]`.
    pub z_e: Tensor,
    /// Codebook ids, `[batch, frames]` (u32).
    pub codes: Tensor,
    /// Out-projected quantized latent, `[batch, latent_dim, frames]`.
    pub zq: Tensor,
}

/// The frozen SAC acoustic codec of X-VC: [`AcousticEncoder`] +
/// [`FactorizedVectorQuantize`] (encode) and [`AcousticDecoder`]
/// (decode, waveform synthesis).
#[derive(Debug)]
pub struct SacCodec {
    pub encoder: AcousticEncoder,
    pub quantizer: FactorizedVectorQuantize,
    pub decoder: AcousticDecoder,
    config: SacCodecConfig,
}

impl SacCodec {
    /// Loads the codec from `xvc_codec.safetensors` (produced by
    /// `tools/convert_xvc_generator.py`, official tensor names with
    /// weight_norm folded) with the default [`SacCodecConfig`].
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(SacCodecConfig::default(), vb)
    }

    /// Builds the codec from a [`VarBuilder`] rooted at the generator
    /// (expects the `acoustic_{encoder,quantizer,decoder}.*` trees).
    pub fn new(config: SacCodecConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: AcousticEncoder::new(&config, vb.pp("acoustic_encoder"))?,
            quantizer: FactorizedVectorQuantize::new(&config, vb.pp("acoustic_quantizer"))?,
            decoder: AcousticDecoder::new(&config, vb.pp("acoustic_decoder"))?,
            config,
        })
    }

    pub fn config(&self) -> &SacCodecConfig {
        &self.config
    }

    /// Encodes a waveform `[batch, 1, samples]` (a multiple of the 320
    /// hop) into the quantized 50 Hz acoustic latent, returning every
    /// intermediate stage.
    pub fn encode(&self, wav: &Tensor) -> Result<SacEncodeOutput> {
        let z = self.encoder.forward(wav)?;
        let z_e = self.quantizer.project(&z)?;
        let codes = self.quantizer.codes(&z_e)?;
        let zq = self.quantizer.decode_codes(&codes)?;
        Ok(SacEncodeOutput { z, z_e, codes, zq })
    }

    /// Decodes a 50 Hz latent `[batch, latent_dim, frames]` (usually the
    /// converter output) into a waveform `[batch, 1, frames · 320]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        Ok(self.decoder.forward(latent)?)
    }
}

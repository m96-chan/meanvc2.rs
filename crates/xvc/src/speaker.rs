//! X-VC speaker encoder — **ERes2Net** (3D-Speaker /
//! `iic/speech_eres2net_sv_en_voxceleb_16k`, frozen by the official
//! Jerrister/X-VC `SpeakerEmbedder`): Kaldi fbank-80 front end →
//! Res2Net trunk with local (AFF-in-block) and global (bottom-up AFF)
//! feature fusion → temporal statistics pooling → 192-d utterance
//! embedding, the speaker condition of the X-VC converter (paper §2.2).
//!
//! Weights: `ckpt/xvc_speaker.safetensors`, converted 1:1 from the official
//! checkpoint with `tools/convert_xvc_speaker.py`. Golden parity vs the
//! official PyTorch stack is covered by `tests/golden_speaker.rs`.
//!
//! Front-end settings (`torchaudio.compliance.kaldi.fbank` defaults with
//! the official overrides): 16 kHz, 25 ms / 10 ms frames, `snip_edges`,
//! povey window, dither 0, DC removal, pre-emphasis 0.97, 512-point FFT
//! power spectrum, 80 Kaldi mel bins over 20 Hz–Nyquist, `log(max(x, ε))`,
//! utterance mean normalization. Unlike the Fast-U2++ front end
//! (`meanvc::v1`), the waveform is **not** rescaled to the int16 range —
//! the official model consumes the preprocessed float wav directly.

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig,
    Linear, Module, ModuleT, VarBuilder,
};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use vc_core::{Error, Result};

const SAMPLE_RATE: usize = 16_000;
const FRAME_LEN: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const N_FFT: usize = 512;
const N_MELS: usize = 80;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
const EPS: f32 = f32::EPSILON;

/// Base channel count (`m_channels`) of the ERes2Net trunk.
const M_CHANNELS: usize = 32;
/// Residual blocks per stage (`num_blocks`).
const NUM_BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// Block output expansion (`BasicBlockERes2Net.expansion`).
const EXPANSION: usize = 2;
/// Embedding dimensionality (`embedding_size`).
const EMBEDDING_SIZE: usize = 192;

fn hz_to_kaldi_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Kaldi mel banks: triangles in mel space over FFT bins, no normalization,
/// Nyquist bin excluded (torchaudio pads it with a zero weight).
fn kaldi_filterbank() -> Vec<Vec<f32>> {
    let n_bins = N_FFT / 2 + 1;
    let high = SAMPLE_RATE as f32 / 2.0;
    let m_lo = hz_to_kaldi_mel(LOW_FREQ);
    let m_hi = hz_to_kaldi_mel(high);
    let delta = (m_hi - m_lo) / (N_MELS + 1) as f32;
    let mut banks = vec![vec![0f32; n_bins]; N_MELS];
    for (m, bank) in banks.iter_mut().enumerate() {
        let left = m_lo + m as f32 * delta;
        let center = left + delta;
        let right = center + delta;
        for (bin, w) in bank.iter_mut().enumerate().take(n_bins - 1) {
            let mel = hz_to_kaldi_mel(bin as f32 * SAMPLE_RATE as f32 / N_FFT as f32);
            if mel > left && mel < right {
                *w = if mel <= center {
                    (mel - left) / delta
                } else {
                    (right - mel) / delta
                };
            }
        }
    }
    banks
}

/// Kaldi fbank-80 front end of the X-VC speaker encoder
/// (`torchaudio.compliance.kaldi.fbank` with `num_mel_bins=80`, `dither=0`,
/// followed by utterance mean normalization — the official `FBank` wrapper
/// with `mean_nor=True`).
pub struct KaldiFbank {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
}

impl std::fmt::Debug for KaldiFbank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KaldiFbank").finish()
    }
}

impl Default for KaldiFbank {
    fn default() -> Self {
        Self::new()
    }
}

impl KaldiFbank {
    pub fn new() -> Self {
        // Povey window: hann^0.85.
        let window: Vec<f32> = (0..FRAME_LEN)
            .map(|i| {
                let hann = 0.5
                    - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (FRAME_LEN - 1) as f32).cos();
                hann.powf(0.85)
            })
            .collect();
        Self {
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
            window,
            filterbank: kaldi_filterbank(),
        }
    }

    /// Mono `samples` at 16 kHz (preprocessed float wav, **not** rescaled to
    /// int16 range) → `[frames, 80]` mean-normalized log-mel features
    /// (`snip_edges` framing: `frames = 1 + (len - 400) / 160`).
    pub fn compute(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        if samples.len() < FRAME_LEN {
            return Err(Error::Input(format!(
                "need at least {FRAME_LEN} samples, got {}",
                samples.len()
            )));
        }
        let frames = 1 + (samples.len() - FRAME_LEN) / FRAME_SHIFT;
        let mut out = vec![0f32; frames * N_MELS];
        let mut frame = vec![0f32; FRAME_LEN];
        let mut buf = vec![Complex32::default(); N_FFT];
        let mut power = vec![0f32; N_FFT / 2 + 1];

        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            frame.copy_from_slice(&samples[start..start + FRAME_LEN]);
            // Remove DC offset.
            let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }
            // Pre-emphasis (kaldi: x[0] -= p * x[0]).
            for i in (1..FRAME_LEN).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];

            buf.fill(Complex32::default());
            for i in 0..FRAME_LEN {
                buf[i] = Complex32::new(frame[i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (bin, p) in power.iter_mut().enumerate() {
                *p = buf[bin].norm_sqr();
            }
            for (m, bank) in self.filterbank.iter().enumerate() {
                let mel: f32 = bank.iter().zip(&power).map(|(w, p)| w * p).sum();
                out[f * N_MELS + m] = mel.max(EPS).ln();
            }
        }

        // Utterance mean normalization (`feat - feat.mean(0)`).
        for m in 0..N_MELS {
            let mean = (0..frames).map(|f| out[f * N_MELS + m]).sum::<f32>() / frames as f32;
            for f in 0..frames {
                out[f * N_MELS + m] -= mean;
            }
        }
        Ok(Tensor::from_vec(out, (frames, N_MELS), device)?)
    }
}

/// Hardtanh(0, 20) — the `ReLU` used inside every ERes2Net block.
fn relu20(x: &Tensor) -> candle_core::Result<Tensor> {
    x.clamp(0f32, 20f32)
}

/// Attentional feature fusion (`fusion.AFF`): a 1×1 bottleneck attention
/// over the concatenated inputs gates a soft mix of `x` and `ds_y`.
#[derive(Debug)]
struct Aff {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl Aff {
    /// `channels` per input; bottleneck `channels / 4` (`r = 4`).
    fn new(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let inter = channels / 4;
        let vb = vb.pp("local_att");
        Ok(Self {
            conv1: conv2d(2 * channels, inter, 1, Default::default(), vb.pp("0"))?,
            bn1: batch_norm(inter, BatchNormConfig::default(), vb.pp("1"))?,
            conv2: conv2d(inter, channels, 1, Default::default(), vb.pp("3"))?,
            bn2: batch_norm(channels, BatchNormConfig::default(), vb.pp("4"))?,
        })
    }

    /// `x`, `ds_y`: `[B, C, F, T]` → `[B, C, F, T]`.
    fn forward(&self, x: &Tensor, ds_y: &Tensor) -> candle_core::Result<Tensor> {
        let xa = Tensor::cat(&[x, ds_y], 1)?;
        let a = self.bn1.forward_t(&self.conv1.forward(&xa)?, false)?;
        let a = self
            .bn2
            .forward_t(&self.conv2.forward(&a.silu()?)?, false)?;
        let att = (a.tanh()? + 1.0)?;
        // x * att + ds_y * (2 - att)
        (x * &att)? + (ds_y * att.affine(-1.0, 2.0)?)?
    }
}

/// One ERes2Net residual block (`BasicBlockERes2Net` /
/// `BasicBlockERes2Net_diff_AFF`, `baseWidth = 32`, `scale = 2`): a 1×1
/// reduction into two `width`-channel branches, the second fused with the
/// processed first (by addition, or by [`Aff`] in the `diff_AFF` variant),
/// then a 1×1 expansion to `planes * 2` with a residual shortcut.
#[derive(Debug)]
struct Res2NetBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    convs: Vec<Conv2d>,
    bns: Vec<BatchNorm>,
    /// `Some` in the `diff_AFF` variant (layers 3–4).
    fuse: Option<Aff>,
    conv3: Conv2d,
    bn3: BatchNorm,
    shortcut: Option<(Conv2d, BatchNorm)>,
    width: usize,
}

impl Res2NetBlock {
    fn new(
        in_planes: usize,
        planes: usize,
        stride: usize,
        with_aff: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let width = planes / 2; // planes * baseWidth(32) / 64
        let scale = 2;
        let out_planes = planes * EXPANSION;
        let stride_cfg = Conv2dConfig {
            stride,
            ..Default::default()
        };
        let pad_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut convs = Vec::with_capacity(scale);
        let mut bns = Vec::with_capacity(scale);
        for i in 0..scale {
            convs.push(conv2d_no_bias(
                width,
                width,
                3,
                pad_cfg,
                vb.pp(format!("convs.{i}")),
            )?);
            bns.push(batch_norm(
                width,
                BatchNormConfig::default(),
                vb.pp(format!("bns.{i}")),
            )?);
        }
        let fuse = if with_aff {
            Some(Aff::new(width, vb.pp("fuse_models.0"))?)
        } else {
            None
        };
        let shortcut = if stride != 1 || in_planes != out_planes {
            Some((
                conv2d_no_bias(in_planes, out_planes, 1, stride_cfg, vb.pp("shortcut.0"))?,
                batch_norm(out_planes, BatchNormConfig::default(), vb.pp("shortcut.1"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1: conv2d_no_bias(in_planes, width * scale, 1, stride_cfg, vb.pp("conv1"))?,
            bn1: batch_norm(width * scale, BatchNormConfig::default(), vb.pp("bn1"))?,
            convs,
            bns,
            fuse,
            conv3: conv2d_no_bias(
                width * scale,
                out_planes,
                1,
                Default::default(),
                vb.pp("conv3"),
            )?,
            bn3: batch_norm(out_planes, BatchNormConfig::default(), vb.pp("bn3"))?,
            shortcut,
            width,
        })
    }

    /// `x`: `[B, in_planes, F, T]` → `[B, planes * 2, F / stride, T / stride]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let out = relu20(&self.bn1.forward_t(&self.conv1.forward(x)?, false)?)?;
        let spx0 = out.narrow(1, 0, self.width)?;
        let spx1 = out.narrow(1, self.width, self.width)?;
        let sp0 = relu20(&self.bns[0].forward_t(&self.convs[0].forward(&spx0)?, false)?)?;
        let sp1_in = match &self.fuse {
            Some(aff) => aff.forward(&sp0, &spx1)?,
            None => (&sp0 + &spx1)?,
        };
        let sp1 = relu20(&self.bns[1].forward_t(&self.convs[1].forward(&sp1_in)?, false)?)?;
        let out = Tensor::cat(&[&sp0, &sp1], 1)?;
        let out = self.bn3.forward_t(&self.conv3.forward(&out)?, false)?;
        let residual = match &self.shortcut {
            Some((conv, bn)) => bn.forward_t(&conv.forward(x)?, false)?,
            None => x.clone(),
        };
        relu20(&(out + residual)?)
    }
}

/// A stack of [`Res2NetBlock`]s (`ERes2Net._make_layer`): the first block
/// carries the stride, the rest are stride 1.
fn make_layer(
    in_planes: &mut usize,
    planes: usize,
    num_blocks: usize,
    stride: usize,
    with_aff: bool,
    vb: VarBuilder,
) -> candle_core::Result<Vec<Res2NetBlock>> {
    let mut blocks = Vec::with_capacity(num_blocks);
    for (i, s) in std::iter::once(stride)
        .chain(std::iter::repeat(1))
        .take(num_blocks)
        .enumerate()
    {
        blocks.push(Res2NetBlock::new(
            *in_planes,
            planes,
            s,
            with_aff,
            vb.pp(format!("{i}")),
        )?);
        *in_planes = planes * EXPANSION;
    }
    Ok(blocks)
}

fn forward_layer(blocks: &[Res2NetBlock], x: &Tensor) -> candle_core::Result<Tensor> {
    let mut x = x.clone();
    for block in blocks {
        x = block.forward(&x)?;
    }
    Ok(x)
}

/// ERes2Net trunk (`num_blocks = [3, 4, 6, 3]`, `m_channels = 32`,
/// `feat_dim = 80`, TSTP pooling, single embedding layer): four Res2Net
/// stages with bottom-up AFF fusion of the downsampled shallower stages,
/// temporal statistics pooling, and a linear projection to 192-d.
#[derive(Debug)]
pub struct ERes2Net {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<Res2NetBlock>,
    layer2: Vec<Res2NetBlock>,
    layer3: Vec<Res2NetBlock>,
    layer4: Vec<Res2NetBlock>,
    layer1_downsample: Conv2d,
    layer2_downsample: Conv2d,
    layer3_downsample: Conv2d,
    fuse_mode12: Aff,
    fuse_mode123: Aff,
    fuse_mode1234: Aff,
    seg_1: Linear,
}

impl ERes2Net {
    /// Builds the model from a [`VarBuilder`] rooted at the official
    /// checkpoint names (`conv1.weight`, `layer1.0...`, `seg_1...`).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let m = M_CHANNELS;
        let ds_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        // stats_dim = feat_dim / 8 * m_channels * 8 = 80 * 32; TSTP emits
        // mean ∥ std of the expanded final stage.
        let stats_dim = N_MELS / 8 * m * 8 * EXPANSION * 2;
        let mut in_planes = m;
        Ok(Self {
            conv1: conv2d_no_bias(
                1,
                m,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("conv1"),
            )?,
            bn1: batch_norm(m, BatchNormConfig::default(), vb.pp("bn1"))?,
            layer1: make_layer(&mut in_planes, m, NUM_BLOCKS[0], 1, false, vb.pp("layer1"))?,
            layer2: make_layer(
                &mut in_planes,
                m * 2,
                NUM_BLOCKS[1],
                2,
                false,
                vb.pp("layer2"),
            )?,
            layer3: make_layer(
                &mut in_planes,
                m * 4,
                NUM_BLOCKS[2],
                2,
                true,
                vb.pp("layer3"),
            )?,
            layer4: make_layer(
                &mut in_planes,
                m * 8,
                NUM_BLOCKS[3],
                2,
                true,
                vb.pp("layer4"),
            )?,
            layer1_downsample: conv2d_no_bias(m * 2, m * 4, 3, ds_cfg, vb.pp("layer1_downsample"))?,
            layer2_downsample: conv2d_no_bias(m * 4, m * 8, 3, ds_cfg, vb.pp("layer2_downsample"))?,
            layer3_downsample: conv2d_no_bias(
                m * 8,
                m * 16,
                3,
                ds_cfg,
                vb.pp("layer3_downsample"),
            )?,
            fuse_mode12: Aff::new(m * 4, vb.pp("fuse_mode12"))?,
            fuse_mode123: Aff::new(m * 8, vb.pp("fuse_mode123"))?,
            fuse_mode1234: Aff::new(m * 16, vb.pp("fuse_mode1234"))?,
            seg_1: linear(stats_dim, EMBEDDING_SIZE, vb.pp("seg_1"))?,
        })
    }

    /// Loads the converted official weights
    /// (`tools/convert_xvc_speaker.py` → `ckpt/xvc_speaker.safetensors`).
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], DType::F32, device)? };
        Self::new(vb)
    }

    /// Temporal statistics pooling (TSTP): mean ∥ std over the time axis
    /// (unbiased variance + 1e-8, matching `torch.var`).
    /// `x`: `[B, C, F, T]` → `[B, 2 * C * F]`.
    fn pool(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, f, t) = x.dims4()?;
        let mean = x.mean(3)?;
        let centered = x.broadcast_sub(&mean.unsqueeze(3)?)?;
        let var = (centered.sqr()?.sum(3)? / (t - 1) as f64)?;
        let std = (var + 1e-8)?.sqrt()?;
        Tensor::cat(&[mean.reshape((b, c * f))?, std.reshape((b, c * f))?], 1)
    }

    /// `fbank`: `[B, T, 80]` mean-normalized features → `[B, 192]`
    /// embedding (`embed_a`; the model is single-embedding-layer).
    pub fn forward(&self, fbank: &Tensor) -> Result<Tensor> {
        let x = fbank.permute((0, 2, 1))?.unsqueeze(1)?; // [B, 1, F, T]
        let out = self
            .bn1
            .forward_t(&self.conv1.forward(&x)?, false)?
            .relu()?;
        let out1 = forward_layer(&self.layer1, &out)?;
        let out2 = forward_layer(&self.layer2, &out1)?;
        let out1_ds = self.layer1_downsample.forward(&out1)?;
        let fuse12 = self.fuse_mode12.forward(&out2, &out1_ds)?;
        let out3 = forward_layer(&self.layer3, &out2)?;
        let fuse12_ds = self.layer2_downsample.forward(&fuse12)?;
        let fuse123 = self.fuse_mode123.forward(&out3, &fuse12_ds)?;
        let out4 = forward_layer(&self.layer4, &out3)?;
        let fuse123_ds = self.layer3_downsample.forward(&fuse123)?;
        let fuse1234 = self.fuse_mode1234.forward(&out4, &fuse123_ds)?;
        let stats = self.pool(&fuse1234)?;
        Ok(self.seg_1.forward(&stats)?)
    }
}

/// The full X-VC speaker encoder: [`KaldiFbank`] front end + [`ERes2Net`].
#[derive(Debug)]
pub struct SpeakerEncoder {
    fbank: KaldiFbank,
    model: ERes2Net,
    device: Device,
}

impl SpeakerEncoder {
    /// Loads `ckpt/xvc_speaker.safetensors`
    /// (see `tools/convert_xvc_speaker.py`).
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        Ok(Self {
            fbank: KaldiFbank::new(),
            model: ERes2Net::load(path, device)?,
            device: device.clone(),
        })
    }

    /// Mono reference `samples` at 16 kHz (preprocessed float wav) →
    /// `[1, 192]` speaker embedding.
    pub fn embed(&self, samples: &[f32]) -> Result<Tensor> {
        let feats = self.fbank.compute(samples, &self.device)?.unsqueeze(0)?;
        self.model.forward(&feats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn fbank_shape_and_finiteness() {
        let fb = KaldiFbank::new();
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin() * 0.3)
            .collect();
        let t = fb.compute(&samples, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1 + (16_000 - 400) / 160, 80]);
        let v: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn fbank_mean_normalized() {
        let fb = KaldiFbank::new();
        let samples: Vec<f32> = (0..8_000)
            .map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.2)
            .collect();
        let t = fb.compute(&samples, &Device::Cpu).unwrap();
        // Per-bin mean over frames is ~0 after utterance mean-norm.
        let mean = t.mean(0).unwrap();
        let v: Vec<f32> = mean.to_vec1().unwrap();
        assert!(
            v.iter().all(|x| x.abs() < 1e-4),
            "means not centered: {v:?}"
        );
    }

    #[test]
    fn fbank_rejects_short_input() {
        let fb = KaldiFbank::new();
        assert!(fb.compute(&[0.0; 399], &Device::Cpu).is_err());
    }

    #[test]
    fn eres2net_output_shape() {
        // Random-init model (VarMap creates missing tensors on demand):
        // checks wiring, shapes and the [B, 192] contract.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = ERes2Net::new(vb).unwrap();
        let fbank = Tensor::randn(0f32, 1f32, (2, 200, 80), &Device::Cpu).unwrap();
        let emb = model.forward(&fbank).unwrap();
        assert_eq!(emb.dims(), &[2, EMBEDDING_SIZE]);
        let v: Vec<f32> = emb.flatten_all().unwrap().to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

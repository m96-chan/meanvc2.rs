//! Live-microphone streaming driver.
//!
//! **This does not reproduce the official streaming scheme.** CosyVoice2's
//! own `inference_vc(..., stream=True)` renders mel/audio incrementally
//! from an already-fully-known source token sequence (`frontend_vc`
//! tokenizes the *entire* source clip up front, in one non-causal FSQ
//! tokenizer pass, before any chunked rendering begins) — it streams the
//! *output*, not the *input*, and has no notion of unbounded live mic
//! audio. That's a dead end for babiniku's live-mic use case (flagged as
//! an open risk in the issue #71 recon: "the FSQ tokenizer is a
//! full-attention non-causal encoder... unproven until tried").
//!
//! Instead this follows the same shape as [`crate` sibling]
//! `crates/seedvc/src/stream.rs`: a sliding source window is
//! **re-tokenized and re-rendered from scratch every hop**, and only the
//! newly-settled tail is emitted, joined to the previous hop's tail with
//! a short raised-cosine crossfade (24 kHz). Unlike Seed-VC's DiT, HiFT's
//! GAN decoder has no run-to-run diffusion variance, so a plain crossfade
//! (no SOLA phase search) is enough to hide the seam — re-verify against
//! the demo before assuming that holds under all conditions.
//!
//! Cost: the flow (encoder + 10-step CFM) reruns over the whole window
//! every hop — O(window) per hop, not incremental. Combined with CosyVoice2
//! being GPU-bound even offline (issue #71: CPU RTF 1.3+), this engine is
//! **CUDA/Metal-only for live use** (documented non-goal in issue #75).

use candle_core::Tensor;
use vc_core::profile::resample_analysis;
use vc_core::Result;

use crate::hift::NoiseMode;
use crate::pipeline::{CosyVoiceEngine, Reference};
use crate::{MEL_SR, TOKEN_RATE, TOKEN_SR};

/// Samples per speech token at 16 kHz (`TOKEN_SR / TOKEN_RATE`).
const SAMPLES_PER_TOKEN: usize = (TOKEN_SR as usize) / TOKEN_RATE;

/// Streaming parameters (16 kHz sample units unless noted).
#[derive(Clone, Copy)]
pub struct StreamConfig {
    /// New source audio consumed per hop, in 16 kHz samples. Must be a
    /// multiple of [`SAMPLES_PER_TOKEN`]. 16 000 = 1.0 s ⇒ 25 tokens,
    /// matching CosyVoice2's own training `chunk_size`.
    pub block: usize,
    /// Left context re-tokenized alongside each new block, in 16 kHz
    /// samples (also a multiple of [`SAMPLES_PER_TOKEN`]). Gives the
    /// non-causal tokenizer and conformer encoder enough history for
    /// stable tokens/mel at the block boundary. 48 000 = 3.0 s.
    pub context: usize,
    /// Crossfade length at 24 kHz samples where consecutive hops'
    /// renders are joined. 1 920 ≈ 80 ms.
    pub crossfade_24k: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            block: 16_000,
            context: 48_000,
            crossfade_24k: 1_920,
        }
    }
}

fn raised_cosine(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::PI * i as f32 / n.max(1) as f32).cos())
        .collect()
}

/// A live streaming session against a fixed reference speaker.
pub struct CosyVoiceStream<'a> {
    engine: &'a CosyVoiceEngine,
    reference: Reference,
    cfg: StreamConfig,
    /// Rolling raw 16 kHz source buffer (all audio seen so far, trimmed
    /// to `context + block` once a hop consumes it).
    buf: Vec<f32>,
    /// Last `crossfade_24k` samples of the previous hop's render, held
    /// back for blending; `None` before the first hop.
    tail: Option<Vec<f32>>,
    fade_in: Vec<f32>,
    fade_out: Vec<f32>,
}

impl CosyVoiceEngine {
    /// Open a streaming session against `reference_audio` (any sample
    /// rate — resampled internally).
    pub fn stream(
        &self,
        reference_audio: &[f32],
        reference_sr: u32,
        cfg: StreamConfig,
    ) -> Result<CosyVoiceStream<'_>> {
        assert_eq!(
            cfg.block % SAMPLES_PER_TOKEN,
            0,
            "block must be a multiple of {SAMPLES_PER_TOKEN}"
        );
        assert_eq!(
            cfg.context % SAMPLES_PER_TOKEN,
            0,
            "context must be a multiple of {SAMPLES_PER_TOKEN}"
        );
        let reference = self.prepare_reference(reference_audio, reference_sr)?;
        let fade_in = raised_cosine(cfg.crossfade_24k);
        let fade_out: Vec<f32> = fade_in.iter().map(|v| 1.0 - v).collect();
        Ok(CosyVoiceStream {
            engine: self,
            reference,
            cfg,
            buf: Vec::new(),
            tail: None,
            fade_in,
            fade_out,
        })
    }
}

impl CosyVoiceStream<'_> {
    /// Feed new source audio (any sample rate — resampled internally to
    /// 16 kHz and appended to the rolling buffer).
    pub fn push(&mut self, samples: &[f32], sr: u32) {
        let s16k = resample_analysis(samples, sr as usize, TOKEN_SR as usize);
        self.buf.extend_from_slice(&s16k);
    }

    /// Whether a full hop's worth of new audio is buffered.
    pub fn ready(&self) -> bool {
        self.buf.len() >= self.cfg.block
    }

    /// Render one hop. Returns `None` if [`Self::ready`] is false.
    /// Output is 48 kHz mono audio (block-sized, minus the held-back
    /// crossfade tail — call again / drain at end-of-stream to flush it).
    /// The crossfade itself runs at HiFT's native 24 kHz for precision;
    /// the exact ×2 upsample to [`STREAM_OUT_SR`] happens last, matching
    /// how `crates/seedvc/src/stream.rs` hands the TUI ready-to-play
    /// audio instead of leaving resampling to the caller.
    pub fn step(&mut self) -> Result<Option<Vec<f32>>> {
        if !self.ready() {
            return Ok(None);
        }
        let window_len = (self.cfg.context + self.cfg.block).min(self.buf.len());
        let window = &self.buf[self.buf.len() - window_len..];

        let src_tokens = self.engine.tokenize(window)?;
        let tokens = Tensor::cat(&[&self.reference.tokens, &src_tokens], 1)?;
        let mu = self.engine.flow_ref().mu(&tokens, false, true)?;
        let mel = self.engine.flow_ref().cfm(
            &mu,
            &self.reference.embedding,
            &self.reference.feat,
            false,
        )?;
        let (audio, _source) = self.engine.hift_ref().vocode(&mel, NoiseMode::Random)?;

        // Consume the block from the rolling buffer now that it's rendered.
        let consumed = self.cfg.block.min(self.buf.len());
        self.buf.drain(0..consumed);

        let cf = self.cfg.crossfade_24k.min(audio.len() / 2);
        let out_24k = match self.tail.take() {
            Some(prev_tail) if cf > 0 => {
                let mut blended = Vec::with_capacity(audio.len() - cf);
                for i in 0..cf {
                    blended.push(prev_tail[i] * self.fade_out[i] + audio[i] * self.fade_in[i]);
                }
                blended.extend_from_slice(&audio[cf..audio.len() - cf]);
                self.tail = Some(audio[audio.len() - cf..].to_vec());
                blended
            }
            _ => {
                if cf > 0 {
                    self.tail = Some(audio[audio.len() - cf..].to_vec());
                    audio[..audio.len() - cf].to_vec()
                } else {
                    audio
                }
            }
        };
        Ok(Some(resample_analysis(
            &out_24k,
            MEL_SR as usize,
            STREAM_OUT_SR as usize,
        )))
    }

    /// Flush the held-back crossfade tail at end-of-stream, resampled to
    /// [`STREAM_OUT_SR`] like [`Self::step`].
    pub fn finish(&mut self) -> Option<Vec<f32>> {
        let tail = self.tail.take()?;
        Some(resample_analysis(
            &tail,
            MEL_SR as usize,
            STREAM_OUT_SR as usize,
        ))
    }
}

/// Sample rate of [`CosyVoiceStream::step`]'s output — the TUI's shared
/// 48 kHz output-thread domain (playback device, `--out` writer, output
/// NR/leveler/EQ/exciter/limiter), same rate Seed-VC's stream emits at.
pub const STREAM_OUT_SR: u32 = 48_000;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ckpt_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ckpt")
    }

    fn have_ckpts() -> bool {
        [
            "cosyvoice_mel.safetensors",
            "cosyvoice_tokenizer.safetensors",
            "cosyvoice_campplus.safetensors",
            "cosyvoice_flow.safetensors",
            "cosyvoice_hift.safetensors",
        ]
        .iter()
        .all(|f| ckpt_dir().join(f).exists())
    }

    fn read_wav_16k(path: &std::path::Path) -> Vec<f32> {
        let mut r = hound::WavReader::open(path).unwrap();
        let spec = r.spec();
        assert_eq!(spec.sample_rate, 16_000);
        r.samples::<i32>()
            .step_by(spec.channels as usize)
            .map(|v| v.unwrap() as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32)
            .collect()
    }

    /// Mechanics smoke test: push/ready/step across a few hops of real
    /// audio produces finite, correctly-shaped, non-degenerate output.
    /// There is no official streaming-from-live-mic baseline to golden
    /// against (see module docs).
    #[test]
    fn streaming_produces_valid_audio() {
        if !have_ckpts() {
            return;
        }
        let ref_path = ckpt_dir().join("F19_01_16k.wav");
        let src_path = ckpt_dir().join("ref_trimmed.wav");
        if !ref_path.exists() || !src_path.exists() {
            return;
        }
        let engine = candle_core::Device::Cpu;
        let engine = CosyVoiceEngine::load(ckpt_dir(), &engine).unwrap();
        let reference_audio = read_wav_16k(&ref_path);
        let source_audio = read_wav_16k(&src_path);

        let cfg = StreamConfig {
            block: 16_000,
            context: 32_000,
            crossfade_24k: 960,
        };
        let mut stream = engine.stream(&reference_audio, TOKEN_SR, cfg).unwrap();

        let mut out = Vec::new();
        for chunk in source_audio.chunks(4_000) {
            stream.push(chunk, TOKEN_SR);
            while stream.ready() {
                if let Some(block) = stream.step().unwrap() {
                    out.extend(block);
                }
            }
        }
        if let Some(tail) = stream.finish() {
            out.extend(tail);
        }

        assert!(!out.is_empty());
        assert!(out.iter().all(|s| s.is_finite()));
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 1e-4 && rms < 1.0, "implausible RMS {rms}");
    }
}

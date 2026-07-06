//! The X-VC end-to-end pipeline: offline conversion and the official
//! stateless chunk-streaming driver.
//!
//! Mirrors `bins/infer_utils.py` of the official implementation
//! (X-VC arXiv:2604.12456):
//!
//! * [`XvcEngine`] loads every frozen stage once (tokenizer + semantic
//!   adapter, speaker encoder, SAC codec, prenet, MMDiT converter) and
//!   exposes [`XvcEngine::convert_offline`] (= `run_offline`) plus
//!   [`XvcEngine::forward_window`] (= `run_stream_chunk_forward`).
//! * [`XvcStream`] is the chunk-streaming state machine
//!   (= `run_streaming`): every hop re-encodes a whole
//!   `chunk_ms` window `[history | current | smooth | future]` through the
//!   full stack, keeps the `current` slice and cross-fades `smooth_ms`
//!   (raised cosine) with the previous window's tail. The default
//!   [`StreamConfig`] is the CPU-viable 640/240/100/20 preset from the
//!   issue #30 recon; the official GPU preset is 2400/120/100/20.
//!
//! The prenet stage (`prenet`, a `Decoder_with_upsample` like the semantic
//! adapter but with `sample_ratios = [1, 1]`, plain LayerNorm — the
//! speaker-condition argument is unused because the official config leaves
//! `condition_dim = None`) fuses the 1024-d semantic and 1024-d acoustic
//! latents (2048 → 1024) before the converter.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::{Device, IndexOp, Tensor};
use vc_core::{Error, Result};

use crate::codec::SacCodec;
use crate::converter::AcousticConverter;
use crate::preprocess::{preprocess, FrameMelExtractor, PreprocessConfig, WhisperFeatureExtractor};
use crate::speaker::SpeakerEncoder;
use crate::tokenizer::{SemanticAdapter, SemanticAdapterConfig, WhisperVqEncoder};

/// Sample rate of every official X-VC checkpoint.
pub const SAMPLE_RATE: usize = 16_000;

/// The prenet configuration (`configs/xvc.yaml` `prenet`): fuse
/// `[semantic 1024 ‖ acoustic 1024]` → 1024 at 50 Hz, Vocos dim 768,
/// 12 + 2·2 ConvNeXt blocks, no upsampling (`sample_ratios = [1, 1]`).
pub fn prenet_config() -> SemanticAdapterConfig {
    SemanticAdapterConfig {
        input_channels: 2048,
        vocos_dim: 768,
        vocos_intermediate_dim: 2048,
        vocos_num_layers: 12,
        out_channels: 1024,
        sample_ratios: vec![1, 1],
    }
}

/// Precomputed target-speaker conditions (`precompute_conditions`):
/// computed once per reference, shared by every window.
#[derive(Debug, Clone)]
pub struct Reference {
    /// ERes2Net utterance embedding, `[1, 192]`.
    pub speaker_condition: Tensor,
    /// Target dB-mel (`FrameMelExtractor`), `[1, 128, frames]`.
    pub frame_condition: Tensor,
}

/// Wall time of the three pipeline stages of one window forward.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    /// Whisper mel + tokenizer + semantic adapter.
    pub semantic: Duration,
    /// Codec encode + prenet + converter.
    pub acoustic: Duration,
    /// Codec decode (waveform synthesis).
    pub decode: Duration,
}

impl StageTimings {
    pub fn total(&self) -> Duration {
        self.semantic + self.acoustic + self.decode
    }
}

/// Every intermediate of one window forward
/// (`run_stream_chunk_forward`) — cheap to return, tensors are refcounted.
#[derive(Debug)]
pub struct WindowOutput {
    /// Semantic VQ ids, `[1, tokens]` i64 (12.5 Hz).
    pub token_ids: Tensor,
    /// Semantic adapter output, `[1, tokens · 4, 1024]` (50 Hz).
    pub sem_adapter_out: Tensor,
    /// Quantized acoustic latent, `[1, 1024, frames]` (50 Hz).
    pub acoustic_zq: Tensor,
    /// Prenet fusion output, `[1, 1024, frames]`.
    pub prenet_out: Tensor,
    /// Converter output latent, `[1, 1024, frames]`.
    pub converter_out: Tensor,
    /// Decoded waveform, `[1, 1, frames · 320]`.
    pub wav: Tensor,
    pub timings: StageTimings,
}

/// The X-VC engine: every frozen component loaded once.
pub struct XvcEngine {
    pub whisper_mel: WhisperFeatureExtractor,
    pub tokenizer: WhisperVqEncoder,
    pub semantic_adapter: SemanticAdapter,
    pub speaker: SpeakerEncoder,
    pub codec: SacCodec,
    pub prenet: SemanticAdapter,
    pub converter: AcousticConverter,
    pub frame_mel: FrameMelExtractor,
    device: Device,
}

impl std::fmt::Debug for XvcEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XvcEngine").finish()
    }
}

impl XvcEngine {
    /// Loads all converted checkpoints from `ckpt_dir`
    /// (`xvc_{tokenizer,speaker,codec,converter,prenet}.safetensors`,
    /// produced by `tools/convert_xvc_tokenizer.py`,
    /// `tools/convert_xvc_speaker.py` and
    /// `tools/convert_xvc_generator.py`). Fails with a clear message when
    /// a file is missing.
    pub fn load(ckpt_dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let dir = ckpt_dir.as_ref();
        let need = |name: &str| -> Result<PathBuf> {
            let p = dir.join(name);
            if !p.exists() {
                return Err(Error::Input(format!(
                    "X-VC checkpoint {} not found — convert the official weights with \
                     tools/convert_xvc_tokenizer.py, tools/convert_xvc_speaker.py and \
                     tools/convert_xvc_generator.py (see docs/xvc.md)",
                    p.display()
                )));
            }
            Ok(p)
        };
        let (tokenizer, semantic_adapter) =
            crate::tokenizer::load(need("xvc_tokenizer.safetensors")?, device)?;
        let speaker = SpeakerEncoder::load(need("xvc_speaker.safetensors")?, device)?;
        let codec = SacCodec::load(need("xvc_codec.safetensors")?, device)?;
        let converter = AcousticConverter::load(need("xvc_converter.safetensors")?, device)?;
        let prenet_path = need("xvc_prenet.safetensors")?;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[prenet_path],
                candle_core::DType::F32,
                device,
            )?
        };
        let prenet = SemanticAdapter::new(prenet_config(), vb.pp("prenet"))?;
        Ok(Self {
            whisper_mel: WhisperFeatureExtractor::default(),
            tokenizer,
            semantic_adapter,
            speaker,
            codec,
            prenet,
            converter,
            frame_mel: FrameMelExtractor::default(),
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Runs the official preprocessing (volume norm / 40 Hz high-pass /
    /// pad to 1280) on raw float64 samples in `[-1, 1]`.
    pub fn preprocess(&self, samples: &[f64]) -> Vec<f32> {
        preprocess(samples, &PreprocessConfig::default())
    }

    /// Precomputes the target conditions (`precompute_conditions`) from
    /// **preprocessed** reference samples ([`Self::preprocess`]).
    pub fn prepare_reference(&self, target: &[f32]) -> Result<Reference> {
        Ok(Reference {
            speaker_condition: self.speaker.embed(target)?,
            frame_condition: self.frame_mel.extract(target, &self.device)?,
        })
    }

    /// One full-stack forward (`run_stream_chunk_forward`): preprocessed
    /// source samples (a multiple of 1280) → converted waveform of the
    /// same length, with every intermediate and per-stage wall time.
    pub fn forward_window(&self, samples: &[f32], reference: &Reference) -> Result<WindowOutput> {
        if samples.is_empty() || samples.len() % 1280 != 0 {
            return Err(Error::Input(format!(
                "window length {} must be a non-zero multiple of 1280 samples",
                samples.len()
            )));
        }
        let wav_in = Tensor::from_vec(samples.to_vec(), (1, 1, samples.len()), &self.device)?;

        // Semantic branch: Whisper mel → VQ ids → embeddings → 50 Hz.
        let t0 = Instant::now();
        let feats = self.whisper_mel.extract(samples, &self.device)?;
        let tok = self
            .tokenizer
            .forward(&feats.input_features, &feats.attention_mask)?;
        let sem_emb = self.tokenizer.embed_ids(&tok.token_ids)?;
        let sem_up = self
            .semantic_adapter
            .forward(&sem_emb.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?; // [1, T50, 1024]
        let t_semantic = t0.elapsed();

        // Acoustic branch + fusion + conversion.
        let t0 = Instant::now();
        let enc = self.codec.encode(&wav_in)?;
        let acu_emb = enc.zq.transpose(1, 2)?.contiguous()?; // [1, T50, 1024]
        let combined = Tensor::cat(&[&sem_up, &acu_emb], 2)?; // [1, T50, 2048]
        let prenet_out = self
            .prenet
            .forward(&combined.transpose(1, 2)?.contiguous()?)?;
        let converter_out = self.converter.forward(
            &prenet_out,
            &reference.frame_condition,
            &reference.speaker_condition,
        )?;
        let t_acoustic = t0.elapsed();

        // Waveform synthesis.
        let t0 = Instant::now();
        let wav = self.codec.decode(&converter_out)?;
        let t_decode = t0.elapsed();

        Ok(WindowOutput {
            token_ids: tok.token_ids,
            sem_adapter_out: sem_up,
            acoustic_zq: enc.zq,
            prenet_out,
            converter_out,
            wav,
            timings: StageTimings {
                semantic: t_semantic,
                acoustic: t_acoustic,
                decode: t_decode,
            },
        })
    }

    /// Offline conversion (`run_offline` = `XVC.inference`): raw float64
    /// source samples → converted waveform (length = source padded to
    /// 1280). The reference conditions come from
    /// [`Self::prepare_reference`].
    pub fn convert_offline(&self, source: &[f64], reference: &Reference) -> Result<Vec<f32>> {
        let processed = self.preprocess(source);
        let out = self.forward_window(&processed, reference)?;
        Ok(out.wav.flatten_all()?.to_vec1()?)
    }

    /// Starts a chunk-streaming session over `reference` with the given
    /// window parameters.
    pub fn stream(&self, reference: Reference, cfg: StreamConfig) -> Result<XvcStream<'_>> {
        XvcStream::new(self, reference, cfg)
    }
}

/// Streaming window parameters (`run_streaming` arguments), all in
/// milliseconds. Window layout per hop:
/// `[history | current | smooth | future]` with
/// `history = chunk − current − smooth − future`.
#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    /// Full re-encoded window (`chunk_ms`). Must be a multiple of 80 ms
    /// (the 1280-sample latent hop).
    pub chunk_ms: usize,
    /// New samples emitted per hop (`current_ms`).
    pub current_ms: usize,
    /// Lookahead (`future_ms`).
    pub future_ms: usize,
    /// Raised-cosine crossfade at chunk joins (`smooth_ms`).
    pub smooth_ms: usize,
}

impl Default for StreamConfig {
    /// The CPU-viable preset from the issue #30 recon (640 ms window,
    /// 240 ms hop): algorithmic latency = current + smooth + future
    /// = 360 ms.
    fn default() -> Self {
        Self {
            chunk_ms: 640,
            current_ms: 240,
            future_ms: 100,
            smooth_ms: 20,
        }
    }
}

impl StreamConfig {
    /// The official GPU streaming preset (2.4 s window, 120 ms hop).
    pub fn official() -> Self {
        Self {
            chunk_ms: 2400,
            current_ms: 120,
            future_ms: 100,
            smooth_ms: 20,
        }
    }

    fn samples(ms: usize) -> usize {
        ms * SAMPLE_RATE / 1000
    }

    pub fn chunk_len(&self) -> usize {
        Self::samples(self.chunk_ms)
    }

    pub fn current_len(&self) -> usize {
        Self::samples(self.current_ms)
    }

    pub fn smooth_len(&self) -> usize {
        Self::samples(self.smooth_ms)
    }

    pub fn future_len(&self) -> usize {
        Self::samples(self.future_ms)
    }

    pub fn history_ms(&self) -> usize {
        self.chunk_ms - self.current_ms - self.smooth_ms - self.future_ms
    }

    pub fn history_len(&self) -> usize {
        Self::samples(self.history_ms())
    }

    fn validate(&self) -> Result<()> {
        if self.current_ms == 0 {
            return Err(Error::Input("current_ms must be > 0".into()));
        }
        if self.chunk_ms < self.current_ms + self.smooth_ms + self.future_ms {
            return Err(Error::Input(
                "chunk_ms must cover current + smooth + future".into(),
            ));
        }
        if self.chunk_len() % 1280 != 0 {
            return Err(Error::Input(format!(
                "chunk_ms {} must align to the 1280-sample latent hop (multiple of 80 ms)",
                self.chunk_ms
            )));
        }
        Ok(())
    }
}

/// One emitted streaming hop.
#[derive(Debug)]
pub struct StreamStep {
    /// `current_len` converted samples.
    pub samples: Vec<f32>,
    /// Per-stage wall time of the window forward (zero when the window
    /// was silent and the forward was skipped).
    pub timings: StageTimings,
}

/// The stateless-window streaming driver (`run_streaming`) as an
/// incremental state machine: [`XvcStream::push`] input samples, then
/// [`XvcStream::step`] until it returns `None`; [`XvcStream::finish`]
/// zero-pads and drains the remainder at end of input.
///
/// Window `i` covers source samples
/// `[i·current − history, i·current + current + smooth + future)`
/// (zero-padded on the left for the first windows), so a hop becomes
/// ready `smooth + future` ms after its `current` span has been pushed.
pub struct XvcStream<'a> {
    engine: &'a XvcEngine,
    reference: Reference,
    cfg: StreamConfig,
    /// Not-yet-dropped input; `buf[0]` is absolute sample `buf_offset`.
    buf: Vec<f32>,
    buf_offset: usize,
    /// Total samples pushed so far.
    pushed: usize,
    /// Next window index.
    next: usize,
    /// Crossfade tail of the previous window (`smooth_len` samples).
    tail: Vec<f32>,
    /// Raised-cosine fade-in table (`smooth_len` samples).
    fade_in: Vec<f32>,
}

impl<'a> XvcStream<'a> {
    fn new(engine: &'a XvcEngine, reference: Reference, cfg: StreamConfig) -> Result<Self> {
        cfg.validate()?;
        let smooth = cfg.smooth_len();
        // torch.linspace(0, 1, n) includes both endpoints.
        let fade_in: Vec<f32> = (0..smooth)
            .map(|i| {
                let t = if smooth > 1 {
                    i as f64 / (smooth - 1) as f64
                } else {
                    0.0
                };
                (0.5 * (1.0 - (std::f64::consts::PI * t).cos())) as f32
            })
            .collect();
        Ok(Self {
            engine,
            reference,
            cfg,
            buf: Vec::new(),
            buf_offset: 0,
            pushed: 0,
            next: 0,
            tail: vec![0.0; smooth],
            fade_in,
        })
    }

    pub fn config(&self) -> &StreamConfig {
        &self.cfg
    }

    /// Appends preprocessed input samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
        self.pushed += samples.len();
    }

    /// Absolute `[start, end)` source span of window `i` (start may be
    /// negative → zero left-pad).
    fn window_span(&self, i: usize) -> (isize, usize) {
        let cur = self.cfg.current_len();
        let start = (i * cur) as isize - self.cfg.history_len() as isize;
        let end = i * cur + cur + self.cfg.smooth_len() + self.cfg.future_len();
        (start, end)
    }

    /// Whether the next window's full span (incl. lookahead) is buffered.
    pub fn ready(&self) -> bool {
        self.window_span(self.next).1 <= self.pushed
    }

    /// Runs one window if enough input is buffered.
    pub fn step(&mut self) -> Result<Option<StreamStep>> {
        if !self.ready() {
            return Ok(None);
        }
        self.step_padded()
    }

    /// Runs the next window unconditionally, zero-padding missing input
    /// on the right (used by [`Self::finish`], mirroring the official
    /// driver's `right_pad`).
    fn step_padded(&mut self) -> Result<Option<StreamStep>> {
        let (start, end) = self.window_span(self.next);
        let mut window = vec![0f32; (end as isize - start) as usize];
        let left_pad = (-start).max(0) as usize;
        let from = start.max(0) as usize;
        for (k, slot) in window.iter_mut().enumerate().skip(left_pad) {
            let abs = from + (k - left_pad);
            if abs < self.pushed {
                *slot = self.buf[abs - self.buf_offset];
            }
        }

        let cur = self.cfg.current_len();
        let smooth = self.cfg.smooth_len();
        let hist = self.cfg.history_len();

        // Silent windows skip the forward (the demo's input gate emits
        // zero chunks): the decoded window is not exactly zero for zero
        // input, but the emitted slice is silence by construction here.
        let silent = window.iter().all(|&s| s == 0.0);
        let (mut out, tail_next, timings) = if silent {
            (vec![0f32; cur], vec![0f32; smooth], StageTimings::default())
        } else {
            let fwd = self.engine.forward_window(&window, &self.reference)?;
            let wav = fwd.wav.i((0, 0))?;
            let out: Vec<f32> = wav.narrow(0, hist, cur)?.to_vec1()?;
            let tail: Vec<f32> = wav.narrow(0, hist + cur, smooth)?.to_vec1()?;
            (out, tail, fwd.timings)
        };

        // Raised-cosine crossfade with the previous tail (skipped on the
        // very first window like the official driver).
        if self.next > 0 {
            for k in 0..smooth.min(out.len()) {
                let w = self.fade_in[k];
                out[k] = self.tail[k] * (1.0 - w) + out[k] * w;
            }
        }
        self.tail = tail_next;
        self.next += 1;

        // Drop input no future window can reach.
        let (next_start, _) = self.window_span(self.next);
        let keep_from = next_start.max(0) as usize;
        if keep_from > self.buf_offset {
            self.buf.drain(..keep_from - self.buf_offset);
            self.buf_offset = keep_from;
        }

        Ok(Some(StreamStep {
            samples: out,
            timings,
        }))
    }

    /// End of input: emits the remaining windows
    /// (`ceil(pushed / current)` in total, matching the official
    /// `total_n_chunks`) with zero right-padding, and returns their
    /// concatenation trimmed so the whole session output equals the
    /// pushed length.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        let cur = self.cfg.current_len();
        let total_windows = self.pushed.div_ceil(cur);
        let mut out = Vec::new();
        while self.next < total_windows {
            let step = self
                .step_padded()?
                .expect("step_padded always yields below total_windows");
            out.extend_from_slice(&step.samples);
        }
        // Total emitted = total_windows · current ≥ pushed: trim the tail.
        let emitted_before = (total_windows - out.len() / cur) * cur;
        out.truncate(self.pushed.saturating_sub(emitted_before));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_config_validates_alignment() {
        assert!(StreamConfig::default().validate().is_ok());
        assert!(StreamConfig::official().validate().is_ok());
        let bad = StreamConfig {
            chunk_ms: 600, // 9600 samples — not a 1280 multiple
            ..StreamConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = StreamConfig {
            current_ms: 0,
            ..StreamConfig::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn default_preset_matches_recon() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.chunk_len(), 10_240);
        assert_eq!(cfg.current_len(), 3_840);
        assert_eq!(cfg.smooth_len(), 320);
        assert_eq!(cfg.history_ms(), 280);
    }
}

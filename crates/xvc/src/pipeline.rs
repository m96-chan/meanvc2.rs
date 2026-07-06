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

/// The intermediates of the acoustic stage ([`XvcEngine::acoustic_forward`]).
#[derive(Debug)]
pub struct AcousticOutput {
    /// Quantized acoustic latent, `[1, 1024, frames]` (50 Hz).
    pub acoustic_zq: Tensor,
    /// Prenet fusion output, `[1, 1024, frames]`.
    pub prenet_out: Tensor,
    /// Converter output latent, `[1, 1024, frames]`.
    pub converter_out: Tensor,
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

    /// The semantic stage of one window forward: Whisper mel → VQ ids →
    /// embeddings → 50 Hz semantic latents. Returns
    /// `(token_ids [1, tokens] i64, sem_up [1, tokens · 4, 1024])`.
    pub fn semantic_forward(&self, samples: &[f32]) -> Result<(Tensor, Tensor)> {
        if samples.is_empty() || samples.len() % 1280 != 0 {
            return Err(Error::Input(format!(
                "window length {} must be a non-zero multiple of 1280 samples",
                samples.len()
            )));
        }
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
        Ok((tok.token_ids, sem_up))
    }

    /// The acoustic stage: codec encode + prenet fusion + MMDiT
    /// conversion. `sem_up` is the second output of
    /// [`Self::semantic_forward`] over the same `samples`.
    pub fn acoustic_forward(
        &self,
        samples: &[f32],
        sem_up: &Tensor,
        reference: &Reference,
    ) -> Result<AcousticOutput> {
        let wav_in = Tensor::from_vec(samples.to_vec(), (1, 1, samples.len()), &self.device)?;
        let enc = self.codec.encode(&wav_in)?;
        let acu_emb = enc.zq.transpose(1, 2)?.contiguous()?; // [1, T50, 1024]
        let combined = Tensor::cat(&[sem_up, &acu_emb], 2)?; // [1, T50, 2048]
        let prenet_out = self
            .prenet
            .forward(&combined.transpose(1, 2)?.contiguous()?)?;
        let converter_out = self.converter.forward(
            &prenet_out,
            &reference.frame_condition,
            &reference.speaker_condition,
        )?;
        Ok(AcousticOutput {
            acoustic_zq: enc.zq,
            prenet_out,
            converter_out,
        })
    }

    /// The waveform-synthesis stage: converted 50 Hz latents
    /// `[1, 1024, frames]` → waveform `[1, 1, frames · 320]`.
    pub fn decode_forward(&self, converter_out: &Tensor) -> Result<Tensor> {
        self.codec.decode(converter_out)
    }

    /// One full-stack forward (`run_stream_chunk_forward`): preprocessed
    /// source samples (a multiple of 1280) → converted waveform of the
    /// same length, with every intermediate and per-stage wall time.
    /// Composes [`Self::semantic_forward`], [`Self::acoustic_forward`] and
    /// [`Self::decode_forward`] — the pipelined driver runs exactly these
    /// three on separate threads, so outputs are bit-identical.
    pub fn forward_window(&self, samples: &[f32], reference: &Reference) -> Result<WindowOutput> {
        // Semantic branch: Whisper mel → VQ ids → embeddings → 50 Hz.
        let t0 = Instant::now();
        let (token_ids, sem_up) = self.semantic_forward(samples)?;
        let t_semantic = t0.elapsed();

        // Acoustic branch + fusion + conversion.
        let t0 = Instant::now();
        let acu = self.acoustic_forward(samples, &sem_up, reference)?;
        let t_acoustic = t0.elapsed();

        // Waveform synthesis.
        let t0 = Instant::now();
        let wav = self.decode_forward(&acu.converter_out)?;
        let t_decode = t0.elapsed();

        Ok(WindowOutput {
            token_ids,
            sem_adapter_out: sem_up,
            acoustic_zq: acu.acoustic_zq,
            prenet_out: acu.prenet_out,
            converter_out: acu.converter_out,
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
    /// Cross-window needle suppression (issue #42): delay emission by
    /// one hop and verify each hop against the NEXT window's rendering
    /// of the same region. The SAC decoder's needle pulses are
    /// window-local, so a short run that is much sharper than the
    /// neighbouring window's rendering is the needle, and the other
    /// rendering replaces it. Costs one hop (`current_ms`) of latency.
    pub cross_check: bool,
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
            cross_check: false,
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
            cross_check: false,
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

    /// Algorithmic latency in milliseconds: a hop becomes ready
    /// `smooth + future` ms after its `current` span has been pushed,
    /// plus the `current` ms of input accumulation itself. NOTE: the
    /// window size (`chunk_ms`) does NOT appear here — growing the
    /// window only grows `history` (= compute per hop), never the
    /// latency. `longer_window_does_not_add_latency` pins this.
    pub fn algorithmic_latency_ms(&self) -> usize {
        self.current_ms + self.smooth_ms + self.future_ms
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
        if self.cross_check && self.history_len() < self.current_len() {
            return Err(Error::Input(
                "cross_check needs history >= current (the next window must cover the held hop)"
                    .into(),
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
    /// Needle runs replaced by the cross-window check (0 unless
    /// [`StreamConfig::cross_check`] is on).
    pub cross_repairs: u32,
}

/// Input-side window assembly shared by [`XvcStream`] and
/// [`XvcPipelinedStream`]: buffers pushed samples and slices out the
/// `[history | current | smooth | future]` window of each hop
/// (zero-padded on the left for the first windows and on the right past
/// the end of input).
#[derive(Debug)]
struct Windower {
    cfg: StreamConfig,
    /// Not-yet-dropped input; `buf[0]` is absolute sample `buf_offset`.
    buf: Vec<f32>,
    buf_offset: usize,
    /// Total samples pushed so far.
    pushed: usize,
    /// Next window index.
    next: usize,
}

impl Windower {
    fn new(cfg: StreamConfig) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            buf: Vec::new(),
            buf_offset: 0,
            pushed: 0,
            next: 0,
        })
    }

    /// Appends preprocessed input samples.
    fn push(&mut self, samples: &[f32]) {
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
    fn ready(&self) -> bool {
        self.window_span(self.next).1 <= self.pushed
    }

    /// Assembles the next window unconditionally (zero right-pad past the
    /// pushed input, mirroring the official driver's `right_pad`),
    /// advances, and drops input no future window can reach. Returns
    /// `(window index, window samples)`.
    fn take_next(&mut self) -> (usize, Vec<f32>) {
        let index = self.next;
        let (start, end) = self.window_span(index);
        let mut window = vec![0f32; (end as isize - start) as usize];
        let left_pad = (-start).max(0) as usize;
        let from = start.max(0) as usize;
        for (k, slot) in window.iter_mut().enumerate().skip(left_pad) {
            let abs = from + (k - left_pad);
            if abs < self.pushed {
                *slot = self.buf[abs - self.buf_offset];
            }
        }
        self.next += 1;

        // Drop input no future window can reach.
        let (next_start, _) = self.window_span(self.next);
        let keep_from = next_start.max(0) as usize;
        if keep_from > self.buf_offset {
            self.buf.drain(..keep_from - self.buf_offset);
            self.buf_offset = keep_from;
        }
        (index, window)
    }

    /// Total windows of the whole session (`ceil(pushed / current)`,
    /// matching the official `total_n_chunks`).
    fn total_windows(&self) -> usize {
        self.pushed.div_ceil(self.cfg.current_len())
    }
}

/// The raised-cosine crossfade state at chunk joins, shared by both
/// drivers: `smooth_len` samples of the previous window's tail are faded
/// into the head of the next window's `current` slice (skipped on the
/// very first window like the official driver).
#[derive(Debug)]
struct Crossfader {
    /// Crossfade tail of the previous window (`smooth_len` samples).
    tail: Vec<f32>,
    /// Raised-cosine fade-in table (`smooth_len` samples).
    fade_in: Vec<f32>,
}

impl Crossfader {
    fn new(smooth: usize) -> Self {
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
        Self {
            tail: vec![0.0; smooth],
            fade_in,
        }
    }

    /// Fades `out`'s head with the stored tail (window 0 is emitted
    /// untouched) and stores `tail_next` for the next window.
    fn apply(&mut self, index: usize, out: &mut [f32], tail_next: Vec<f32>) {
        if index > 0 {
            for k in 0..self.fade_in.len().min(out.len()) {
                let w = self.fade_in[k];
                out[k] = self.tail[k] * (1.0 - w) + out[k] * w;
            }
        }
        self.tail = tail_next;
    }
}

/// One sliced window: the emitted `current` samples, the next crossfade
/// tail, and — when [`StreamConfig::cross_check`] is on — this window's
/// rendering of the PREVIOUS hop's region.
type SlicedWindow = (Vec<f32>, Vec<f32>, Option<Vec<f32>>);

/// Extracts a [`SlicedWindow`] from one decoded window waveform
/// `[1, 1, chunk_len]`.
fn slice_window_wav(wav: &Tensor, cfg: &StreamConfig) -> Result<SlicedWindow> {
    let wav = wav.i((0, 0))?;
    let out: Vec<f32> = wav
        .narrow(0, cfg.history_len(), cfg.current_len())?
        .to_vec1()?;
    let tail: Vec<f32> = wav
        .narrow(0, cfg.history_len() + cfg.current_len(), cfg.smooth_len())?
        .to_vec1()?;
    let prev = if cfg.cross_check {
        Some(
            wav.narrow(0, cfg.history_len() - cfg.current_len(), cfg.current_len())?
                .to_vec1()?,
        )
    } else {
        None
    };
    Ok((out, tail, prev))
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
    windower: Windower,
    fader: Crossfader,
    checker: CrossChecker,
}

impl<'a> XvcStream<'a> {
    fn new(engine: &'a XvcEngine, reference: Reference, cfg: StreamConfig) -> Result<Self> {
        Ok(Self {
            engine,
            reference,
            windower: Windower::new(cfg)?,
            fader: Crossfader::new(cfg.smooth_len()),
            checker: CrossChecker::new(cfg.cross_check),
        })
    }

    pub fn config(&self) -> &StreamConfig {
        &self.windower.cfg
    }

    /// Appends preprocessed input samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.windower.push(samples);
    }

    /// Whether the next window's full span (incl. lookahead) is buffered.
    pub fn ready(&self) -> bool {
        self.windower.ready()
    }

    /// Runs one window if enough input is buffered. With
    /// [`StreamConfig::cross_check`] the very first window yields no hop
    /// (it is held for verification), so this loops until a hop is ready
    /// or input runs out.
    pub fn step(&mut self) -> Result<Option<StreamStep>> {
        while self.ready() {
            if let Some(step) = self.step_padded()? {
                return Ok(Some(step));
            }
        }
        Ok(None)
    }

    /// Runs the next window unconditionally, zero-padding missing input
    /// on the right (used by [`Self::finish`]).
    fn step_padded(&mut self) -> Result<Option<StreamStep>> {
        let cfg = *self.config();
        let (index, window) = self.windower.take_next();

        // Silent windows skip the forward (the demo's input gate emits
        // zero chunks): the decoded window is not exactly zero for zero
        // input, but the emitted slice is silence by construction here.
        let silent = window.iter().all(|&s| s == 0.0);
        let (mut out, tail_next, prev, timings) = if silent {
            (
                vec![0f32; cfg.current_len()],
                vec![0f32; cfg.smooth_len()],
                None,
                StageTimings::default(),
            )
        } else {
            let fwd = self.engine.forward_window(&window, &self.reference)?;
            let (out, tail, prev) = slice_window_wav(&fwd.wav, &cfg)?;
            (out, tail, prev, fwd.timings)
        };

        self.fader.apply(index, &mut out, tail_next);

        Ok(self.checker.push(out, timings, prev.as_deref()))
    }

    /// End of input: emits the remaining windows
    /// (`ceil(pushed / current)` in total, matching the official
    /// `total_n_chunks`) with zero right-padding, and returns their
    /// concatenation trimmed so the whole session output equals the
    /// pushed length.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        let cur = self.config().current_len();
        let total_windows = self.windower.total_windows();
        let mut out = Vec::new();
        while self.windower.next < total_windows {
            if let Some(step) = self.step_padded()? {
                out.extend_from_slice(&step.samples);
            }
        }
        if let Some(step) = self.checker.flush() {
            out.extend_from_slice(&step.samples);
        }
        // Total emitted = total_windows · current ≥ pushed: trim the tail.
        let emitted_before = (total_windows - out.len() / cur) * cur;
        out.truncate(self.windower.pushed.saturating_sub(emitted_before));
        Ok(out)
    }
}

/// One-hop hold-and-verify state for [`StreamConfig::cross_check`],
/// shared by the sequential and pipelined drivers.
///
/// The SAC decoder occasionally emits a needle pulse (issue #42) whose
/// position is a property of the WHOLE re-encoded window — the
/// neighbouring window renders the same audio instant cleanly. Each hop
/// is therefore held back until the next window has rendered, and short
/// divergent runs where the held hop is much sharper than the other
/// rendering are replaced by the other rendering. Both candidates are
/// genuine model output, so — unlike an amplitude-threshold declicker —
/// natural speech can never be "repaired" into something it isn't.
struct CrossChecker {
    enabled: bool,
    held: Option<(Vec<f32>, StageTimings)>,
}

impl CrossChecker {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            held: None,
        }
    }

    /// Feeds a freshly rendered hop plus the same window's rendering of
    /// the previous hop's region (`None` for silent windows). Returns
    /// the hop to emit now, if any.
    fn push(
        &mut self,
        samples: Vec<f32>,
        timings: StageTimings,
        prev_render: Option<&[f32]>,
    ) -> Option<StreamStep> {
        if !self.enabled {
            return Some(StreamStep {
                samples,
                timings,
                cross_repairs: 0,
            });
        }
        let emit = self.held.take().map(|(mut held, t)| {
            let repairs = match prev_render {
                Some(other) => cross_check_repair(&mut held, other),
                None => 0,
            };
            StreamStep {
                samples: held,
                timings: t,
                cross_repairs: repairs,
            }
        });
        self.held = Some((samples, timings));
        emit
    }

    /// End of input: emits the still-held final hop unverified.
    fn flush(&mut self) -> Option<StreamStep> {
        self.held.take().map(|(samples, timings)| StreamStep {
            samples,
            timings,
            cross_repairs: 0,
        })
    }
}

/// Maximum replaceable divergence run (seconds) — needles are ≲2.5 ms.
const XCHK_MAX_RUN_SECS: f32 = 0.0025;
/// Margin crossfaded around a replaced run (samples).
const XCHK_MARGIN: usize = 8;
/// A run qualifies only when the held hop's sharpness exceeds the other
/// rendering's by this factor — genuine re-rendering differences (phase
/// drift, level) are symmetric in sharpness and never qualify.
///
/// The gate is deliberately LIBERAL (~2 replacements/second on live
/// content). A replacement swaps in the other window's genuine
/// rendering, crossfaded — the eighth field recording confirmed the
/// audible ticks tracked the NeedleGuard's bridge repairs (dc), never
/// the cross-check (xr), so liberal firing here is safe by
/// construction while stricter gates measurably leak real needles
/// (hardened variants let 5–21 needles through the worst-case suite).
const XCHK_SHARPNESS: f32 = 2.5;
/// Divergence floor: |held − other| must exceed this to open a run.
const XCHK_DIV_FLOOR: f32 = 0.04;

/// Replaces needle-suspect runs of `held` by `other` (the next window's
/// rendering of the same samples). Returns the number of replaced runs.
fn cross_check_repair(held: &mut [f32], other: &[f32]) -> u32 {
    if held.len() != other.len() || held.is_empty() {
        return 0;
    }
    let max_run = (XCHK_MAX_RUN_SECS * SAMPLE_RATE as f32) as usize;
    // Divergence threshold: well above the typical re-rendering
    // difference of THIS hop (median is immune to the short needle).
    let mut mag: Vec<f32> = held.iter().zip(other).map(|(a, b)| (a - b).abs()).collect();
    let mid = mag.len() / 2;
    mag.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let thr = (4.0 * mag[mid]).max(XCHK_DIV_FLOOR);

    let sharpness = |x: &[f32], lo: usize, hi: usize| -> f32 {
        let lo = lo.saturating_sub(XCHK_MARGIN);
        let hi = (hi + XCHK_MARGIN).min(x.len());
        x[lo..hi]
            .windows(3)
            .map(|w| (w[0] - 2.0 * w[1] + w[2]).abs())
            .fold(0f32, f32::max)
    };

    let mut repairs = 0;
    let mut i = 0;
    while i < held.len() {
        if (held[i] - other[i]).abs() <= thr {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < held.len() && j - i <= max_run && (held[j] - other[j]).abs() > 0.5 * thr {
            j += 1;
        }
        if j - i <= max_run && sharpness(held, i, j) > XCHK_SHARPNESS * sharpness(other, i, j) {
            let lo = i.saturating_sub(XCHK_MARGIN);
            let hi = (j + XCHK_MARGIN).min(held.len());
            for k in lo..hi {
                // Linear crossfade at both edges, full replacement inside.
                let edge = (k - lo + 1).min(hi - k) as f32;
                let w = (edge / (XCHK_MARGIN + 1) as f32).min(1.0);
                held[k] = (1.0 - w) * held[k] + w * other[k];
            }
            repairs += 1;
            i = hi;
        } else {
            i = j;
        }
    }
    repairs
}

/// A window flowing between two pipeline stages: silent windows bypass
/// the model but keep their slot in the (order-preserving) channels.
enum StageMsg<T> {
    Work {
        index: usize,
        payload: T,
        timings: StageTimings,
    },
    Silent {
        index: usize,
    },
}

/// The pipelined streaming driver (issue #38): the same stateless-window
/// stream as [`XvcStream`], but asynchronous behind channels.
///
/// On **CPU** the three stages of [`XvcEngine::forward_window`] —
/// semantic (mel, tokenizer, adapter), acoustic (codec encode, prenet,
/// converter) and decode (waveform synthesis) — run on three dedicated
/// threads connected by bounded channels, so consecutive windows overlap.
/// Throughput becomes `max(stage)` instead of `sum(stages)` at the cost
/// of up to two extra hops of latency while the pipeline is full.
///
/// On **accelerator devices** (CUDA/Metal) the whole forward runs on a
/// single worker thread instead: device kernels serialize on one stream
/// anyway so stage overlap buys nothing, and concurrent op submission
/// from several host threads is not safe in candle — measured on CUDA it
/// corrupted the audio non-deterministically (host-side locking around
/// the forwards is not enough, because kernel launches are asynchronous
/// and mutex order does not imply stream order for the tensors crossing
/// threads).
///
/// Every window runs exactly the ops of the sequential driver in the same
/// order, so the output is **bit-identical** to [`XvcStream`]
/// (`tests/golden_pipeline.rs::pipelined_stream_matches_sequential` on
/// CPU, `examples/bench_stages.rs` on CUDA).
///
/// [`XvcPipelinedStream::push`] input samples (ready windows are enqueued
/// automatically), drain finished hops with
/// [`XvcPipelinedStream::try_next`], and call
/// [`XvcPipelinedStream::finish`] at end of input.
pub struct XvcPipelinedStream {
    windower: Windower,
    /// `None` after `finish` closed the pipeline.
    tx_job: Option<std::sync::mpsc::SyncSender<StageMsg<Vec<f32>>>>,
    rx_step: std::sync::mpsc::Receiver<Result<StreamStep>>,
    /// Steps already handed to the caller (or drained by `finish`).
    received: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl XvcPipelinedStream {
    /// Capacity of the inter-stage channels: enough to keep every stage
    /// busy without letting the backlog (= added latency) grow unbounded.
    const STAGE_QUEUE: usize = 2;

    pub fn new(
        engine: std::sync::Arc<XvcEngine>,
        reference: Reference,
        cfg: StreamConfig,
    ) -> Result<Self> {
        use std::sync::mpsc::{channel, sync_channel};
        let windower = Windower::new(cfg)?;

        let (tx_job, rx_job) = sync_channel::<StageMsg<Vec<f32>>>(Self::STAGE_QUEUE);

        // Accelerator devices: single worker thread (see the type docs —
        // multi-thread op submission is unsafe in candle, and stage
        // overlap buys nothing when kernels serialize on one stream).
        if !matches!(engine.device(), Device::Cpu) {
            let (tx_step, rx_step) = channel::<Result<StreamStep>>();
            let worker = std::thread::Builder::new()
                .name("xvc-window".into())
                .spawn(move || {
                    let mut fader = Crossfader::new(cfg.smooth_len());
                    let mut checker = CrossChecker::new(cfg.cross_check);
                    while let Ok(msg) = rx_job.recv() {
                        let out = match msg {
                            StageMsg::Silent { index } => {
                                let mut out = vec![0f32; cfg.current_len()];
                                fader.apply(index, &mut out, vec![0.0; cfg.smooth_len()]);
                                Ok(checker.push(out, StageTimings::default(), None))
                            }
                            StageMsg::Work { index, payload, .. } => {
                                engine.forward_window(&payload, &reference).and_then(|fwd| {
                                    let (mut out, tail, prev) = slice_window_wav(&fwd.wav, &cfg)?;
                                    fader.apply(index, &mut out, tail);
                                    Ok(checker.push(out, fwd.timings, prev.as_deref()))
                                })
                            }
                        };
                        match out {
                            Ok(None) => {}
                            Ok(Some(step)) => {
                                if tx_step.send(Ok(step)).is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = tx_step.send(Err(e));
                                break;
                            }
                        }
                    }
                    if let Some(step) = checker.flush() {
                        let _ = tx_step.send(Ok(step));
                    }
                })
                .map_err(|e| Error::Input(format!("cannot spawn the xvc worker: {e}")))?;
            return Ok(Self {
                windower,
                tx_job: Some(tx_job),
                rx_step,
                received: 0,
                handles: vec![worker],
            });
        }
        let (tx_sem, rx_sem) =
            sync_channel::<Result<StageMsg<(Vec<f32>, Tensor)>>>(Self::STAGE_QUEUE);
        let (tx_acu, rx_acu) = sync_channel::<Result<StageMsg<Tensor>>>(Self::STAGE_QUEUE);
        // Unbounded: finished audio must never block the decode thread.
        let (tx_step, rx_step) = channel::<Result<StreamStep>>();

        // Stage 1: semantic (Whisper mel → tokenizer → adapter).
        let eng = engine.clone();
        let semantic = std::thread::Builder::new()
            .name("xvc-semantic".into())
            .spawn(move || {
                while let Ok(msg) = rx_job.recv() {
                    let out = match msg {
                        StageMsg::Silent { index } => Ok(StageMsg::Silent { index }),
                        StageMsg::Work { index, payload, .. } => {
                            let t0 = Instant::now();
                            match eng.semantic_forward(&payload) {
                                Ok((_ids, sem_up)) => Ok(StageMsg::Work {
                                    index,
                                    payload: (payload, sem_up),
                                    timings: StageTimings {
                                        semantic: t0.elapsed(),
                                        ..Default::default()
                                    },
                                }),
                                Err(e) => Err(e),
                            }
                        }
                    };
                    let failed = out.is_err();
                    if tx_sem.send(out).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(|e| Error::Input(format!("cannot spawn the semantic stage: {e}")))?;

        // Stage 2: acoustic (codec encode + prenet + converter).
        let eng = engine.clone();
        let acoustic = std::thread::Builder::new()
            .name("xvc-acoustic".into())
            .spawn(move || {
                while let Ok(msg) = rx_sem.recv() {
                    let out = match msg {
                        Err(e) => Err(e),
                        Ok(StageMsg::Silent { index }) => Ok(StageMsg::Silent { index }),
                        Ok(StageMsg::Work {
                            index,
                            payload: (window, sem_up),
                            mut timings,
                        }) => {
                            let t0 = Instant::now();
                            match eng.acoustic_forward(&window, &sem_up, &reference) {
                                Ok(acu) => {
                                    timings.acoustic = t0.elapsed();
                                    Ok(StageMsg::Work {
                                        index,
                                        payload: acu.converter_out,
                                        timings,
                                    })
                                }
                                Err(e) => Err(e),
                            }
                        }
                    };
                    let failed = out.is_err();
                    if tx_acu.send(out).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(|e| Error::Input(format!("cannot spawn the acoustic stage: {e}")))?;

        // Stage 3: decode + crossfade (sequential by construction: the
        // channels preserve window order).
        let eng = engine;
        let decode = std::thread::Builder::new()
            .name("xvc-decode".into())
            .spawn(move || {
                let mut fader = Crossfader::new(cfg.smooth_len());
                let mut checker = CrossChecker::new(cfg.cross_check);
                while let Ok(msg) = rx_acu.recv() {
                    let out = match msg {
                        Err(e) => Err(e),
                        Ok(StageMsg::Silent { index }) => {
                            let mut out = vec![0f32; cfg.current_len()];
                            fader.apply(index, &mut out, vec![0.0; cfg.smooth_len()]);
                            Ok(checker.push(out, StageTimings::default(), None))
                        }
                        Ok(StageMsg::Work {
                            index,
                            payload: converter_out,
                            mut timings,
                        }) => {
                            let t0 = Instant::now();
                            eng.decode_forward(&converter_out)
                                .and_then(|wav| slice_window_wav(&wav, &cfg))
                                .map(|(mut out, tail, prev)| {
                                    timings.decode = t0.elapsed();
                                    fader.apply(index, &mut out, tail);
                                    checker.push(out, timings, prev.as_deref())
                                })
                        }
                    };
                    match out {
                        Ok(None) => {}
                        Ok(Some(step)) => {
                            if tx_step.send(Ok(step)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx_step.send(Err(e));
                            break;
                        }
                    }
                }
                if let Some(step) = checker.flush() {
                    let _ = tx_step.send(Ok(step));
                }
            })
            .map_err(|e| Error::Input(format!("cannot spawn the decode stage: {e}")))?;

        Ok(Self {
            windower,
            tx_job: Some(tx_job),
            rx_step,
            received: 0,
            handles: vec![semantic, acoustic, decode],
        })
    }

    pub fn config(&self) -> &StreamConfig {
        &self.windower.cfg
    }

    /// Sends one assembled window into the pipeline (blocks while the
    /// input queue is full — i.e. when the pipeline is saturated).
    fn enqueue(&mut self) -> Result<()> {
        let (index, window) = self.windower.take_next();
        let msg = if window.iter().all(|&s| s == 0.0) {
            StageMsg::Silent { index }
        } else {
            StageMsg::Work {
                index,
                payload: window,
                timings: StageTimings::default(),
            }
        };
        self.tx_job
            .as_ref()
            .ok_or_else(|| Error::Input("stream already finished".into()))?
            .send(msg)
            .map_err(|_| Error::Input("the xvc pipeline stopped (stage thread gone)".into()))
    }

    /// Appends preprocessed input samples and enqueues every window that
    /// became ready.
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.windower.push(samples);
        while self.windower.ready() {
            self.enqueue()?;
        }
        Ok(())
    }

    /// Returns the next finished hop without blocking (`None` when no hop
    /// is ready yet). Hops come out in window order.
    pub fn try_next(&mut self) -> Result<Option<StreamStep>> {
        match self.rx_step.try_recv() {
            Ok(step) => {
                self.received += 1;
                step.map(Some)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err(Error::Input(
                "the xvc pipeline stopped (stage thread gone)".into(),
            )),
        }
    }

    /// Blocks up to `timeout` for the next finished hop.
    pub fn next_timeout(&mut self, timeout: Duration) -> Result<Option<StreamStep>> {
        match self.rx_step.recv_timeout(timeout) {
            Ok(step) => {
                self.received += 1;
                step.map(Some)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(Error::Input(
                "the xvc pipeline stopped (stage thread gone)".into(),
            )),
        }
    }

    /// End of input: enqueues the remaining zero-right-padded windows,
    /// closes the pipeline, drains every hop not yet consumed and returns
    /// their concatenation trimmed so the whole session output equals the
    /// pushed length (same semantics as [`XvcStream::finish`]).
    pub fn finish(mut self) -> Result<Vec<f32>> {
        let cur = self.config().current_len();
        let pushed = self.windower.pushed;
        let total_windows = self.windower.total_windows();
        while self.windower.next < total_windows {
            self.enqueue()?;
        }
        // Close the input so the stage threads drain and exit.
        self.tx_job = None;
        let mut out = Vec::new();
        while self.received < total_windows {
            let step = self
                .rx_step
                .recv()
                .map_err(|_| Error::Input("the xvc pipeline stopped (stage thread gone)".into()))?;
            out.extend_from_slice(&step?.samples);
            self.received += 1;
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
        // Total emitted = total_windows · current ≥ pushed: trim the tail.
        let emitted_before = (total_windows - out.len() / cur) * cur;
        out.truncate(pushed.saturating_sub(emitted_before));
        Ok(out)
    }
}

impl Drop for XvcPipelinedStream {
    fn drop(&mut self) {
        // Close the input channel and let the stage threads drain their
        // in-flight windows and exit; finished steps are discarded by the
        // dropped receiver.
        self.tx_job = None;
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longer_window_does_not_add_latency() {
        // Issue #42: the CUDA default enlarges the window to shed
        // decoder needles. The enlargement must land entirely in the
        // HISTORY part — if a refactor ever grows future/smooth/current
        // along with the window, latency silently increases and this
        // fails.
        let base = StreamConfig::default();
        for chunk_ms in [640, 960, 1280, 1920, 2400] {
            let cfg = StreamConfig {
                chunk_ms,
                ..Default::default()
            };
            cfg.validate().unwrap();
            assert_eq!(cfg.current_ms, base.current_ms);
            assert_eq!(cfg.smooth_ms, base.smooth_ms);
            assert_eq!(cfg.future_ms, base.future_ms);
            assert_eq!(
                cfg.algorithmic_latency_ms(),
                base.algorithmic_latency_ms(),
                "window {chunk_ms} ms changed the algorithmic latency"
            );
            assert_eq!(
                cfg.history_ms(),
                chunk_ms - base.algorithmic_latency_ms(),
                "window growth must go entirely into history"
            );
        }
    }

    #[test]
    fn cross_check_repairs_sharp_lone_run() {
        // Two renderings of the same vowel: slightly different (phase
        // drift), but the held one carries a decoder needle.
        let n = 3_840;
        let f = 300.0 / SAMPLE_RATE as f32;
        let mut held: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * f * i as f32).sin())
            .collect();
        let other: Vec<f32> = (0..n)
            .map(|i| 0.29 * (2.0 * std::f32::consts::PI * f * (i as f32 + 3.0)).sin())
            .collect();
        for k in 0..5 {
            held[2_000 + k] = if k % 2 == 0 { 0.9 } else { -0.85 };
        }
        let repairs = cross_check_repair(&mut held, &other);
        assert_eq!(repairs, 1);
        let peak = held[1_980..2_030].iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak < 0.4, "needle survived the cross-check: {peak}");
    }

    #[test]
    fn cross_check_leaves_symmetric_divergence_alone() {
        // Genuine re-rendering differences are symmetric in sharpness:
        // nothing may be replaced even where the divergence is large.
        let n = 3_840;
        let f = 300.0 / SAMPLE_RATE as f32;
        let held: Vec<f32> = (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * f * i as f32).sin())
            .collect();
        let other: Vec<f32> = (0..n)
            .map(|i| 0.22 * (2.0 * std::f32::consts::PI * f * (i as f32 + 11.0)).sin())
            .collect();
        let mut checked = held.clone();
        let repairs = cross_check_repair(&mut checked, &other);
        assert_eq!(repairs, 0);
        assert_eq!(checked, held);
    }

    #[test]
    fn cross_check_keeps_genuine_plosives() {
        // A real transient (plosive burst / attack) appears in BOTH
        // renderings — sharp in both, so the sharpness-asymmetry gate
        // must leave it bit-untouched even though it is short and the
        // renderings differ slightly.
        let n = 3_840;
        let f = 300.0 / SAMPLE_RATE as f32;
        let burst = |i: usize, k0: usize, amp: f32| -> f32 {
            let d = i as f32 - k0 as f32;
            if (0.0..24.0).contains(&d) {
                amp * (0.7 * d).sin() * (1.0 - d / 24.0)
            } else {
                0.0
            }
        };
        let held: Vec<f32> = (0..n)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * f * i as f32).sin() + burst(i, 2_000, 0.5))
            .collect();
        // The other rendering: slightly different level and a 2-sample
        // shifted burst (window re-render jitter).
        let other: Vec<f32> = (0..n)
            .map(|i| {
                0.19 * (2.0 * std::f32::consts::PI * f * (i as f32 + 2.0)).sin()
                    + burst(i, 2_002, 0.47)
            })
            .collect();
        let mut checked = held.clone();
        let repairs = cross_check_repair(&mut checked, &other);
        assert_eq!(repairs, 0, "genuine plosive was repaired away");
        assert_eq!(checked, held);
    }

    #[test]
    fn cross_checker_holds_one_hop() {
        let mut c = CrossChecker::new(true);
        let a = vec![1.0f32; 8];
        let b = vec![2.0f32; 8];
        assert!(c.push(a.clone(), StageTimings::default(), None).is_none());
        let first = c.push(b, StageTimings::default(), None).unwrap();
        assert_eq!(first.samples, a);
        let last = c.flush().unwrap();
        assert_eq!(last.samples, vec![2.0f32; 8]);
        assert!(c.flush().is_none());
    }

    #[test]
    fn cross_checker_disabled_is_passthrough() {
        let mut c = CrossChecker::new(false);
        let step = c
            .push(vec![1.0f32; 8], StageTimings::default(), None)
            .unwrap();
        assert_eq!(step.samples, vec![1.0f32; 8]);
        assert!(c.flush().is_none());
    }

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
    fn windower_spans_and_padding() {
        let cfg = StreamConfig::default();
        let mut w = Windower::new(cfg).unwrap();
        // Window 0 spans [-history, current + smooth + future).
        assert_eq!(w.window_span(0), (-(cfg.history_len() as isize), 5_760));
        assert_eq!(
            w.window_span(1),
            (
                cfg.current_len() as isize - cfg.history_len() as isize,
                2 * cfg.current_len() + cfg.smooth_len() + cfg.future_len()
            )
        );
        assert!(!w.ready());

        // Ramp input: sample k has value k+1 (nonzero to expose padding).
        let input: Vec<f32> = (0..2 * cfg.current_len()).map(|k| (k + 1) as f32).collect();
        w.push(&input[..cfg.current_len()]);
        assert!(!w.ready(), "window 0 needs smooth+future lookahead");
        w.push(&input[cfg.current_len()..]);
        assert!(w.ready());

        let (i0, win0) = w.take_next();
        assert_eq!(i0, 0);
        assert_eq!(win0.len(), cfg.chunk_len());
        // Left zero-pad over the missing history…
        assert!(win0[..cfg.history_len()].iter().all(|&s| s == 0.0));
        // …then the pushed samples verbatim.
        let n = cfg.chunk_len() - cfg.history_len();
        assert_eq!(win0[cfg.history_len()..], input[..n]);

        // total_windows = ceil(pushed / current).
        assert_eq!(w.total_windows(), 2);
        // Window 1 starts at current − history = −640 (still left-padded
        // with this preset) and is not fully buffered: right side
        // zero-padded too.
        assert!(!w.ready());
        let (i1, win1) = w.take_next();
        assert_eq!(i1, 1);
        let left = cfg.history_len() - cfg.current_len();
        assert!(win1[..left].iter().all(|&s| s == 0.0));
        assert_eq!(win1[left..left + input.len()], input[..]);
        assert!(win1[left + input.len()..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn crossfader_first_window_untouched_then_fades() {
        let mut f = Crossfader::new(4);
        let mut w0 = vec![1.0f32; 8];
        f.apply(0, &mut w0, vec![0.5; 4]);
        assert_eq!(w0, vec![1.0; 8], "window 0 must not be faded");
        let mut w1 = vec![1.0f32; 8];
        f.apply(1, &mut w1, vec![0.0; 4]);
        // Head fades from the stored tail (0.5) to the new window (1.0);
        // fade_in(0) = 0 → exactly the previous tail at k = 0.
        assert_eq!(w1[0], 0.5);
        assert_eq!(w1[3], 1.0, "fade_in end = 1 → fully the new window");
        assert!(w1[1] > 0.5 && w1[1] < 1.0);
        assert_eq!(w1[4..], [1.0; 4]);
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

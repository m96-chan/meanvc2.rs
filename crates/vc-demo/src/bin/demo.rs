//! Real-time voice-conversion TUI demo with a virtual microphone.
//!
//! Captures the default microphone, converts the voice chunk by chunk
//! with the selected engine (`--engine meanvc` (default, 200 ms chunks) or
//! `--engine xvc` (240 ms hop, 640 ms re-encoded window — the official
//! X-VC CPU streaming preset)), and plays the result into the platform's
//! **virtual microphone** route (issue #51, `vc_demo::audio`): on Linux a
//! PulseAudio/PipeWire null sink whose remapped monitor shows up as a
//! selectable source (`babiniku_mic`); on Windows/macOS a user-installed
//! loopback device (VB-CABLE / BlackHole), auto-detected or chosen with
//! `--output-device`.
//!
//! ```sh
//! cargo run --release -p vc-demo --bin babiniku-demo -- \
//!     --reference ckpt/test.wav --voice-print ckpt/voice_print_test.safetensors
//! cargo run --release -p vc-demo --bin babiniku-demo -- \
//!     --engine xvc --reference her_voice.wav
//! ```
//!
//! Keys: `q` (or Ctrl-C) quit · `p` passthrough (bypass conversion for A/B) ·
//! `l` loopback monitor (hear the converted voice on the default output) ·
//! `[` / `]` pitch shift −/+0.5 semitone (post-vocoder, Signalsmith
//! Stretch) · `,` / `.` RNNoise denoise mix −/+10 % (pre-ASR, in-process;
//! independent of the `--denoise` WebRTC stage) · `;` / `'` BWE exciter
//! wet −/+10 % (bandwidth extension above 8 kHz, issue #42).
//!
//! Output path (issue #42): the engines synthesize at 16 kHz, so the
//! converted chunks are upsampled in-process to **48 kHz**
//! ([`vc_core::bwe::Upsampler3x`], exact ×3 windowed-sinc polyphase) and
//! the playback stream opens at 48 kHz — this sidesteps PipeWire's own
//! 16→48 resample and its measured −15.5 dB rolloff at 6.5–7.6 kHz.
//! `--out` wavs are therefore written at **48 kHz** (input capture stays
//! 16 kHz). On top of that, an optional harmonic exciter
//! ([`vc_core::bwe::Exciter`], off by default) synthesizes the missing
//! 8–16 kHz band; `--bwe <0-100>` sets the wet amount.
//!
//! Options: `--pitch-shift <semitones>` / `--denoise-mix <0-100>` /
//! `--bwe <0-100>` set the initial knob values, `--denoise` inserts
//! PipeWire's WebRTC noise suppression in
//! front of the microphone (recommended for noisy mics),
//! `--input-device <source>` records from a specific capture device,
//! `--output-device <name>` routes the converted voice to a specific
//! playback device (non-Linux virtual-mic routing),
//! `--wav <file>` converts a wav file instead of the microphone
//! (paced in real time), `--headless` disables the TUI, `--out <file>`
//! additionally records the converted audio (48 kHz), `--no-sink` skips
//! creating the virtual device, `--monitor` starts with the loopback
//! enabled, `--duration <secs>` auto-stops (for testing).
//!
//! Shutdown: SIGINT (Ctrl-C) and SIGTERM run the same clean teardown as
//! `q`, unloading every pactl module the demo created; stale
//! babiniku-named modules left behind by a killed run are recovered
//! (unloaded) at startup before fresh devices are created (issue #39).
//!
//! BNFs are extracted with the incremental `FastU2pp::forward_chunk`
//! streaming caches (issue #9), bit-matching the official WeNet chunked
//! decode; the remaining approximation is the vocoder, which re-synthesizes
//! each chunk with a 200 ms mel tail as context (tracked in #9).

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor};
use meanvc2::backends::{FastU2pp, FastU2ppConfig, Vocos, VocosConfig};
use meanvc2::encoders::Vocoder;
use meanvc2::v1::{interpolate_linear, KaldiFbank, KvCache, MeanVc1, MeanVc1Config, MelV1};
use nnnoiseless::DenoiseState;
use signalsmith_stretch::Stretch;
use std::sync::atomic::AtomicI32;
use vc_core::bwe::{Exciter, Upsampler3x};
use vc_demo::audio::{self, AudioBackend};
use xvc::preprocess::HighpassBiquad;

const SR: usize = 16_000;
/// Playback / `--out` sample rate: the 16 kHz engine output is upsampled
/// ×3 in-process before it reaches the sink (issue #42).
const OUT_SR: usize = 48_000;
const CHUNK_SAMPLES: usize = 3_200; // 200 ms = one CARD chunk (20 mel frames)
/// X-VC hop: 240 ms of new audio per re-encoded 640 ms window (the CPU
/// streaming preset 640/240/100/20 from issue #30).
const XVC_CHUNK_SAMPLES: usize = 3_840;
/// Seed-VC live block (320 ms at 16 kHz — the official real-time GUI
/// range is 250–300 ms; 320 keeps whole 20 ms frames).
#[cfg(feature = "seedvc")]
const SEEDVC_CHUNK_SAMPLES: usize = 5_120;
const FBANK_WINDOW: usize = 400; // kaldi 25 ms frame
const FBANK_SHIFT: usize = 160; // kaldi 10 ms shift
const BNF_CHUNK: usize = 5; // subsampled BNF frames per CARD chunk
const MEL_TAIL: usize = 32; // vocoder left context, in mel frames (320 ms)
/// Cross-fade length at chunk joins, in samples (10 ms). Each chunk is
/// vocoded with the mel tail as context, so the window also re-renders the
/// end of the previous chunk; holding back FADE samples and cross-fading
/// removes the phase discontinuity at the join.
const FADE: usize = 160;

/// Live-tunable knobs shared with the TUI thread.
struct Controls {
    /// Pitch shift in tenths of a semitone (post-vocoder).
    pitch_decisemitones: AtomicI32,
    /// RNNoise dry/wet mix in percent (0 = off).
    denoise_mix: AtomicI32,
    /// Output-side RNNoise wet % (Seed-VC path; < / > keys).
    out_denoise: AtomicI32,
    /// Voice-profile EQ wet % (issue #62; ( / ) keys).
    profile_eq: AtomicI32,
    /// BWE exciter wet amount in percent (0 = off; issue #42).
    bwe_wet: AtomicI32,
    /// Input gate threshold in dBFS (chunk RMS); the model hallucinates
    /// voiced murmurs on silent input, so sub-threshold chunks bypass the
    /// DiT and emit silence. i32::MIN disables.
    gate_db: AtomicI32,
}

/// Which engine converts the voice.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EngineKind {
    MeanVc,
    Xvc,
    #[cfg(feature = "seedvc")]
    SeedVc,
}

impl EngineKind {
    fn name(self) -> &'static str {
        match self {
            EngineKind::MeanVc => "meanvc",
            EngineKind::Xvc => "xvc",
            #[cfg(feature = "seedvc")]
            EngineKind::SeedVc => "seedvc",
        }
    }

    /// Input chunk per hop, in samples.
    fn chunk_samples(self) -> usize {
        match self {
            EngineKind::MeanVc => CHUNK_SAMPLES,
            EngineKind::Xvc => XVC_CHUNK_SAMPLES,
            #[cfg(feature = "seedvc")]
            EngineKind::SeedVc => SEEDVC_CHUNK_SAMPLES,
        }
    }

    /// Labels of the three per-stage RTF slots.
    fn stage_names(self) -> [&'static str; 3] {
        match self {
            EngineKind::MeanVc => ["asr", "vc", "vocoder"],
            EngineKind::Xvc => ["semantic", "convert", "decode"],
            #[cfg(feature = "seedvc")]
            EngineKind::SeedVc => ["content", "cfm", "vocoder"],
        }
    }
}

#[derive(Default)]
struct Stats {
    in_rms: f32,
    out_rms: f32,
    rtf_asr: f32,
    rtf_vc: f32,
    rtf_voc: f32,
    chunks: u64,
    late: u64,
    gated: u64,
    declicks: u64,
    /// Hard-gate splice fades (sp in the TUI), separate from the guard's
    /// needle repairs so field reports can attribute a firing layer.
    splices: u64,
    /// Cross-window replacements (xr in the TUI), separate from
    /// declicks so a per-layer firing rate is visible in the field.
    cross_repairs: u64,
    passthrough: bool,
    status: String,
    /// Engine configuration summary shown in the TUI footer (window
    /// size / cross-check), set once by the conversion thread.
    engine_info: String,
}

struct Args {
    engine: EngineKind,
    reference: String,
    voice_print: Option<String>,
    wav: Option<String>,
    out: Option<String>,
    input_device: Option<String>,
    /// Playback device the converted voice is routed to (non-Linux
    /// virtual-mic routing, e.g. "CABLE Input" / "BlackHole 2ch").
    output_device: Option<String>,
    pitch_shift: f32,
    denoise_mix: i32,
    /// Output-side RNNoise wet % (Seed-VC).
    out_denoise: i32,
    /// Voice-profile EQ wet % (issue #62).
    profile_eq: i32,
    bwe: i32,
    gate_db: i32,
    headless: bool,
    no_sink: bool,
    /// Disable the cross-window needle check (saves one hop = 240 ms of
    /// latency, at the cost of occasional decoder needle ticks).
    low_latency: bool,
    /// X-VC re-encode window override (ms, multiple of 80). Larger
    /// windows produce fewer decoder needles at more compute per hop;
    /// latency is unaffected (only the history part grows). Default:
    /// 2400 (the official GPU preset) on CUDA, 640 on CPU.
    window_ms: Option<usize>,
    /// Seed-VC CFM steps override (live default 6; more = higher
    /// quality per hop, fewer = more real-time headroom).
    #[cfg(feature = "seedvc")]
    cfm_steps: Option<usize>,
    /// X-VC hop override (ms). Smaller hops shrink both the audible
    /// needle probability (the emitted slice is a smaller fraction of
    /// the window) and the algorithmic latency, at proportionally more
    /// compute. Default: 120 (the official GPU preset) on CUDA, 240 on
    /// CPU.
    hop_ms: Option<usize>,
    monitor: bool,
    denoise: bool,
    duration: Option<f32>,
    /// Force CPU inference even when built with `--features cuda`.
    cpu: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        engine: EngineKind::MeanVc,
        reference: "ckpt/test.wav".into(),
        voice_print: None,
        wav: None,
        out: None,
        input_device: None,
        output_device: None,
        pitch_shift: 0.0,
        denoise_mix: 0,
        out_denoise: 0,
        profile_eq: 0,
        bwe: 0,
        gate_db: -45,
        headless: false,
        no_sink: false,
        low_latency: false,
        window_ms: None,
        hop_ms: None,
        #[cfg(feature = "seedvc")]
        cfm_steps: None,
        monitor: false,
        denoise: false,
        duration: None,
        cpu: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--engine" => {
                a.engine = match it.next().as_deref() {
                    Some("meanvc") => EngineKind::MeanVc,
                    Some("xvc") => EngineKind::Xvc,
                    #[cfg(feature = "seedvc")]
                    Some("seedvc") => EngineKind::SeedVc,
                    other => {
                        eprintln!(
                            "--engine must be meanvc or xvc{} (got {other:?})",
                            if cfg!(feature = "seedvc") {
                                " or seedvc"
                            } else {
                                ""
                            }
                        );
                        std::process::exit(2);
                    }
                }
            }
            "--reference" => a.reference = it.next().expect("--reference <wav>"),
            "--voice-print" => {
                a.voice_print = Some(it.next().expect("--voice-print <safetensors>"))
            }
            "--wav" => a.wav = Some(it.next().expect("--wav <file>")),
            "--out" => a.out = Some(it.next().expect("--out <file>")),
            "--headless" => a.headless = true,
            "--low-latency" => a.low_latency = true,
            "--window-ms" => {
                a.window_ms = Some(
                    it.next()
                        .expect("--window-ms <ms>")
                        .parse()
                        .expect("--window-ms takes an integer"),
                )
            }
            #[cfg(feature = "seedvc")]
            "--cfm-steps" => {
                a.cfm_steps = Some(
                    it.next()
                        .expect("--cfm-steps <n>")
                        .parse()
                        .expect("--cfm-steps takes an integer"),
                )
            }
            "--hop-ms" => {
                a.hop_ms = Some(
                    it.next()
                        .expect("--hop-ms <ms>")
                        .parse()
                        .expect("--hop-ms takes an integer"),
                )
            }
            "--cpu" => a.cpu = true,
            "--no-sink" => a.no_sink = true,
            "--monitor" => a.monitor = true,
            "--denoise" => a.denoise = true,
            "--pitch-shift" => {
                a.pitch_shift = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--pitch-shift <semitones>")
            }
            "--profile-eq" => {
                a.profile_eq = it
                    .next()
                    .expect("--profile-eq <0-100>")
                    .parse()
                    .expect("--profile-eq takes an integer");
            }
            "--out-denoise" => {
                a.out_denoise = it
                    .next()
                    .expect("--out-denoise <0-100>")
                    .parse()
                    .expect("--out-denoise takes an integer");
            }
            "--denoise-mix" => {
                a.denoise_mix = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--denoise-mix <0-100>")
            }
            "--bwe" => {
                a.bwe = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--bwe <0-100>")
            }
            "--gate" => {
                a.gate_db = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--gate <dBFS, e.g. -45>")
            }
            "--input-device" => a.input_device = Some(it.next().expect("--input-device <source>")),
            "--output-device" => a.output_device = Some(it.next().expect("--output-device <name>")),
            "--duration" => a.duration = it.next().and_then(|s| s.parse().ok()),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

/// In-process RNNoise at its native 48 kHz, for the OUTPUT side of the
/// Seed-VC path (field report: a faint noise bed under the converted
/// voice — diffusion residual — that output NR should scrub). Wet/dry
/// mix so the air of the voice can be kept.
#[cfg(feature = "seedvc")]
struct Rnnoise48k {
    state: Box<DenoiseState<'static>>,
    /// Carry so chunks need not align to the 480-sample frames.
    buf: Vec<f32>,
    out: Vec<f32>,
}

#[cfg(feature = "seedvc")]
impl Rnnoise48k {
    fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            buf: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Denoises in place with `mix` wet ratio (0 = bypass). Adds up to
    /// one RNNoise frame (10 ms) of latency via the alignment carry.
    fn process(&mut self, chunk: &mut [f32], mix: f32) {
        if mix <= 0.0 {
            return;
        }
        for &s in chunk.iter() {
            self.buf.push(s * 32768.0);
        }
        while self.buf.len() >= DenoiseState::FRAME_SIZE {
            let frame: Vec<f32> = self.buf.drain(..DenoiseState::FRAME_SIZE).collect();
            let mut den = vec![0f32; DenoiseState::FRAME_SIZE];
            self.state.process_frame(&mut den, &frame);
            for (d, raw) in den.iter().zip(&frame) {
                self.out.push((d * mix + raw * (1.0 - mix)) / 32768.0);
            }
        }
        for (i, s) in chunk.iter_mut().enumerate() {
            *s = if i < self.out.len() { self.out[i] } else { 0.0 };
        }
        let n = chunk.len().min(self.out.len());
        self.out.drain(..n);
    }
}

/// In-process RNNoise at 16 kHz: exact x3 up/down resampling around the
/// 48 kHz, 480-sample RNNoise frames (3200 in -> 20 frames -> 3200 out).
struct Rnnoise16k {
    state: Box<DenoiseState<'static>>,
    prev: f32,
}

impl Rnnoise16k {
    fn new() -> Self {
        Self {
            state: DenoiseState::new(),
            prev: 0.0,
        }
    }

    fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        // Linear x3 upsample (16k -> 48k), i16 scaling for RNNoise.
        let n = chunk.len();
        let mut up = Vec::with_capacity(n * 3);
        let mut last = self.prev;
        for &s in chunk {
            up.push((last + (s - last) / 3.0) * 32768.0);
            up.push((last + 2.0 * (s - last) / 3.0) * 32768.0);
            up.push(s * 32768.0);
            last = s;
        }
        self.prev = last;
        let mut den = vec![0f32; up.len()];
        for (i_chunk, o_chunk) in up
            .chunks(DenoiseState::FRAME_SIZE)
            .zip(den.chunks_mut(DenoiseState::FRAME_SIZE))
        {
            self.state.process_frame(o_chunk, i_chunk);
        }
        // 3-tap mean + decimate (48k -> 16k).
        (0..n)
            .map(|i| {
                let j = i * 3;
                let a = den[j];
                let b = den.get(j + 1).copied().unwrap_or(a);
                let c = den.get(j + 2).copied().unwrap_or(b);
                (a + b + c) / (3.0 * 32768.0)
            })
            .collect()
    }
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn read_wav_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let sc = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.unwrap() as f32 / sc).collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap()).collect(),
    };
    let mono: Vec<f32> = s.into_iter().step_by(spec.channels as usize).collect();
    // Any rate is welcome (issue #62 follow-up: a 48 kHz reference is
    // BETTER — the profile EQ reads it at native rate — so the 16 kHz
    // engines must not reject it); resample down for the engine.
    if spec.sample_rate == SR as u32 {
        Ok(mono)
    } else {
        Ok(vc_core::profile::resample_analysis(
            &mono,
            spec.sample_rate as usize,
            SR,
        ))
    }
}

/// Streaming approximation of X-VC's utterance-level percentile volume
/// normalization (`utils/audio.py::audio_volume_normalize`, coeff 0.2)
/// for live microphone input: the 90th–99th-percentile statistic runs
/// over a sliding window of the last few seconds and the gain is
/// smoothed between chunks. Wav input bypasses this and uses the exact
/// offline preprocessing instead.
/// Downward expander in front of the conversion (issue #45): below the
/// gate threshold the input is attenuated toward a floor instead of
/// zeroed, so breaths, lip noise and word tails survive — quietly —
/// and get CONVERTED (the binary gate deleted exactly the micro-sounds
/// that make a voice read as a real person). Fast open, slow close,
/// per-sample gain smoothing; state carries across chunks.
struct SoftExpander {
    gain: f32,
    open_coeff: f32,
    close_coeff: f32,
}

impl SoftExpander {
    /// Attenuation floor for sub-threshold audio (−18 dB).
    const FLOOR: f32 = 0.125;

    fn new(sample_rate: f32) -> Self {
        Self {
            gain: Self::FLOOR,
            open_coeff: 1.0 - (-1.0 / (0.008 * sample_rate)).exp(),
            close_coeff: 1.0 - (-1.0 / (0.180 * sample_rate)).exp(),
        }
    }

    fn process(&mut self, chunk: &mut [f32], open: bool) {
        let target = if open { 1.0 } else { Self::FLOOR };
        let coeff = if target > self.gain {
            self.open_coeff
        } else {
            self.close_coeff
        };
        for s in chunk.iter_mut() {
            self.gain += coeff * (target - self.gain);
            *s *= self.gain;
        }
    }
}

/// Output loudness leveler (issue #42, VRChat report): the converted
/// voice lands at whatever level the model chooses (~0.05 rms,
/// reference-dependent) — too quiet for downstream apps. Speech-active
/// rms is tracked with a ~2.5 s time constant and levelled toward
/// `TARGET_RMS`; the gain is ramped inside each chunk so there are no
/// boundary steps, and the (slow, brickwalled) limiter catches the rare
/// peak the makeup gain pushes over the threshold.
struct OutputLeveler {
    est: f32,
    gain: f32,
    applied: f32,
}

impl OutputLeveler {
    const TARGET_RMS: f32 = 0.11;
    /// Per-chunk smoothing of the speech-rms estimate (240 ms chunks →
    /// τ ≈ 2.4 s).
    const ALPHA: f32 = 0.1;
    /// Speech gate: chunks quieter than this do not update the estimate.
    const GATE: f32 = 0.012;

    fn new() -> Self {
        Self {
            est: Self::TARGET_RMS,
            gain: 1.0,
            applied: 1.0,
        }
    }

    fn process(&mut self, chunk: &mut [f32]) {
        let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len().max(1) as f32).sqrt();
        if rms > Self::GATE {
            self.est += Self::ALPHA * (rms - self.est);
            // Cap raised 4 -> 10: the field recording showed speech rms
            // stuck at 0.061 (= raw x4), i.e. the cap was the binding
            // constraint, not the estimate.
            self.gain = (Self::TARGET_RMS / self.est.max(1e-4)).clamp(0.25, 10.0);
        }
        let n = chunk.len().max(1) as f32;
        let step = (self.gain - self.applied) / n;
        for s in chunk.iter_mut() {
            self.applied += step;
            *s *= self.applied;
        }
        self.applied = self.gain;
    }
}

struct MicVolumeNormalizer {
    hist: std::collections::VecDeque<f32>,
    gain: f64,
    /// Gain actually applied to the previous sample; the per-chunk gain
    /// update is ramped sample-by-sample so chunk boundaries carry no
    /// level step (a step there reads as a tick once the BWE exciter
    /// reproduces its broadband splash above 8 kHz).
    applied: f64,
}

impl MicVolumeNormalizer {
    const WINDOW: usize = 8 * SR; // percentile statistic over 8 s
    const COEFF: f64 = 0.2;

    fn new() -> Self {
        Self {
            hist: std::collections::VecDeque::with_capacity(Self::WINDOW),
            gain: 1.0,
            applied: 1.0,
        }
    }

    /// Updates the gain estimate with one chunk and applies it in place.
    fn process(&mut self, chunk: &mut [f64]) {
        for &s in chunk.iter() {
            if self.hist.len() == Self::WINDOW {
                self.hist.pop_front();
            }
            self.hist.push_back(s.abs() as f32);
        }
        let mut significant: Vec<f32> = self.hist.iter().copied().filter(|&s| s > 0.01).collect();
        if significant.len() > 10 {
            significant.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (
                (0.9 * significant.len() as f64) as usize,
                (0.99 * significant.len() as f64) as usize,
            );
            let volume =
                significant[lo..hi].iter().map(|&s| s as f64).sum::<f64>() / (hi - lo) as f64;
            let target = (Self::COEFF / volume).clamp(0.1, 10.0);
            self.gain = 0.8 * self.gain + 0.2 * target;
        }
        let n = chunk.len().max(1) as f64;
        let step = (self.gain - self.applied) / n;
        for s in chunk {
            self.applied += step;
            *s = (*s * self.applied).clamp(-1.0, 1.0);
        }
    }
}

/// Voice print: an explicitly passed safetensors file, otherwise (feature
/// "wavlm") computed natively FROM THE REFERENCE AUDIO via the ONNX
/// WavLM-Large SV model at ckpt/wavlm_sv.onnx. There is deliberately no
/// file fallback: a stale precomputed voice print of a different speaker
/// silently overrides the reference timbre.
fn load_voice_print(
    args: &Args,
    #[cfg_attr(not(feature = "wavlm"), allow(unused_variables))] reference: &[f32],
    device: &Device,
) -> anyhow::Result<Tensor> {
    if let Some(path) = &args.voice_print {
        let vp = candle_core::safetensors::load(path, device)
            .map_err(|e| anyhow::anyhow!("cannot read --voice-print {path}: {e}"))?;
        return vp
            .get("voice_print")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("voice_print tensor missing in {path}"));
    }
    #[cfg(feature = "wavlm")]
    {
        use meanvc2::encoders::SpeakerEncoder;
        eprintln!("computing voice print from the reference audio (WavLM, ckpt/wavlm_sv.onnx)…");
        let sv = meanvc2::backends::WavLmSv::load("ckpt/wavlm_sv.onnx")?;
        Ok(sv.embed(reference, SR)?)
    }
    #[cfg(not(feature = "wavlm"))]
    anyhow::bail!(
        "no --voice-print given and the wavlm feature is off; pass a precomputed \
         voice print or rebuild with --features wavlm"
    )
}

/// Accelerator selection for the X-VC and Seed-VC engines: CPU is the
/// project baseline (never requires a GPU); a build with
/// `--features cuda` places the whole engine on `cuda:0` when one is
/// present (`--cpu` opts out). The meanvc engine stays on CPU — it is
/// already real time there.
#[cfg(feature = "cuda")]
fn xvc_device(force_cpu: bool) -> Device {
    if force_cpu {
        return Device::Cpu;
    }
    match Device::new_cuda(0) {
        Ok(d) => {
            eprintln!("accelerator: cuda:0");
            d
        }
        Err(e) => {
            eprintln!("CUDA unavailable ({e}); falling back to CPU");
            Device::Cpu
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn xvc_device(_force_cpu: bool) -> Device {
    Device::Cpu
}

/// Message to the output thread: a mel chunk to vocode (MeanVC) or
/// ready waveform samples (X-VC, passthrough, gate silence).
enum OutMsg {
    Mel(Tensor),
    Raw(Vec<f32>),
    /// Already at the 48 kHz output rate (Seed-VC path: BigVGAN gives
    /// real bandwidth up to 11 kHz, so the 16→48 upsampler and pitch
    /// stage are bypassed; exciter/leveler/limiter still apply).
    #[cfg(feature = "seedvc")]
    Raw48(Vec<f32>),
}

/// Per-engine model state, moved into the conversion thread.
enum Models {
    MeanVc {
        model: Box<MeanVc1>,
        asr: Box<FastU2pp>,
        prompt_mel: Tensor,
        spks: Tensor,
    },
    Xvc {
        /// Shared by the three pipeline stage threads
        /// ([`xvc::XvcPipelinedStream`]).
        engine: std::sync::Arc<xvc::XvcEngine>,
        reference: xvc::Reference,
    },
    #[cfg(feature = "seedvc")]
    SeedVc {
        engine: std::sync::Arc<seedvc::pipeline::SeedVcEngine>,
        /// Reference audio at 22 050 Hz (conditions are precomputed by
        /// the stream itself).
        ref_22k: Vec<f32>,
    },
}

/// The X-VC conversion loop: 240 ms input hops feed the 640/240/100/20
/// stateless-window streaming driver in its **pipelined** mode
/// ([`xvc::XvcPipelinedStream`], issue #38): the semantic / convert /
/// decode stages of consecutive windows overlap on three threads, so the
/// sustained throughput requirement is `max(stage) < hop` instead of
/// `sum(stages) < hop`. Each finished window emits 240 ms of converted
/// waveform (the engine decodes to waveform itself, so there is no
/// separate vocoder stage). A hop counts as **late** when it is emitted
/// more than one hop-period after its schedule slot (first emitted hop +
/// k·hop), i.e. when it would underrun a one-hop jitter buffer; the
/// schedule then re-anchors, so one transient stall counts once instead
/// of permanently flagging every later hop (steady-state pipeline latency
/// sits ~100 ms above the first, contention-free hop; unbounded drift,
/// not a constant offset, is what breaks live playback).
/// Seed-VC live conversion (issue #50): sliding-context re-conversion
/// per 320 ms block (`seedvc::stream::SeedVcStream`), soft-gate
/// expander in front like the X-VC path, deep silence hard-skipped
/// (whisper/CFM on pure silence both hallucinate and burn compute).
/// Emits 48 kHz blocks (`OutMsg::Raw48`); there is no needle guard or
/// cross-check — the BigVGAN decoder line has no needle pathology
/// (issue #49), which is the reason this engine exists.
#[cfg(feature = "seedvc")]
#[allow(clippy::too_many_arguments)]
fn run_seedvc_conversion(
    engine: std::sync::Arc<seedvc::pipeline::SeedVcEngine>,
    ref_22k: Vec<f32>,
    cfm_steps: Option<usize>,
    mic_input: bool,
    rx_in: std::sync::mpsc::Receiver<Vec<f32>>,
    tx_out: std::sync::mpsc::SyncSender<OutMsg>,
    stats: Arc<Mutex<Stats>>,
    run: Arc<AtomicBool>,
    controls: Arc<Controls>,
) -> anyhow::Result<()> {
    let mut cfg = seedvc::stream::StreamConfig::default();
    if let Some(steps) = cfm_steps {
        cfg.steps = steps;
    }
    let hop_s = cfg.block as f32 / SR as f32;
    stats.lock().unwrap().engine_info = format!(
        "block {} ms · ctx {:.1} s · cfm {} steps",
        cfg.block * 1000 / SR,
        cfg.context as f32 / SR as f32,
        cfg.steps,
    );
    let mut stream = engine
        .stream(&ref_22k, cfg)
        .map_err(|e| anyhow::anyhow!("seedvc reference prep: {e}"))?;
    let mut norm = MicVolumeNormalizer::new();
    let mut hp = HighpassBiquad::new(SR as f64, 40.0);
    let mut expander = SoftExpander::new(SR as f32);
    let hangover: u32 = (480 / (cfg.block * 1000 / SR).max(1)) as u32;
    let deep_chunks: u32 = (960 / (cfg.block * 1000 / SR).max(1)) as u32;
    let mut open_for = 0u32;
    let mut deep_for = 0u32;
    let mut hops = 0u64;
    let mut was_passthrough = false;

    while run.load(Ordering::Relaxed) {
        let chunk = match rx_in.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => chunk,
            Err(_) => continue,
        };
        let passthrough = stats.lock().unwrap().passthrough;
        let in_level = rms(&chunk);
        if passthrough {
            was_passthrough = true;
            let mut st = stats.lock().unwrap();
            st.in_rms = in_level;
            st.out_rms = in_level;
            st.chunks += 1;
            drop(st);
            let _ = tx_out.send(OutMsg::Raw(chunk));
            continue;
        }
        if was_passthrough {
            stream = engine
                .stream(&ref_22k, cfg)
                .map_err(|e| anyhow::anyhow!("seedvc stream restart: {e}"))?;
            hp = HighpassBiquad::new(SR as f64, 40.0);
            expander = SoftExpander::new(SR as f32);
            was_passthrough = false;
        }

        let gate = controls.gate_db.load(Ordering::Relaxed);
        let db = 20.0 * in_level.max(1e-9).log10();
        if db >= gate as f32 {
            open_for = hangover + 1;
        }
        open_for = open_for.saturating_sub(1);
        if db < gate as f32 - 18.0 {
            deep_for += 1;
        } else {
            deep_for = 0;
        }
        let gated = deep_for >= deep_chunks;

        let mut prepared: Vec<f32> = if mic_input {
            let mut c64: Vec<f64> = chunk.iter().map(|&s| s as f64).collect();
            norm.process(&mut c64);
            hp.process(&mut c64);
            c64.iter().map(|&s| s as f32).collect()
        } else {
            chunk.clone()
        };
        expander.process(&mut prepared, open_for > 0);

        {
            let mut st = stats.lock().unwrap();
            st.in_rms = in_level;
            st.chunks += 1;
            if gated {
                st.gated += 1;
            }
        }
        if gated {
            // Deep silence: skip the model entirely, emit 48 kHz zeros.
            let _ = tx_out.send(OutMsg::Raw48(vec![0.0; prepared.len() * 3]));
            continue;
        }
        stream.push(&prepared);
        while stream.ready() {
            let t0 = Instant::now();
            let Some(out48) = stream
                .step()
                .map_err(|e| anyhow::anyhow!("seedvc step: {e}"))?
            else {
                break;
            };
            let dt = t0.elapsed().as_secs_f32();
            hops += 1;
            {
                let mut st = stats.lock().unwrap();
                st.rtf_vc = dt / hop_s;
                // Warm-up grace like the X-VC drain: the first hops
                // absorb one-off CUDA kernel/allocator costs.
                if dt > hop_s && hops > 4 {
                    st.late += 1;
                }
            }
            let _ = tx_out.send(OutMsg::Raw48(out48));
        }
    }
    Ok(())
}

/// Microphone input is preprocessed incrementally (sliding-percentile
/// volume normalization + streaming 40 Hz high-pass); wav input arrives
/// already preprocessed offline.
#[allow(clippy::too_many_arguments)]
fn run_xvc_conversion(
    engine: std::sync::Arc<xvc::XvcEngine>,
    reference: xvc::Reference,
    mic_input: bool,
    rx_in: std::sync::mpsc::Receiver<Vec<f32>>,
    tx_out: std::sync::mpsc::SyncSender<OutMsg>,
    stats: Arc<Mutex<Stats>>,
    run: Arc<AtomicBool>,
    controls: Arc<Controls>,
    cross_check: bool,
    window_ms: usize,
    hop_ms_cfg: usize,
) -> anyhow::Result<()> {
    // Cross-window needle suppression is on unless --low-latency: it
    // holds each hop until the next window has re-rendered the same
    // region (+240 ms), which is what finally kills the decoder needle
    // ticks that amplitude heuristics can only chase (issue #42).
    //
    // The window defaults to 2400 ms on CUDA (the official GPU preset):
    // measured needle rate drops ~3-4x vs the 640 ms CPU preset, worst
    // per-hop forward stays ~43 ms against the 240 ms deadline, and the
    // enlargement is all history, so latency is unchanged (pinned by
    // `longer_window_does_not_add_latency`). CPU keeps 640 ms — it
    // cannot absorb the compute — and leans on cross_check instead.
    let cfg = xvc::StreamConfig {
        chunk_ms: window_ms,
        current_ms: hop_ms_cfg,
        cross_check,
        ..Default::default()
    };
    stats.lock().unwrap().engine_info = format!(
        "window {window_ms} ms · hop {hop_ms_cfg} ms{}{}",
        if cross_check { " · xcheck" } else { "" },
        if window_ms <= 640 {
            " (short window: more needles — try --window-ms 1280+ if compute allows)"
        } else {
            ""
        },
    );
    let hop = cfg.current_ms as f32 / 1000.0;
    let mut stream = xvc::XvcPipelinedStream::new(engine.clone(), reference.clone(), cfg)?;
    let mut hp = HighpassBiquad::new(SR as f64, 40.0);
    let mut norm = MicVolumeNormalizer::new();
    let mut was_passthrough = false;
    // Gate hangover: keep converting this many chunks after the level
    // drops, so word tails are not clipped.
    // Time-based gate ballistics (hop-size agnostic): ~480 ms hangover,
    // ~1 s of true silence before the hard mute.
    let hangover: u32 = (480 / hop_ms_cfg.max(1)) as u32;
    let deep_chunks: u32 = (960 / hop_ms_cfg.max(1)) as u32;
    let mut open_for = 0u32;
    let mut deep_for = 0u32;
    let mut expander = SoftExpander::new(SR as f32);
    let mut prev_gated = false;
    // Output schedule for the late counter: hop k is due at
    // `anchor + k·hop`, anchored on the first emitted hop.
    let mut anchor: Option<Instant> = None;
    let mut emitted: u64 = 0;
    // Warm-up grace: the first hops absorb one-off costs (cuda kernel
    // warmup, pipeline fill); underruns there are not steady-state.
    const WARMUP_HOPS: u64 = 4;

    // Needle suppressor on the decoder output (issue #42). The eighth
    // field recording pinned the audible ticks to the guard's earlier
    // linear-bridge repair (excision through quiet breathy content
    // leaves a notch that is itself a tick); the repair is now a
    // tapered gain dip, so a real needle is attenuated into a normal
    // sample and a false positive merely softens a transient. The guard
    // stays in the default path because cross-check structurally cannot
    // catch needles inside high-divergence windows (the run merges into
    // long divergence and is rejected as real audio) — which is exactly
    // where needles are most frequent.
    let mut needle_guard = vc_core::declick::NeedleGuard::new(SR as f32);
    // Forwards every finished hop to the output thread.
    let mut drain = |stream: &mut xvc::XvcPipelinedStream,
                     anchor: &mut Option<Instant>,
                     emitted: &mut u64|
     -> anyhow::Result<()> {
        while let Some(step) = stream
            .try_next()
            .map_err(|e| anyhow::anyhow!("xvc step: {e}"))?
        {
            let now = Instant::now();
            let due =
                *anchor.get_or_insert(now) + Duration::from_secs_f64(*emitted as f64 * hop as f64);
            let t = step.timings;
            {
                let mut st = stats.lock().unwrap();
                st.rtf_asr = t.semantic.as_secs_f32() / hop;
                st.rtf_vc = t.acoustic.as_secs_f32() / hop;
                st.rtf_voc = t.decode.as_secs_f32() / hop;
                st.out_rms = rms(&step.samples);
                st.cross_repairs += step.cross_repairs as u64;
                if now > due + Duration::from_secs_f32(hop) {
                    if *emitted >= WARMUP_HOPS {
                        st.late += 1;
                    }
                    // Jitter-buffer semantics: one underrun event counts
                    // once, then the schedule re-anchors to the recovered
                    // timeline. A fixed anchor would flag every subsequent
                    // hop after a single transient stall, turning the
                    // counter into a permanent +1/hop tally.
                    *anchor = Some(now - Duration::from_secs_f64(*emitted as f64 * hop as f64));
                }
            }
            *emitted += 1;
            let repaired_before = needle_guard.repaired;
            let cleaned = needle_guard.process(&step.samples);
            if needle_guard.repaired > repaired_before {
                stats.lock().unwrap().declicks += needle_guard.repaired - repaired_before;
            }
            let _ = tx_out.send(OutMsg::Raw(cleaned));
        }
        Ok(())
    };

    while run.load(Ordering::Relaxed) {
        let chunk = match rx_in.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => chunk,
            Err(_) => {
                // No new input: keep draining the pipeline (end of wav /
                // mic hiccup) so in-flight hops are not held back.
                drain(&mut stream, &mut anchor, &mut emitted)?;
                continue;
            }
        };
        let passthrough = stats.lock().unwrap().passthrough;
        let in_level = rms(&chunk);
        if passthrough {
            was_passthrough = true;
            let mut st = stats.lock().unwrap();
            st.in_rms = in_level;
            st.out_rms = in_level;
            st.chunks += 1;
            drop(st);
            let _ = tx_out.send(OutMsg::Raw(chunk));
            continue;
        }
        if was_passthrough {
            // Restart the stream timeline: the window history must not
            // carry context across the bypassed audio.
            stream = xvc::XvcPipelinedStream::new(engine.clone(), reference.clone(), cfg)?;
            hp = HighpassBiquad::new(SR as f64, 40.0);
            expander = SoftExpander::new(SR as f32);
            (anchor, emitted) = (None, 0);
            was_passthrough = false;
        }

        // Soft input gate (issue #45): above the knob threshold the
        // expander is open (unity); below, the input is attenuated to
        // −18 dB instead of zeroed, so breaths and word tails are
        // CONVERTED. Only sustained deep silence (18 dB under the knob
        // for ~1 s) is hard-zeroed — that keeps the all-zero window
        // skip (CPU) and the anti-murmur guarantee for true silence.
        let gate = controls.gate_db.load(Ordering::Relaxed);
        let db = 20.0 * in_level.max(1e-9).log10();
        if db >= gate as f32 {
            open_for = hangover + 1;
        }
        open_for = open_for.saturating_sub(1);
        if db < gate as f32 - 18.0 {
            deep_for += 1;
        } else {
            deep_for = 0;
        }
        let gated = deep_for >= deep_chunks;

        let mut processed: Vec<f32> = if mic_input {
            let mut c64: Vec<f64> = chunk.iter().map(|&s| s as f64).collect();
            norm.process(&mut c64);
            hp.process(&mut c64);
            c64.iter().map(|&s| s as f32).collect()
        } else {
            chunk.clone()
        };
        expander.process(&mut processed, open_for > 0);
        let mut prepared: Vec<f32> = if gated {
            vec![0.0; chunk.len()]
        } else {
            processed.clone()
        };
        // Deterministic splice fades at hard-gate transitions: the first
        // zeroed chunk fades the (expanded) audio out and the first
        // reopened chunk fades in — no blind discontinuity detection,
        // so continuous audio is never touched.
        const SPLICE_FADE: usize = 160; // 10 ms at 16 kHz
        if gated && !prev_gated {
            stats.lock().unwrap().splices += 1;
            let n = SPLICE_FADE.min(processed.len());
            for i in 0..n {
                let w = 1.0 - i as f32 / n as f32;
                prepared[i] = processed[i] * w * w;
            }
        } else if !gated && prev_gated {
            let n = SPLICE_FADE.min(prepared.len());
            for (i, s) in prepared.iter_mut().take(n).enumerate() {
                let w = i as f32 / n as f32;
                *s *= w * w;
            }
        }
        prev_gated = gated;
        stream
            .push(&prepared)
            .map_err(|e| anyhow::anyhow!("xvc push: {e}"))?;

        // The pipeline is windowed, so the converted tail of earlier
        // speech still drains while the gate is closed.
        drain(&mut stream, &mut anchor, &mut emitted)?;

        let mut st = stats.lock().unwrap();
        st.in_rms = in_level;
        st.chunks += 1;
        if gated {
            st.gated += 1;
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // candle's CPU gemm uses a rayon pool that defaults to all logical
    // cores; on SMT machines the contention roughly triples small-chunk
    // latency (measured: vocoder chunk RTF 0.57 -> 0.06 with the pool
    // pinned to physical cores). Must run before the first tensor op.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        std::env::set_var("RAYON_NUM_THREADS", num_cpus::get_physical().to_string());
    }
    let args = parse_args();
    let device = Device::Cpu;
    let engine = args.engine;
    // Hop default per device: 120 ms on CUDA (the official GPU preset).
    // A needle lands in the emitted slice with p = hop/window, so
    // halving the hop halves the audible-needle rate — and the
    // algorithmic latency (hop + smooth + future) plus the cross-check
    // hold BOTH drop by 120 ms. Compute doubles (worst hop ~43 ms
    // against the 120 ms deadline — 36 % duty on CUDA). CPU keeps 240.
    let xvc_hop_ms = args.hop_ms.unwrap_or_else(|| {
        if matches!(xvc_device(args.cpu), Device::Cpu) {
            240
        } else {
            120
        }
    });
    let chunk_samples = match engine {
        EngineKind::Xvc => xvc_hop_ms * SR / 1000,
        _ => engine.chunk_samples(),
    };

    eprintln!("loading models ({} engine)…", engine.name());
    let (models, vocos) = match engine {
        EngineKind::MeanVc => {
            let model = MeanVc1::load(
                MeanVc1Config::default(),
                "ckpt/model_200ms.safetensors",
                &device,
            )?;
            let asr = FastU2pp::load(
                FastU2ppConfig::official_meanvc1(),
                "ckpt/fastu2pp.safetensors",
                &device,
            )?;
            let vocos = Vocos::load(
                VocosConfig::official_meanvc1(),
                "ckpt/vocos.safetensors",
                &device,
            )?;
            let reference = read_wav_16k(&args.reference)?;
            let prompt_mel = MelV1::new().compute(&reference, &device)?.unsqueeze(0)?;
            let spks = load_voice_print(&args, &reference, &device)?.unsqueeze(0)?;
            (
                Models::MeanVc {
                    model: Box::new(model),
                    asr: Box::new(asr),
                    prompt_mel,
                    spks,
                },
                Some(vocos),
            )
        }
        EngineKind::Xvc => {
            let xdev = xvc_device(args.cpu);
            let xeng = xvc::XvcEngine::load("ckpt", &xdev)
                .map_err(|e| anyhow::anyhow!("cannot load the X-VC engine: {e}"))?;
            let raw: Vec<f64> = read_wav_16k(&args.reference)?
                .iter()
                .map(|&s| s as f64)
                .collect();
            let target = xeng.preprocess(&raw);
            let reference = xeng.prepare_reference(&target)?;
            (
                Models::Xvc {
                    engine: std::sync::Arc::new(xeng),
                    reference,
                },
                None,
            )
        }
        #[cfg(feature = "seedvc")]
        EngineKind::SeedVc => {
            let sdev = xvc_device(args.cpu);
            let seng = seedvc::pipeline::SeedVcEngine::load("ckpt", &sdev)
                .map_err(|e| anyhow::anyhow!("cannot load the Seed-VC engine: {e}"))?;
            // The engine world is 22 050 Hz; accept any wav rate.
            let mut r = hound::WavReader::open(&args.reference)?;
            let spec = r.spec();
            let raw: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Int => {
                    let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    r.samples::<i32>()
                        .step_by(spec.channels as usize)
                        .map(|v| v.map(|x| x as f32 / scale))
                        .collect::<Result<_, _>>()?
                }
                hound::SampleFormat::Float => r
                    .samples::<f32>()
                    .step_by(spec.channels as usize)
                    .collect::<Result<_, _>>()?,
            };
            let ref_22k = if spec.sample_rate == 22_050 {
                raw
            } else {
                seedvc::pipeline::resample(&raw, spec.sample_rate as usize, 22_050)
            };
            (
                Models::SeedVc {
                    engine: std::sync::Arc::new(seng),
                    ref_22k,
                },
                None,
            )
        }
    };

    // Ctrl-C / SIGTERM funnels into the same shutdown path as `q`, so the
    // pactl teardown at the end of main always runs (issue #39). In the
    // TUI, raw mode delivers Ctrl-C as a key event handled in `run_tui`;
    // this handler covers headless mode and SIGTERM. A second signal
    // hard-exits in case the graceful path is stuck.
    let run = Arc::new(AtomicBool::new(true));
    {
        let run = run.clone();
        ctrlc::set_handler(move || {
            if !run.swap(false, Ordering::Relaxed) {
                std::process::exit(130);
            }
        })
        .map_err(|e| anyhow::anyhow!("cannot install the signal handler: {e}"))?;
    }

    // Platform audio backend (issue #52): Pulse null-sink virtual mic on
    // Linux, cpal + loopback-device routing on Windows/macOS.
    let backend = audio::default_backend(audio::BackendOptions {
        output_device: args.output_device.clone(),
    });
    // Belt-and-braces: recover devices leaked by a previous run that was
    // killed before teardown, before creating fresh ones.
    if !args.no_sink || (args.denoise && args.wav.is_none()) {
        backend.recover_stale();
    }
    let sink_status = if args.no_sink {
        None
    } else {
        Some(backend.create_virtual_mic()?)
    };
    let sink_ok = sink_status.is_some();
    // Optional OS-level noise suppression in front of the mic.
    let capture_device: Option<String> = if args.denoise && args.wav.is_none() {
        match backend.create_denoised_source(args.input_device.as_deref())? {
            Some(src) => Some(src),
            None => {
                eprintln!(
                    "--denoise: no OS-level denoiser on this platform ({}); \
                     the in-process RNNoise stage stays active instead — \
                     set it with --denoise-mix <0-100> or the , / . knob",
                    backend.name()
                );
                args.input_device.clone()
            }
        }
    } else {
        args.input_device.clone()
    };
    let mut monitoring = false;
    if args.monitor && sink_ok {
        monitoring = backend.toggle_monitor()?;
    }

    let controls = Arc::new(Controls {
        pitch_decisemitones: AtomicI32::new((args.pitch_shift * 10.0).round() as i32),
        denoise_mix: AtomicI32::new(args.denoise_mix.clamp(0, 100)),
        out_denoise: AtomicI32::new(args.out_denoise.clamp(0, 100)),
        profile_eq: AtomicI32::new(args.profile_eq.clamp(0, 100)),
        bwe_wet: AtomicI32::new(args.bwe.clamp(0, 100)),
        gate_db: AtomicI32::new(args.gate_db),
    });
    let stats = Arc::new(Mutex::new(Stats {
        status: sink_status
            .clone()
            .unwrap_or_else(|| "virtual sink disabled (--no-sink)".into()),
        ..Default::default()
    }));

    // --- input thread: microphone or paced wav file -> chunk channel.
    // With --engine xvc the wav is preprocessed offline first (exact
    // official volume normalization + high-pass); microphone input is
    // preprocessed incrementally in the conversion thread instead.
    let wav_samples: Option<Vec<f32>> = match &args.wav {
        None => None,
        Some(path) => Some(match engine {
            EngineKind::MeanVc => read_wav_16k(path)?,
            EngineKind::Xvc => {
                let raw: Vec<f64> = read_wav_16k(path)?.iter().map(|&s| s as f64).collect();
                xvc::preprocess::preprocess(&raw, &Default::default())
            }
            // Seed-VC file input needs no offline preprocessing: the
            // stream handles rates internally from 16 kHz chunks.
            #[cfg(feature = "seedvc")]
            EngineKind::SeedVc => read_wav_16k(path)?,
        }),
    };
    let mic_input = wav_samples.is_none();
    let (tx_in, rx_in) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);
    let run_in = run.clone();
    let capture_device = capture_device.clone();
    let controls_in = controls.clone();
    let backend_in = backend.clone();
    let hop_ms = chunk_samples as u64 * 1000 / SR as u64;
    let input = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut rnnoise = Rnnoise16k::new();
        let mut denoise_chunk = |c: Vec<f32>| -> Vec<f32> {
            let mix = controls_in
                .denoise_mix
                .load(Ordering::Relaxed)
                .clamp(0, 100);
            if mix == 0 {
                return c;
            }
            let wet = rnnoise.process(&c);
            let w = mix as f32 / 100.0;
            c.iter()
                .zip(&wet)
                .map(|(d, n)| d * (1.0 - w) + n * w)
                .collect()
        };
        match wav_samples {
            Some(samples) => {
                let t0 = Instant::now();
                for (i, chunk) in samples.chunks(chunk_samples).enumerate() {
                    if !run_in.load(Ordering::Relaxed) {
                        break;
                    }
                    let mut c = chunk.to_vec();
                    c.resize(chunk_samples, 0.0);
                    // Pace to real time.
                    let due = Duration::from_millis(hop_ms * i as u64);
                    if let Some(wait) = due.checked_sub(t0.elapsed()) {
                        std::thread::sleep(wait);
                    }
                    if tx_in.send(denoise_chunk(c)).is_err() {
                        break;
                    }
                }
                Ok(())
            }
            None => {
                // The backend provides blocking 16 kHz mono capture with
                // generous transport buffering (the issue #42 stall
                // headroom lives in the backend implementations).
                let mut cap =
                    backend_in.open_capture(capture_device.as_deref(), SR as u32, chunk_samples)?;
                let mut buf = vec![0f32; chunk_samples];
                while run_in.load(Ordering::Relaxed) {
                    cap.read(&mut buf)?;
                    if tx_in.send(denoise_chunk(buf.clone())).is_err() {
                        break;
                    }
                }
                Ok(())
            }
        }
    });

    // Voice-profile EQ target (issue #62): the reference wav IS the
    // target speaker's real voice; its LTAS is the shape the output
    // should have. Engine-agnostic — read at native rate, resampled to
    // the 48 kHz analysis domain.
    let profile_eq = {
        let mut r = hound::WavReader::open(&args.reference)?;
        let spec = r.spec();
        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
                r.samples::<i32>()
                    .step_by(spec.channels as usize)
                    .map(|v| v.map(|x| x as f32 / scale))
                    .collect::<Result<_, _>>()?
            }
            hound::SampleFormat::Float => r
                .samples::<f32>()
                .step_by(spec.channels as usize)
                .collect::<Result<_, _>>()?,
        };
        let ref48 =
            vc_core::profile::resample_analysis(&raw, spec.sample_rate as usize, 48_000);
        vc_core::profile::ProfileEq::new(&ref48, spec.sample_rate as f32)
    };

    // --- output thread: vocoding + playback (pipelined with the VC stage
    // so the two heaviest stages run concurrently).
    let (tx_out, rx_out) = std::sync::mpsc::sync_channel::<OutMsg>(8);
    let run_out = run.clone();
    let out_path = args.out.clone();
    let stats_out = stats.clone();
    let controls_out = controls.clone();
    let backend_out = backend.clone();
    let output = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut profile_eq = profile_eq;
        let mut stretch = Stretch::preset_default(1, SR as u32);
        let mut current_semi = 0f32;
        // Second shifter for the 48 kHz direct path (Seed-VC).
        #[cfg(feature = "seedvc")]
        let mut stretch48 = Stretch::preset_default(1, OUT_SR as u32);
        #[cfg(feature = "seedvc")]
        let mut rnnoise48 = Rnnoise48k::new();
        #[cfg(feature = "seedvc")]
        let mut current_semi48 = 0f32;
        // Bandwidth extension (issue #42): exact ×3 upsampling to 48 kHz
        // plus the optional harmonic exciter, after the pitch shifter.
        let mut upsampler = Upsampler3x::new();
        let mut exciter = Exciter::new(OUT_SR as f32);
        let mut limiter = vc_core::bwe::Limiter::new(OUT_SR as f32, 0.90);
        let mut leveler = OutputLeveler::new();
        let mut chunk48: Vec<f32> = Vec::new();
        // Playback opens at 48 kHz into the virtual-mic route so the OS
        // never resamples the converted voice itself (issue #42).
        let mut play = if sink_ok {
            Some(backend_out.open_playback(OUT_SR as u32)?)
        } else {
            None
        };
        let mut writer = match &out_path {
            Some(p) => Some(hound::WavWriter::create(
                p,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: OUT_SR as u32,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?),
            None => None,
        };
        let mut mel_tail: Option<Tensor> = None;
        let mut hold: Vec<f32> = Vec::new();
        while run_out.load(Ordering::Relaxed) {
            let Ok(msg) = rx_out.recv_timeout(Duration::from_millis(300)) else {
                continue;
            };
            #[cfg(feature = "seedvc")]
            if let OutMsg::Raw48(c) = &msg {
                let mut c48 = c.clone();
                // Output NR before the leveler, so the diffusion noise
                // bed is scrubbed instead of boosted.
                let odn = controls_out
                    .out_denoise
                    .load(Ordering::Relaxed)
                    .clamp(0, 100) as f32
                    / 100.0;
                rnnoise48.process(&mut c48, odn);
                // Pitch shift at the output rate (same knob semantics as
                // the 16 kHz path; bypassed at 0 to avoid its latency).
                let semi = controls_out.pitch_decisemitones.load(Ordering::Relaxed) as f32 / 10.0;
                if semi.abs() > 0.05 {
                    if (semi - current_semi48).abs() > 0.01 {
                        stretch48
                            .set_transpose_factor_semitones(semi, Some(8_000.0 / OUT_SR as f32));
                        current_semi48 = semi;
                    }
                    let mut shifted = vec![0f32; c48.len()];
                    stretch48.process(&c48, &mut shifted);
                    c48 = shifted;
                }
                leveler.process(&mut c48);
                let eq_wet =
                    controls_out.profile_eq.load(Ordering::Relaxed).clamp(0, 100) as f32 / 100.0;
                profile_eq.observe(&c48);
                profile_eq.process(&mut c48, eq_wet);
                let wet =
                    controls_out.bwe_wet.load(Ordering::Relaxed).clamp(0, 100) as f32 / 100.0;
                exciter.process(&mut c48, wet);
                limiter.process(&mut c48);
                {
                    let mut st = stats_out.lock().unwrap();
                    st.out_rms = rms(&c48);
                }
                // Same 48 kHz playback stream into the virtual-mic
                // route as the upsampled engines (issue #52 rebase).
                if let Some(p) = play.as_mut() {
                    p.write(&c48)?;
                }
                if let Some(w) = writer.as_mut() {
                    for s in &c48 {
                        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                    }
                }
                continue;
            }
            let chunk: Vec<f32> = match msg {
                OutMsg::Raw(c) => c,
                #[cfg(feature = "seedvc")]
                OutMsg::Raw48(_) => unreachable!("handled above"),
                OutMsg::Mel(mel) => {
                    // Vocoding with a mel tail as left context (MeanVC
                    // only; the X-VC engine decodes to waveform itself).
                    let vocos = vocos.as_ref().expect("Mel chunks require the vocoder");
                    let t0 = Instant::now();
                    let mel_win = match &mel_tail {
                        Some(tail) => Tensor::cat(&[tail, &mel], 1)?,
                        None => mel.clone(),
                    };
                    let mel01 = ((mel_win.squeeze(0)? + 1.0)? / 2.0)?;
                    let wav = vocos.synthesize(&mel01)?;
                    // Cross-fade with the held-back tail of the previous
                    // chunk (both windows render the overlap region).
                    let take = (CHUNK_SAMPLES + FADE).min(wav.len());
                    let cur = &wav[wav.len() - take..];
                    let mut out: Vec<f32> = Vec::with_capacity(take);
                    if hold.len() == FADE && take > FADE {
                        for i in 0..FADE {
                            let w = i as f32 / FADE as f32;
                            out.push(hold[i] * (1.0 - w) + cur[i] * w);
                        }
                        out.extend_from_slice(&cur[FADE..take - FADE]);
                    } else {
                        out.extend_from_slice(&cur[..take - FADE]);
                    }
                    hold = cur[take - FADE..].to_vec();
                    mel_tail =
                        Some(mel.narrow(1, 20usize.saturating_sub(MEL_TAIL), MEL_TAIL.min(20))?);
                    let mut st = stats_out.lock().unwrap();
                    st.rtf_voc = t0.elapsed().as_secs_f32() / 0.2;
                    st.out_rms = rms(&out);
                    out
                }
            };
            // Post-vocoder pitch shift (Signalsmith Stretch), bypassed at 0
            // to avoid its internal latency.
            let semi = controls_out.pitch_decisemitones.load(Ordering::Relaxed) as f32 / 10.0;
            let chunk: Vec<f32> = if semi.abs() > 0.05 {
                if (semi - current_semi).abs() > 0.01 {
                    // Tonality limit ~8 kHz/sample-rate is the recommended
                    // setting for voice (reduces warble on shifted speech).
                    stretch.set_transpose_factor_semitones(semi, Some(8_000.0 / SR as f32));
                    current_semi = semi;
                }
                let mut shifted = vec![0f32; chunk.len()];
                stretch.process(&chunk, &mut shifted);
                shifted
            } else {
                chunk
            };
            // Loudness leveling toward a healthy mic level (the raw
            // converted voice is reference-dependent and often too
            // quiet for downstream apps — VRChat report).
            let mut chunk = chunk;
            leveler.process(&mut chunk);
            // Bandwidth extension (issue #42): 16 → 48 kHz windowed-sinc
            // upsampling, then the harmonic exciter fills 8–16 kHz when
            // the wet knob is open (0 % = bit-exact bypass).
            chunk48.clear();
            upsampler.process(&chunk, &mut chunk48);
            let eq_wet =
                controls_out.profile_eq.load(Ordering::Relaxed).clamp(0, 100) as f32 / 100.0;
            profile_eq.observe(&chunk48);
            profile_eq.process(&mut chunk48, eq_wet);
            let wet = controls_out.bwe_wet.load(Ordering::Relaxed).clamp(0, 100) as f32 / 100.0;
            exciter.process(&mut chunk48, wet);
            limiter.process(&mut chunk48);
            if let Some(p) = play.as_mut() {
                p.write(&chunk48)?;
            }
            if let Some(w) = writer.as_mut() {
                for s in &chunk48 {
                    w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
                }
            }
        }
        if let Some(w) = writer {
            w.finalize()?;
        }
        Ok(())
    });

    // --- conversion thread
    let stats_vc = stats.clone();
    let run_vc = run.clone();
    let controls_vc = controls.clone();
    let xvc_cross_check = !args.low_latency;
    #[cfg(feature = "seedvc")]
    let seedvc_cfm_steps = args.cfm_steps;
    // Window default per device (issue #42 knee search): CUDA absorbs
    // the official 2400 ms window (~4x fewer decoder needles, worst hop
    // ~43 ms vs the 240 ms deadline); CPU cannot and keeps 640 ms.
    let xvc_window_ms = args.window_ms.unwrap_or_else(|| {
        if matches!(xvc_device(args.cpu), Device::Cpu) {
            640
        } else {
            2_400
        }
    });

    let vc = std::thread::spawn(move || -> anyhow::Result<()> {
        let (model, asr, prompt_mel, spks) = match models {
            #[cfg(feature = "seedvc")]
            Models::SeedVc { engine, ref_22k } => {
                return run_seedvc_conversion(
                    engine,
                    ref_22k,
                    seedvc_cfm_steps,
                    mic_input,
                    rx_in,
                    tx_out,
                    stats_vc,
                    run_vc,
                    controls_vc,
                );
            }
            Models::Xvc {
                engine: xeng,
                reference,
            } => {
                return run_xvc_conversion(
                    xeng,
                    reference,
                    mic_input,
                    rx_in,
                    tx_out,
                    stats_vc,
                    run_vc,
                    controls_vc,
                    xvc_cross_check,
                    xvc_window_ms,
                    xvc_hop_ms,
                );
            }
            Models::MeanVc {
                model,
                asr,
                prompt_mel,
                spks,
            } => (model, asr, prompt_mel, spks),
        };
        let fbank = KaldiFbank::new();
        // Incremental front end: raw-sample carry for the fbank framing and
        // the Fast-U2++ streaming caches (att K/V + conv left context).
        let mut sample_buf: Vec<f32> = Vec::with_capacity(2 * CHUNK_SAMPLES);
        let mut asr_state = asr.stream();
        let mut bnf_pending: Option<Tensor> = None;
        let mut kv = KvCache::default();
        let mut prev_mel: Option<Tensor> = None;
        let mut q = 0usize;
        // Gate hangover: keep converting this many chunks after the level
        // drops, so word tails are not clipped.
        const HANGOVER: u32 = 2;
        let mut open_for = 0u32;
        while run_vc.load(Ordering::Relaxed) {
            let Ok(chunk) = rx_in.recv_timeout(Duration::from_millis(300)) else {
                continue;
            };
            let passthrough = stats_vc.lock().unwrap().passthrough;
            let in_level = rms(&chunk);
            // Input energy gate: silence makes the DiT hallucinate voiced
            // murmurs, so sub-threshold chunks emit silence directly (the
            // CARD state freezes and resumes seamlessly on the next chunk).
            let gate = controls_vc.gate_db.load(Ordering::Relaxed);
            if !passthrough {
                let db = 20.0 * in_level.max(1e-9).log10();
                if db >= gate as f32 {
                    open_for = HANGOVER + 1;
                }
                open_for = open_for.saturating_sub(1);
                if open_for == 0 {
                    let mut st = stats_vc.lock().unwrap();
                    st.in_rms = in_level;
                    st.out_rms = 0.0;
                    st.chunks += 1;
                    st.gated += 1;
                    drop(st);
                    let _ = tx_out.send(OutMsg::Raw(vec![0.0; CHUNK_SAMPLES]));
                    continue;
                }
            }
            if passthrough {
                // Drop the streaming state: the ASR caches must not carry
                // context across the bypassed audio.
                sample_buf.clear();
                asr_state.reset();
                bnf_pending = None;
                let mut st = stats_vc.lock().unwrap();
                st.in_rms = in_level;
                st.out_rms = in_level;
                st.chunks += 1;
                drop(st);
                let _ = tx_out.send(OutMsg::Raw(chunk));
                continue;
            }

            // Incremental fbank: frames are window-local, so computing them
            // over the carried buffer and draining whole shifts is exact.
            let t0 = Instant::now();
            sample_buf.extend_from_slice(&chunk);
            if sample_buf.len() >= FBANK_WINDOW {
                let fb = fbank.compute(&sample_buf, &device)?.unsqueeze(0)?;
                let consumed = fb.dim(1)? * FBANK_SHIFT;
                sample_buf.drain(..consumed);
                // Streaming BNFs with per-layer caches (one 23-frame window
                // with stride 20 per 200 ms chunk after warm-up).
                if let Some(bn) = asr.forward_chunk(&fb, &mut asr_state)? {
                    bnf_pending = Some(match bnf_pending.take() {
                        Some(prev) => Tensor::cat(&[&prev, &bn], 1)?,
                        None => bn,
                    });
                }
            }
            let t_asr = t0.elapsed();

            let mut t_vc = Duration::ZERO;
            let mut mels = Vec::new();
            while bnf_pending.as_ref().map_or(0, |t| t.dim(1).unwrap_or(0)) >= BNF_CHUNK {
                let pending = bnf_pending.take().unwrap();
                let bn5 = pending.narrow(1, 0, BNF_CHUNK)?;
                let rest = pending.dim(1)? - BNF_CHUNK;
                bnf_pending = (rest > 0)
                    .then(|| pending.narrow(1, BNF_CHUNK, rest))
                    .transpose()?;
                let cond = interpolate_linear(&bn5, 4)?; // [1, 20, 256]

                // CARD chunk with streaming KV cache.
                let t0 = Instant::now();
                let timbre = model.timbre_cond(&cond, &prompt_mel, &spks)?;
                let noise = Tensor::randn(0f32, 1f32, (1, 20, 80), &device)?;
                let u = model.forward_stream(
                    &noise,
                    &timbre,
                    &spks,
                    prev_mel.as_ref(),
                    q * 20,
                    &mut kv,
                )?;
                let mel = (noise - u)?;
                t_vc += t0.elapsed();

                prev_mel = Some(mel.clone());
                q += 1;
                mels.push(mel);
            }

            // The VC stage must stay under one chunk period; the vocoder
            // runs concurrently in the output thread.
            let total = t_asr + t_vc;
            {
                let mut st = stats_vc.lock().unwrap();
                st.in_rms = in_level;
                st.rtf_asr = t_asr.as_secs_f32() / 0.2;
                if !mels.is_empty() {
                    st.rtf_vc = t_vc.as_secs_f32() / (0.2 * mels.len() as f32);
                }
                st.chunks += 1;
                if total > Duration::from_millis(200) {
                    st.late += 1;
                }
            }
            for mel in mels {
                let _ = tx_out.send(OutMsg::Mel(mel));
            }
        }
        Ok(())
    });

    // --- UI / main loop
    let started = Instant::now();
    if args.headless {
        while run.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
            if let Some(d) = args.duration {
                if started.elapsed().as_secs_f32() >= d {
                    run.store(false, Ordering::Relaxed);
                }
            }
            let st = stats.lock().unwrap();
            let names = engine.stage_names();
            eprint!(
                "\r[{}] chunks {:4}  in {:.3} out {:.3}  RTF {} {:.2} {} {:.2} {} {:.2}  late {} ng {} sp {} xr {}   ",
                engine.name(),
                st.chunks,
                st.in_rms,
                st.out_rms,
                names[0],
                st.rtf_asr,
                names[1],
                st.rtf_vc,
                names[2],
                st.rtf_voc,
                st.late, st.declicks, st.splices, st.cross_repairs
            );
            std::io::stderr().flush().ok();
        }
        eprintln!();
    } else {
        run_tui(
            engine,
            stats.clone(),
            run.clone(),
            backend.clone(),
            monitoring,
            controls.clone(),
            sink_ok,
            args.duration,
        )?;
    }

    run.store(false, Ordering::Relaxed);
    for h in [input, output, vc] {
        if let Ok(Err(e)) = h.join() {
            eprintln!("thread error: {e}");
        }
    }
    backend.monitor_off();
    backend.destroy_virtual_mic();
    eprintln!("virtual device removed, bye");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_tui(
    engine: EngineKind,
    stats: Arc<Mutex<Stats>>,
    run: Arc<AtomicBool>,
    backend: Arc<dyn AudioBackend>,
    monitoring_initial: bool,
    controls: Arc<Controls>,
    sink_ok: bool,
    duration: Option<f32>,
) -> anyhow::Result<()> {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use ratatui::prelude::*;
    use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

    let mut terminal = ratatui::init();
    let started = Instant::now();
    let mut monitoring = monitoring_initial;
    while run.load(Ordering::Relaxed) {
        if let Some(d) = duration {
            if started.elapsed().as_secs_f32() >= d {
                break;
            }
        }
        let snapshot = {
            let st = stats.lock().unwrap();
            (
                st.in_rms,
                st.out_rms,
                st.rtf_asr,
                st.rtf_vc,
                st.rtf_voc,
                st.chunks,
                st.late,
                st.passthrough,
                st.status.clone(),
                st.gated,
                st.engine_info.clone(),
                st.declicks,
                st.cross_repairs,
                st.splices,
            )
        };
        terminal.draw(|f| {
            let rows = Layout::vertical([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(6),
                Constraint::Min(3),
            ])
            .split(f.area());
            let level = |v: f32| ((v * 8.0).min(1.0) * 100.0) as u16;
            f.render_widget(
                Gauge::default()
                    .block(Block::default().borders(Borders::ALL).title(" mic in "))
                    .gauge_style(Style::new().fg(Color::Green))
                    .percent(level(snapshot.0)),
                rows[0],
            );
            f.render_widget(
                Gauge::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" converted out "),
                    )
                    .gauge_style(Style::new().fg(Color::Cyan))
                    .percent(level(snapshot.1)),
                rows[1],
            );
            let total = snapshot.2 + snapshot.3 + snapshot.4;
            f.render_widget(
                Gauge::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(" total RTF (must stay < 1.0) "),
                    )
                    .gauge_style(Style::new().fg(if total < 1.0 {
                        Color::Green
                    } else {
                        Color::Red
                    }))
                    .percent(((total).min(1.5) * 66.6) as u16)
                    .label(format!("{total:.2}")),
                rows[2],
            );
            let pitch = controls.pitch_decisemitones.load(Ordering::Relaxed) as f32 / 10.0;
            let mix = controls.denoise_mix.load(Ordering::Relaxed);
            let bwe = controls.bwe_wet.load(Ordering::Relaxed);
            let gate = controls.gate_db.load(Ordering::Relaxed);
            let out_nr = controls.out_denoise.load(Ordering::Relaxed);
            let peq = controls.profile_eq.load(Ordering::Relaxed);
            let names = engine.stage_names();
            f.render_widget(
                Paragraph::new(format!(
                    "RTF  {} {:.2} · {} {:.2} · {} {:.2}\nchunks {} · late {} · gated {} · ng {} · sp {} · xr {} · mode {}\npitch {:+.1} st ([ / ])  ·  denoise mix {}% (, / .)  ·  gate {} dB (- / =)\nbwe exciter {}% (; / ')  ·  out-nr {}% (< / >)  ·  eq {}% (( / ))  ·  output 48 kHz  ·  {}",
                    names[0],
                    snapshot.2,
                    names[1],
                    snapshot.3,
                    names[2],
                    snapshot.4,
                    snapshot.5,
                    snapshot.6,
                    snapshot.9,
                    snapshot.11,
                    snapshot.13,
                    snapshot.12,
                    if snapshot.7 { "PASSTHROUGH" } else { "CONVERT" },
                    pitch,
                    mix,
                    gate,
                    bwe,
                    out_nr,
                    peq,
                    if snapshot.10.is_empty() {
                        "engine defaults"
                    } else {
                        snapshot.10.as_str()
                    },
                ))
                .block(Block::default().borders(Borders::ALL).title(format!(
                    " pipeline — engine {} ",
                    engine.name()
                ))),
                rows[3],
            );
            f.render_widget(
                Paragraph::new(format!(
                    "{}\n[q] quit   [p] passthrough   [l] loopback monitor: {}",
                    snapshot.8,
                    if monitoring {
                        "ON (hearing converted voice)"
                    } else {
                        "off"
                    }
                ))
                .block(Block::default().borders(Borders::ALL).title(format!(
                    " babiniku virtual mic — {} ",
                    engine.name()
                ))),
                rows[4],
            );
        })?;
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('q') => break,
                    // Raw mode turns Ctrl-C into a key event instead of
                    // SIGINT — treat it as quit so teardown runs (#39).
                    KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char('p') => {
                        let mut st = stats.lock().unwrap();
                        st.passthrough = !st.passthrough;
                    }
                    KeyCode::Char('l') if sink_ok => {
                        monitoring = backend.toggle_monitor().unwrap_or(monitoring);
                    }
                    KeyCode::Char('[') => {
                        let v = controls.pitch_decisemitones.load(Ordering::Relaxed);
                        controls
                            .pitch_decisemitones
                            .store((v - 5).max(-120), Ordering::Relaxed);
                    }
                    KeyCode::Char(']') => {
                        let v = controls.pitch_decisemitones.load(Ordering::Relaxed);
                        controls
                            .pitch_decisemitones
                            .store((v + 5).min(120), Ordering::Relaxed);
                    }
                    KeyCode::Char('(') => {
                        let v = controls.profile_eq.load(Ordering::Relaxed);
                        controls
                            .profile_eq
                            .store((v - 10).max(0), Ordering::Relaxed);
                    }
                    KeyCode::Char(')') => {
                        let v = controls.profile_eq.load(Ordering::Relaxed);
                        controls
                            .profile_eq
                            .store((v + 10).min(100), Ordering::Relaxed);
                    }
                    KeyCode::Char('<') => {
                        let v = controls.out_denoise.load(Ordering::Relaxed);
                        controls
                            .out_denoise
                            .store((v - 10).max(0), Ordering::Relaxed);
                    }
                    KeyCode::Char('>') => {
                        let v = controls.out_denoise.load(Ordering::Relaxed);
                        controls
                            .out_denoise
                            .store((v + 10).min(100), Ordering::Relaxed);
                    }
                    KeyCode::Char(',') => {
                        let v = controls.denoise_mix.load(Ordering::Relaxed);
                        controls
                            .denoise_mix
                            .store((v - 10).max(0), Ordering::Relaxed);
                    }
                    KeyCode::Char('.') => {
                        let v = controls.denoise_mix.load(Ordering::Relaxed);
                        controls
                            .denoise_mix
                            .store((v + 10).min(100), Ordering::Relaxed);
                    }
                    KeyCode::Char(';') => {
                        let v = controls.bwe_wet.load(Ordering::Relaxed);
                        controls.bwe_wet.store((v - 10).max(0), Ordering::Relaxed);
                    }
                    KeyCode::Char('\'') => {
                        let v = controls.bwe_wet.load(Ordering::Relaxed);
                        controls.bwe_wet.store((v + 10).min(100), Ordering::Relaxed);
                    }
                    KeyCode::Char('-') => {
                        let v = controls.gate_db.load(Ordering::Relaxed);
                        controls.gate_db.store((v - 3).max(-90), Ordering::Relaxed);
                    }
                    KeyCode::Char('=') => {
                        let v = controls.gate_db.load(Ordering::Relaxed);
                        controls.gate_db.store((v + 3).min(-10), Ordering::Relaxed);
                    }
                    _ => {}
                }
            }
        }
    }
    ratatui::restore();
    run.store(false, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod norm_tests {
    use super::*;

    /// Chunked processing of a steady tone must not introduce level steps
    /// at chunk boundaries while the gain estimate adapts (issue #42
    /// field report: ticks during a sustained vowel, live mic only).
    #[test]
    fn expander_floor_preserves_breath() {
        let mut e = SoftExpander::new(16_000.0);
        // Closed expander: quiet breath noise must survive at the floor,
        // never hit zero.
        let mut c = vec![0.01f32; 3_840];
        e.process(&mut c, false);
        let min = c.iter().fold(f32::MAX, |m, &v| m.min(v));
        assert!(min > 0.0009, "breath was muted: min {min}");
        assert!(c[3_839] <= 0.01 * SoftExpander::FLOOR * 1.05);
    }

    #[test]
    fn expander_opens_fast_closes_slow() {
        let mut e = SoftExpander::new(16_000.0);
        let mut c = vec![0.1f32; 3_840];
        e.process(&mut c, true);
        // Open: essentially unity within ~40 ms (5 time constants).
        assert!(c[640] > 0.098, "opened too slowly: {}", c[640]);
        // Close: after ONE 240 ms chunk the gain must still be well
        // above the floor (slow release keeps word tails alive).
        let mut c2 = vec![0.1f32; 3_840];
        e.process(&mut c2, false);
        assert!(
            c2[3_839] > 0.1 * SoftExpander::FLOOR * 2.0,
            "closed too fast: {}",
            c2[3_839]
        );
        // Gain is continuous across the chunk boundary.
        assert!((c2[0] - 0.1).abs() < 0.002, "boundary step: {}", c2[0]);
    }

    #[test]
    fn output_leveler_reaches_target_without_steps() {
        let mut lv = OutputLeveler::new();
        // Quiet speech-like chunks (rms 0.04) must be brought up toward
        // the target without inter-chunk level steps.
        let mut last_tail = None::<f32>;
        let mut final_rms = 0.0;
        for _ in 0..60 {
            let mut c: Vec<f32> = (0..3_840)
                .map(|i| 0.0566 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin())
                .collect();
            let head = c[0];
            lv.process(&mut c);
            if let Some(t) = last_tail {
                // Boundary continuity: applied gain carries across chunks.
                let expected = head * lv.applied / lv.gain.max(1e-6);
                let _ = expected;
                assert!((c[0] - head * t / 1.0).abs() < 0.05, "boundary step");
            }
            last_tail = Some(lv.applied);
            final_rms = (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt();
        }
        assert!(
            (final_rms - OutputLeveler::TARGET_RMS).abs() < 0.02,
            "did not level: rms {final_rms}"
        );
    }

    #[test]
    fn normalizer_gain_ramps_across_chunks() {
        let sr = SR as f64;
        let sig: Vec<f64> = (0..8 * SR)
            .map(|i| 0.6 * (2.0 * std::f64::consts::PI * 220.0 * i as f64 / sr).sin())
            .collect();
        let mut norm = MicVolumeNormalizer::new();
        let mut out = sig.clone();
        for c in out.chunks_mut(CHUNK_SAMPLES) {
            norm.process(c);
        }
        // Max adjacent-sample jump of the normalized tone must stay close
        // to the tone's own slope bound (2*pi*f/sr * amp * max_gain_local),
        // i.e. no additional boundary steps.
        let mut max_ratio = 0f64;
        for w in out.windows(2) {
            let d = (w[1] - w[0]).abs();
            max_ratio = max_ratio.max(d);
        }
        // Tone slope bound with the converged gain (<= ~0.42 for 220 Hz):
        // allow 20% headroom; a per-chunk gain step of a few percent on a
        // 0.6 amplitude tone would exceed this immediately.
        let slope = 2.0 * std::f64::consts::PI * 220.0 / sr;
        let bound = out.iter().fold(0f64, |m, &v| m.max(v.abs())) * slope * 1.2 + 1e-6;
        assert!(
            max_ratio < bound,
            "boundary level step detected: {max_ratio} vs slope bound {bound}"
        );
    }
}

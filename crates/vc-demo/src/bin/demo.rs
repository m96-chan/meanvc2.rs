//! Real-time voice-conversion TUI demo with a virtual microphone.
//!
//! Captures the default microphone, converts the voice chunk by chunk
//! with the selected engine (`--engine meanvc` (default, 200 ms chunks) or
//! `--engine xvc` (240 ms hop, 640 ms re-encoded window — the official
//! X-VC CPU streaming preset)), and plays the result into a
//! PulseAudio/PipeWire null sink whose remapped monitor shows up as a
//! selectable **virtual microphone** (`babiniku_mic`) in other apps.
//!
//! ```sh
//! cargo run --release -p vc-demo --bin babiniku-demo -- \
//!     --reference ckpt/test.wav --voice-print ckpt/voice_print_test.safetensors
//! cargo run --release -p vc-demo --bin babiniku-demo -- \
//!     --engine xvc --reference her_voice.wav
//! ```
//!
//! Keys: `q` quit · `p` passthrough (bypass conversion for A/B) ·
//! `l` loopback monitor (hear the converted voice on the default output) ·
//! `[` / `]` pitch shift −/+0.5 semitone (post-vocoder, Signalsmith
//! Stretch) · `,` / `.` RNNoise denoise mix −/+10 % (pre-ASR, in-process;
//! independent of the `--denoise` WebRTC stage).
//!
//! Options: `--pitch-shift <semitones>` / `--denoise-mix <0-100>` set the
//! initial knob values, `--denoise` inserts PipeWire's WebRTC noise suppression in
//! front of the microphone (recommended for noisy mics),
//! `--input-device <source>` records from a specific PulseAudio source,
//! `--wav <file>` converts a wav file instead of the microphone
//! (paced in real time), `--headless` disables the TUI, `--out <file>`
//! additionally records the converted audio, `--no-sink` skips creating
//! the virtual device, `--monitor` starts with the loopback enabled,
//! `--duration <secs>` auto-stops (for testing).
//!
//! BNFs are extracted with the incremental `FastU2pp::forward_chunk`
//! streaming caches (issue #9), bit-matching the official WeNet chunked
//! decode; the remaining approximation is the vocoder, which re-synthesizes
//! each chunk with a 200 ms mel tail as context (tracked in #9).

use std::io::Write as _;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use candle_core::{Device, Tensor};
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use meanvc2::backends::{FastU2pp, FastU2ppConfig, Vocos, VocosConfig};
use meanvc2::encoders::Vocoder;
use meanvc2::v1::{interpolate_linear, KaldiFbank, KvCache, MeanVc1, MeanVc1Config, MelV1};
use nnnoiseless::DenoiseState;
use signalsmith_stretch::Stretch;
use std::sync::atomic::AtomicI32;
use xvc::preprocess::HighpassBiquad;

const SR: usize = 16_000;
const CHUNK_SAMPLES: usize = 3_200; // 200 ms = one CARD chunk (20 mel frames)
/// X-VC hop: 240 ms of new audio per re-encoded 640 ms window (the CPU
/// streaming preset 640/240/100/20 from issue #30).
const XVC_CHUNK_SAMPLES: usize = 3_840;
const FBANK_WINDOW: usize = 400; // kaldi 25 ms frame
const FBANK_SHIFT: usize = 160; // kaldi 10 ms shift
const BNF_CHUNK: usize = 5; // subsampled BNF frames per CARD chunk
const MEL_TAIL: usize = 32; // vocoder left context, in mel frames (320 ms)
/// Cross-fade length at chunk joins, in samples (10 ms). Each chunk is
/// vocoded with the mel tail as context, so the window also re-renders the
/// end of the previous chunk; holding back FADE samples and cross-fading
/// removes the phase discontinuity at the join.
const FADE: usize = 160;
const SINK: &str = "babiniku";
const VIRT_MIC: &str = "babiniku_mic";

/// Live-tunable knobs shared with the TUI thread.
struct Controls {
    /// Pitch shift in tenths of a semitone (post-vocoder).
    pitch_decisemitones: AtomicI32,
    /// RNNoise dry/wet mix in percent (0 = off).
    denoise_mix: AtomicI32,
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
}

impl EngineKind {
    fn name(self) -> &'static str {
        match self {
            EngineKind::MeanVc => "meanvc",
            EngineKind::Xvc => "xvc",
        }
    }

    /// Input chunk per hop, in samples.
    fn chunk_samples(self) -> usize {
        match self {
            EngineKind::MeanVc => CHUNK_SAMPLES,
            EngineKind::Xvc => XVC_CHUNK_SAMPLES,
        }
    }

    /// Labels of the three per-stage RTF slots.
    fn stage_names(self) -> [&'static str; 3] {
        match self {
            EngineKind::MeanVc => ["asr", "vc", "vocoder"],
            EngineKind::Xvc => ["semantic", "convert", "decode"],
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
    passthrough: bool,
    status: String,
}

struct Args {
    engine: EngineKind,
    reference: String,
    voice_print: Option<String>,
    wav: Option<String>,
    out: Option<String>,
    input_device: Option<String>,
    pitch_shift: f32,
    denoise_mix: i32,
    gate_db: i32,
    headless: bool,
    no_sink: bool,
    monitor: bool,
    denoise: bool,
    duration: Option<f32>,
}

fn parse_args() -> Args {
    let mut a = Args {
        engine: EngineKind::MeanVc,
        reference: "ckpt/test.wav".into(),
        voice_print: None,
        wav: None,
        out: None,
        input_device: None,
        pitch_shift: 0.0,
        denoise_mix: 0,
        gate_db: -45,
        headless: false,
        no_sink: false,
        monitor: false,
        denoise: false,
        duration: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--engine" => {
                a.engine = match it.next().as_deref() {
                    Some("meanvc") => EngineKind::MeanVc,
                    Some("xvc") => EngineKind::Xvc,
                    other => {
                        eprintln!("--engine must be meanvc or xvc (got {other:?})");
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
            "--no-sink" => a.no_sink = true,
            "--monitor" => a.monitor = true,
            "--denoise" => a.denoise = true,
            "--pitch-shift" => {
                a.pitch_shift = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--pitch-shift <semitones>")
            }
            "--denoise-mix" => {
                a.denoise_mix = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--denoise-mix <0-100>")
            }
            "--gate" => {
                a.gate_db = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--gate <dBFS, e.g. -45>")
            }
            "--input-device" => a.input_device = Some(it.next().expect("--input-device <source>")),
            "--duration" => a.duration = it.next().and_then(|s| s.parse().ok()),
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

/// Creates the null sink + remapped virtual microphone; returns module ids.
fn create_virtual_device() -> anyhow::Result<Vec<String>> {
    let load = |args: &[&str]| -> anyhow::Result<String> {
        let out = Command::new("pactl")
            .arg("load-module")
            .args(args)
            .output()?;
        anyhow::ensure!(
            out.status.success(),
            "pactl load-module failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    let sink = load(&[
        "module-null-sink",
        &format!("sink_name={SINK}"),
        "sink_properties=device.description=Babiniku-Output",
    ])?;
    let mic = load(&[
        "module-remap-source",
        &format!("source_name={VIRT_MIC}"),
        &format!("master={SINK}.monitor"),
        "source_properties=device.description=Babiniku-Virtual-Mic",
    ])?;
    Ok(vec![sink, mic])
}

const DENOISED_SRC: &str = "babiniku_denoised";

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

/// PipeWire/Pulse WebRTC noise suppression in front of the microphone:
/// creates a cleaned source the input thread records from. Returns the
/// pactl module id.
fn create_denoiser(master: Option<&str>) -> anyhow::Result<String> {
    let mut cmd = Command::new("pactl");
    cmd.args([
        "load-module",
        "module-echo-cancel",
        &format!("source_name={DENOISED_SRC}"),
        "aec_method=webrtc",
        "source_properties=device.description=Babiniku-Denoised-Input",
    ]);
    if let Some(m) = master {
        cmd.arg(format!("source_master={m}"));
    }
    let out = cmd.output()?;
    anyhow::ensure!(
        out.status.success(),
        "pactl module-echo-cancel failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Loopback monitor: routes the converted audio to the default output so
/// the user can hear it. Returns the pactl module id.
struct Monitor(Mutex<Option<String>>);

impl Monitor {
    fn toggle(&self) -> anyhow::Result<bool> {
        let mut slot = self.0.lock().unwrap();
        match slot.take() {
            Some(id) => {
                let _ = Command::new("pactl").args(["unload-module", &id]).status();
                Ok(false)
            }
            None => {
                let out = Command::new("pactl")
                    .args([
                        "load-module",
                        "module-loopback",
                        &format!("source={SINK}.monitor"),
                        "latency_msec=60",
                    ])
                    .output()?;
                anyhow::ensure!(
                    out.status.success(),
                    "pactl module-loopback failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
                *slot = Some(String::from_utf8_lossy(&out.stdout).trim().to_string());
                Ok(true)
            }
        }
    }

    fn off(&self) {
        let mut slot = self.0.lock().unwrap();
        if let Some(id) = slot.take() {
            let _ = Command::new("pactl").args(["unload-module", &id]).status();
        }
    }
}

fn destroy_virtual_device(modules: &[String]) {
    for m in modules.iter().rev() {
        let _ = Command::new("pactl").args(["unload-module", m]).status();
    }
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn read_wav_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    anyhow::ensure!(spec.sample_rate == SR as u32, "expected 16 kHz wav: {path}");
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let sc = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.unwrap() as f32 / sc).collect()
        }
        hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap()).collect(),
    };
    Ok(s.into_iter().step_by(spec.channels as usize).collect())
}

/// Streaming approximation of X-VC's utterance-level percentile volume
/// normalization (`utils/audio.py::audio_volume_normalize`, coeff 0.2)
/// for live microphone input: the 90th–99th-percentile statistic runs
/// over a sliding window of the last few seconds and the gain is
/// smoothed between chunks. Wav input bypasses this and uses the exact
/// offline preprocessing instead.
struct MicVolumeNormalizer {
    hist: std::collections::VecDeque<f32>,
    gain: f64,
}

impl MicVolumeNormalizer {
    const WINDOW: usize = 8 * SR; // percentile statistic over 8 s
    const COEFF: f64 = 0.2;

    fn new() -> Self {
        Self {
            hist: std::collections::VecDeque::with_capacity(Self::WINDOW),
            gain: 1.0,
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
        for s in chunk {
            *s = (*s * self.gain).clamp(-1.0, 1.0);
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

fn pulse_spec() -> Spec {
    Spec {
        format: Format::FLOAT32NE,
        channels: 1,
        rate: SR as u32,
    }
}

/// Message to the output thread: a mel chunk to vocode (MeanVC) or
/// ready waveform samples (X-VC, passthrough, gate silence).
enum OutMsg {
    Mel(Tensor),
    Raw(Vec<f32>),
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
        engine: Box<xvc::XvcEngine>,
        reference: xvc::Reference,
    },
}

/// The X-VC conversion loop: 240 ms input hops feed the 640/240/100/20
/// stateless-window streaming driver; each ready window emits 240 ms of
/// converted waveform (the engine decodes to waveform itself, so there is
/// no separate vocoder stage). Microphone input is preprocessed
/// incrementally (sliding-percentile volume normalization + streaming
/// 40 Hz high-pass); wav input arrives already preprocessed offline.
#[allow(clippy::too_many_arguments)]
fn run_xvc_conversion(
    engine: &xvc::XvcEngine,
    reference: xvc::Reference,
    mic_input: bool,
    rx_in: std::sync::mpsc::Receiver<Vec<f32>>,
    tx_out: std::sync::mpsc::SyncSender<OutMsg>,
    stats: Arc<Mutex<Stats>>,
    run: Arc<AtomicBool>,
    controls: Arc<Controls>,
) -> anyhow::Result<()> {
    let cfg = xvc::StreamConfig::default();
    let hop = cfg.current_ms as f32 / 1000.0;
    let mut stream = engine.stream(reference.clone(), cfg)?;
    let mut hp = HighpassBiquad::new(SR as f64, 40.0);
    let mut norm = MicVolumeNormalizer::new();
    let mut was_passthrough = false;
    // Gate hangover: keep converting this many chunks after the level
    // drops, so word tails are not clipped.
    const HANGOVER: u32 = 2;
    let mut open_for = 0u32;
    while run.load(Ordering::Relaxed) {
        let Ok(chunk) = rx_in.recv_timeout(Duration::from_millis(300)) else {
            continue;
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
            stream = engine.stream(reference.clone(), cfg)?;
            hp = HighpassBiquad::new(SR as f64, 40.0);
            was_passthrough = false;
        }

        // Input energy gate: silent windows skip the whole model forward
        // (the converter hallucinates voiced murmurs on silence).
        let gate = controls.gate_db.load(Ordering::Relaxed);
        let db = 20.0 * in_level.max(1e-9).log10();
        if db >= gate as f32 {
            open_for = HANGOVER + 1;
        }
        open_for = open_for.saturating_sub(1);
        let gated = open_for == 0;

        let prepared: Vec<f32> = if gated {
            vec![0.0; chunk.len()]
        } else if mic_input {
            let mut c64: Vec<f64> = chunk.iter().map(|&s| s as f64).collect();
            norm.process(&mut c64);
            hp.process(&mut c64);
            c64.iter().map(|&s| s as f32).collect()
        } else {
            chunk
        };
        stream.push(&prepared);

        // The pipeline is windowed, so the converted tail of earlier
        // speech still drains while the gate is closed.
        while let Some(step) = stream
            .step()
            .map_err(|e| anyhow::anyhow!("xvc step: {e}"))?
        {
            let t = step.timings;
            {
                let mut st = stats.lock().unwrap();
                st.rtf_asr = t.semantic.as_secs_f32() / hop;
                st.rtf_vc = t.acoustic.as_secs_f32() / hop;
                st.rtf_voc = t.decode.as_secs_f32() / hop;
                st.out_rms = rms(&step.samples);
                if t.total() > Duration::from_secs_f32(hop) {
                    st.late += 1;
                }
            }
            let _ = tx_out.send(OutMsg::Raw(step.samples));
        }

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
    let chunk_samples = engine.chunk_samples();

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
            let xeng = xvc::XvcEngine::load("ckpt", &device)
                .map_err(|e| anyhow::anyhow!("cannot load the X-VC engine: {e}"))?;
            let raw: Vec<f64> = read_wav_16k(&args.reference)?
                .iter()
                .map(|&s| s as f64)
                .collect();
            let target = xeng.preprocess(&raw);
            let reference = xeng.prepare_reference(&target)?;
            (
                Models::Xvc {
                    engine: Box::new(xeng),
                    reference,
                },
                None,
            )
        }
    };

    let mut modules = if args.no_sink {
        Vec::new()
    } else {
        create_virtual_device()?
    };
    let sink_ok = !modules.is_empty();
    // Optional noise suppression in front of the mic.
    let capture_device: Option<String> = if args.denoise && args.wav.is_none() {
        modules.push(create_denoiser(args.input_device.as_deref())?);
        Some(DENOISED_SRC.to_string())
    } else {
        args.input_device.clone()
    };
    let monitor = Arc::new(Monitor(Mutex::new(None)));
    if args.monitor && sink_ok {
        monitor.toggle()?;
    }

    let controls = Arc::new(Controls {
        pitch_decisemitones: AtomicI32::new((args.pitch_shift * 10.0).round() as i32),
        denoise_mix: AtomicI32::new(args.denoise_mix.clamp(0, 100)),
        gate_db: AtomicI32::new(args.gate_db),
    });
    let stats = Arc::new(Mutex::new(Stats {
        status: if sink_ok {
            format!(
                "virtual mic \"{VIRT_MIC}\" is live — select \"Babiniku-Virtual-Mic\" in your app"
            )
        } else {
            "virtual sink disabled (--no-sink)".into()
        },
        ..Default::default()
    }));
    let run = Arc::new(AtomicBool::new(true));

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
        }),
    };
    let mic_input = wav_samples.is_none();
    let (tx_in, rx_in) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);
    let run_in = run.clone();
    let capture_device = capture_device.clone();
    let controls_in = controls.clone();
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
                let s = Simple::new(
                    None,
                    "babiniku-demo",
                    Direction::Record,
                    capture_device.as_deref(),
                    "capture",
                    &pulse_spec(),
                    None,
                    None,
                )
                .map_err(|e| anyhow::anyhow!("pulse record: {e}"))?;
                let mut buf = vec![0u8; chunk_samples * 4];
                while run_in.load(Ordering::Relaxed) {
                    s.read(&mut buf).map_err(|e| anyhow::anyhow!("read: {e}"))?;
                    let c: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    if tx_in.send(denoise_chunk(c)).is_err() {
                        break;
                    }
                }
                Ok(())
            }
        }
    });

    // --- output thread: vocoding + playback (pipelined with the VC stage
    // so the two heaviest stages run concurrently).
    let (tx_out, rx_out) = std::sync::mpsc::sync_channel::<OutMsg>(8);
    let run_out = run.clone();
    let out_path = args.out.clone();
    let stats_out = stats.clone();
    let controls_out = controls.clone();
    let output = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut stretch = Stretch::preset_default(1, SR as u32);
        let mut current_semi = 0f32;
        let play = if sink_ok {
            Some(
                Simple::new(
                    None,
                    "babiniku-demo",
                    Direction::Playback,
                    Some(SINK),
                    "converted",
                    &pulse_spec(),
                    None,
                    None,
                )
                .map_err(|e| anyhow::anyhow!("pulse playback: {e}"))?,
            )
        } else {
            None
        };
        let mut writer = match &out_path {
            Some(p) => Some(hound::WavWriter::create(
                p,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: SR as u32,
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
            let chunk: Vec<f32> = match msg {
                OutMsg::Raw(c) => c,
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
            if let Some(p) = &play {
                let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_ne_bytes()).collect();
                p.write(&bytes).map_err(|e| anyhow::anyhow!("write: {e}"))?;
            }
            if let Some(w) = writer.as_mut() {
                for s in &chunk {
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
    let vc = std::thread::spawn(move || -> anyhow::Result<()> {
        let (model, asr, prompt_mel, spks) = match models {
            Models::Xvc {
                engine: xeng,
                reference,
            } => {
                return run_xvc_conversion(
                    &xeng,
                    reference,
                    mic_input,
                    rx_in,
                    tx_out,
                    stats_vc,
                    run_vc,
                    controls_vc,
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
                "\r[{}] chunks {:4}  in {:.3} out {:.3}  RTF {} {:.2} {} {:.2} {} {:.2}  late {}   ",
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
                st.late
            );
            std::io::stderr().flush().ok();
        }
        eprintln!();
    } else {
        run_tui(
            engine,
            stats.clone(),
            run.clone(),
            monitor.clone(),
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
    monitor.off();
    destroy_virtual_device(&modules);
    eprintln!("virtual device removed, bye");
    Ok(())
}

fn run_tui(
    engine: EngineKind,
    stats: Arc<Mutex<Stats>>,
    run: Arc<AtomicBool>,
    monitor: Arc<Monitor>,
    controls: Arc<Controls>,
    sink_ok: bool,
    duration: Option<f32>,
) -> anyhow::Result<()> {
    use crossterm::event::{self, Event, KeyCode};
    use ratatui::prelude::*;
    use ratatui::widgets::{Block, Borders, Gauge, Paragraph};

    let mut terminal = ratatui::init();
    let started = Instant::now();
    let mut monitoring = {
        let m = monitor.0.lock().unwrap();
        m.is_some()
    };
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
            )
        };
        terminal.draw(|f| {
            let rows = Layout::vertical([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(5),
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
            let gate = controls.gate_db.load(Ordering::Relaxed);
            let names = engine.stage_names();
            f.render_widget(
                Paragraph::new(format!(
                    "RTF  {} {:.2} · {} {:.2} · {} {:.2}\nchunks {} · late {} · gated {} · mode {}\npitch {:+.1} st ([ / ])  ·  denoise mix {}% (, / .)  ·  gate {} dB (- / =)",
                    names[0],
                    snapshot.2,
                    names[1],
                    snapshot.3,
                    names[2],
                    snapshot.4,
                    snapshot.5,
                    snapshot.6,
                    snapshot.9,
                    if snapshot.7 { "PASSTHROUGH" } else { "CONVERT" },
                    pitch,
                    mix,
                    gate,
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
                    KeyCode::Char('p') => {
                        let mut st = stats.lock().unwrap();
                        st.passthrough = !st.passthrough;
                    }
                    KeyCode::Char('l') if sink_ok => {
                        monitoring = monitor.toggle().unwrap_or(monitoring);
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

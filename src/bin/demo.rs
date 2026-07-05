//! MeanVC real-time TUI demo with a virtual microphone.
//!
//! Captures the default microphone, converts the voice with the official
//! MeanVC v1 checkpoints chunk by chunk (200 ms), and plays the result
//! into a PulseAudio/PipeWire null sink whose remapped monitor shows up as
//! a selectable **virtual microphone** (`meanvc_mic`) in other apps.
//!
//! ```sh
//! cargo run --release --features demo --bin meanvc-demo -- \
//!     --reference ckpt/test.wav --voice-print ckpt/voice_print_test.safetensors
//! ```
//!
//! Keys: `q` quit · `p` passthrough (bypass conversion for A/B) ·
//! `l` loopback monitor (hear the converted voice on the default output).
//!
//! Options: `--wav <file>` converts a wav file instead of the microphone
//! (paced in real time), `--headless` disables the TUI, `--out <file>`
//! additionally records the converted audio, `--no-sink` skips creating
//! the virtual device, `--monitor` starts with the loopback enabled,
//! `--duration <secs>` auto-stops (for testing).
//!
//! Approximations vs the offline path (tracked in #9): BNFs come from a
//! rolling 1.6 s analysis window instead of full incremental caches, and
//! the vocoder re-synthesizes each chunk with a 200 ms mel tail as
//! context.

use std::collections::VecDeque;
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

const SR: usize = 16_000;
const CHUNK_SAMPLES: usize = 3_200; // 200 ms = one CARD chunk (20 mel frames)
const HISTORY_SAMPLES: usize = 19_200; // 1.2 s rolling ASR window
const MEL_TAIL: usize = 20; // vocoder left context, in mel frames
const SINK: &str = "meanvc";
const VIRT_MIC: &str = "meanvc_mic";

#[derive(Default)]
struct Stats {
    in_rms: f32,
    out_rms: f32,
    rtf_asr: f32,
    rtf_vc: f32,
    rtf_voc: f32,
    chunks: u64,
    late: u64,
    passthrough: bool,
    status: String,
}

struct Args {
    reference: String,
    voice_print: String,
    wav: Option<String>,
    out: Option<String>,
    headless: bool,
    no_sink: bool,
    monitor: bool,
    duration: Option<f32>,
}

fn parse_args() -> Args {
    let mut a = Args {
        reference: "ckpt/test.wav".into(),
        voice_print: "ckpt/voice_print_test.safetensors".into(),
        wav: None,
        out: None,
        headless: false,
        no_sink: false,
        monitor: false,
        duration: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--reference" => a.reference = it.next().expect("--reference <wav>"),
            "--voice-print" => a.voice_print = it.next().expect("--voice-print <safetensors>"),
            "--wav" => a.wav = Some(it.next().expect("--wav <file>")),
            "--out" => a.out = Some(it.next().expect("--out <file>")),
            "--headless" => a.headless = true,
            "--no-sink" => a.no_sink = true,
            "--monitor" => a.monitor = true,
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
        let out = Command::new("pactl").arg("load-module").args(args).output()?;
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
        "sink_properties=device.description=MeanVC-Output",
    ])?;
    let mic = load(&[
        "module-remap-source",
        &format!("source_name={VIRT_MIC}"),
        &format!("master={SINK}.monitor"),
        "source_properties=device.description=MeanVC-Virtual-Mic",
    ])?;
    Ok(vec![sink, mic])
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

fn pulse_spec() -> Spec {
    Spec {
        format: Format::FLOAT32NE,
        channels: 1,
        rate: SR as u32,
    }
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

    eprintln!("loading models…");
    let model = MeanVc1::load(MeanVc1Config::default(), "ckpt/model_200ms.safetensors", &device)?;
    let asr = FastU2pp::load(
        FastU2ppConfig::official_meanvc1(),
        "ckpt/fastu2pp.safetensors",
        &device,
    )?;
    let vocos = Vocos::load(VocosConfig::official_meanvc1(), "ckpt/vocos.safetensors", &device)?;
    let reference = read_wav_16k(&args.reference)?;
    let prompt_mel = MelV1::new().compute(&reference, &device)?.unsqueeze(0)?;
    let vp = candle_core::safetensors::load(&args.voice_print, &device)?;
    let spks = vp
        .get("voice_print")
        .ok_or_else(|| anyhow::anyhow!("voice_print tensor missing in {}", args.voice_print))?
        .unsqueeze(0)?;

    let modules = if args.no_sink {
        Vec::new()
    } else {
        create_virtual_device()?
    };
    let sink_ok = !modules.is_empty();
    let monitor = Arc::new(Monitor(Mutex::new(None)));
    if args.monitor && sink_ok {
        monitor.toggle()?;
    }

    let stats = Arc::new(Mutex::new(Stats {
        status: if sink_ok {
            format!("virtual mic \"{VIRT_MIC}\" is live — select \"MeanVC-Virtual-Mic\" in your app")
        } else {
            "virtual sink disabled (--no-sink)".into()
        },
        ..Default::default()
    }));
    let run = Arc::new(AtomicBool::new(true));

    // --- input thread: microphone or paced wav file -> chunk channel
    let (tx_in, rx_in) = std::sync::mpsc::sync_channel::<Vec<f32>>(8);
    let run_in = run.clone();
    let wav_src = args.wav.clone();
    let input = std::thread::spawn(move || -> anyhow::Result<()> {
        match wav_src {
            Some(path) => {
                let samples = read_wav_16k(&path)?;
                let t0 = Instant::now();
                for (i, chunk) in samples.chunks(CHUNK_SAMPLES).enumerate() {
                    if !run_in.load(Ordering::Relaxed) {
                        break;
                    }
                    let mut c = chunk.to_vec();
                    c.resize(CHUNK_SAMPLES, 0.0);
                    // Pace to real time.
                    let due = Duration::from_millis(200 * i as u64);
                    if let Some(wait) = due.checked_sub(t0.elapsed()) {
                        std::thread::sleep(wait);
                    }
                    if tx_in.send(c).is_err() {
                        break;
                    }
                }
                Ok(())
            }
            None => {
                let s = Simple::new(
                    None,
                    "meanvc-demo",
                    Direction::Record,
                    None,
                    "capture",
                    &pulse_spec(),
                    None,
                    None,
                )
                .map_err(|e| anyhow::anyhow!("pulse record: {e}"))?;
                let mut buf = vec![0u8; CHUNK_SAMPLES * 4];
                while run_in.load(Ordering::Relaxed) {
                    s.read(&mut buf).map_err(|e| anyhow::anyhow!("read: {e}"))?;
                    let c: Vec<f32> = buf
                        .chunks_exact(4)
                        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    if tx_in.send(c).is_err() {
                        break;
                    }
                }
                Ok(())
            }
        }
    });

    // --- output thread: vocoding + playback (pipelined with the VC stage
    // so the two heaviest stages run concurrently).
    enum OutMsg {
        Mel(Tensor),
        Raw(Vec<f32>),
    }
    let (tx_out, rx_out) = std::sync::mpsc::sync_channel::<OutMsg>(8);
    let run_out = run.clone();
    let out_path = args.out.clone();
    let stats_out = stats.clone();
    let output = std::thread::spawn(move || -> anyhow::Result<()> {
        let play = if sink_ok {
            Some(
                Simple::new(
                    None,
                    "meanvc-demo",
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
        while run_out.load(Ordering::Relaxed) {
            let Ok(msg) = rx_out.recv_timeout(Duration::from_millis(300)) else {
                continue;
            };
            let chunk: Vec<f32> = match msg {
                OutMsg::Raw(c) => c,
                OutMsg::Mel(mel) => {
                    // Vocoding with a mel tail as left context.
                    let t0 = Instant::now();
                    let mel_win = match &mel_tail {
                        Some(tail) => Tensor::cat(&[tail, &mel], 1)?,
                        None => mel.clone(),
                    };
                    let mel01 = ((mel_win.squeeze(0)? + 1.0)? / 2.0)?;
                    let wav = vocos.synthesize(&mel01)?;
                    let emit = CHUNK_SAMPLES.min(wav.len());
                    let out: Vec<f32> = wav[wav.len() - emit..].to_vec();
                    mel_tail =
                        Some(mel.narrow(1, 20usize.saturating_sub(MEL_TAIL), MEL_TAIL.min(20))?);
                    let mut st = stats_out.lock().unwrap();
                    st.rtf_voc = t0.elapsed().as_secs_f32() / 0.2;
                    st.out_rms = rms(&out);
                    out
                }
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
    let vc = std::thread::spawn(move || -> anyhow::Result<()> {
        let fbank = KaldiFbank::new();
        let mut history: VecDeque<f32> = VecDeque::with_capacity(HISTORY_SAMPLES);
        let mut kv = KvCache::default();
        let mut prev_mel: Option<Tensor> = None;
        let mut q = 0usize;
        while run_vc.load(Ordering::Relaxed) {
            let Ok(chunk) = rx_in.recv_timeout(Duration::from_millis(300)) else {
                continue;
            };
            let passthrough = stats_vc.lock().unwrap().passthrough;
            let in_level = rms(&chunk);
            if passthrough {
                let mut st = stats_vc.lock().unwrap();
                st.in_rms = in_level;
                st.out_rms = in_level;
                st.chunks += 1;
                drop(st);
                let _ = tx_out.send(OutMsg::Raw(chunk));
                continue;
            }
            history.extend(chunk.iter().copied());
            while history.len() > HISTORY_SAMPLES {
                history.pop_front();
            }
            let hist: Vec<f32> = history.iter().copied().collect();
            if hist.len() < CHUNK_SAMPLES * 2 {
                continue; // need context before the first chunk
            }

            // BNFs for the newest 200 ms from the rolling window.
            let t0 = Instant::now();
            let fb = fbank.compute(&hist, &device)?.unsqueeze(0)?;
            let bn = asr.forward(&fb)?;
            let n_sub = bn.dim(1)?;
            let bn5 = bn.narrow(1, n_sub - 5, 5)?;
            let cond = interpolate_linear(&bn5, 4)?; // [1, 20, 256]
            let t_asr = t0.elapsed();

            // CARD chunk with streaming KV cache.
            let t0 = Instant::now();
            let timbre = model.timbre_cond(&cond, &prompt_mel, &spks)?;
            let noise = Tensor::randn(0f32, 1f32, (1, 20, 80), &device)?;
            let u = model.forward_stream(&noise, &timbre, &spks, prev_mel.as_ref(), q * 20, &mut kv)?;
            let mel = (noise - u)?;
            let t_vc = t0.elapsed();

            prev_mel = Some(mel.clone());
            q += 1;

            // The VC stage must stay under one chunk period; the vocoder
            // runs concurrently in the output thread.
            let total = t_asr + t_vc;
            {
                let mut st = stats_vc.lock().unwrap();
                st.in_rms = in_level;
                st.rtf_asr = t_asr.as_secs_f32() / 0.2;
                st.rtf_vc = t_vc.as_secs_f32() / 0.2;
                st.chunks += 1;
                if total > Duration::from_millis(200) {
                    st.late += 1;
                }
            }
            let _ = tx_out.send(OutMsg::Mel(mel));
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
            eprint!(
                "\rchunks {:4}  in {:.3} out {:.3}  RTF asr {:.2} vc {:.2} voc {:.2}  late {}   ",
                st.chunks, st.in_rms, st.out_rms, st.rtf_asr, st.rtf_vc, st.rtf_voc, st.late
            );
            std::io::stderr().flush().ok();
        }
        eprintln!();
    } else {
        run_tui(stats.clone(), run.clone(), monitor.clone(), sink_ok, args.duration)?;
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
    stats: Arc<Mutex<Stats>>,
    run: Arc<AtomicBool>,
    monitor: Arc<Monitor>,
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
            )
        };
        terminal.draw(|f| {
            let rows = Layout::vertical([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(4),
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
                    .block(Block::default().borders(Borders::ALL).title(" converted out "))
                    .gauge_style(Style::new().fg(Color::Cyan))
                    .percent(level(snapshot.1)),
                rows[1],
            );
            let total = snapshot.2 + snapshot.3 + snapshot.4;
            f.render_widget(
                Gauge::default()
                    .block(Block::default().borders(Borders::ALL).title(" total RTF (must stay < 1.0) "))
                    .gauge_style(Style::new().fg(if total < 1.0 { Color::Green } else { Color::Red }))
                    .percent(((total).min(1.5) * 66.6) as u16)
                    .label(format!("{total:.2}")),
                rows[2],
            );
            f.render_widget(
                Paragraph::new(format!(
                    "RTF  asr {:.2} · vc {:.2} · vocoder {:.2}\nchunks {} · late {} · mode {}",
                    snapshot.2,
                    snapshot.3,
                    snapshot.4,
                    snapshot.5,
                    snapshot.6,
                    if snapshot.7 { "PASSTHROUGH" } else { "CONVERT" },
                ))
                .block(Block::default().borders(Borders::ALL).title(" pipeline ")),
                rows[3],
            );
            f.render_widget(
                Paragraph::new(format!(
                    "{}\n[q] quit   [p] passthrough   [l] loopback monitor: {}",
                    snapshot.8,
                    if monitoring { "ON (hearing converted voice)" } else { "off" }
                ))
                    .block(Block::default().borders(Borders::ALL).title(" MeanVC virtual mic ")),
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
                    _ => {}
                }
            }
        }
    }
    ratatui::restore();
    run.store(false, Ordering::Relaxed);
    Ok(())
}

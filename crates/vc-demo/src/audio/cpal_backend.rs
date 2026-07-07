//! Portable backend: capture/playback via `cpal` (issues #52/#53/#54).
//!
//! One implementation covers WASAPI (Windows), CoreAudio (macOS) and —
//! behind the `cpal-backend` feature, groundwork for #55 — ALSA (Linux).
//! These platforms have no OS null sink, so the "virtual microphone" is a
//! **route**: the converted voice plays into the input end of a
//! user-installed loopback driver (VB-CABLE, VoiceMeeter, BlackHole) and
//! the app records the driver's output end. [`pick_route_device`] chooses
//! the route: `--output-device` wins, a known loopback device is
//! auto-detected, otherwise the default output plays with a setup hint
//! (see `docs/windows.md` / `docs/macos.md`).
//!
//! Devices rarely run at the engine rates (16 kHz capture / 48 kHz
//! playback) — WASAPI shared mode and CoreAudio default to the device mix
//! rate (44.1/48 kHz) — so both directions go through [`Resampler`].
//! `cpal` streams deliver audio in callbacks; a bounded ring buffer
//! adapts them to the pipeline's blocking chunk reads/writes (capture
//! drops the *oldest* samples on overrun so latency cannot grow without
//! bound; playback write blocks, letting the device pace the pipeline).
//! `cpal::Stream` is not `Send`, so streams are created on the thread
//! that uses them (see the module contract in [`super`]).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::{
    match_device, pick_route_device, AudioBackend, BackendOptions, CaptureStream, DeviceList,
    PlaybackStream, Resampler, Route,
};

/// cpal implementation of [`AudioBackend`].
pub struct CpalBackend {
    opts: BackendOptions,
    /// Route device name resolved by [`Self::create_virtual_mic`];
    /// `None` = default output.
    route: Mutex<Option<String>>,
    /// Self-monitor request, picked up by the playback stream on its next
    /// write (streams are thread-affine, so the toggle cannot create the
    /// monitor stream itself).
    monitor_on: Arc<AtomicBool>,
}

impl CpalBackend {
    pub fn new(opts: BackendOptions) -> Self {
        Self {
            opts,
            route: Mutex::new(None),
            monitor_on: Arc::new(AtomicBool::new(false)),
        }
    }

    fn output_device_by_name(name: Option<&str>) -> Result<cpal::Device> {
        let host = cpal::default_host();
        match name {
            None => host
                .default_output_device()
                .ok_or_else(|| anyhow!("no default output device")),
            Some(n) => host
                .output_devices()
                .context("cannot enumerate output devices")?
                .find(|d| device_name(d).as_deref() == Some(n))
                .ok_or_else(|| anyhow!("output device {n:?} disappeared")),
        }
    }
}

/// Human-readable device name (cpal 0.18 moved it into
/// [`cpal::DeviceDescription`]).
fn device_name(d: &cpal::Device) -> Option<String> {
    d.description().ok().map(|desc| desc.name().to_string())
}

impl AudioBackend for CpalBackend {
    fn name(&self) -> &'static str {
        "cpal"
    }

    fn list_devices(&self) -> Result<DeviceList> {
        let host = cpal::default_host();
        let names = |it: Result<_, cpal::Error>| -> Vec<String> {
            it.map(|ds: Vec<cpal::Device>| ds.iter().filter_map(device_name).collect())
                .unwrap_or_default()
        };
        Ok(DeviceList {
            inputs: names(host.input_devices().map(Iterator::collect)),
            outputs: names(host.output_devices().map(Iterator::collect)),
        })
    }

    /// Resolves the playback route. Nothing is created OS-side — the
    /// loopback driver *is* the virtual device — so this only picks the
    /// device and tells the user which end to select as their mic.
    fn create_virtual_mic(&self) -> Result<String> {
        let outputs = self.list_devices()?.outputs;
        let status = match pick_route_device(&outputs, self.opts.output_device.as_deref())? {
            Route::Requested(d) | Route::AutoDetected(d) => {
                let msg = format!(
                    "routing the converted voice to \"{d}\" — select the loopback's \
                     capture end (e.g. \"CABLE Output\" / \"BlackHole 2ch\") as the mic in your app"
                );
                *self.route.lock().unwrap() = Some(d);
                msg
            }
            Route::DefaultOutput => {
                *self.route.lock().unwrap() = None;
                "no loopback device found — playing on the default output. Install \
                 VB-CABLE (Windows) or BlackHole (macOS) for a virtual mic; see docs/"
                    .to_string()
            }
        };
        Ok(status)
    }

    fn destroy_virtual_mic(&self) {
        // Route only; the loopback driver belongs to the user.
        *self.route.lock().unwrap() = None;
    }

    fn open_capture(
        &self,
        device: Option<&str>,
        rate: u32,
        chunk_samples: usize,
    ) -> Result<Box<dyn CaptureStream>> {
        let host = cpal::default_host();
        let dev = match device {
            None => host
                .default_input_device()
                .ok_or_else(|| anyhow!("no default input device"))?,
            Some(req) => {
                let devices: Vec<cpal::Device> = host
                    .input_devices()
                    .context("cannot enumerate input devices")?
                    .collect();
                let names: Vec<String> = devices.iter().filter_map(device_name).collect();
                let picked = match_device(&names, req)?.to_string();
                devices
                    .into_iter()
                    .find(|d| device_name(d).as_deref() == Some(&picked))
                    .ok_or_else(|| anyhow!("input device {picked:?} disappeared"))?
            }
        };
        let config = dev
            .default_input_config()
            .context("no default input config")?;
        let dev_rate = config.sample_rate();
        let channels = config.channels() as usize;
        // 4 s of device-rate slack mirrors the pulse BufferAttr headroom
        // (issue #42): a transient pipeline stall must not drop samples.
        let ring = Arc::new(Ring::new(dev_rate as usize * 4));
        let stream = build_input_stream(&dev, &config, channels, ring.clone())?;
        stream.play().context("cannot start the capture stream")?;
        eprintln!(
            "capture: {:?} @ {dev_rate} Hz x{channels} -> {rate} Hz mono",
            device_name(&dev).unwrap_or_default()
        );
        Ok(Box::new(CpalCapture {
            _stream: stream,
            ring,
            resampler: Resampler::new(dev_rate, rate),
            staging: Vec::with_capacity(chunk_samples),
            pending: Vec::with_capacity(2 * chunk_samples),
        }))
    }

    fn open_playback(&self, rate: u32) -> Result<Box<dyn PlaybackStream>> {
        let route = self.route.lock().unwrap().clone();
        let dev = Self::output_device_by_name(route.as_deref())?;
        let (stream, ring, dev_rate) = build_output_on(&dev)?;
        eprintln!(
            "playback: {rate} Hz mono -> {:?} @ {dev_rate} Hz",
            device_name(&dev).unwrap_or_default()
        );
        Ok(Box::new(CpalPlayback {
            _stream: stream,
            ring,
            resampler: Resampler::new(rate, dev_rate),
            scratch: Vec::new(),
            rate,
            route_is_default: route.is_none(),
            monitor_on: self.monitor_on.clone(),
            monitor: None,
        }))
    }

    fn toggle_monitor(&self) -> Result<bool> {
        let on = !self.monitor_on.load(Ordering::Relaxed);
        self.monitor_on.store(on, Ordering::Relaxed);
        Ok(on)
    }

    fn monitor_off(&self) {
        self.monitor_on.store(false, Ordering::Relaxed);
    }
}

/// Bounded mono f32 ring between a cpal callback and a blocking caller.
struct Ring {
    q: Mutex<VecDeque<f32>>,
    cv: Condvar,
    cap: usize,
}

impl Ring {
    fn new(cap: usize) -> Self {
        Self {
            q: Mutex::new(VecDeque::with_capacity(cap)),
            cv: Condvar::new(),
            cap,
        }
    }

    /// Capture side: append, dropping the *oldest* samples on overrun so a
    /// stalled pipeline resumes near real time instead of seconds behind —
    /// the same policy a full pulse server buffer applies to a slow reader,
    /// so #42-style input-splice artifacts present identically on every
    /// backend (the 4 s cap makes an overrun just as unlikely here).
    fn push_capture(&self, samples: impl Iterator<Item = f32>) {
        let mut q = self.q.lock().unwrap();
        q.extend(samples);
        let cap = self.cap;
        if q.len() > cap {
            let excess = q.len() - cap;
            q.drain(..excess);
        }
        drop(q);
        self.cv.notify_one();
    }

    /// Blocking pop of at most `max` samples (at least one) into `out`.
    fn pop_blocking(&self, out: &mut Vec<f32>, max: usize) {
        let mut q = self.q.lock().unwrap();
        while q.is_empty() {
            q = self.cv.wait(q).unwrap();
        }
        let n = q.len().min(max);
        out.extend(q.drain(..n));
    }

    /// Playback side: blocking append (the device callback drains and
    /// notifies, so the device paces the pipeline like `pa_simple_write`).
    fn push_blocking(&self, samples: &[f32]) {
        let mut i = 0;
        while i < samples.len() {
            let mut q = self.q.lock().unwrap();
            while q.len() >= self.cap {
                q = self.cv.wait(q).unwrap();
            }
            let room = self.cap - q.len();
            let n = room.min(samples.len() - i);
            q.extend(&samples[i..i + n]);
            i += n;
        }
    }

    /// Non-blocking append that drops on overrun (self-monitor: a full
    /// monitor must never stall the virtual-mic route).
    fn push_lossy(&self, samples: &[f32]) {
        let mut q = self.q.lock().unwrap();
        let room = self.cap.saturating_sub(q.len());
        q.extend(&samples[..samples.len().min(room)]);
    }

    /// Device callback: fill `data` (interleaved, `channels` wide) from
    /// the queue, padding with silence on underrun.
    fn fill_frames(&self, data: &mut [f32], channels: usize) {
        let mut q = self.q.lock().unwrap();
        for frame in data.chunks_mut(channels) {
            let s = q.pop_front().unwrap_or(0.0);
            frame.fill(s);
        }
        drop(q);
        self.cv.notify_one();
    }
}

/// Builds a mono-downmixing input stream in the device's native sample
/// format, feeding `ring` at the device rate.
fn build_input_stream(
    dev: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    channels: usize,
    ring: Arc<Ring>,
) -> Result<cpal::Stream> {
    let err_cb = |e: cpal::Error| eprintln!("capture stream error: {e}");
    let sc: cpal::StreamConfig = config.config();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            sc,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                ring.push_capture(
                    data.chunks(channels)
                        .map(|f| f.iter().sum::<f32>() / channels as f32),
                );
            },
            err_cb,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream(
            sc,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                ring.push_capture(data.chunks(channels).map(|f| {
                    f.iter().map(|&s| s as f32 / 32_768.0).sum::<f32>() / channels as f32
                }));
            },
            err_cb,
            None,
        )?,
        cpal::SampleFormat::U16 => dev.build_input_stream(
            sc,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                ring.push_capture(data.chunks(channels).map(|f| {
                    f.iter().map(|&s| s as f32 / 32_768.0 - 1.0).sum::<f32>() / channels as f32
                }));
            },
            err_cb,
            None,
        )?,
        other => anyhow::bail!("unsupported capture sample format {other:?}"),
    };
    Ok(stream)
}

/// Opens an f32 output stream on `dev` at its default config; returns the
/// stream, its feeding ring (1 s cap — enough slack to absorb jitter,
/// small enough to backpressure the pipeline) and the device rate.
fn build_output_on(dev: &cpal::Device) -> Result<(cpal::Stream, Arc<Ring>, u32)> {
    let config = dev
        .default_output_config()
        .context("no default output config")?;
    let dev_rate = config.sample_rate();
    let channels = config.channels() as usize;
    let ring = Arc::new(Ring::new(dev_rate as usize));
    let cb_ring = ring.clone();
    let err_cb = |e: cpal::Error| eprintln!("playback stream error: {e}");
    let sc: cpal::StreamConfig = config.config();
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => dev.build_output_stream(
            sc,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                cb_ring.fill_frames(data, channels);
            },
            err_cb,
            None,
        )?,
        cpal::SampleFormat::I16 => {
            let mut mono: Vec<f32> = Vec::new();
            dev.build_output_stream(
                sc,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    mono.resize(data.len(), 0.0);
                    cb_ring.fill_frames(&mut mono, channels);
                    for (d, s) in data.iter_mut().zip(&mono) {
                        *d = (s.clamp(-1.0, 1.0) * 32_767.0) as i16;
                    }
                },
                err_cb,
                None,
            )?
        }
        other => anyhow::bail!("unsupported playback sample format {other:?}"),
    };
    stream.play().context("cannot start the playback stream")?;
    Ok((stream, ring, dev_rate))
}

struct CpalCapture {
    _stream: cpal::Stream,
    ring: Arc<Ring>,
    resampler: Resampler,
    /// Device-rate staging pulled from the ring (reused, issue #6).
    staging: Vec<f32>,
    /// Engine-rate samples not yet handed to the caller.
    pending: Vec<f32>,
}

impl CaptureStream for CpalCapture {
    fn read(&mut self, buf: &mut [f32]) -> Result<()> {
        while self.pending.len() < buf.len() {
            self.staging.clear();
            self.ring.pop_blocking(&mut self.staging, 1024);
            let staged = std::mem::take(&mut self.staging);
            self.resampler.process(&staged, &mut self.pending);
            self.staging = staged;
        }
        buf.copy_from_slice(&self.pending[..buf.len()]);
        self.pending.drain(..buf.len());
        Ok(())
    }
}

struct CpalPlayback {
    _stream: cpal::Stream,
    ring: Arc<Ring>,
    resampler: Resampler,
    /// Device-rate scratch (reused, issue #6).
    scratch: Vec<f32>,
    /// Pipeline rate, for the lazily created monitor stream.
    rate: u32,
    /// When the route already is the default output, the monitor would
    /// duplicate the same audio on the same device — skip it.
    route_is_default: bool,
    monitor_on: Arc<AtomicBool>,
    /// Lazily created on this thread when the TUI toggles the flag
    /// (`cpal::Stream` is not `Send`, so the toggle cannot do it).
    monitor: Option<Monitor>,
}

struct Monitor {
    _stream: cpal::Stream,
    ring: Arc<Ring>,
    resampler: Resampler,
    scratch: Vec<f32>,
}

impl PlaybackStream for CpalPlayback {
    fn write(&mut self, samples: &[f32]) -> Result<()> {
        // Reconcile the monitor with the TUI flag before pushing audio.
        let want = self.monitor_on.load(Ordering::Relaxed) && !self.route_is_default;
        if want && self.monitor.is_none() {
            match Self::open_monitor(self.rate) {
                Ok(m) => self.monitor = Some(m),
                Err(e) => eprintln!("monitor unavailable: {e}"),
            }
        } else if !want {
            self.monitor = None;
        }
        if let Some(m) = &mut self.monitor {
            m.scratch.clear();
            m.resampler.process(samples, &mut m.scratch);
            m.ring.push_lossy(&m.scratch);
        }
        self.scratch.clear();
        self.resampler.process(samples, &mut self.scratch);
        self.ring.push_blocking(&self.scratch);
        Ok(())
    }
}

impl CpalPlayback {
    fn open_monitor(rate: u32) -> Result<Monitor> {
        let dev = CpalBackend::output_device_by_name(None)?;
        let (stream, ring, dev_rate) = build_output_on(&dev)?;
        Ok(Monitor {
            _stream: stream,
            ring,
            resampler: Resampler::new(rate, dev_rate),
            scratch: Vec::new(),
        })
    }
}

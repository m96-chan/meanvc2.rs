//! Audio backend abstraction (issue #52 — #51 step 0).
//!
//! The demo's platform surface is three things: blocking mono capture at
//! the engine rate, blocking playback of the converted signal, and the
//! **virtual microphone** route those samples travel through. Each OS
//! implements them behind [`AudioBackend`]:
//!
//! - **Linux** — [`pulse::PulseBackend`]: PulseAudio/PipeWire null sink +
//!   remapped source via `pactl` (the original implementation; the #39
//!   teardown / stale-recovery semantics are preserved unchanged).
//! - **Windows / macOS** — [`cpal_backend::CpalBackend`]: capture and
//!   playback via WASAPI / CoreAudio (`cpal`); there is no OS null sink,
//!   so the "virtual mic" is a **route** to a user-installed loopback
//!   device (VB-CABLE, BlackHole, VoiceMeeter), auto-detected from the
//!   output-device list or forced with `--output-device` (#53, #54).
//!
//! Threading contract: the backend handle is `Send + Sync` and shared
//! across the pipeline threads, but streams are opened **on the thread
//! that uses them** and never move (`cpal::Stream` is not `Send` on every
//! platform; the pulse `Simple` streams were already per-thread).

use std::sync::Arc;

use anyhow::Result;

#[cfg(any(not(target_os = "linux"), feature = "cpal-backend"))]
pub mod cpal_backend;
#[cfg(target_os = "linux")]
pub mod pulse;

/// Blocking mono capture stream at the rate given to
/// [`AudioBackend::open_capture`].
pub trait CaptureStream {
    /// Fills `buf` with the next `buf.len()` samples, blocking until they
    /// are available.
    fn read(&mut self, buf: &mut [f32]) -> Result<()>;
}

/// Blocking mono playback stream at the rate given to
/// [`AudioBackend::open_playback`].
pub trait PlaybackStream {
    /// Writes `samples`, blocking on the device's own pacing.
    fn write(&mut self, samples: &[f32]) -> Result<()>;
}

/// Input/output device names visible to the backend.
#[derive(Debug, Default)]
pub struct DeviceList {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// One per-platform audio implementation. Shared across the pipeline
/// threads; see the module docs for the stream-affinity contract.
pub trait AudioBackend: Send + Sync {
    /// Short backend name for logs/TUI (e.g. `"pulse"`, `"cpal"`).
    fn name(&self) -> &'static str;

    /// Enumerates capture/playback device names (for `--list-devices`
    /// style probing and route selection diagnostics).
    fn list_devices(&self) -> Result<DeviceList>;

    /// Best-effort recovery of devices leaked by a previous killed run
    /// (issue #39). A no-op on backends that create no OS objects.
    fn recover_stale(&self) {}

    /// Sets up the virtual-microphone route and returns a user-facing
    /// status line ("select X as your mic in the app").
    fn create_virtual_mic(&self) -> Result<String>;

    /// Tears down whatever [`Self::create_virtual_mic`] created.
    fn destroy_virtual_mic(&self);

    /// Opens blocking mono capture at `rate`. `device` is a backend
    /// device name (Pulse source name / cpal name substring); `None` is
    /// the default input. `chunk_samples` sizes the transport buffering.
    fn open_capture(
        &self,
        device: Option<&str>,
        rate: u32,
        chunk_samples: usize,
    ) -> Result<Box<dyn CaptureStream>>;

    /// Opens blocking mono playback at `rate` into the virtual-mic route.
    fn open_playback(&self, rate: u32) -> Result<Box<dyn PlaybackStream>>;

    /// Toggles the self-monitor (hear the converted voice on the default
    /// output); returns the new state.
    fn toggle_monitor(&self) -> Result<bool>;

    /// Forces the self-monitor off (shutdown path).
    fn monitor_off(&self);

    /// Inserts an OS-level noise-suppressed source in front of the
    /// microphone (`--denoise`); returns the source name to capture from,
    /// or `None` when the platform has no such facility (the in-process
    /// RNNoise knob still works there).
    fn create_denoised_source(&self, _master: Option<&str>) -> Result<Option<String>> {
        Ok(None)
    }
}

/// Options shared by every backend constructor.
#[derive(Debug, Default, Clone)]
pub struct BackendOptions {
    /// Non-Linux: name (substring) of the playback device the converted
    /// voice is routed to — the loopback driver's input end. Auto-detected
    /// when omitted. Ignored on Linux (the null sink *is* the route).
    pub output_device: Option<String>,
}

/// The platform's default backend: Pulse on Linux, cpal elsewhere.
#[cfg(target_os = "linux")]
pub fn default_backend(opts: BackendOptions) -> Arc<dyn AudioBackend> {
    Arc::new(pulse::PulseBackend::new(opts))
}

/// The platform's default backend: Pulse on Linux, cpal elsewhere.
#[cfg(not(target_os = "linux"))]
pub fn default_backend(opts: BackendOptions) -> Arc<dyn AudioBackend> {
    Arc::new(cpal_backend::CpalBackend::new(opts))
}

/// Output devices that mark the input end of a user-installed loopback
/// driver, i.e. a virtual-mic route (lower-case markers, substring match):
/// VB-CABLE / VoiceMeeter on Windows, BlackHole on macOS.
pub const LOOPBACK_MARKERS: &[&str] =
    &["cable input", "vb-audio", "voicemeeter input", "blackhole"];

/// How the virtual-mic route was chosen on backends without an OS null
/// sink (see [`pick_route_device`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// The device the user asked for with `--output-device`.
    Requested(String),
    /// A known loopback driver found in the output-device list.
    AutoDetected(String),
    /// Nothing suitable found: play on the default output and tell the
    /// user how to install a loopback driver.
    DefaultOutput,
}

/// Picks the playback device the converted voice is routed to, from the
/// available output-device `names`. An explicit `requested` name matches
/// case-insensitively as a substring and it is an error when it matches
/// nothing (a typo must not silently fall back to the speakers); with no
/// request, a known loopback driver ([`LOOPBACK_MARKERS`]) is
/// auto-detected, else the default output is used.
pub fn pick_route_device(names: &[String], requested: Option<&str>) -> Result<Route> {
    if let Some(req) = requested {
        return Ok(Route::Requested(match_device(names, req)?.to_string()));
    }
    for marker in LOOPBACK_MARKERS {
        if let Some(d) = names.iter().find(|d| d.to_lowercase().contains(marker)) {
            return Ok(Route::AutoDetected(d.clone()));
        }
    }
    Ok(Route::DefaultOutput)
}

/// Case-insensitive substring device match for `--input-device` on
/// name-list backends: `None` selects the default device, a request that
/// matches nothing is an error listing what exists.
pub fn match_device<'a>(names: &'a [String], requested: &str) -> Result<&'a str> {
    let lc = requested.to_lowercase();
    names
        .iter()
        .find(|d| d.to_lowercase().contains(&lc))
        .map(String::as_str)
        .ok_or_else(|| anyhow::anyhow!("device {requested:?} matches none of: {names:?}"))
}

/// Streaming windowed-sinc resampler between two fixed rates.
///
/// The desktop backends capture/play at whatever rate the device runs
/// (44.1/48 kHz on CoreAudio/WASAPI) while the engines consume 16 kHz and
/// emit 48 kHz, so the cpal transport resamples in both directions. A
/// 32-tap Hann-windowed sinc with the cutoff at 90 % of the narrower
/// Nyquist keeps aliasing out of the ASR band; state carries across
/// [`Resampler::process`] calls so chunk boundaries are seamless.
pub struct Resampler {
    from: u32,
    to: u32,
    /// Input samples per output sample.
    step: f64,
    /// Anti-alias cutoff relative to the *input* Nyquist.
    cutoff: f64,
    /// Fractional read position into `hist`, in input samples.
    pos: f64,
    /// Carried input, pre-padded with half a window of silence so the
    /// first output samples have full left context.
    hist: Vec<f32>,
}

impl Resampler {
    /// Number of sinc taps (window support, in input samples).
    pub const TAPS: usize = 32;

    pub fn new(from: u32, to: u32) -> Self {
        let half = Self::TAPS as f64 / 2.0;
        Self {
            from,
            to,
            step: from as f64 / to as f64,
            cutoff: 0.9 * (to as f64 / from as f64).min(1.0),
            pos: half,
            hist: vec![0.0; Self::TAPS / 2],
        }
    }

    /// True when input passes through untouched (`from == to`).
    pub fn is_identity(&self) -> bool {
        self.from == self.to
    }

    /// Resamples `input`, appending to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.is_identity() {
            out.extend_from_slice(input);
            return;
        }
        self.hist.extend_from_slice(input);
        let half = Self::TAPS as f64 / 2.0;
        // Emit while the sinc support around `pos` is fully buffered.
        while self.pos + half < self.hist.len() as f64 {
            let lo = (self.pos - half).ceil() as usize;
            let hi = (self.pos + half).floor() as usize;
            let mut acc = 0.0f64;
            for (i, &s) in self.hist[lo..=hi].iter().enumerate() {
                let t = (lo + i) as f64 - self.pos;
                // Hann-windowed sinc, low-passed at `cutoff`.
                let sinc = if t == 0.0 {
                    self.cutoff
                } else {
                    (std::f64::consts::PI * self.cutoff * t).sin() / (std::f64::consts::PI * t)
                };
                let w = 0.5 + 0.5 * (std::f64::consts::PI * t / half).cos();
                acc += s as f64 * sinc * w;
            }
            out.push(acc as f32);
            self.pos += self.step;
        }
        // Drop input the window can no longer reach.
        let done = (self.pos - half).floor().max(0.0) as usize;
        if done > 0 {
            self.hist.drain(..done);
            self.pos -= done as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // --- virtual-mic route selection (#53 VB-CABLE / #54 BlackHole) ---

    #[test]
    fn requested_output_device_matches_case_insensitive_substring() {
        let out = names(&["Speakers (Realtek)", "CABLE Input (VB-Audio Virtual Cable)"]);
        let r = pick_route_device(&out, Some("cable input")).unwrap();
        assert_eq!(
            r,
            Route::Requested("CABLE Input (VB-Audio Virtual Cable)".into())
        );
    }

    #[test]
    fn requested_output_device_that_matches_nothing_is_an_error() {
        let out = names(&["Speakers (Realtek)"]);
        let err = pick_route_device(&out, Some("blackhole")).unwrap_err();
        // A typo must not silently fall back to the speakers.
        assert!(err.to_string().contains("blackhole"), "{err}");
        assert!(err.to_string().contains("Speakers"), "{err}");
    }

    #[test]
    fn vb_cable_is_auto_detected_without_a_request() {
        let out = names(&["Speakers (Realtek)", "CABLE Input (VB-Audio Virtual Cable)"]);
        let r = pick_route_device(&out, None).unwrap();
        assert_eq!(
            r,
            Route::AutoDetected("CABLE Input (VB-Audio Virtual Cable)".into())
        );
    }

    #[test]
    fn blackhole_is_auto_detected_without_a_request() {
        let out = names(&["MacBook Pro Speakers", "BlackHole 2ch"]);
        let r = pick_route_device(&out, None).unwrap();
        assert_eq!(r, Route::AutoDetected("BlackHole 2ch".into()));
    }

    #[test]
    fn no_loopback_device_falls_back_to_default_output() {
        let out = names(&["MacBook Pro Speakers"]);
        assert_eq!(pick_route_device(&out, None).unwrap(), Route::DefaultOutput);
    }

    #[test]
    fn input_device_match_is_substring_and_errors_on_miss() {
        let ins = names(&["MacBook Pro Microphone", "USB Audio CODEC"]);
        assert_eq!(match_device(&ins, "usb").unwrap(), "USB Audio CODEC");
        let err = match_device(&ins, "focusrite").unwrap_err();
        assert!(err.to_string().contains("MacBook Pro Microphone"), "{err}");
    }

    // --- resampler (device rate <-> engine rates) ---

    /// Frequency estimate via zero crossings, in Hz.
    fn zero_cross_hz(x: &[f32], rate: f32) -> f32 {
        let n = x.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        n as f32 / (x.len() as f32 / rate)
    }

    fn tone(freq: f32, rate: f32, secs: f32) -> Vec<f32> {
        (0..(rate * secs) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / rate).sin())
            .collect()
    }

    #[test]
    fn identity_rate_passes_samples_through_bit_exact() {
        let x = tone(440.0, 16_000.0, 0.1);
        let mut r = Resampler::new(16_000, 16_000);
        let mut out = Vec::new();
        r.process(&x, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn downsample_48k_to_16k_preserves_a_1khz_tone() {
        let x = tone(1_000.0, 48_000.0, 0.5);
        let mut r = Resampler::new(48_000, 16_000);
        let mut out = Vec::new();
        // Chunked like the live capture path: state must carry across calls.
        for c in x.chunks(480) {
            r.process(c, &mut out);
        }
        let expected = x.len() / 3;
        assert!(
            (out.len() as i64 - expected as i64).unsigned_abs() < 64,
            "length {} vs {expected}",
            out.len()
        );
        let hz = zero_cross_hz(&out[800..], 16_000.0);
        assert!((hz - 1_000.0).abs() < 20.0, "tone shifted to {hz} Hz");
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!((rms - 0.707).abs() < 0.07, "amplitude changed: rms {rms}");
    }

    #[test]
    fn downsample_rejects_frequencies_above_the_target_nyquist() {
        // 10 kHz is inaudible garbage after a 16 kHz resample (Nyquist
        // 8 kHz); without the low-pass it aliases into the speech band.
        let x = tone(10_000.0, 48_000.0, 0.5);
        let mut r = Resampler::new(48_000, 16_000);
        let mut out = Vec::new();
        r.process(&x, &mut out);
        let rms = (out[800..].iter().map(|s| s * s).sum::<f32>() / (out.len() - 800) as f32).sqrt();
        assert!(rms < 0.05, "alias leaked through: rms {rms}");
    }

    #[test]
    fn upsample_16k_to_48k_preserves_a_1khz_tone() {
        let x = tone(1_000.0, 16_000.0, 0.5);
        let mut r = Resampler::new(16_000, 48_000);
        let mut out = Vec::new();
        for c in x.chunks(160) {
            r.process(c, &mut out);
        }
        let hz = zero_cross_hz(&out[2400..], 48_000.0);
        assert!((hz - 1_000.0).abs() < 20.0, "tone shifted to {hz} Hz");
    }

    #[test]
    fn fractional_ratio_44100_to_16000_preserves_pitch() {
        let x = tone(1_000.0, 44_100.0, 0.5);
        let mut r = Resampler::new(44_100, 16_000);
        let mut out = Vec::new();
        for c in x.chunks(441) {
            r.process(c, &mut out);
        }
        let hz = zero_cross_hz(&out[800..], 16_000.0);
        assert!((hz - 1_000.0).abs() < 20.0, "tone shifted to {hz} Hz");
    }
}

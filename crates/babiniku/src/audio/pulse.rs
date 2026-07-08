//! Linux backend: PulseAudio/PipeWire via `libpulse-simple` + `pactl`.
//!
//! The virtual microphone is a null sink (`babiniku`) whose monitor is
//! remapped into a selectable source (`babiniku_mic`); the converted
//! voice plays into the sink and every app sees the source. Teardown and
//! stale-device recovery follow issue #39: SIGINT/SIGTERM unload every
//! module this process created, and leftovers of a killed run are
//! unloaded at the next startup before fresh devices are created.
//!
//! This is the original `demo.rs` implementation moved behind
//! [`AudioBackend`] (issue #52) — behavior unchanged.

use std::process::Command;
use std::sync::Mutex;

use anyhow::Result;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;

use super::{AudioBackend, BackendOptions, CaptureStream, DeviceList, PlaybackStream};

pub const SINK: &str = "babiniku";
pub const VIRT_MIC: &str = "babiniku_mic";
pub const DENOISED_SRC: &str = "babiniku_denoised";

/// PulseAudio/PipeWire implementation of [`AudioBackend`].
pub struct PulseBackend {
    /// pactl module ids owned by this process, unloaded in reverse on
    /// teardown (null sink, remap source, echo-cancel denoiser).
    modules: Mutex<Vec<String>>,
    /// Loopback-monitor module id while the self-monitor is on.
    monitor: Mutex<Option<String>>,
}

impl PulseBackend {
    pub fn new(opts: BackendOptions) -> Self {
        if let Some(dev) = &opts.output_device {
            eprintln!(
                "--output-device {dev:?} is ignored on Linux: the converted voice \
                 always plays into the \"{SINK}\" null sink (the virtual mic)"
            );
        }
        Self {
            modules: Mutex::new(Vec::new()),
            monitor: Mutex::new(None),
        }
    }

    fn load_module(args: &[&str]) -> Result<String> {
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
    }
}

/// Capture spec at the engine rate.
fn spec(rate: u32) -> Spec {
    Spec {
        format: Format::FLOAT32NE,
        channels: 1,
        rate,
    }
}

impl AudioBackend for PulseBackend {
    fn name(&self) -> &'static str {
        "pulse"
    }

    fn list_devices(&self) -> Result<DeviceList> {
        let names = |kind: &str| -> Result<Vec<String>> {
            let out = Command::new("pactl")
                .args(["list", "short", kind])
                .output()?;
            anyhow::ensure!(
                out.status.success(),
                "pactl list {kind} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            Ok(String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter_map(|l| l.split('\t').nth(1).map(str::to_string))
                .collect())
        };
        Ok(DeviceList {
            inputs: names("sources")?,
            outputs: names("sinks")?,
        })
    }

    /// Belt-and-braces startup recovery: unloads the stale modules found
    /// by [`stale_babiniku_modules`] before fresh devices are created, and
    /// logs what was recovered. Dependents (remap source, loopback) were
    /// loaded after the null sink, so unloading in reverse listing order
    /// tears them down first; a module that already disappeared is not an
    /// error.
    fn recover_stale(&self) {
        let out = Command::new("pactl")
            .args(["list", "modules", "short"])
            .output();
        let listing = match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).into_owned(),
            // No pulse daemon — device creation will report the real error.
            _ => return,
        };
        for (id, name) in stale_babiniku_modules(&listing).iter().rev() {
            let ok = Command::new("pactl")
                .args(["unload-module", id])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            eprintln!(
                "recovered stale {name} (module {id}) left by a previous run{}",
                if ok { "" } else { " — unload failed" }
            );
        }
    }

    /// Creates the null sink + remapped virtual microphone.
    fn create_virtual_mic(&self) -> Result<String> {
        let sink = Self::load_module(&[
            "module-null-sink",
            &format!("sink_name={SINK}"),
            "sink_properties=device.description=Babiniku-Output",
        ])?;
        self.modules.lock().unwrap().push(sink);
        let mic = Self::load_module(&[
            "module-remap-source",
            &format!("source_name={VIRT_MIC}"),
            &format!("master={SINK}.monitor"),
            "source_properties=device.description=Babiniku-Virtual-Mic",
        ])?;
        self.modules.lock().unwrap().push(mic);
        Ok(format!(
            "virtual mic \"{VIRT_MIC}\" is live — select \"Babiniku-Virtual-Mic\" in your app"
        ))
    }

    fn destroy_virtual_mic(&self) {
        let mut modules = self.modules.lock().unwrap();
        for m in modules.iter().rev() {
            let _ = Command::new("pactl").args(["unload-module", m]).status();
        }
        modules.clear();
    }

    fn open_capture(
        &self,
        device: Option<&str>,
        rate: u32,
        chunk_samples: usize,
    ) -> Result<Box<dyn CaptureStream>> {
        let s = Simple::new(
            None,
            "babiniku",
            Direction::Record,
            device,
            "capture",
            &spec(rate),
            None,
            // Generous capture buffering: a transient pipeline stall (GPU
            // hiccup, scheduler burp) back-pressures the input thread;
            // with the default fragsize the server then drops mic samples,
            // splicing a discontinuity into the INPUT that converts into
            // an audible tick (issue #42 third field recording: mid-level
            // clicks with 0.27 sample steps, no clipping, live only).
            // 2 s of slack absorbs any realistic stall.
            Some(&libpulse_binding::def::BufferAttr {
                maxlength: rate * 4 * 2, // 2 s of f32 mono
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: (chunk_samples * 4) as u32,
            }),
        )
        .map_err(|e| anyhow::anyhow!("pulse record: {e}"))?;
        Ok(Box::new(PulseCapture {
            s,
            bytes: Vec::new(),
        }))
    }

    fn open_playback(&self, rate: u32) -> Result<Box<dyn PlaybackStream>> {
        let s = Simple::new(
            None,
            "babiniku",
            Direction::Playback,
            Some(SINK),
            "converted",
            &spec(rate),
            None,
            None,
        )
        .map_err(|e| anyhow::anyhow!("pulse playback: {e}"))?;
        Ok(Box::new(PulsePlayback {
            s,
            bytes: Vec::new(),
        }))
    }

    /// Loopback monitor: routes the sink monitor to the default output so
    /// the user hears the converted voice.
    fn toggle_monitor(&self) -> Result<bool> {
        let mut slot = self.monitor.lock().unwrap();
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

    fn monitor_off(&self) {
        let mut slot = self.monitor.lock().unwrap();
        if let Some(id) = slot.take() {
            let _ = Command::new("pactl").args(["unload-module", &id]).status();
        }
    }

    /// PipeWire/Pulse WebRTC noise suppression in front of the microphone:
    /// creates a cleaned source the input thread records from.
    fn create_denoised_source(&self, master: Option<&str>) -> Result<Option<String>> {
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
        self.modules
            .lock()
            .unwrap()
            .push(String::from_utf8_lossy(&out.stdout).trim().to_string());
        Ok(Some(DENOISED_SRC.to_string()))
    }
}

struct PulseCapture {
    s: Simple,
    /// Reusable transfer buffer (no per-chunk allocation, issue #6).
    bytes: Vec<u8>,
}

impl CaptureStream for PulseCapture {
    fn read(&mut self, buf: &mut [f32]) -> Result<()> {
        self.bytes.resize(buf.len() * 4, 0);
        self.s
            .read(&mut self.bytes)
            .map_err(|e| anyhow::anyhow!("read: {e}"))?;
        for (o, b) in buf.iter_mut().zip(self.bytes.chunks_exact(4)) {
            *o = f32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }
}

struct PulsePlayback {
    s: Simple,
    /// Reusable transfer buffer (no per-chunk allocation, issue #6).
    bytes: Vec<u8>,
}

impl PlaybackStream for PulsePlayback {
    fn write(&mut self, samples: &[f32]) -> Result<()> {
        self.bytes.clear();
        for s in samples {
            self.bytes.extend_from_slice(&s.to_ne_bytes());
        }
        self.s
            .write(&self.bytes)
            .map_err(|e| anyhow::anyhow!("write: {e}"))
    }
}

/// Parses `pactl list modules short` output (`id\tname\targuments`) and
/// returns the `(id, name)` of every module that owns one of our device
/// names — leftovers from a previous run that was killed before teardown
/// (issue #39). Matching is exact on whole argument tokens
/// (`sink_name=babiniku` etc.), so devices that merely contain the word
/// are left alone.
pub fn stale_babiniku_modules(listing: &str) -> Vec<(String, String)> {
    let owned = [
        format!("sink_name={SINK}"),
        format!("source_name={VIRT_MIC}"),
        format!("source_name={DENOISED_SRC}"),
        format!("source={SINK}.monitor"),
    ];
    listing
        .lines()
        .filter_map(|line| {
            let mut cols = line.splitn(3, '\t');
            let id = cols.next()?.trim();
            let name = cols.next()?.trim();
            let args = cols.next().unwrap_or("");
            args.split_whitespace()
                .any(|kv| owned.iter().any(|o| kv == o))
                .then(|| (id.to_string(), name.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::stale_babiniku_modules;

    /// A realistic `pactl list modules short` listing: the four modules a
    /// killed demo leaks, plus lookalikes that must be left alone.
    const LISTING: &str = "\
5\tmodule-native-protocol-unix\t
21\tmodule-null-sink\tsink_name=babiniku sink_properties=device.description=Babiniku-Output
22\tmodule-remap-source\tsource_name=babiniku_mic master=babiniku.monitor source_properties=device.description=Babiniku-Virtual-Mic
23\tmodule-loopback\tsource=babiniku.monitor latency_msec=60
24\tmodule-echo-cancel\tsource_name=babiniku_denoised aec_method=webrtc source_properties=device.description=Babiniku-Denoised-Input
25\tmodule-null-sink\tsink_name=babiniku2
26\tmodule-null-sink\tsink_name=other sink_properties=device.description=babiniku
";

    #[test]
    fn finds_exactly_the_leaked_babiniku_modules() {
        let stale = stale_babiniku_modules(LISTING);
        let ids: Vec<&str> = stale.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["21", "22", "23", "24"]);
        assert_eq!(stale[0].1, "module-null-sink");
        assert_eq!(stale[2].1, "module-loopback");
    }

    #[test]
    fn empty_or_malformed_listings_match_nothing() {
        assert!(stale_babiniku_modules("").is_empty());
        assert!(stale_babiniku_modules("garbage without tabs\n\n").is_empty());
    }
}

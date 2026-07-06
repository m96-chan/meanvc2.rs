//! Bandwidth-extension (BWE) post-processing for 16 kHz engine output
//! (issue #42).
//!
//! Every current engine synthesizes at 16 kHz, so the converted voice has
//! no energy above 8 kHz — perceived as a "gauzy/veiled" timbre,
//! especially on bright voices. This module provides the two pure-DSP
//! levers evaluated in issue #42:
//!
//! * [`Upsampler3x`] — exact ×3 windowed-sinc polyphase upsampling
//!   (16 kHz → 48 kHz). Playing the demo output at 48 kHz avoids
//!   PipeWire's own 16→48 resample and its measured −15.5 dB rolloff at
//!   6.5–7.6 kHz, and gives the exciter a spectrum to write into.
//! * [`Exciter`] — classic harmonic exciter running in the 48 kHz
//!   domain: band-pass 3–8 kHz → full-wave rectifier (generates even
//!   harmonics at 6–16 kHz) → high-pass ≈7.5 kHz (keeps only the new
//!   high band) → envelope-matched gain → wet mix added to the dry
//!   signal. Zero added sample latency (IIR biquads only), amplitude of
//!   the synthetic band tracks the 3–8 kHz source band.
//!
//! Both stages are engine-agnostic (they see only the output waveform),
//! fp32, allocation-free in the streaming path, and add well under the
//! 10 ms live-latency budget of issue #42 (the upsampler's linear-phase
//! group delay is ≈2.5 ms; the exciter adds none).

use std::f32::consts::PI;

/// Polyphase decomposition factor (16 kHz → 48 kHz).
const PHASES: usize = 3;
/// Prototype low-pass FIR taps per polyphase branch.
const TAPS_PER_PHASE: usize = 80;
/// Total prototype length (multiple of [`PHASES`]).
const PROTO_TAPS: usize = PHASES * TAPS_PER_PHASE;
/// Kaiser window shape: ≈90 dB stop-band attenuation.
const KAISER_BETA: f32 = 8.96;

/// Zeroth-order modified Bessel function of the first kind (power
/// series), for the Kaiser window.
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let x2 = (x as f64 / 2.0) * (x as f64 / 2.0);
    for k in 1..32 {
        term *= x2 / (k as f64 * k as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum as f32
}

/// Exact ×3 windowed-sinc polyphase upsampler (16 kHz → 48 kHz).
///
/// The prototype is a 240-tap Kaiser-windowed sinc low-pass at 8 kHz
/// (output-rate cutoff = fs_in/2): pass-band flat within ±0.05 dB up to
/// ≈7.3 kHz, images rejected by ≈90 dB above ≈8.6 kHz. Linear phase;
/// group delay (= added latency) is (240−1)/2 samples at 48 kHz ≈ 2.5 ms.
///
/// Streaming: [`Upsampler3x::process`] carries the FIR history across
/// calls, so chunked processing is bit-identical to one-shot processing.
pub struct Upsampler3x {
    /// Prototype FIR, reordered per polyphase branch:
    /// `phase[p][k] = 3 · h[3k + p]`.
    phase: Box<[[f32; TAPS_PER_PHASE]; PHASES]>,
    /// Ring buffer of the last [`TAPS_PER_PHASE`] input samples;
    /// `hist[pos]` is the newest.
    hist: [f32; TAPS_PER_PHASE],
    pos: usize,
}

impl Upsampler3x {
    pub fn new() -> Self {
        // Kaiser-windowed sinc prototype, cutoff fc = fs_out/6 (= 8 kHz
        // at 48 kHz), gain ×3 to compensate the zero-stuffing.
        let c = (PROTO_TAPS - 1) as f32 / 2.0;
        let fc = 1.0 / (2.0 * PHASES as f32); // normalized to fs_out
        let i0b = bessel_i0(KAISER_BETA);
        let mut h = [0f32; PROTO_TAPS];
        for (n, tap) in h.iter_mut().enumerate() {
            let t = n as f32 - c;
            let sinc = if t.abs() < 1e-6 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * t).sin() / (PI * t)
            };
            let r = t / c;
            let win = bessel_i0(KAISER_BETA * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            *tap = PHASES as f32 * sinc * win;
        }
        let mut phase = Box::new([[0f32; TAPS_PER_PHASE]; PHASES]);
        for (n, &tap) in h.iter().enumerate() {
            phase[n % PHASES][n / PHASES] = tap;
        }
        Self {
            phase,
            hist: [0f32; TAPS_PER_PHASE],
            pos: 0,
        }
    }

    /// Upsamples `input` (16 kHz) and **appends** `3 · input.len()`
    /// samples at 48 kHz to `output`.
    pub fn process(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.reserve(input.len() * PHASES);
        for &x in input {
            // Push the newest sample into the ring buffer.
            self.pos = (self.pos + 1) % TAPS_PER_PHASE;
            self.hist[self.pos] = x;
            // One output sample per polyphase branch:
            // y[3n + p] = Σ_k h[3k + p] · x[n − k].
            for p in 0..PHASES {
                let coeffs = &self.phase[p];
                let mut acc = 0f32;
                let mut idx = self.pos;
                for &c in coeffs.iter() {
                    acc += c * self.hist[idx];
                    idx = if idx == 0 {
                        TAPS_PER_PHASE - 1
                    } else {
                        idx - 1
                    };
                }
                output.push(acc);
            }
        }
    }

    /// Added latency in seconds: the linear-phase group delay of the
    /// prototype at the output rate.
    pub fn latency_secs(&self) -> f32 {
        (PROTO_TAPS - 1) as f32 / 2.0 / 48_000.0
    }
}

impl Default for Upsampler3x {
    fn default() -> Self {
        Self::new()
    }
}

/// Transposed direct-form-II biquad (RBJ cookbook coefficients).
#[derive(Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn highpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * f0 / fs;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos) / 2.0 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn lowpass(fs: f32, f0: f32, q: f32) -> Self {
        let w0 = 2.0 * PI * f0 / fs;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 - cos) / 2.0 / a0,
            b1: (1.0 - cos) / a0,
            b2: (1.0 - cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// One-pole peak envelope follower with separate attack/release times.
#[derive(Clone, Copy)]
struct EnvFollower {
    attack: f32,
    release: f32,
    env: f32,
}

impl EnvFollower {
    fn new(fs: f32, attack_s: f32, release_s: f32) -> Self {
        Self {
            attack: 1.0 - (-1.0 / (fs * attack_s)).exp(),
            release: 1.0 - (-1.0 / (fs * release_s)).exp(),
            env: 0.0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let a = x.abs();
        let coef = if a > self.env {
            self.attack
        } else {
            self.release
        };
        self.env += coef * (a - self.env);
        self.env
    }
}

/// Steep linear-phase FIR high-pass for the exciter's harmonic branch.
///
/// The full-wave rectifier lands products **inside** the untouched
/// 0–8 kHz spectrum too (baseband intermodulation, 2f of sub-4 kHz
/// content); an IIR high-pass leaks several dB of them just below its
/// cutoff. This 241-tap Kaiser high-pass (spectral inversion of an
/// 8 kHz low-pass, ≈80 dB stop-band, ≈1 kHz transition) keeps
/// everything the exciter adds below ≈7.5 kHz at least 80 dB down, so
/// the dry conversion output stays measurably bit-clean there.
struct FirHighpass {
    taps: Vec<f32>,
    hist: Vec<f32>,
    pos: usize,
}

impl FirHighpass {
    /// `fc` is the transition-center frequency in Hz.
    fn new(fs: f32, fc: f32, ntaps: usize, beta: f32) -> Self {
        debug_assert!(ntaps % 2 == 1, "type-I FIR needed for spectral inversion");
        let c = (ntaps - 1) as f32 / 2.0;
        let fcn = fc / fs;
        let i0b = bessel_i0(beta);
        let mut taps = vec![0f32; ntaps];
        // Unity-gain windowed-sinc low-pass…
        for (n, tap) in taps.iter_mut().enumerate() {
            let t = n as f32 - c;
            let sinc = if t.abs() < 1e-6 {
                2.0 * fcn
            } else {
                (2.0 * PI * fcn * t).sin() / (PI * t)
            };
            let r = t / c;
            *tap = sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
        }
        // …spectrally inverted into a high-pass.
        for tap in taps.iter_mut() {
            *tap = -*tap;
        }
        taps[(ntaps - 1) / 2] += 1.0;
        Self {
            hist: vec![0f32; ntaps],
            taps,
            pos: 0,
        }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let n = self.taps.len();
        self.pos = (self.pos + 1) % n;
        self.hist[self.pos] = x;
        let mut acc = 0f32;
        let mut idx = self.pos;
        for &t in &self.taps {
            acc += t * self.hist[idx];
            idx = if idx == 0 { n - 1 } else { idx - 1 };
        }
        acc
    }
}

/// Classic harmonic exciter for bandwidth extension above 8 kHz
/// (issue #42 lever 2). Runs at 48 kHz on the upsampled engine output.
///
/// Chain: band-pass 3–8 kHz (source band) → full-wave rectifier (the
/// nonlinearity; even harmonics of the 3–8 kHz band land at 6–16 kHz)
/// → steep linear-phase high-pass at ≈8 kHz ([`FirHighpass`]: keeps
/// only the newly synthesized band; the dry 0–7.5 kHz spectrum stays
/// clean to ≈−80 dB) → envelope-matched gain (the synthetic band tracks
/// the source-band level at a fixed −6 dB target ratio) → `wet`-scaled
/// addition.
///
/// * The dry path is untouched — the exciter adds **zero** pipeline
///   latency. The additive wet branch is delayed by its FIR group delay
///   (2.5 ms), far inside both the 10 ms live budget of issue #42 and
///   the perceptual fusion window for added high-band content.
/// * `wet == 0.0` is a bit-exact bypass (the filters still run so the
///   knob can be opened live without a transient).
pub struct Exciter {
    band_hp: Biquad,
    band_lp: Biquad,
    harm_hp: FirHighpass,
    src_env: EnvFollower,
    harm_env: EnvFollower,
    /// One-pole smoothed envelope-matching gain (prevents the ×MAX_GAIN
    /// burst at speech onsets, where the harmonic envelope is still ~0).
    gain_state: f32,
    gain_up: f32,
    gain_down: f32,
    /// Previous chunk's wet amount; knob changes ramp across the chunk.
    prev_wet: f32,
}

impl Exciter {
    /// Level of the synthetic 8–16 kHz band relative to the 3–8 kHz
    /// source band at `wet = 1.0` (−6 dB).
    const TARGET_RATIO: f32 = 0.5;
    /// Envelope-matching gain ceiling (guards the rectifier's small-
    /// signal noise floor from being amplified into audible hiss).
    const MAX_GAIN: f32 = 8.0;

    /// `sample_rate` is the rate the exciter runs at (48 kHz in the
    /// demo's output path).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            band_hp: Biquad::highpass(sample_rate, 3_000.0, std::f32::consts::FRAC_1_SQRT_2),
            band_lp: Biquad::lowpass(sample_rate, 8_000.0, std::f32::consts::FRAC_1_SQRT_2),
            // Transition ≈7.5 → 8.5 kHz, stop-band ≈80 dB below 7.5 kHz.
            harm_hp: FirHighpass::new(sample_rate, 8_050.0, 241, 7.857),
            src_env: EnvFollower::new(sample_rate, 0.005, 0.050),
            harm_env: EnvFollower::new(sample_rate, 0.005, 0.050),
            gain_state: 0.0,
            // Asymmetric gain slew: rising gain is slewed (~15 ms) so a
            // ceiling-parked gain cannot slam into a fresh harmonic burst,
            // falling gain is fast (~3 ms) so bursts de-amplify instantly.
            // 15 ms keeps onsets bright — the original 80 ms rise dulled
            // every word onset, and the burst it guarded against turned
            // out to be the decoder needle (now repaired upstream by
            // `declick::NeedleGuard`).
            gain_up: 1.0 - (-1.0 / (0.015 * sample_rate)).exp(),
            gain_down: 1.0 - (-1.0 / (0.003 * sample_rate)).exp(),
            prev_wet: 0.0,
        }
    }

    /// Processes one chunk in place. `wet` is the mix amount in
    /// `0.0..=1.0`; `0.0` leaves `samples` bit-identical (the filter
    /// state still advances so live knob changes are transient-free).
    pub fn process(&mut self, samples: &mut [f32], wet: f32) {
        let wet = wet.clamp(0.0, 1.0);
        let n = samples.len().max(1) as f32;
        let wet_step = (wet - self.prev_wet) / n;
        let mut wet_now = self.prev_wet;
        for s in samples.iter_mut() {
            let x = *s;
            wet_now += wet_step;
            // Source band 3–8 kHz.
            let band = self.band_lp.tick(self.band_hp.tick(x));
            // Full-wave rectification: even harmonics + DC/baseband,
            // amplitude-linear in the input (envelope is preserved).
            let rect = band.abs();
            // Keep only the synthesized high band.
            let harm = self.harm_hp.tick(rect);
            // Envelope matching: scale the harmonics so their level
            // tracks the source band at TARGET_RATIO.
            let se = self.src_env.tick(band);
            let he = self.harm_env.tick(harm);
            // Silence guard: fade the wet branch out below −54 dBFS source
            // level instead of amplifying the noise floor at word onsets.
            let guard = (se / 2e-3).clamp(0.0, 1.0);
            let target = (se / (he + 1e-9)).min(Self::MAX_GAIN) * guard;
            let alpha = if target > self.gain_state {
                self.gain_up
            } else {
                self.gain_down
            };
            self.gain_state += alpha * (target - self.gain_state);
            if wet_now > 0.0 {
                // Bound the synthetic band by the source-band envelope:
                // whatever the gain state, the added sample can never
                // exceed the band's own level (field report: ticks during
                // sustained vowels came from gain*harm bursts, not from
                // waveform steps). Soft saturation, not a hard clamp — a
                // hard clamp flat-tops the band at every glottal pulse,
                // and those edges are themselves broadband ticks.
                // Identity below the knee (no intermodulation on normal
                // content), C1-smooth saturation above it.
                let add = Self::TARGET_RATIO * self.gain_state * harm;
                let bound = 1.5 * se + 1e-12;
                let u = add / bound;
                let knee = 0.75f32;
                let soft = if u.abs() <= knee {
                    u
                } else {
                    u.signum() * (knee + (1.0 - knee) * ((u.abs() - knee) / (1.0 - knee)).tanh())
                };
                *s = x + wet_now * bound * soft;
            }
        }
        self.prev_wet = wet;
    }
}

/// Look-ahead peak limiter: keeps the output under `threshold` without
/// hard-clipping (hard clips read as "kachi-kachi" ticks — issue #42
/// field analysis found every reported tick coincided with |y| > 0.97).
///
/// A `lookahead`-sample delay line lets the gain reach any upcoming peak
/// before it plays: the gain envelope takes the minimum required gain
/// over the look-ahead window (sliding-minimum via a monotonic deque),
/// then releases exponentially (~80 ms). Below threshold it is exactly
/// unity — bit-transparent.
pub struct Limiter {
    threshold: f32,
    delay: std::collections::VecDeque<f32>,
    lookahead: usize,
    /// (index, required_gain) monotonic deque for the sliding minimum.
    window: std::collections::VecDeque<(u64, f32)>,
    pos: u64,
    gain: f32,
    attack: f32,
    release: f32,
}

impl Limiter {
    pub fn new(sample_rate: f32, threshold: f32) -> Self {
        let lookahead = (sample_rate * 0.005) as usize; // 5 ms
        Self {
            threshold,
            delay: std::collections::VecDeque::from(vec![0.0; lookahead]),
            lookahead,
            window: std::collections::VecDeque::new(),
            pos: 0,
            gain: 1.0,
            // ~1.5 ms attack: ≥3 time constants inside the 5 ms
            // look-ahead (settled ≥96 % before the peak plays), slow
            // enough that per-glottal-pulse gain moves are inaudible —
            // the 0.4 ms attack read as カタカタ rattling when the
            // converted voice (crest ≈ 19) worked the threshold on
            // every loud syllable (eleventh field report).
            attack: 1.0 - (-1.0 / (0.0015 * sample_rate)).exp(),
            release: 1.0 - (-1.0 / (0.150 * sample_rate)).exp(),
        }
    }

    /// Delay introduced by the look-ahead, in samples.
    pub fn latency(&self) -> usize {
        self.lookahead
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let x = *s;
            // Required gain for the incoming sample.
            let need = if x.abs() > self.threshold {
                self.threshold / x.abs()
            } else {
                1.0
            };
            // Sliding minimum over the look-ahead window.
            while let Some(&(_, g)) = self.window.back() {
                if g >= need {
                    self.window.pop_back();
                } else {
                    break;
                }
            }
            self.window.push_back((self.pos, need));
            while let Some(&(i, _)) = self.window.front() {
                if i + self.lookahead as u64 <= self.pos {
                    self.window.pop_front();
                } else {
                    break;
                }
            }
            let target = self.window.front().map_or(1.0, |&(_, g)| g);
            // Attack: move to the (lower) target immediately — the peak is
            // still `lookahead` samples away, so this is a 2.5 ms ramp by
            // construction. Release: exponential.
            let coeff = if target < self.gain { self.attack } else { self.release };
            self.gain += coeff * (target - self.gain);
            self.delay.push_back(x);
            let out = self.delay.pop_front().unwrap_or(0.0);
            // Brickwall clamp: the smoothed gain settles ≥96 % inside the
            // look-ahead; the remaining sliver is clamped exactly at the
            // peak samples (a ≤2 % dip on isolated samples — inaudible,
            // and the output can never exceed the threshold).
            let need_now = if out.abs() * self.gain > self.threshold {
                self.threshold / out.abs()
            } else {
                self.gain
            };
            *s = out * self.gain.min(need_now);
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    const SR_IN: f32 = 16_000.0;
    const SR_OUT: f32 = 48_000.0;

    fn sine(freq: f32, sr: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// Goertzel power of `x` at `freq` (Hann-windowed DFT bin magnitude).
    fn tone_mag(x: &[f32], freq: f32, sr: f32) -> f32 {
        let n = x.len();
        let mut re = 0f64;
        let mut im = 0f64;
        for (i, &s) in x.iter().enumerate() {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos();
            let ph = 2.0 * std::f64::consts::PI * freq as f64 * i as f64 / sr as f64;
            re += w * s as f64 * ph.cos();
            im += w * s as f64 * ph.sin();
        }
        ((re * re + im * im).sqrt() * 2.0 / n as f64) as f32
    }

    /// Word-onset regression (issue #42 field report): silence followed by
    /// a burst must not inject a one-sample harmonic spike — the smoothed
    /// gain has ~10 ms to rise, so the wet output may not jump faster than
    /// the dry input by more than a small factor.
    #[test]
    fn exciter_onset_produces_no_click() {
        let mut ex = Exciter::new(SR_OUT);
        let mut sig = vec![0.0f32; 4_800]; // 100 ms silence
        sig.extend(sine(5_000.0, SR_OUT, 9_600, 0.6)); // sudden voiced burst
        let dry = sig.clone();
        ex.process(&mut sig, 1.0);
        let added: Vec<f32> = sig.iter().zip(&dry).map(|(w, d)| w - d).collect();
        // Added high band within the first 2 ms of the onset stays small
        // relative to its steady-state level.
        let onset = rms(&added[4_800..4_896]);
        let steady = rms(&added[9_600..14_000]);
        assert!(
            onset < 0.35 * steady + 1e-6,
            "onset burst: {onset} vs steady {steady}"
        );
        // And in pure silence the wet branch adds nothing audible.
        assert!(rms(&added[..4_700]) < 1e-5);
    }

    /// Chunked processing must equal one-shot processing exactly — any
    /// difference means per-chunk state loss, i.e. periodic clicks at hop
    /// boundaries in live use.
    #[test]
    fn chunked_equals_oneshot_through_full_chain() {
        let sig: Vec<f32> = (0..48_000)
            .map(|i| {
                0.4 * (2.0 * PI * 4_500.0 * i as f32 / SR_OUT).sin()
                    + 0.2 * (2.0 * PI * 6_300.0 * i as f32 / SR_OUT).sin()
            })
            .collect();
        // Identical warm-up on both instances settles the wet ramp
        // (prev_wet) and filter state before the comparison window.
        let warm = &sig[..11_520];
        let mut ex_a = Exciter::new(SR_OUT);
        let mut w = warm.to_vec();
        ex_a.process(&mut w, 0.6);
        let mut a = sig[11_520..].to_vec();
        ex_a.process(&mut a, 0.6);
        let mut ex_b = Exciter::new(SR_OUT);
        let mut w = warm.to_vec();
        ex_b.process(&mut w, 0.6);
        let mut b = sig[11_520..].to_vec();
        for chunk in b.chunks_mut(11_520) {
            ex_b.process(chunk, 0.6);
        }
        let d = a[11_520..]
            .iter()
            .zip(&b[11_520..])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-6, "chunk-boundary state loss: max diff {d}");

        // Same for the upsampler.
        let x16: Vec<f32> = (0..16_000)
            .map(|i| 0.5 * (2.0 * PI * 3_000.0 * i as f32 / SR_IN).sin())
            .collect();
        let mut up1 = Upsampler3x::new();
        let mut whole = Vec::new();
        up1.process(&x16, &mut whole);
        let mut up2 = Upsampler3x::new();
        let mut parts = Vec::new();
        for c in x16.chunks(3_840) {
            let mut o = Vec::new();
            up2.process(c, &mut o);
            parts.extend(o);
        }
        let m = whole.len().min(parts.len());
        let du = whole[..m]
            .iter()
            .zip(&parts[..m])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(du < 1e-6, "upsampler chunk-boundary loss: max diff {du}");
    }

    /// Sustained-vowel spike regression (issue #42, real-recording
    /// analysis): with a vowel-like signal whose 3-8 kHz band is weak and
    /// fluctuating, the envelope-matching gain must not turn band bursts
    /// into high-band clicks. Bound the added band's crest factor.
    #[test]
    fn exciter_vowel_shimmer_produces_no_spikes() {
        // 250 Hz glottal-ish harmonic series with slow shimmer on the
        // upper partials (deterministic).
        let n = 48_000 * 3;
        let mut sig = vec![0f32; n];
        for h in 1..=30 {
            let f = 250.0 * h as f32;
            let a = 0.5 / h as f32;
            for (i, s) in sig.iter_mut().enumerate() {
                let t = i as f32 / SR_OUT;
                let shim = if f > 3_000.0 {
                    0.5 + 0.5 * (2.0 * PI * (1.7 + 0.13 * h as f32) * t).sin()
                } else {
                    1.0
                };
                *s += a * shim * (2.0 * PI * f * t).sin();
            }
        }
        let dry = sig.clone();
        let mut ex = Exciter::new(SR_OUT);
        for c in sig.chunks_mut(11_520) {
            ex.process(c, 1.0);
        }
        let added: Vec<f32> = sig.iter().zip(&dry).map(|(w, d)| w - d).collect();
        // Envelope crest factor of the added band after settling.
        let tail = &added[24_000..];
        let fr = 96;
        let env: Vec<f32> = tail
            .chunks(fr)
            .map(|c| c.iter().fold(0f32, |m, &v| m.max(v.abs())))
            .collect();
        let mut sorted = env.clone();
        sorted.sort_by(f32::total_cmp);
        let med = sorted[sorted.len() / 2] + 1e-9;
        let crest = env.iter().fold(0f32, |m, &v| m.max(v)) / med;
        assert!(crest < 4.0, "high-band click crest factor {crest}");
    }

    #[test]
    fn limiter_is_transparent_below_threshold() {
        let mut l = Limiter::new(SR_OUT, 0.9);
        let sig = sine(440.0, SR_OUT, 9_600, 0.5);
        let mut out = sig.clone();
        l.process(&mut out);
        // Compare with the look-ahead delay compensated.
        let d = l.latency();
        let diff = out[d..]
            .iter()
            .zip(&sig[..sig.len() - d])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-6, "not transparent: {diff}");
    }

    #[test]
    fn limiter_prevents_clipping_without_steps() {
        let mut l = Limiter::new(SR_OUT, 0.9);
        // 0.5 sine with a 1.4x burst in the middle.
        let mut sig = sine(300.0, SR_OUT, 48_000, 0.5);
        for s in sig[16_000..20_000].iter_mut() {
            *s *= 2.8;
        }
        l.process(&mut sig);
        let peak = sig.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(peak <= 0.905, "peak {peak}");
        // Smooth gain: no added discontinuities.
        let dmax = sig.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0f32, f32::max);
        assert!(dmax < 0.06, "step in limited output: {dmax}");
    }

    /// Energy of `x` in the band `[lo, hi)` Hz via FFT.
    fn band_energy(x: &[f32], sr: f32, lo: f32, hi: f32) -> f32 {
        let n = x.len();
        let mut buf: Vec<Complex<f32>> = x
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
                Complex::new(s * w, 0.0)
            })
            .collect();
        FftPlanner::new().plan_fft_forward(n).process(&mut buf);
        let hz_per_bin = sr / n as f32;
        buf.iter()
            .take(n / 2)
            .enumerate()
            .filter(|(i, _)| {
                let f = *i as f32 * hz_per_bin;
                f >= lo && f < hi
            })
            .map(|(_, c)| c.norm_sqr())
            .sum()
    }

    // ------------------------------------------------------------------
    // Upsampler3x

    #[test]
    fn upsampler_output_is_exactly_3x_len() {
        let mut up = Upsampler3x::new();
        let mut out = Vec::new();
        up.process(&vec![0.25; 1234], &mut out);
        assert_eq!(out.len(), 3 * 1234);
    }

    #[test]
    fn upsampler_passband_is_flat() {
        // Pass-band probes: gain must be 0 dB within ±0.1 dB.
        for freq in [500.0, 1_000.0, 2_000.0, 4_000.0, 6_000.0, 7_000.0] {
            let x = sine(freq, SR_IN, 16_000, 0.5);
            let mut up = Upsampler3x::new();
            let mut y = Vec::new();
            up.process(&x, &mut y);
            // Skip the FIR warm-up on both signals, compare steady state.
            let xs = &x[2_000..14_000];
            let ys = &y[6_000..42_000];
            let gain_db = 20.0 * (rms(ys) / rms(xs)).log10();
            assert!(
                gain_db.abs() < 0.1,
                "{freq} Hz pass-band gain {gain_db:.3} dB"
            );
        }
    }

    #[test]
    fn upsampler_rejects_images() {
        // A tone at f (≤ 7 kHz) images at 16k − f and 16k + f after
        // zero-stuffing; the prototype must crush them by ≥ 70 dB.
        for freq in [3_000.0f32, 5_000.0, 6_500.0] {
            let x = sine(freq, SR_IN, 16_000, 0.5);
            let mut up = Upsampler3x::new();
            let mut y = Vec::new();
            up.process(&x, &mut y);
            let ys = &y[6_000..42_000];
            let fundamental = tone_mag(ys, freq, SR_OUT);
            for image in [16_000.0 - freq, 16_000.0 + freq] {
                let img = tone_mag(ys, image, SR_OUT);
                let rej_db = 20.0 * (img / fundamental).log10();
                assert!(
                    rej_db < -70.0,
                    "{freq} Hz image at {image} Hz only {rej_db:.1} dB down"
                );
            }
        }
    }

    #[test]
    fn upsampler_streaming_matches_batch() {
        // Chunked processing must be bit-identical to one-shot.
        let x: Vec<f32> = (0..4_800)
            .map(|i| ((i * 2654435761u64 as usize) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut up1 = Upsampler3x::new();
        let mut batch = Vec::new();
        up1.process(&x, &mut batch);
        let mut up2 = Upsampler3x::new();
        let mut streamed = Vec::new();
        for chunk in x.chunks(160) {
            up2.process(chunk, &mut streamed);
        }
        assert_eq!(batch, streamed);
    }

    #[test]
    fn upsampler_latency_is_under_10ms() {
        let up = Upsampler3x::new();
        assert!(up.latency_secs() < 0.010, "{}", up.latency_secs());
        // Measured: the impulse response peak sits at the reported group
        // delay.
        let mut up = Upsampler3x::new();
        let mut imp = vec![0f32; 400];
        imp[0] = 1.0;
        let mut y = Vec::new();
        up.process(&imp, &mut y);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        let measured = peak as f32 / SR_OUT;
        assert!(measured < 0.010, "measured latency {measured} s");
        assert!((measured - up.latency_secs()).abs() < 0.001);
    }

    // ------------------------------------------------------------------
    // Exciter

    /// A speech-band-ish test signal at 48 kHz: sines inside the 3–8 kHz
    /// source band plus a low-frequency component.
    fn source_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SR_OUT;
                0.3 * (2.0 * PI * 400.0 * t).sin()
                    + 0.2 * (2.0 * PI * 3_500.0 * t).sin()
                    + 0.2 * (2.0 * PI * 5_000.0 * t).sin()
                    + 0.2 * (2.0 * PI * 6_500.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn exciter_wet_zero_is_bit_exact_bypass() {
        let x = source_signal(48_000);
        let mut y = x.clone();
        Exciter::new(SR_OUT).process(&mut y, 0.0);
        assert_eq!(x, y);
    }

    #[test]
    fn exciter_populates_the_high_band() {
        let x = source_signal(96_000);
        let mut dry = x.clone();
        Exciter::new(SR_OUT).process(&mut dry, 0.0);
        let mut wet = x.clone();
        Exciter::new(SR_OUT).process(&mut wet, 0.6);
        // Steady state only.
        let dry_hi = band_energy(&dry[24_000..], SR_OUT, 8_000.0, 16_000.0);
        let wet_hi = band_energy(&wet[24_000..], SR_OUT, 8_000.0, 16_000.0);
        let src = band_energy(&x[24_000..], SR_OUT, 3_000.0, 8_000.0);
        // Input has essentially nothing above 8 kHz…
        assert!(dry_hi < 1e-6 * src, "dry high band {dry_hi} vs src {src}");
        // …and the exciter puts real energy there (> −30 dB re: source
        // band at wet 0.6).
        assert!(
            wet_hi > 1e-3 * src,
            "wet high band {wet_hi} vs src {src} ({:.1} dB)",
            10.0 * (wet_hi / src).log10()
        );
    }

    #[test]
    fn exciter_leaves_the_low_band_unchanged() {
        let x = source_signal(96_000);
        let mut wet = x.clone();
        Exciter::new(SR_OUT).process(&mut wet, 1.0);
        let lo_in = band_energy(&x[24_000..], SR_OUT, 0.0, 7_000.0);
        let lo_out = band_energy(&wet[24_000..], SR_OUT, 0.0, 7_000.0);
        let diff_db = 10.0 * (lo_out / lo_in).log10();
        assert!(
            diff_db.abs() < 0.05,
            "0–7 kHz band changed by {diff_db:.3} dB at wet 1.0"
        );
    }

    #[test]
    fn exciter_leakage_below_the_passband_edge_is_negligible() {
        // Everything the exciter adds below the upsampler's pass-band
        // edge (≈7.4 kHz) must be at least 60 dB below the dry content
        // there, so the 0–8 kHz conversion output stays measurably
        // unchanged (issue #42 requires the A/B diff below the
        // upsampler's ±0.1 dB pass-band ripple). Note the rectifier
        // lands products **inside** this region (e.g. 2×3.5 kHz = 7 kHz,
        // 5 − 3.5 kHz intermodulation), so this bounds the harmonic
        // high-pass stop-band, not just roll-off.
        let x = source_signal(96_000);
        let mut wet = x.clone();
        Exciter::new(SR_OUT).process(&mut wet, 1.0);
        let diff: Vec<f32> = x.iter().zip(&wet).map(|(a, b)| b - a).collect();
        let leak = band_energy(&diff[24_000..], SR_OUT, 0.0, 7_400.0);
        let dry = band_energy(&x[24_000..], SR_OUT, 0.0, 7_400.0);
        assert!(
            leak < 1e-6 * dry,
            "sub-7.4 kHz leakage only {:.1} dB below the dry content",
            10.0 * (dry / leak).log10()
        );
    }

    #[test]
    fn exciter_is_causal_and_latency_free() {
        // Outputs up to sample k depend only on inputs up to sample k.
        let a = source_signal(48_000);
        let mut b = a.clone();
        for s in &mut b[24_000..] {
            *s = -*s;
        }
        let mut ya = a.clone();
        Exciter::new(SR_OUT).process(&mut ya, 0.8);
        let mut yb = b.clone();
        Exciter::new(SR_OUT).process(&mut yb, 0.8);
        assert_eq!(&ya[..24_000], &yb[..24_000]);
        assert_ne!(&ya[24_000..24_100], &yb[24_000..24_100]);
    }

    #[test]
    fn exciter_silence_stays_silent() {
        let mut x = vec![0f32; 48_000];
        Exciter::new(SR_OUT).process(&mut x, 1.0);
        assert!(x.iter().all(|s| s.abs() < 1e-6 && s.is_finite()));
    }
}

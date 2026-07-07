//! Voice profile EQ (issue #62): adapt the output spectrum toward the
//! reference speaker's real long-term average spectrum (LTAS).
//!
//! The ground truth is free: the `--reference` wav IS the target
//! speaker's real voice. Measured on the live stack, the converted
//! output is 11–21 dB thin below 800 Hz and 5–19 dB dull between
//! 3–11 kHz against the reference's LTAS — a static coloration that
//! reads as "voice changer". This module measures both sides and
//! corrects with a slowly adapting linear-phase FIR:
//!
//! - [`Ltas`] accumulates voiced-frame power spectra (1024-point FFT,
//!   Hann, energy-gated so silence never skews the estimate).
//! - [`ProfileEq`] holds the reference LTAS as the target, accumulates
//!   the (pre-EQ) output LTAS, and every [`ProfileEq::UPDATE_SECS`] of
//!   voiced audio rebuilds a 128-tap linear-phase FIR from the
//!   1/3-octave-smoothed dB difference, clamped and band-capped:
//!   ±10 dB overall, +6 dB above 5 kHz (the diffusion noise bed lives
//!   there — issue #62 guardrail), zero beyond 95 % of the reference's
//!   Nyquist (that band is the BWE exciter's job), zero below 70 Hz.
//!   Coefficients slew toward each rebuild, so there is no zipper
//!   noise; adaptation is open-loop (measured before the EQ), hence
//!   unconditionally stable.
//!
//! `wet` mixes the corrected signal; `0` is bit-transparent bypass.

use rustfft::{num_complex::Complex64, FftPlanner};

/// Analysis rate: the demo's output domain.
pub const RATE: f32 = 48_000.0;
const N_FFT: usize = 1_024;
const N_BINS: usize = N_FFT / 2 + 1;
/// FIR length (linear phase; ~1.3 ms group delay at 48 kHz).
pub const TAPS: usize = 128;

/// Windowed-sinc rational resampler for ANALYSIS use (reference wavs
/// arrive at 16 / 22.05 / 44.1 kHz; the LTAS target lives at 48 kHz).
/// Textbook polyphase design, width 12, Hann window.
pub fn resample_analysis(input: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return input.to_vec();
    }
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let g = gcd(from, to);
    let (up, down) = (to / g, from / g);
    let width = 12f64;
    let cutoff = 0.95 * (to as f64 / from as f64).min(1.0);
    let n_out = input.len() * up / down;
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        // Output instant in input-sample units.
        let pos = n as f64 * down as f64 / up as f64;
        let lo = (pos - width).ceil() as isize;
        let hi = (pos + width).floor() as isize;
        let mut acc = 0f64;
        for j in lo..=hi {
            if j < 0 || j as usize >= input.len() {
                continue;
            }
            let t = j as f64 - pos;
            let sinc = if t == 0.0 {
                cutoff
            } else {
                (std::f64::consts::PI * cutoff * t).sin() / (std::f64::consts::PI * t)
            };
            let w = 0.5 + 0.5 * (std::f64::consts::PI * t / width).cos();
            acc += input[j as usize] as f64 * sinc * w;
        }
        out.push(acc as f32);
    }
    out
}

/// Energy-gated LTAS accumulator.
pub struct Ltas {
    acc: Vec<f64>,
    frames: u64,
    window: Vec<f64>,
    /// Carry for chunked feeding.
    buf: Vec<f32>,
    /// Frame gate on rms.
    gate: f32,
}

impl Ltas {
    pub fn new(gate: f32) -> Self {
        Self {
            acc: vec![0.0; N_BINS],
            frames: 0,
            window: (0..N_FFT)
                .map(|i| {
                    0.5 - 0.5
                        * (2.0 * std::f64::consts::PI * i as f64 / (N_FFT - 1) as f64).cos()
                })
                .collect(),
            buf: Vec::new(),
            gate,
        }
    }

    /// Feeds samples (any chunking); voiced frames accumulate.
    pub fn feed(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut scratch = vec![Complex64::new(0.0, 0.0); N_FFT];
        while self.buf.len() >= N_FFT {
            let frame: Vec<f32> = self.buf.drain(..N_FFT).collect();
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / N_FFT as f32).sqrt();
            if rms < self.gate {
                continue;
            }
            for (i, (s, b)) in frame.iter().zip(scratch.iter_mut()).enumerate() {
                *b = Complex64::new(*s as f64 * self.window[i], 0.0);
            }
            fft.process(&mut scratch);
            for (a, c) in self.acc.iter_mut().zip(&scratch[..N_BINS]) {
                *a += c.norm_sqr();
            }
            self.frames += 1;
        }
    }

    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Mean power spectrum in dB (None until any frame accumulated).
    pub fn db(&self) -> Option<Vec<f64>> {
        if self.frames == 0 {
            return None;
        }
        Some(
            self.acc
                .iter()
                .map(|a| 10.0 * (a / self.frames as f64 + 1e-18).log10())
                .collect(),
        )
    }
}

/// 1/3-octave (constant-Q) smoothing of a per-bin dB curve.
fn smooth_third_octave(db: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; db.len()];
    for (k, o) in out.iter_mut().enumerate() {
        let f = (k.max(1)) as f64 * RATE as f64 / N_FFT as f64;
        let half = f * (2f64.powf(1.0 / 6.0) - 1.0);
        let lo = ((f - half) * N_FFT as f64 / RATE as f64).floor().max(0.0) as usize;
        let hi = (((f + half) * N_FFT as f64 / RATE as f64).ceil() as usize).min(db.len() - 1);
        let span = &db[lo..=hi.max(lo)];
        *o = span.iter().sum::<f64>() / span.len() as f64;
    }
    out
}

/// The adaptive corrective EQ.
pub struct ProfileEq {
    target_db: Vec<f64>,
    /// Correction zeroed above this bin (95 % of the reference Nyquist).
    max_bin: usize,
    out_ltas: Ltas,
    /// Current / goal FIR coefficients (linear phase).
    taps: Vec<f32>,
    goal: Vec<f32>,
    /// FIR history for streaming convolution.
    hist: Vec<f32>,
    next_update: u64,
}

impl ProfileEq {
    /// Voiced audio between coefficient rebuilds (seconds).
    pub const UPDATE_SECS: f64 = 3.0;
    /// Overall correction clamp (dB).
    const CLAMP_DB: f64 = 10.0;
    /// Boost cap above 5 kHz (noise-bed guardrail, dB).
    const HF_BOOST_CAP_DB: f64 = 6.0;
    /// No correction below this frequency (rumble guardrail, Hz).
    const F_LO: f64 = 70.0;
    /// Coefficient slew per rebuild (1 = jump).
    const SLEW: f32 = 0.35;

    /// `reference_48k`: the reference voice resampled to 48 kHz (band
    /// limited by its native rate is fine); `ref_native_rate` bounds
    /// the correction band.
    pub fn new(reference_48k: &[f32], ref_native_rate: f32) -> Self {
        let mut ref_ltas = Ltas::new(0.01);
        ref_ltas.feed(reference_48k);
        let target_db = ref_ltas
            .db()
            .map(|d| smooth_third_octave(&d))
            .unwrap_or_else(|| vec![0.0; N_BINS]);
        let nyq = (ref_native_rate as f64 / 2.0) * 0.95;
        let max_bin = ((nyq * N_FFT as f64 / RATE as f64) as usize).min(N_BINS - 1);
        let mut passthrough = vec![0f32; TAPS];
        passthrough[TAPS / 2] = 1.0;
        Self {
            target_db,
            max_bin,
            out_ltas: Ltas::new(0.01),
            taps: passthrough.clone(),
            goal: passthrough,
            hist: vec![0.0; TAPS - 1],
            next_update: (Self::UPDATE_SECS * RATE as f64 / N_FFT as f64) as u64,
        }
    }

    /// Group delay of the linear-phase FIR, in samples.
    pub fn latency(&self) -> usize {
        TAPS / 2
    }

    /// Feeds the PRE-EQ output for adaptation (open loop) and rebuilds
    /// the goal coefficients when enough voiced audio accumulated.
    pub fn observe(&mut self, pre_eq: &[f32]) {
        self.out_ltas.feed(pre_eq);
        if self.out_ltas.frames() < self.next_update {
            return;
        }
        self.next_update =
            self.out_ltas.frames() + (Self::UPDATE_SECS * RATE as f64 / N_FFT as f64) as u64;
        let Some(cur) = self.out_ltas.db() else {
            return;
        };
        let cur = smooth_third_octave(&cur);
        // Level-normalize the diff in the 1–3 kHz band (the leveler owns
        // loudness; the EQ owns shape).
        let (b1, b3) = (
            (1_000.0 * N_FFT as f64 / RATE as f64) as usize,
            (3_000.0 * N_FFT as f64 / RATE as f64) as usize,
        );
        let bias: f64 = (b1..b3)
            .map(|k| self.target_db[k] - cur[k])
            .sum::<f64>()
            / (b3 - b1) as f64;
        let lo_bin = (Self::F_LO * N_FFT as f64 / RATE as f64) as usize;
        let hf_bin = (5_000.0 * N_FFT as f64 / RATE as f64) as usize;
        let mut gain_db = vec![0.0f64; N_BINS];
        for k in lo_bin..=self.max_bin {
            let mut g = (self.target_db[k] - cur[k] - bias)
                .clamp(-Self::CLAMP_DB, Self::CLAMP_DB);
            if k >= hf_bin {
                g = g.min(Self::HF_BOOST_CAP_DB);
            }
            gain_db[k] = g;
        }
        // Taper the band edges over ~3 bins to avoid ringing.
        for e in 0..3usize {
            let w = e as f64 / 3.0;
            if lo_bin + e < N_BINS {
                gain_db[lo_bin + e] *= w;
            }
            if self.max_bin >= e {
                gain_db[self.max_bin - e] *= w;
            }
        }
        self.goal = design_fir(&gain_db);
        // Slew towards the goal (zipper-free adaptation).
        for (t, g) in self.taps.iter_mut().zip(&self.goal) {
            *t += Self::SLEW * (g - *t);
        }
    }

    /// Applies the EQ with `wet` in `0..=1` (0 = bit-transparent).
    pub fn process(&mut self, chunk: &mut [f32], wet: f32) {
        // Keep the FIR history warm even when dry, so enabling the knob
        // mid-session is click-free.
        let dry: Vec<f32> = chunk.to_vec();
        let mut input = Vec::with_capacity(self.hist.len() + dry.len());
        input.extend_from_slice(&self.hist);
        input.extend_from_slice(&dry);
        if wet > 0.0 {
            for (i, c) in chunk.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (j, t) in self.taps.iter().enumerate() {
                    acc += t * input[i + TAPS - 1 - j];
                }
                // The FIR delays by TAPS/2; mix against equally delayed
                // dry so wet changes never phase-comb.
                let delayed_dry = input[i + TAPS / 2 - 1];
                *c = delayed_dry * (1.0 - wet) + acc * wet;
            }
        } else {
            // Bit-transparent bypass still pays the group delay so the
            // knob can move live without a timeline jump.
            for (i, c) in chunk.iter_mut().enumerate() {
                *c = input[i + TAPS / 2 - 1];
            }
        }
        let keep = input.len() - (TAPS - 1);
        self.hist = input[keep..].to_vec();
    }
}

/// Linear-phase FIR from a target magnitude (dB per FFT bin) via the
/// window method: inverse real FFT of the magnitude, Hann-windowed
/// around the centre tap.
fn design_fir(gain_db: &[f64]) -> Vec<f32> {
    let mut spec = vec![Complex64::new(0.0, 0.0); N_FFT];
    for k in 0..N_BINS {
        let mag = 10f64.powf(gain_db[k] / 20.0);
        spec[k] = Complex64::new(mag, 0.0);
        if k > 0 && k < N_FFT / 2 {
            spec[N_FFT - k] = Complex64::new(mag, 0.0);
        }
    }
    let mut planner = FftPlanner::new();
    planner.plan_fft_inverse(N_FFT).process(&mut spec);
    // Impulse response is real and centred at 0; rotate to the middle
    // and window to TAPS.
    let mut taps = vec![0f32; TAPS];
    for (i, t) in taps.iter_mut().enumerate() {
        let n = (i as isize - (TAPS / 2) as isize).rem_euclid(N_FFT as isize) as usize;
        let w = 0.5 - 0.5
            * (2.0 * std::f64::consts::PI * i as f64 / (TAPS - 1) as f64).cos();
        *t = (spec[n].re / N_FFT as f64 * w) as f32;
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                ((rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
                    - 0.5)
                    * 0.2
            })
            .collect()
    }

    /// Simple one-pole low-pass to colour a signal.
    fn lowpassed(x: &[f32], a: f32) -> Vec<f32> {
        let mut y = Vec::with_capacity(x.len());
        let mut s = 0f32;
        for &v in x {
            s += a * (v - s);
            y.push(s);
        }
        y
    }

    #[test]
    fn bypass_is_delay_only() {
        let x = noise(48_000, 7);
        let mut eq = ProfileEq::new(&x, 48_000.0);
        let mut y = x.clone();
        eq.process(&mut y, 0.0);
        let d = eq.latency();
        for i in 0..x.len() - d {
            assert_eq!(y[i + d], x[i], "bypass altered sample {i}");
        }
    }

    #[test]
    fn converges_towards_reference_colouration() {
        // Reference = bright noise; output = the same noise low-passed.
        // The EQ must recover most of the lost high band.
        let bright = noise(48_000 * 4, 1);
        let dull = lowpassed(&bright, 0.25);
        let mut eq = ProfileEq::new(&bright, 48_000.0);
        // ~16 s of observation = several rebuilds; the slew converges to
        // >80 % of the design.
        for _ in 0..4 {
            for c in dull.chunks(4_800) {
                eq.observe(c);
            }
        }
        let mut out = dull.clone();
        eq.process(&mut out, 1.0);
        let band = |y: &[f32], lo: f64, hi: f64| -> f64 {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(N_FFT);
            let mut acc = 0.0;
            let mut cnt = 0;
            for f in y.chunks_exact(N_FFT) {
                let mut b: Vec<Complex64> =
                    f.iter().map(|&s| Complex64::new(s as f64, 0.0)).collect();
                fft.process(&mut b);
                for k in 0..N_BINS {
                    let fr = k as f64 * RATE as f64 / N_FFT as f64;
                    if fr >= lo && fr < hi {
                        acc += b[k].norm_sqr();
                        cnt += 1;
                    }
                }
            }
            10.0 * (acc / cnt as f64).log10()
        };
        let deficit_before = band(&bright, 6_000.0, 11_000.0) - band(&dull, 6_000.0, 11_000.0);
        let deficit_after = band(&bright, 6_000.0, 11_000.0) - band(&out, 6_000.0, 11_000.0);
        println!("6-11k deficit: before {deficit_before:.1} dB, after {deficit_after:.1} dB");
        assert!(
            deficit_after < deficit_before - 5.0,
            "EQ did not recover the high band: {deficit_before:.1} -> {deficit_after:.1} dB"
        );
        // The clamp bounds how much a single stage may do.
        assert!(deficit_after > deficit_before - 2.0 - ProfileEq::CLAMP_DB);
    }

    #[test]
    fn silence_never_updates() {
        let x = noise(48_000, 3);
        let mut eq = ProfileEq::new(&x, 48_000.0);
        let before = eq.taps.clone();
        eq.observe(&vec![0.0; 48_000 * 4]);
        assert_eq!(eq.taps, before, "silence moved the coefficients");
    }

    #[test]
    fn chunked_equals_oneshot() {
        let x = noise(19_200, 9);
        let mk = || {
            let mut eq = ProfileEq::new(&noise(48_000, 1), 48_000.0);
            // Force a non-trivial response.
            for c in lowpassed(&noise(48_000 * 4, 2), 0.3).chunks(4_800) {
                eq.observe(c);
            }
            eq
        };
        let mut a = mk();
        let mut one = x.clone();
        a.process(&mut one, 1.0);
        let mut b = mk();
        let mut chunked = Vec::new();
        for c in x.chunks(1_536) {
            let mut c = c.to_vec();
            b.process(&mut c, 1.0);
            chunked.extend(c);
        }
        let d = one
            .iter()
            .zip(&chunked)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-6, "chunked mismatch {d}");
    }
}

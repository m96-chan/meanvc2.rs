//! Surgical needle-pulse suppressor for neural-decoder output.
//!
//! The X-VC SAC decoder occasionally emits a single needle pulse — a
//! ≲2.5 ms burst driven into the final `tanh` (|y| ≈ 1) that sits an
//! order of magnitude above the surrounding waveform (issue #42; the
//! stage probe shows the converter/prenet/codec features are smooth and
//! the spike is born inside the decoder, identically on CPU and CUDA).
//! It is a property of the whole re-encoded window, so overlapping
//! windows do not reproduce it — it cannot be cross-checked away, only
//! repaired in the time domain.
//!
//! Detection is deliberately conservative, so ordinary speech is never
//! touched (the lesson of the earlier blind inter-chunk declicker, which
//! false-positived on normal waveform slopes):
//!
//! - the local statistic is the **median** absolute sample over a ±8 ms
//!   context, which the needle itself cannot inflate (a 2.5 ms run is
//!   < 1/6 of the window);
//! - a sample is a needle candidate only if `|y| > ABS_FLOOR` (0.30)
//!   **and** `|y| > RATIO × median` (8× — natural glottal pulses in
//!   converted speech stay under ~5× their local median);
//! - only contiguous runs up to 2.5 ms are repaired; anything longer is
//!   real audio by definition and is left alone.
//!
//! Repair ATTENUATES the run to the local level with a tapered gain
//! dip (the eighth field recording pinned audible ticks to the earlier
//! linear-bridge repair: excising through quiet breathy content leaves
//! a notch that is itself a tick; a gain dip keeps the waveform
//! continuous, and a false positive merely softens a transient).
//!
//! The suppressor is streaming: [`NeedleGuard::process`] consumes a
//! chunk and returns the same number of samples delayed by the context
//! length, with state carried across chunks (chunked output equals
//! one-shot output exactly).

/// Streaming needle suppressor (see the module docs).
pub struct NeedleGuard {
    /// Context half-window (samples).
    ctx: usize,
    /// Max repairable run length (samples).
    max_run: usize,
    /// Isolation context length (samples).
    iso: usize,
    /// Pending samples: `[processed-context | unprocessed]`.
    buf: Vec<f32>,
    /// Number of needle runs repaired so far.
    pub repaired: u64,
}

/// Absolute amplitude floor for a needle candidate. The decoder emits
/// needles scaled with the local signal, down to ~0.12 in quiet speech
/// (sixth field recording); 0.10 catches those while staying above any
/// plausible noise-floor content.
const ABS_FLOOR: f32 = 0.10;
/// Candidate threshold as a multiple of the local median |y|. Natural
/// glottal peaks in converted speech stay under ~5x their local median;
/// field-recording needles measure 6.2–8x.
const RATIO: f32 = 6.0;
/// Floor for the local median, so a silent context (median ≈ 0, e.g.
/// right after the input gate opens) cannot make every voiced sample
/// look like an infinite-ratio needle.
const MED_FLOOR: f32 = 0.012;
/// Minimum context level for action on SMALL candidates: in
/// near-silence the guard's false-positive class (breath/mouth/
/// environment transients) is naked to the ear — even a 6 dB, 0.5 ms
/// dip reads as a soft knock (ninth/tenth field reports) — while a
/// small needle there is barely audible in the first place. The guard
/// therefore acts only where the voice masks the repair…
const VOICED_FLOOR: f32 = 0.018;
/// …EXCEPT for loud needles: a pulse this big is clearly audible even
/// against silence (live capture leaked a 0.9 needle in a quiet
/// passage when the context gate was unconditional), and removing a
/// loud pop from silence leaves nothing audible behind.
const LOUD_NEEDLE: f32 = 0.25;
/// Guard margin (gain-ramp length) around a detected run (samples at
/// 16 kHz): 1 ms ramps — the ninth field report heard the earlier
/// 0.5 ms ramps as a soft knock.
const MARGIN: usize = 16;
/// Isolation context checked on both sides of a run (seconds): a needle
/// is a lone pulse, so its neighbourhood stays well below the run peak.
/// Speech onsets fail this check — the next glottal cycle follows at a
/// comparable amplitude within 2 ms — and are left untouched.
const ISO_SECS: f32 = 0.002;
/// Neighbourhood-to-peak bound for the isolation check.
const ISO_RATIO: f32 = 0.65;

impl NeedleGuard {
    /// `sample_rate` is the rate of the processed stream (16 kHz for the
    /// X-VC decoder output).
    pub fn new(sample_rate: f32) -> Self {
        Self {
            ctx: (0.008 * sample_rate) as usize,
            // Action is restricted to needle-width runs (≤ 0.5 ms): the
            // decoder needle is a 3–5 sample pulse at 16 kHz, while
            // breath/consonant transients — the guard's false-positive
            // class in quiet passages, audible as soft knocks when
            // touched — are broader and now pass untouched.
            max_run: (0.0005 * sample_rate) as usize,
            iso: (ISO_SECS * sample_rate) as usize,
            buf: Vec::new(),
            repaired: 0,
        }
    }

    /// Latency introduced by the guard, in samples: median context plus
    /// enough slack that a run touching the emit horizon is always fully
    /// judged before its samples leave the buffer.
    pub fn latency(&self) -> usize {
        self.ctx + self.max_run + MARGIN
    }

    /// Median absolute value over `[c-ctx, c+ctx]` clamped to the buffer.
    fn local_median(&self, c: usize) -> f32 {
        let lo = c.saturating_sub(self.ctx);
        let hi = (c + self.ctx + 1).min(self.buf.len());
        let mut mag: Vec<f32> = self.buf[lo..hi].iter().map(|s| s.abs()).collect();
        let mid = mag.len() / 2;
        mag.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        mag[mid]
    }

    /// Feeds `chunk` and returns the next `chunk.len()` processed
    /// samples (delayed by [`Self::latency`]; the first call is padded
    /// with leading zeros).
    pub fn process(&mut self, chunk: &[f32]) -> Vec<f32> {
        if self.buf.is_empty() {
            // Prime the delay line so output length always matches input.
            self.buf = vec![0.0; self.latency()];
        }
        self.buf.extend_from_slice(chunk);

        // Candidates are scanned wherever the median context is complete
        // (up to buf.len() - ctx). The emit horizon holds back a further
        // max_run + MARGIN samples so any run overlapping emitted audio
        // has already been fully judged (and repaired) before it leaves.
        let emit = self
            .buf
            .len()
            .saturating_sub(self.latency())
            .min(chunk.len());
        let scan_end = self.buf.len().saturating_sub(self.ctx);
        let mut i = 0;
        while i < scan_end {
            let y = self.buf[i].abs();
            if y < ABS_FLOOR {
                i += 1;
                continue;
            }
            let raw_med = self.local_median(i);
            let med = raw_med.max(MED_FLOOR);
            if (raw_med < VOICED_FLOOR && y < LOUD_NEEDLE) || y <= RATIO * med {
                i += 1;
                continue;
            }
            // Grow the run while samples stay above a relaxed bound.
            let run_thr = (RATIO * 0.5) * med;
            let mut j = i + 1;
            while j < scan_end && j - i <= self.max_run && self.buf[j].abs() > run_thr {
                j += 1;
            }
            let iso = self.iso;
            let run_peak = self.buf[i..j].iter().fold(0f32, |m, s| m.max(s.abs()));
            let side_peak = |lo: usize, hi: usize| -> f32 {
                self.buf[lo.min(self.buf.len())..hi.min(self.buf.len())]
                    .iter()
                    .fold(0f32, |m, s| m.max(s.abs()))
            };
            let left = side_peak(i.saturating_sub(MARGIN + iso), i.saturating_sub(MARGIN));
            let right = side_peak(j + MARGIN, j + MARGIN + iso);
            let isolated = left < ISO_RATIO * run_peak && right < ISO_RATIO * run_peak;
            if isolated && j - i <= self.max_run && j < scan_end {
                let lo = i.saturating_sub(MARGIN);
                let hi = (j + MARGIN).min(self.buf.len() - 1);
                let (a, b) = (self.buf[lo], self.buf[hi]);
                let n = hi - lo;
                for (k, s) in self.buf[lo..=hi].iter_mut().enumerate() {
                    let w = k as f32 / n as f32;
                    *s = a * (1.0 - w) + b * w;
                }
                self.repaired += 1;
                i = hi + 1;
            } else {
                // Too long, not isolated (a speech onset), or context
                // still incomplete: skip past it untouched.
                i = j;
            }
        }

        let out: Vec<f32> = self.buf.drain(..emit).collect();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vowel-like test signal: 300 Hz with harmonics, rms ≈ 0.1.
    fn vowel(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                let f = 300.0;
                0.09 * (2.0 * std::f32::consts::PI * f * t).sin()
                    + 0.05 * (4.0 * std::f32::consts::PI * f * t).sin()
                    + 0.03 * (6.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect()
    }

    #[test]
    fn transparent_on_clean_speechlike_audio() {
        let x = vowel(16_000);
        let mut g = NeedleGuard::new(16_000.0);
        let y = g.process(&x);
        assert_eq!(y.len(), x.len());
        let d = g.latency();
        // Output is the input delayed by `latency()`, bit-exact.
        for i in 0..x.len() - d {
            assert_eq!(y[i + d], x[i], "sample {i} modified on clean audio");
        }
        assert_eq!(g.repaired, 0);
    }

    #[test]
    fn removes_decoder_needle() {
        let mut x = vowel(16_000);
        // A 7-sample tanh-saturated needle like the SAC decoder emits.
        for k in 0..7 {
            x[8_000 + k] = if k % 2 == 0 { 0.95 } else { -0.9 };
        }
        let mut g = NeedleGuard::new(16_000.0);
        let y = g.process(&x);
        let d = g.latency();
        let peak = y[8_000 + d - 32..8_000 + d + 32]
            .iter()
            .fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak < 0.3, "needle survived: local peak {peak}");
        assert_eq!(g.repaired, 1);
    }

    #[test]
    fn keeps_natural_glottal_peaks() {
        // A sharp but natural pulse: 4x the local median is loud speech,
        // not a needle.
        let mut x = vowel(16_000);
        let med: f32 = {
            let mut m: Vec<f32> = x.iter().map(|s| s.abs()).collect();
            m.sort_by(|a, b| a.total_cmp(b));
            m[m.len() / 2]
        };
        for k in 0..12 {
            x[8_000 + k] += 3.5 * med * (std::f32::consts::PI * k as f32 / 12.0).sin();
        }
        let expected = x.clone();
        let mut g = NeedleGuard::new(16_000.0);
        let y = g.process(&x);
        let d = g.latency();
        for i in 7_900..8_100 {
            assert_eq!(y[i + d], expected[i], "natural pulse modified at {i}");
        }
    }

    #[test]
    fn removes_small_needle_in_quiet_speech() {
        // Sixth field recording: needles scale down with the local
        // signal — a 0.2 pulse over a soft-voiced passage still clicks.
        let mut x: Vec<f32> = vowel(16_000).iter().map(|s| s * 0.5).collect();
        for k in 0..5 {
            x[8_000 + k] = 0.2;
        }
        let mut g = NeedleGuard::new(16_000.0);
        let y = g.process(&x);
        let d = g.latency();
        let peak = y[8_000 + d - 32..8_000 + d + 32]
            .iter()
            .fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak < 0.13, "small needle survived: local peak {peak}");
        assert_eq!(g.repaired, 1);
    }

    #[test]
    fn silent_context_is_never_touched() {
        // Tenth field report: repairs in near-silence are naked to the
        // ear (soft keyboard-like knocks) while needles there are barely
        // audible — below VOICED_FLOOR the guard must not act at all.
        let mut x: Vec<f32> = vowel(16_000).iter().map(|s| s * 0.1).collect();
        for k in 0..5 {
            x[8_000 + k] = 0.12;
        }
        let expected = x.clone();
        let mut g = NeedleGuard::new(16_000.0);
        let y = g.process(&x);
        let d = g.latency();
        assert_eq!(g.repaired, 0, "guard acted in near-silence");
        for i in 7_900..8_100 {
            assert_eq!(y[i + d], expected[i]);
        }
    }

    #[test]
    fn keeps_genuine_plosive_attack() {
        // A plosive: near-instant attack followed by a sustained burst.
        // Longer than any needle (> 2.5 ms) and not isolated — must be
        // bit-untouched even at high amplitude over a quiet context.
        let mut x = vec![0.001f32; 16_000];
        for k in 0..160 {
            // 10 ms decaying burst at 4 kHz over near-silence.
            let t = k as f32;
            x[8_000 + k] = 0.6 * (1.571 * t).sin() * (-t / 80.0).exp();
        }
        let expected = x.clone();
        let mut g = NeedleGuard::new(16_000.0);
        let y = g.process(&x);
        let d = g.latency();
        assert_eq!(g.repaired, 0, "plosive attack was repaired");
        for i in 7_900..8_300 {
            assert_eq!(y[i + d], expected[i], "plosive modified at {i}");
        }
    }

    #[test]
    fn chunked_equals_oneshot() {
        let mut x = vowel(32_000);
        for k in 0..7 {
            x[10_000 + k] = 0.92;
            x[20_011 + k] = -0.94;
        }
        let mut g1 = NeedleGuard::new(16_000.0);
        let one = g1.process(&x);
        let mut g2 = NeedleGuard::new(16_000.0);
        let mut chunked = Vec::new();
        for c in x.chunks(3_840) {
            chunked.extend(g2.process(c));
        }
        assert_eq!(one.len(), chunked.len());
        let d = one
            .iter()
            .zip(&chunked)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-7, "chunked/oneshot mismatch {d}");
        assert_eq!(g1.repaired, 2);
        assert_eq!(g2.repaired, 2);
    }

    #[test]
    fn needle_straddling_chunk_boundary_is_repaired() {
        let mut x = vowel(16_000);
        // Needle right at a 3840-sample chunk edge.
        for k in 0..7 {
            x[3_837 + k] = 0.93;
        }
        let mut g = NeedleGuard::new(16_000.0);
        let mut y = Vec::new();
        for c in x.chunks(3_840) {
            y.extend(g.process(c));
        }
        let d = g.latency();
        let peak = y[3_837 + d - 16..3_844 + d + 16]
            .iter()
            .fold(0f32, |m, s| m.max(s.abs()));
        assert!(peak < 0.3, "boundary needle survived: {peak}");
        assert_eq!(g.repaired, 1);
    }
}

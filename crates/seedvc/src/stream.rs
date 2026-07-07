//! Streaming driver for the live demo, following the official
//! real-time GUI's scheme: a sliding source context is re-converted
//! every block and only the tail block is emitted, joined with a short
//! raised-cosine crossfade. There is no per-window decoder pathology to
//! defend against here (issue #49: the BigVGAN line has no needle
//! artifacts), so no cross-check / needle-guard layers exist.
//!
//! Rates: `push` takes 16 kHz mic samples (the demo's world);
//! internally the mel/vocoder domain is 22 050 Hz; `emit` returns
//! 48 kHz output ready for the demo's post chain.

use candle_core::{Device, Tensor};

use crate::pipeline::{resample, SeedVcEngine};
use crate::Result;

/// Streaming parameters (16 kHz sample units where applicable).
#[derive(Clone, Copy)]
pub struct StreamConfig {
    /// New input consumed per hop (16 k samples). 5120 = 320 ms.
    pub block: usize,
    /// Whisper (content-encoder) left context (16 k samples). 40 000 =
    /// 2.5 s like the official GUI's `extra_time_ce`.
    pub context: usize,
    /// DiT/vocoder left context (16 k samples). 8 000 = 0.5 s like the
    /// official GUI's `extra_time`: only this short window flows
    /// through the CFM and BigVGAN every hop — the long context feeds
    /// whisper alone. (Re-converting the full 2.5 s tripled the DiT
    /// sequence and blew the real-time budget.)
    pub dit_context: usize,
    /// Crossfade at block joins, in 22 050 Hz samples (~40 ms).
    pub crossfade_22k: usize,
    /// CFM steps / cfg rate. 8 steps (offline default 10) keeps the
    /// step time ~25 % under the block budget on CUDA.
    pub steps: usize,
    pub cfg_rate: f64,
    /// Reference prompt cap in seconds: the prompt occupies the DiT
    /// sequence every step of every hop, so a long reference dominates
    /// compute. 4 s of timbre prompt is plenty for conditioning.
    pub max_prompt_s: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            block: 5_120,
            context: 40_000,
            dit_context: 8_000,
            crossfade_22k: 882,
            steps: 8,
            cfg_rate: 0.7,
            max_prompt_s: 4.0,
        }
    }
}

/// Reference conditions, computed once per session.
pub struct Reference {
    prompt_condition: Tensor,
    mel2: Tensor,
    style: Tensor,
    ref_mel_frames: usize,
}

pub struct SeedVcStream<'a> {
    engine: &'a SeedVcEngine,
    cfg: StreamConfig,
    reference: Reference,
    /// Sliding 16 kHz source buffer (up to context + block).
    buf: Vec<f32>,
    /// Pending un-consumed input (16 k samples).
    pending: usize,
    /// Crossfade tail of the previous emission (22 050 Hz).
    tail: Vec<f32>,
    /// Deterministic per-stream noise counter (fresh gaussian per hop
    /// via candle's seeded device RNG would break CUDA/CPU parity, so
    /// the stream carries its own tiny xorshift for reproducibility).
    rng: u64,
}

impl SeedVcEngine {
    /// Precomputes the reference conditions and opens a stream.
    pub fn stream(&self, ref_22k: &[f32], cfg: StreamConfig) -> Result<SeedVcStream<'_>> {
        let cap = (cfg.max_prompt_s * 22_050.0) as usize;
        let ref_22k = if ref_22k.len() > cap {
            &ref_22k[..cap]
        } else {
            ref_22k
        };
        let ref16 = resample(ref_22k, 22_050, 16_000);
        let s_ori = self.whisper_features(&ref16)?;
        let mel2 = self.mel22(ref_22k)?;
        let frames = mel2.dim(2)?;
        let prompt_condition = self.regulate(&s_ori, frames)?;
        let fbank = self.ref_fbank(&ref16);
        let style = self.campplus_embed(&fbank)?;
        Ok(SeedVcStream {
            engine: self,
            cfg,
            reference: Reference {
                prompt_condition,
                mel2,
                style,
                ref_mel_frames: frames,
            },
            buf: Vec::new(),
            pending: 0,
            tail: Vec::new(),
            rng: 0x5eed_5eed_5eed_5eed,
        })
    }
}

impl SeedVcStream<'_> {
    /// Appends 16 kHz input samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
        self.pending += samples.len();
        let keep = self.cfg.context + self.cfg.block.max(self.pending);
        if self.buf.len() > keep {
            let drop = self.buf.len() - keep;
            self.buf.drain(..drop);
        }
    }

    /// Whether a full block is buffered.
    pub fn ready(&self) -> bool {
        self.pending >= self.cfg.block
    }

    fn next_noise(&mut self, shape: (usize, usize, usize), dev: &Device) -> Result<Tensor> {
        // xorshift64* gaussian via Box-Muller — deterministic and
        // device-independent.
        let n = shape.0 * shape.1 * shape.2;
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            let mut next = || {
                self.rng ^= self.rng >> 12;
                self.rng ^= self.rng << 25;
                self.rng ^= self.rng >> 27;
                (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64
                    / (1u64 << 53) as f64
            };
            let (u1, u2) = (next().max(1e-12), next());
            let r = (-2.0 * u1.ln()).sqrt();
            let th = 2.0 * std::f64::consts::PI * u2;
            v.push((r * th.cos()) as f32);
            if v.len() < n {
                v.push((r * th.sin()) as f32);
            }
        }
        Ok(Tensor::from_vec(v, shape, dev)?)
    }

    /// Converts the next block; returns 48 kHz samples covering exactly
    /// `block / 16000` seconds (crossfaded with the previous emission).
    pub fn step(&mut self) -> Result<Option<Vec<f32>>> {
        if !self.ready() {
            return Ok(None);
        }
        self.pending -= self.cfg.block;

        // Long window feeds whisper; only the short tail window flows
        // through the regulator / CFM / vocoder.
        let win16 = &self.buf[self.buf.len().saturating_sub(self.cfg.context + self.cfg.block)..];
        let s_alt = self.engine.whisper_features(win16)?;
        let short_len = (self.cfg.dit_context + self.cfg.block).min(win16.len());
        let win16_short = &win16[win16.len() - short_len..];
        let win22 = resample(win16_short, 16_000, 22_050);
        let mel = self.engine.mel22(&win22)?;
        let t_win = mel.dim(2)?;
        // Whisper features are one per 320 input samples (+1): slice
        // the tail that covers the short window.
        let n_feat = s_alt.dim(1)?;
        let keep = (short_len / 320 + 1).min(n_feat);
        let s_short = s_alt.narrow(1, n_feat - keep, keep)?;
        let cond = self.engine.regulate(&s_short, t_win)?;
        let cat = Tensor::cat(&[&self.reference.prompt_condition, &cond], 1)?;
        let t_ref = self.reference.ref_mel_frames;
        let noise = self.next_noise((1, 80, t_ref + t_win), cat.device())?;
        let vc_mel = self.engine.cfm_inference(
            &cat,
            &self.reference.mel2,
            &self.reference.style,
            &noise,
            self.cfg.steps,
            self.cfg.cfg_rate,
        )?;
        let vc_mel = vc_mel.narrow(2, t_ref, t_win)?;
        let wave22: Vec<f32> = self.engine.vocode(&vc_mel)?;

        // Emit the tail block (+ crossfade lead-in) in the 22 050 domain.
        // Mel framing renders slightly less than the window (up to
        // ~1024 samples of tail), so the rendered timeline runs a
        // constant few tens of ms behind the input — uniform across
        // hops once the sliding window has its steady length. Early
        // hops can be shorter than a full block; pad the front with
        // zeros (a one-off ~35 ms of leading silence at stream start).
        let block22 = self.cfg.block * 441 / 320;
        let xf = self.cfg.crossfade_22k.min(block22);
        let need = block22 + xf;
        let mut out22 = if wave22.len() >= need {
            wave22[wave22.len() - need..].to_vec()
        } else {
            let mut v = vec![0.0; need - wave22.len()];
            v.extend_from_slice(&wave22);
            v
        };
        if self.tail.len() == xf {
            for i in 0..xf {
                let w = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / xf as f64).cos();
                out22[i] = self.tail[i] * (1.0 - w as f32) + out22[i] * w as f32;
            }
        }
        // Hold back the crossfade tail for the next block: the emitted
        // region is [xf, xf + block22), the lead-in [0, xf) was blended.
        let emitted: Vec<f32> = out22[..block22].to_vec();
        self.tail = out22[block22..].to_vec();

        Ok(Some(resample(&emitted, 22_050, 48_000)))
    }
}

//! Streaming driver, following the same scheme validated for
//! `SeedVcStream` (issue #50/#74 recon): a sliding source window is
//! re-converted every hop and only the tail block is emitted, SOLA-
//! spliced against the previous emission's held-back tail. Unlike
//! Seed-VC, Vevo-Timbre has no separate "long context for content,
//! short context for the diffusion model" split — HuBERT, RepCodec,
//! and the CFM all run on the same window every hop, since there is
//! no length-regulation step (content-style codes feed `cond_emb`
//! directly, frame-for-frame with the mel grid).
//!
//! Rates: `push` takes 16 kHz mic samples; `emit` returns 48 kHz
//! output (Vocos synthesizes at 24 kHz internally, resampled up).
//!
//! ## Live status: not yet real-time on this port (candle CUDA)
//!
//! The #72 recon's "80 ms/block @ 6 steps" number was measured against
//! the *official PyTorch* implementation — it does **not** carry over
//! to this port. Per-hop profiling on an RTX 5090 (`cargo run
//! --release -p vevo --features cuda --example stream_probe`) puts a
//! 320 ms block at **~0.56 s wall time regardless of CFM step count**
//! (2 vs 6 steps measured within noise of each other): content-style
//! extraction ~70 ms, `reverse_diffusion` ~70 ms, **`Vocos::synthesize`
//! ~420 ms** for the 30-layer/dim-1024 ConvNeXt backbone on ~40 mel
//! frames. The CFM loop is *not* the bottleneck here (unlike Seed-VC's
//! DiT), so lowering `steps` doesn't reclaim the deficit — the fix is
//! in the vocoder path (candle's CUDA backend appears to pay a fixed
//! per-kernel-launch cost across the 30-block ConvNeXt stack that
//! PyTorch's cuDNN/fused path amortizes away). Tracked as a follow-up
//! tuning item alongside #44/#48; offline `VevoEngine::inference_fm`
//! is unaffected (no latency budget to miss).

use candle_core::{Device, Tensor};

use crate::pipeline::{resample, VevoEngine};
use crate::Result;

/// Streaming parameters (16 kHz sample units unless noted).
#[derive(Clone, Copy)]
pub struct StreamConfig {
    /// New input consumed per hop. 5_120 = 320 ms.
    pub block: usize,
    /// Left context re-processed alongside each block. 8_000 = 0.5 s,
    /// the window the #72 recon validated for content quality (see
    /// the module docs for this port's actual measured latency).
    pub context: usize,
    /// Crossfade at block joins, in 24 kHz samples (~40 ms).
    pub crossfade_24k: usize,
    /// SOLA search range (24 kHz samples, ~10 ms): each hop's render
    /// is spliced at the offset that best phase-aligns with the
    /// previous tail (same rationale as `SeedVcStream`: adjacent
    /// diffusion renders agree in envelope but not phase).
    pub sola_search_24k: usize,
    /// CFM steps. Offline default is 32; 6 keeps live quality close to
    /// that while the vocoder path (not the CFM loop) is the actual
    /// bottleneck on this port — see the module docs.
    pub steps: usize,
    /// Reference prompt cap in seconds — the prompt occupies the CFM
    /// sequence on every step of every hop.
    pub max_prompt_s: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            block: 5_120,
            context: 8_000,
            crossfade_24k: 960,
            sola_search_24k: 240,
            steps: 6,
            max_prompt_s: 4.0,
        }
    }
}

struct Reference {
    codes: Tensor,
    mel: Tensor,
}

pub struct VevoStream<'a> {
    engine: &'a VevoEngine,
    cfg: StreamConfig,
    reference: Reference,
    /// Sliding 16 kHz source buffer (up to `context + block`).
    buf: Vec<f32>,
    pending: usize,
    /// Crossfade tail of the previous emission (24 kHz).
    tail: Vec<f32>,
    rng: u64,
    /// Fixed CFM noise, cached per shape — fresh noise per hop
    /// decorrelates the texture at block joints (same field-tested
    /// rationale as `SeedVcStream`).
    noise_cache: Option<Tensor>,
}

impl VevoEngine {
    /// Precomputes the reference conditions and opens a stream.
    pub fn stream(&self, ref_24k: &[f32], cfg: StreamConfig) -> Result<VevoStream<'_>> {
        let cap = (cfg.max_prompt_s * 24_000.0) as usize;
        let ref_24k = if ref_24k.len() > cap {
            &ref_24k[..cap]
        } else {
            ref_24k
        };
        let ref16 = resample(ref_24k, 24_000, 16_000);
        let codes = self.content_style_codes(&ref16)?;
        let mel = self.mel_feature(ref_24k)?;
        Ok(VevoStream {
            engine: self,
            cfg,
            reference: Reference { codes, mel },
            buf: Vec::new(),
            pending: 0,
            tail: Vec::new(),
            rng: 0x5eed_5eed_5eed_5eed,
            noise_cache: None,
        })
    }
}

impl VevoStream<'_> {
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

    pub fn ready(&self) -> bool {
        self.pending >= self.cfg.block
    }

    fn next_noise(&mut self, shape: (usize, usize, usize), dev: &Device) -> Result<Tensor> {
        // xorshift64* gaussian via Box-Muller — deterministic and
        // device-independent (matches SeedVcStream's rationale).
        let n = shape.0 * shape.1 * shape.2;
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            let mut next = || {
                self.rng ^= self.rng >> 12;
                self.rng ^= self.rng << 25;
                self.rng ^= self.rng >> 27;
                (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
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

    /// Converts the next block; returns 48 kHz samples covering
    /// `block / 16000` seconds (crossfaded with the previous emission).
    pub fn step(&mut self) -> Result<Option<Vec<f32>>> {
        if !self.ready() {
            return Ok(None);
        }
        self.pending -= self.cfg.block;

        let win16 = &self.buf[self
            .buf
            .len()
            .saturating_sub(self.cfg.context + self.cfg.block)..];
        let win_codes = self.engine.content_style_codes(win16)?;
        let codes = Tensor::cat(&[&self.reference.codes, &win_codes], 1)?;
        let cond = self.engine.fmt.cond_embed(&codes)?;

        let prompt_len = self.reference.mel.dim(1)?;
        let target_len = codes.dim(1)? - prompt_len;
        let want = (1, target_len, self.reference.mel.dim(2)?);
        let noise = match &self.noise_cache {
            Some(n) if n.dims3()? == want => n.clone(),
            _ => {
                let n = self.next_noise(want, cond.device())?;
                self.noise_cache = Some(n.clone());
                n
            }
        };
        let mel = self.engine.fmt.reverse_diffusion(
            &cond,
            &self.reference.mel,
            noise,
            self.cfg.steps,
            1.0,
            0.75,
        )?;
        let wave24: Vec<f32> = self.engine.vocos.synthesize(&mel)?;

        // Emit the tail block (+ crossfade lead-in), 24 kHz domain.
        // block16 -> block24k samples: 24000/16000 = 3/2.
        let block24 = self.cfg.block * 3 / 2;
        let xf = self.cfg.crossfade_24k.min(block24);
        let search = self.cfg.sola_search_24k;
        let need = block24 + xf + search;
        let mut out24 = if wave24.len() >= need {
            wave24[wave24.len() - need..].to_vec()
        } else {
            let mut v = vec![0.0; need - wave24.len()];
            v.extend_from_slice(&wave24);
            v
        };

        // SOLA: pick the splice offset whose lead-in best correlates
        // with the previous tail, then crossfade there.
        let k_best = if self.tail.len() == xf && search > 0 {
            let mut best = (f32::MIN, 0usize);
            for k in 0..=search {
                let seg = &out24[k..k + xf];
                let (mut dot, mut en) = (0f32, 1e-9f32);
                for (t, s) in self.tail.iter().zip(seg) {
                    dot += t * s;
                    en += s * s;
                }
                let score = dot / en.sqrt();
                if score > best.0 {
                    best = (score, k);
                }
            }
            best.1
        } else {
            0
        };
        if self.tail.len() == xf {
            for i in 0..xf {
                let w = 0.5 - 0.5 * (std::f64::consts::PI * i as f64 / xf as f64).cos();
                out24[k_best + i] = self.tail[i] * (1.0 - w as f32) + out24[k_best + i] * w as f32;
            }
        }
        let end = (k_best + block24).min(out24.len());
        let mut emitted: Vec<f32> = out24[k_best..end].to_vec();
        emitted.resize(block24, *emitted.last().unwrap_or(&0.0));
        let t_end = (k_best + block24 + xf).min(out24.len());
        self.tail = out24[k_best + block24..t_end].to_vec();
        self.tail.resize(xf, 0.0);

        Ok(Some(resample(&emitted, 24_000, 48_000)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn ckpt_dir() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
        let need = [
            "vevo_hubert.safetensors",
            "vevo_hubert_stats.safetensors",
            "vevo_repcodec.safetensors",
            "vevo_fmt.safetensors",
            "vevo_vocos.safetensors",
        ];
        need.iter().all(|f| path.join(f).exists()).then_some(path)
    }

    #[test]
    fn stream_emits_expected_sample_counts() {
        let Some(ckpt) = ckpt_dir() else { return };
        let dev = Device::Cpu;
        let engine = VevoEngine::load(&ckpt, &dev).unwrap();
        // Short synthetic reference and source (silence is fine for a
        // shape/plumbing check; golden parity is covered by pipeline.rs).
        let ref24: Vec<f32> = vec![0.0; 24_000 * 2];
        let cfg = StreamConfig {
            steps: 2,
            ..StreamConfig::default()
        };
        let mut stream = engine.stream(&ref24, cfg).unwrap();
        let src16: Vec<f32> = vec![0.0; 16_000];
        stream.push(&src16);
        let mut total = 0usize;
        while stream.ready() {
            if let Some(out) = stream.step().unwrap() {
                total += out.len();
            }
        }
        let expected_hops = 16_000 / cfg.block;
        let expected_block48 = cfg.block * 3; // 16k -> 48k
        assert_eq!(total, expected_hops * expected_block48);
    }

    /// Real audio smoke test (rule 2): streams an actual reference/
    /// source pair and checks the output is finite with plausible
    /// energy — not a golden-parity check (the offline pipeline
    /// already covers that), just "does this run and produce speech-
    /// shaped output, not silence or NaNs."
    #[test]
    fn stream_real_audio_is_finite_and_voiced() {
        let Some(ckpt) = ckpt_dir() else { return };
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
        let (Ok(mut ref_r), Ok(mut src_r)) = (
            hound::WavReader::open(root.join("ref_stage1_48k.wav")),
            hound::WavReader::open(root.join("ref_trimmed.wav")),
        ) else {
            return;
        };
        let read16 = |r: &mut hound::WavReader<std::io::BufReader<std::fs::File>>| -> Vec<f32> {
            let spec = r.spec();
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Int => {
                    let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    r.samples::<i32>()
                        .map(|s| s.unwrap() as f32 / scale)
                        .collect()
                }
                hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
            };
            if spec.sample_rate == 16_000 {
                samples
            } else {
                resample(&samples, spec.sample_rate as usize, 16_000)
            }
        };
        let ref_spec = ref_r.spec();
        let ref16 = read16(&mut ref_r);
        let ref24 = resample(&ref16, 16_000, 24_000);
        let _ = ref_spec;
        let src16 = read16(&mut src_r);

        let dev = Device::Cpu;
        let engine = VevoEngine::load(&ckpt, &dev).unwrap();
        let cfg = StreamConfig {
            steps: 4,
            ..StreamConfig::default()
        };
        let mut stream = engine.stream(&ref24, cfg).unwrap();
        stream.push(&src16[..src16.len().min(cfg.context + cfg.block * 3)]);

        let mut out = Vec::new();
        while stream.ready() {
            if let Some(chunk) = stream.step().unwrap() {
                out.extend(chunk);
            }
        }
        assert!(!out.is_empty(), "no output produced");
        assert!(
            out.iter().all(|s| s.is_finite()),
            "non-finite sample in output"
        );
        let rms = (out.iter().map(|s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 1e-4, "output looks silent, rms={rms}");
    }
}

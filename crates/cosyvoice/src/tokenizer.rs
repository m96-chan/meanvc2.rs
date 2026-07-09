//! FSQ supervised speech tokenizer (CosyVoice2 `speech_tokenizer_v2`, 25 Hz).
//!
//! Whisper-style encoder with FSMN memory blocks and rotary attention,
//! finished by an FSQ head (8 dims × 3 levels ⇒ vocab 3⁸ = 6561). CosyVoice 2
//! paper §2.2; architecture follows the Apache-2.0 torch re-implementation
//! in xingchensong/S3Tokenizer (`model_v2.py`), weights extracted from the
//! official ONNX by `tools/convert_cosyvoice.py`.
//!
//! Input: whisper 128-mel `[1, 128, T]` (100 Hz) → conv ×2 stride-2 → 25 Hz
//! tokens `[1, T/4]` (u32). Full (non-causal) self-attention — chunked live
//! use must re-tokenize a sliding window (see `stream.rs`).

use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{
    conv1d, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, LayerNorm, Linear, Module,
    VarBuilder,
};
use vc_core::Result;

const N_STATE: usize = 1280;
const N_HEAD: usize = 20;
const HEAD_DIM: usize = 64;
const N_LAYER: usize = 6;
const FSMN_KERNEL: usize = 31;

struct FsmnAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    fsmn: Tensor, // [1280, 1, 31] depthwise kernel
}

impl FsmnAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            query: linear(N_STATE, N_STATE, vb.pp("query"))?,
            key: linear_no_bias(N_STATE, N_STATE, vb.pp("key"))?,
            value: linear(N_STATE, N_STATE, vb.pp("value"))?,
            out: linear(N_STATE, N_STATE, vb.pp("out"))?,
            fsmn: vb.get((N_STATE, 1, FSMN_KERNEL), "fsmn_block.weight")?,
        })
    }

    /// `x`: [1, T, 1280]; `cos`/`sin`: [T, 64].
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.query.forward(x)?.reshape((b, t, N_HEAD, HEAD_DIM))?;
        let k = self.key.forward(x)?.reshape((b, t, N_HEAD, HEAD_DIM))?;
        let v = self.value.forward(x)?.reshape((b, t, N_HEAD, HEAD_DIM))?;

        let rope = |x: &Tensor| -> Result<Tensor> {
            // rotate-half with duplicated cos/sin (GPT-NeoX layout)
            let l = x.narrow(D::Minus1, 0, HEAD_DIM / 2)?;
            let r = x.narrow(D::Minus1, HEAD_DIM / 2, HEAD_DIM / 2)?;
            let rot = Tensor::cat(&[&r.neg()?, &l], D::Minus1)?;
            let c = cos.reshape((1, t, 1, HEAD_DIM))?;
            let s = sin.reshape((1, t, 1, HEAD_DIM))?;
            Ok(x.broadcast_mul(&c)?.add(&rot.broadcast_mul(&s)?)?)
        };
        let q = rope(&q)?;
        let k = rope(&k)?;

        // FSMN memory over v: depthwise conv k31, same padding, + residual
        let v_flat = v.reshape((b, t, N_STATE))?;
        let vt = v_flat.transpose(1, 2)?.contiguous()?; // [b, 1280, t]
        let pad_l = (FSMN_KERNEL - 1) / 2;
        let pad_r = FSMN_KERNEL - 1 - pad_l;
        let vt = vt.pad_with_zeros(2, pad_l, pad_r)?;
        let mem = vt.conv1d(&self.fsmn, 0, 1, 1, N_STATE)?; // groups = channels
        let fsm = mem.transpose(1, 2)?.add(&v_flat)?;

        let scale = (HEAD_DIM as f64).powf(-0.25);
        let q = (q.permute((0, 2, 1, 3))?.contiguous()? * scale)?;
        let k = (k.permute((0, 2, 3, 1))?.contiguous()? * scale)?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;
        let qk = q.matmul(&k)?;
        let w = softmax(&qk, D::Minus1)?;
        let wv = w.matmul(&v)?; // [b, h, t, d]
        let wv = wv.permute((0, 2, 1, 3))?.reshape((b, t, N_STATE))?;
        Ok(self.out.forward(&wv)?.add(&fsm)?)
    }
}

struct Block {
    attn: FsmnAttention,
    attn_ln: LayerNorm,
    mlp0: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

impl Block {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: FsmnAttention::load(vb.pp("attn"))?,
            attn_ln: layer_norm(N_STATE, 1e-5, vb.pp("attn_ln"))?,
            mlp0: linear(N_STATE, N_STATE * 4, vb.pp("mlp.0"))?,
            mlp2: linear(N_STATE * 4, N_STATE, vb.pp("mlp.2"))?,
            mlp_ln: layer_norm(N_STATE, 1e-5, vb.pp("mlp_ln"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.attn.forward(&self.attn_ln.forward(x)?, cos, sin)?)?;
        let h = self.mlp0.forward(&self.mlp_ln.forward(&x)?)?.gelu_erf()?;
        Ok(x.add(&self.mlp2.forward(&h)?)?)
    }
}

/// The full tokenizer: conv front-end + 6 FSMN blocks + FSQ head.
pub struct SpeechTokenizer {
    conv1: Conv1d,
    conv2: Conv1d,
    blocks: Vec<Block>,
    project_down: Linear,
    device: Device,
}

impl SpeechTokenizer {
    pub fn load(vb: VarBuilder, device: &Device) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let enc = vb.pp("encoder");
        let mut blocks = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            blocks.push(Block::load(enc.pp(format!("blocks.{i}")))?);
        }
        Ok(Self {
            conv1: conv1d(128, N_STATE, 3, cfg, enc.pp("conv1"))?,
            conv2: conv1d(N_STATE, N_STATE, 3, cfg, enc.pp("conv2"))?,
            blocks,
            project_down: linear(N_STATE, 8, vb.pp("quantizer._codebook.project_down"))?,
            device: device.clone(),
        })
    }

    /// `mel`: whisper 128-mel `[1, 128, T]` → FSQ token ids `[1, T/4]` (u32).
    pub fn tokenize(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?.gelu_erf()?;
        let x = self.conv2.forward(&x)?.gelu_erf()?;
        let x = x.transpose(1, 2)?.contiguous()?; // [1, T', 1280]
        let t = x.dim(1)?;

        // rotary tables (theta 10000, per-head dim 64, duplicated halves)
        let half = HEAD_DIM / 2;
        let mut cos = vec![0f32; t * HEAD_DIM];
        let mut sin = vec![0f32; t * HEAD_DIM];
        for pos in 0..t {
            for i in 0..half {
                let freq = 1.0f64 / 10000f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
                let ang = (pos as f64 * freq) as f32;
                cos[pos * HEAD_DIM + i] = ang.cos();
                cos[pos * HEAD_DIM + half + i] = ang.cos();
                sin[pos * HEAD_DIM + i] = ang.sin();
                sin[pos * HEAD_DIM + half + i] = ang.sin();
            }
        }
        let cos = Tensor::from_vec(cos, (t, HEAD_DIM), &self.device)?;
        let sin = Tensor::from_vec(sin, (t, HEAD_DIM), &self.device)?;

        let mut x = x;
        for b in &self.blocks {
            x = b.forward(&x, &cos, &sin)?;
        }

        // FSQ: tanh → ×0.999 → round → {0,1,2} → base-3 digits
        let h = self.project_down.forward(&x)?.tanh()?;
        let h = ((h * 0.9990000128746033f64)?.round()? + 1.0)?;
        let powers = Tensor::from_vec(
            (0..8u32).map(|k| 3f32.powi(k as i32)).collect::<Vec<_>>(),
            (1, 1, 8),
            &self.device,
        )?;
        let ids = h.broadcast_mul(&powers)?.sum(D::Minus1)?; // [1, T']
        Ok(ids.to_dtype(DType::U32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn ckpt(name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt")
            .join(name);
        p.exists().then_some(p)
    }

    fn fixture() -> Option<HashMap<String, Tensor>> {
        candle_core::safetensors::load(ckpt("cosyvoice_e2e_fixture.safetensors")?, &Device::Cpu)
            .ok()
    }

    #[test]
    fn tokens_match_official_onnx() {
        let (Some(fx), Some(w)) = (fixture(), ckpt("cosyvoice_tokenizer.safetensors")) else {
            return;
        };
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &dev).unwrap() };
        let tok = SpeechTokenizer::load(vb, &dev).unwrap();
        for (mel_k, tok_k) in [
            ("source_mel128", "source_tokens"),
            ("prompt_mel128", "prompt_tokens"),
        ] {
            let ids = tok.tokenize(&fx[mel_k]).unwrap();
            let got = ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
            let want = fx[tok_k]
                .flatten_all()
                .unwrap()
                .to_dtype(DType::U32)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(got.len(), want.len(), "{mel_k}: token count");
            let miss = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            // FSQ rounding sits on hard decision boundaries; allow a stray token.
            assert!(
                miss * 100 <= want.len(),
                "{mel_k}: {miss}/{} token mismatches",
                want.len()
            );
        }
    }
}

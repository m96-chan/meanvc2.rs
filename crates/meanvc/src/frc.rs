//! Future-receptive chunking (FRC).
//!
//! FRC partitions the temporal sequence into chunks of `B` consecutive
//! frames and assigns a dedicated attention mask at each DiT layer.
//! At layer `l`, attention from chunk `i` to chunk `j` is allowed iff
//! `j ∈ [i - P_l, i + F_l]`, where `P_l` is the past receptive field and
//! `F_l` the future receptive field of that layer; full intra-chunk
//! attention (`j == i`) is always preserved.
//!
//! Because the masks compose across layers, the effective receptive field
//! expands with depth. With the paper's setting `P = [2, 2, 1, 1]`,
//! `F = [1, 0, 0, 0]`, each output chunk sees 6 past chunks, itself, and
//! 1 future chunk — i.e. a bounded 40 ms look-ahead per chunk.

use candle_core::{DType, Device, Tensor};

/// A very negative value used for masked-out attention logits.
///
/// `f32::NEG_INFINITY` can produce NaNs when a full softmax row is masked;
/// FRC never masks a full row (intra-chunk attention is always allowed),
/// but a large finite value is safer under mixed precision.
const MASK_VALUE: f32 = -1e9;

/// Builds the additive attention mask for one DiT layer.
///
/// Returns a `[seq_len, seq_len]` tensor containing `0` where attention is
/// allowed and a large negative value where it is not, ready to be added to
/// the pre-softmax attention logits. `seq_len` frames are grouped into
/// `ceil(seq_len / chunk_frames)` chunks; a trailing partial chunk is
/// treated as a regular chunk.
pub fn layer_mask(
    seq_len: usize,
    chunk_frames: usize,
    past_chunks: usize,
    future_chunks: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for q in 0..seq_len {
        let qc = (q / chunk_frames) as isize;
        for k in 0..seq_len {
            let kc = (k / chunk_frames) as isize;
            let allowed = kc >= qc - past_chunks as isize && kc <= qc + future_chunks as isize;
            if !allowed {
                data[q * seq_len + k] = MASK_VALUE;
            }
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)?.to_dtype(DType::F32)
}

/// Builds the per-layer FRC masks for a whole decoder.
///
/// `past` and `future` hold `P_l` / `F_l` for each layer and must have the
/// same length (one entry per DiT block).
pub fn decoder_masks(
    seq_len: usize,
    chunk_frames: usize,
    past: &[usize],
    future: &[usize],
    device: &Device,
) -> candle_core::Result<Vec<Tensor>> {
    assert_eq!(
        past.len(),
        future.len(),
        "past/future receptive fields must have the same number of layers"
    );
    past.iter()
        .zip(future.iter())
        .map(|(&p, &f)| layer_mask(seq_len, chunk_frames, p, f, device))
        .collect()
}

/// Total receptive field (in chunks) of the stacked decoder:
/// `(sum(P_l), sum(F_l))`.
///
/// With the paper's defaults this is `(6, 1)`: 6 past chunks and 1 future
/// chunk, so streaming inference needs exactly `sum(F_l)` chunks of
/// look-ahead.
pub fn total_receptive_field(past: &[usize], future: &[usize]) -> (usize, usize) {
    (past.iter().sum(), future.iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra_chunk_is_always_visible() {
        let dev = Device::Cpu;
        let m = layer_mask(8, 4, 0, 0, &dev).unwrap();
        let m: Vec<Vec<f32>> = m.to_vec2().unwrap();
        // Frames 0..4 form chunk 0, frames 4..8 chunk 1.
        assert_eq!(m[0][3], 0.0);
        assert_eq!(m[3][0], 0.0);
        assert!(m[0][4] < -1e8);
        assert!(m[7][3] < -1e8);
    }

    #[test]
    fn past_and_future_windows() {
        let dev = Device::Cpu;
        // 4 chunks of 2 frames, P=1, F=1.
        let m = layer_mask(8, 2, 1, 1, &dev).unwrap();
        let m: Vec<Vec<f32>> = m.to_vec2().unwrap();
        // Query frame 4 (chunk 2) may see chunks 1..=3 (frames 2..8).
        assert!(m[4][1] < -1e8); // chunk 0: blocked
        assert_eq!(m[4][2], 0.0); // chunk 1: past
        assert_eq!(m[4][5], 0.0); // chunk 2: intra
        assert_eq!(m[4][6], 0.0); // chunk 3: future
    }

    #[test]
    fn paper_receptive_field_totals() {
        let (p, f) = total_receptive_field(&[2, 2, 1, 1], &[1, 0, 0, 0]);
        assert_eq!((p, f), (6, 1));
    }
}

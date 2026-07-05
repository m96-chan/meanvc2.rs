//! Chunk-wise autoregressive denoising (CARD) attention mask.
//!
//! Training layout: the sequence is `[N clean cache chunks ‖ N noisy
//! chunks]`, each of `chunk_size` frames. Cache chunks attend only within
//! their own chunk; noisy chunk `q` attends its own chunk plus the cache
//! chunks `[q - max_lookback, q - 1]`. Mirrors `ChunkAttnProcessor` in the
//! official implementation.

use candle_core::{Device, Tensor};

const MASK_VALUE: f32 = -1e9;

/// Builds the additive `[2 * n_chunks * chunk_size; 2]`-shaped CARD mask.
pub fn card_mask(
    n_chunks: usize,
    chunk_size: usize,
    max_lookback: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let len = 2 * n_chunks * chunk_size;
    let mut data = vec![MASK_VALUE; len * len];
    for row in 0..len {
        let cj = row / chunk_size; // query chunk over the full 2N layout
        for col in 0..len {
            let ci = col / chunk_size; // key chunk
            let intra = ci == cj;
            // Noisy query chunk q = cj - N attends cache chunks
            // [q - max_lookback, q - 1].
            let cache_visible = cj >= n_chunks && ci < n_chunks && {
                let q = cj - n_chunks;
                ci < q && ci + max_lookback >= q
            };
            if intra || cache_visible {
                data[row * len + col] = 0.0;
            }
        }
    }
    Tensor::from_vec(data, (len, len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_visibility() {
        let dev = Device::Cpu;
        // 3 chunks of 2 frames, lookback 1 => sequence [c0 c1 c2 | x0 x1 x2].
        let m: Vec<Vec<f32>> = card_mask(3, 2, 1, &dev)
            .unwrap()
            .to_vec2()
            .unwrap();
        let vis = |qc: usize, kc: usize| m[qc * 2][kc * 2] == 0.0;
        // Cache chunks: intra only.
        assert!(vis(0, 0) && !vis(0, 1) && !vis(0, 3));
        // Noisy chunk x0 (chunk 3): itself only (no previous chunks).
        assert!(vis(3, 3) && !vis(3, 0) && !vis(3, 4));
        // Noisy chunk x1 (chunk 4): itself + cache c0, not c1 (that is
        // itself), not beyond lookback.
        assert!(vis(4, 4) && vis(4, 0) && !vis(4, 1) && !vis(4, 3));
        // Noisy chunk x2 (chunk 5): itself + c1 (lookback 1 excludes c0).
        assert!(vis(5, 5) && vis(5, 1) && !vis(5, 0));
    }
}

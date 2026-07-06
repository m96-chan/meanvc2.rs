//! Streaming conversion demo with randomly initialized weights.
//!
//! Feeds synthetic BNF chunks through the full UTTE → FRC-DiT → 1-NFE
//! pipeline and reports the shapes and per-chunk latency. With trained
//! weights the emitted tensors would be mel chunks for a vocoder.
//!
//! ```sh
//! cargo run --release --example streaming_demo
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use meanvc2::{MeanVc2, MeanVc2Config, StreamingConverter};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let cfg = MeanVc2Config::default();

    // Random init; use `MeanVc2::load(cfg, "meanvc2.safetensors", &device)`
    // for trained weights.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MeanVc2::new(cfg.clone(), vb)?;

    // A stand-in for the ECAPA-TDNN reference embedding.
    let speaker = Tensor::randn(0f32, 1f32, (1, cfg.decoder.speaker_dim), &device)?;
    let mut converter = StreamingConverter::new(&model, &speaker)?;

    println!(
        "chunk = {} mel frames (40 ms), look-ahead = {} chunk(s)",
        cfg.decoder.chunk_frames,
        converter.lookahead_chunks()
    );

    let num_chunks = 25; // 1 s of audio at 40 ms per chunk
    let mut emitted = 0;
    let start = std::time::Instant::now();
    for i in 0..num_chunks {
        // A stand-in for one 40 ms Fast-U2++ BNF chunk (already upsampled
        // to the mel frame rate, see `meanvc2::encoders::upsample_bnf`).
        let bnf = Tensor::randn(
            0f32,
            1f32,
            (1, cfg.decoder.chunk_frames, cfg.utte.bnf_dim),
            &device,
        )?;
        for mel in converter.push(&bnf)? {
            emitted += 1;
            if emitted == 1 {
                println!("first packet after {} pushes: {:?}", i + 1, mel.dims());
            }
        }
    }
    emitted += converter.finish()?.len();
    let elapsed = start.elapsed();

    println!(
        "emitted {emitted}/{num_chunks} mel chunks in {elapsed:.2?} \
         ({:.3} ms/chunk, RTF ≈ {:.3})",
        elapsed.as_secs_f64() * 1e3 / emitted as f64,
        elapsed.as_secs_f64() / (num_chunks as f64 * 0.04),
    );
    Ok(())
}

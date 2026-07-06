//! MeanVC v1 (CARD + MRTE) demo with randomly initialized weights at the
//! official 200 ms-checkpoint scale. With converted official weights
//! (`MeanVc1::load(cfg, "model_200ms.safetensors", &dev)`) this becomes
//! real voice conversion.
//!
//! ```sh
//! cargo run --release --example v1_demo
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use meanvc2::v1::{MeanVc1, MeanVc1Config};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let cfg = MeanVc1Config::default(); // official config_200ms.json scale
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MeanVc1::new(cfg.clone(), vb)?;
    let params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("MeanVC v1: {:.1}M params (paper: 14M)", params as f64 / 1e6);

    // 2 s of BNFs at 10 ms frames (100/s), 3 s reference mel, voice print.
    let n = 10 * cfg.chunk_size; // 10 chunks of 200 ms
    let cond = Tensor::randn(0f32, 1f32, (1, n, cfg.bn_dim), &device)?;
    let prompts = Tensor::randn(0f32, 1f32, (1, 300, cfg.n_mels), &device)?;
    let spks = Tensor::randn(0f32, 1f32, (1, cfg.bn_dim), &device)?;

    let t0 = std::time::Instant::now();
    let mel = model.sample(&cond, &prompts, &spks)?;
    let elapsed = t0.elapsed();
    let audio_secs = n as f64 * 0.01;
    println!(
        "CARD sampling: {:?} -> {elapsed:.0?} for {audio_secs:.1} s of mel (RTF ≈ {:.3}, {} chunks of {} frames)",
        mel.dims(),
        elapsed.as_secs_f64() / audio_secs,
        n / cfg.chunk_size,
        cfg.chunk_size,
    );
    let v: Vec<f32> = mel.flatten_all()?.to_vec1()?;
    assert!(v.iter().all(|x| x.is_finite()));
    Ok(())
}

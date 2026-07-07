//! Debug/bench probe for the streaming driver (#50): loads the engine
//! on CUDA when available, feeds the e2e fixture source through
//! `SeedVcStream` and prints per-step wall times.
//!
//! ```sh
//! cargo run --release -p seedvc --features cuda --example stream_probe
//! ```

use candle_core::{Device, IndexOp};
use seedvc::pipeline::{resample, SeedVcEngine};
use seedvc::stream::StreamConfig;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
    let device = if std::env::var("SEEDVC_CPU").is_ok() {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("device: {device:?}");
    let t0 = Instant::now();
    let eng = SeedVcEngine::load(&dir, &device)?;
    println!("load: {:.1}s", t0.elapsed().as_secs_f32());
    let fx = candle_core::safetensors::load(dir.join("seedvc_e2e_fixture.safetensors"), &device)?;
    let rf: Vec<f32> = fx["ref_22k"].i(0)?.to_vec1()?;
    let src22: Vec<f32> = fx["source_22k"].i(0)?.to_vec1()?;
    let src16 = resample(&src22, 22_050, 16_000);

    let cfg = StreamConfig::default();
    let t0 = Instant::now();
    let mut stream = eng.stream(&rf, cfg)?;
    println!("reference prep: {:.2}s", t0.elapsed().as_secs_f32());
    let mut fed = 0;
    let mut n = 0;
    while fed + cfg.block <= src16.len() && n < 6 {
        stream.push(&src16[fed..fed + cfg.block]);
        fed += cfg.block;
        while stream.ready() {
            let t0 = Instant::now();
            let out = stream.step()?;
            let dt = t0.elapsed().as_secs_f32();
            n += 1;
            println!(
                "step {n}: {:.2}s (budget {:.2}s) out {}",
                dt,
                cfg.block as f32 / 16_000.0,
                out.map(|o| o.len()).unwrap_or(0)
            );
        }
    }
    Ok(())
}

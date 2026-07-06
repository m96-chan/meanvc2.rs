//! Per-op stage bench for issue #38 (CUDA slow-path hunt): times every
//! pipeline component on one 640 ms window, with device sync between
//! measurements.
//!
//! ```sh
//! cargo run --release -p xvc --features cuda --example bench_stages
//! cargo run --release -p xvc --example bench_stages   # CPU
//! ```

use std::time::Instant;

use candle_core::{Device, Tensor};
use xvc::XvcEngine;

fn time<T>(dev: &Device, label: &str, iters: usize, mut f: impl FnMut() -> T) -> T {
    // Warmup (the first call includes kernel/cuBLAS init — reported as
    // the cold time).
    let tw = Instant::now();
    let mut out = f();
    dev.synchronize().unwrap();
    let cold = tw.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    for _ in 0..iters {
        out = f();
    }
    dev.synchronize().unwrap();
    println!(
        "{label:24} {:8.2} ms   (cold {cold:8.2} ms)",
        t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
    );
    out
}

fn main() -> anyhow::Result<()> {
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("device: {device:?}");
    let eng = XvcEngine::load("ckpt", &device)?;

    // 640 ms window of real audio from the e2e fixture.
    let fx = candle_core::safetensors::load("ckpt/xvc_e2e_fixture.safetensors", &Device::Cpu)?;
    let src: Vec<f32> = fx["source_wav"].flatten_all()?.to_vec1()?;
    let window: Vec<f32> = src[3200..13440].to_vec();
    let target: Vec<f32> = fx["target_wav"].flatten_all()?.to_vec1()?;
    let reference = eng.prepare_reference(&target)?;
    let n = 5;

    let feats = time(&device, "whisper mel", n, || {
        eng.whisper_mel.extract(&window, &device).unwrap()
    });
    let tok = time(&device, "tokenizer", n, || {
        eng.tokenizer
            .forward(&feats.input_features, &feats.attention_mask)
            .unwrap()
    });
    let sem_emb = eng.tokenizer.embed_ids(&tok.token_ids)?;
    let sem_in = sem_emb.transpose(1, 2)?.contiguous()?;
    let sem_up = time(&device, "semantic adapter", n, || {
        eng.semantic_adapter.forward(&sem_in).unwrap()
    })
    .transpose(1, 2)?
    .contiguous()?;

    let wav_in = Tensor::from_vec(window.clone(), (1, 1, window.len()), &device)?;
    let enc = time(&device, "codec encode", n, || {
        eng.codec.encode(&wav_in).unwrap()
    });
    let acu_emb = enc.zq.transpose(1, 2)?.contiguous()?;
    let combined = Tensor::cat(&[&sem_up, &acu_emb], 2)?
        .transpose(1, 2)?
        .contiguous()?;
    let prenet_out = time(&device, "prenet", n, || {
        eng.prenet.forward(&combined).unwrap()
    });
    let conv_out = time(&device, "converter", n, || {
        eng.converter
            .forward(
                &prenet_out,
                &reference.frame_condition,
                &reference.speaker_condition,
            )
            .unwrap()
    });
    time(&device, "codec decode", n, || {
        eng.codec.decode(&conv_out).unwrap()
    });
    time(&device, "speaker embed", 1, || {
        eng.speaker.embed(&target).unwrap()
    });

    // Correctness vs the chain fixture (same window, same conditions).
    if let Ok(chain) =
        candle_core::safetensors::load("ckpt/xvc_chain_fixture.safetensors", &Device::Cpu)
    {
        let reference = xvc::Reference {
            speaker_condition: chain["speaker_condition"].to_device(&device)?,
            frame_condition: chain["frame_condition"].to_device(&device)?,
        };
        let win: Vec<f32> = chain["chunk_wav"].flatten_all()?.to_vec1()?;
        let out = eng.forward_window(&win, &reference)?;
        for (name, got, want) in [
            (
                "sem_adapter_out",
                &out.sem_adapter_out,
                &chain["sem_adapter_out"],
            ),
            ("acoustic_zq", &out.acoustic_zq, &chain["acoustic_zq"]),
            ("prenet_out", &out.prenet_out, &chain["prenet_out"]),
            ("converter_out", &out.converter_out, &chain["converter_out"]),
            ("wav_out", &out.wav, &chain["wav_out"]),
        ] {
            let g: Vec<f32> = got.flatten_all()?.to_vec1()?;
            let w: Vec<f32> = want.flatten_all()?.to_vec1()?;
            let max = g
                .iter()
                .zip(&w)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("fixture diff {name:16} max abs {max:.3e}");
        }
        let ids: Vec<i64> = out.token_ids.flatten_all()?.to_vec1()?;
        let want_ids: Vec<f32> = chain["token_ids"]
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let flips = ids
            .iter()
            .zip(&want_ids)
            .filter(|(a, b)| **a != **b as i64)
            .count();
        println!("fixture token id flips: {flips}/{}", ids.len());

        // prepare_reference on this device vs the fixture conditions
        // (the e2e fixture's target is the same audio `target_wav`).
        let r = eng.prepare_reference(&target)?;
        for (name, got, want) in [
            (
                "speaker_condition",
                &r.speaker_condition,
                &fx["speaker_condition"],
            ),
            (
                "frame_condition",
                &r.frame_condition,
                &fx["frame_condition"],
            ),
        ] {
            let g: Vec<f32> = got.flatten_all()?.to_vec1()?;
            let w: Vec<f32> = want.flatten_all()?.to_vec1()?;
            let max = g
                .iter()
                .zip(&w)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("reference diff {name:18} max abs {max:.3e}");
        }
    }

    // Demo-direction (target_wav as source) offline forward: this device
    // vs a CPU engine, token ids + waveform correlation.
    if !matches!(device, Device::Cpu) {
        let cpu_eng = XvcEngine::load("ckpt", &Device::Cpu)?;
        let mut pad = target.clone();
        while pad.len() % 1280 != 0 {
            pad.push(0.0);
        }
        let r_dev = eng.prepare_reference(&src)?;
        let r_cpu = cpu_eng.prepare_reference(&src)?;
        let o_dev = eng.forward_window(&pad, &r_dev)?;
        let o_cpu = cpu_eng.forward_window(&pad, &r_cpu)?;
        let ids_dev: Vec<i64> = o_dev.token_ids.flatten_all()?.to_vec1()?;
        let ids_cpu: Vec<i64> = o_cpu.token_ids.flatten_all()?.to_vec1()?;
        let flips = ids_dev.iter().zip(&ids_cpu).filter(|(a, b)| a != b).count();
        let g: Vec<f32> = o_dev.wav.flatten_all()?.to_vec1()?;
        let w: Vec<f32> = o_cpu.wav.flatten_all()?.to_vec1()?;
        let n = g.len() as f64;
        let (mg, mw) = (
            g.iter().map(|&x| x as f64).sum::<f64>() / n,
            w.iter().map(|&x| x as f64).sum::<f64>() / n,
        );
        let (mut num, mut dg, mut dw) = (0f64, 0f64, 0f64);
        for (&a, &b) in g.iter().zip(&w) {
            let (a, b) = (a as f64 - mg, b as f64 - mw);
            num += a * b;
            dg += a * a;
            dw += b * b;
        }
        println!(
            "demo-direction offline dev-vs-cpu: token flips {flips}/{}, wav corr {:.4}",
            ids_dev.len(),
            num / (dg.sqrt() * dw.sqrt()).max(1e-12)
        );
    }

    // Pipelined vs sequential driver on this device.
    {
        let reference = eng.prepare_reference(&target)?;
        let cfg = xvc::StreamConfig::default();
        let eng = std::sync::Arc::new(eng);
        let mut seq = eng.stream(reference.clone(), cfg)?;
        let mut want: Vec<f32> = Vec::new();
        for chunk in src.chunks(cfg.current_len()) {
            seq.push(chunk);
            while let Some(step) = seq.step()? {
                want.extend_from_slice(&step.samples);
            }
        }
        want.extend_from_slice(&seq.finish()?);
        let mut pipe = xvc::XvcPipelinedStream::new(eng.clone(), reference, cfg)?;
        let mut got: Vec<f32> = Vec::new();
        for chunk in src.chunks(cfg.current_len()) {
            pipe.push(chunk)?;
            while let Some(step) = pipe.try_next()? {
                got.extend_from_slice(&step.samples);
            }
        }
        got.extend_from_slice(&pipe.finish()?);
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("pipelined vs sequential ({device:?}): max abs {max:.3e}");
    }
    Ok(())
}

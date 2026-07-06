//! Golden tests of the X-VC end-to-end pipeline (`xvc::pipeline`) against
//! the official-implementation fixtures (`tools/gen_xvc_fixtures.py`):
//!
//! * one streaming window vs `xvc_chain_fixture.safetensors` — tight
//!   per-stage tolerances (this is the golden test of the prenet port);
//! * offline + full CPU-preset stream vs `xvc_e2e_fixture.safetensors` —
//!   loose by design (a single VQ argmin flip legitimately changes the
//!   waveform locally; see issue #30), correlation + envelope-correlation
//!   based;
//! * the frame-condition mel extractor vs the precomputed
//!   `frame_condition`.
//!
//! All tests skip when the converted checkpoints / fixtures are absent.

mod util;

use candle_core::{Device, IndexOp, Tensor};
use xvc::pipeline::{Reference, StreamConfig};
use xvc::preprocess::FrameMelExtractor;
use xvc::XvcEngine;

const CKPTS: &[&str] = &[
    "xvc_tokenizer.safetensors",
    "xvc_speaker.safetensors",
    "xvc_codec.safetensors",
    "xvc_converter.safetensors",
    "xvc_prenet.safetensors",
];

/// Loads the engine, skipping the test when any checkpoint is missing.
fn engine() -> Option<XvcEngine> {
    let mut dir = None;
    for name in CKPTS {
        dir = Some(util::ckpt_path(name)?.parent().unwrap().to_path_buf());
    }
    Some(XvcEngine::load(dir.unwrap(), &Device::Cpu).unwrap())
}

fn samples(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

/// Pearson correlation of two equal-length sample vectors.
fn corr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&x| x as f64).sum::<f64>() / n,
        b.iter().map(|&x| x as f64).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64 - ma, y as f64 - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-12)
}

/// RMS envelope (10 ms frames) correlation.
fn envelope_corr(a: &[f32], b: &[f32]) -> f64 {
    let frame = 160;
    let env = |x: &[f32]| -> Vec<f32> {
        x.chunks(frame)
            .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
            .collect()
    };
    corr(&env(a), &env(b))
}

#[test]
fn frame_condition_matches_e2e_fixture() {
    let Some(fx) = util::ckpt_fixture("xvc_e2e_fixture.safetensors") else {
        return;
    };
    let mel = FrameMelExtractor::default()
        .extract(&samples(&fx["target_wav"]), &Device::Cpu)
        .unwrap();
    assert_eq!(mel.dims(), fx["frame_condition"].dims());
    let diff = util::max_abs_diff(&mel, &fx["frame_condition"]);
    println!("frame_condition max abs diff {diff:.2e}");
    assert!(diff < 1e-3, "frame_condition diff {diff}");
}

#[test]
fn reference_conditions_match_e2e_fixture() {
    let (Some(fx), Some(eng)) = (util::ckpt_fixture("xvc_e2e_fixture.safetensors"), engine())
    else {
        return;
    };
    let target = samples(&fx["target_wav"]);
    let r = eng.prepare_reference(&target).unwrap();
    let d_spk = util::max_abs_diff(&r.speaker_condition, &fx["speaker_condition"]);
    let d_mel = util::max_abs_diff(&r.frame_condition, &fx["frame_condition"]);
    println!("speaker_condition diff {d_spk:.2e}, frame_condition diff {d_mel:.2e}");
    assert!(d_spk < 1e-3, "speaker_condition diff {d_spk}");
    assert!(d_mel < 1e-3, "frame_condition diff {d_mel}");
}

#[test]
fn streaming_step_matches_chain_fixture() {
    let (Some(fx), Some(eng)) = (
        util::ckpt_fixture("xvc_chain_fixture.safetensors"),
        engine(),
    ) else {
        return;
    };
    let reference = Reference {
        speaker_condition: fx["speaker_condition"].clone(),
        frame_condition: fx["frame_condition"].clone(),
    };
    let out = eng
        .forward_window(&samples(&fx["chunk_wav"]), &reference)
        .unwrap();

    let ids: Vec<i64> = samples(&fx["token_ids"].to_dtype(candle_core::DType::F32).unwrap())
        .iter()
        .map(|&v| v as i64)
        .collect();
    let got: Vec<i64> = out.token_ids.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(got, ids, "semantic token ids");

    let d_sem = util::max_abs_diff(&out.sem_adapter_out, &fx["sem_adapter_out"]);
    let d_zq = util::max_abs_diff(&out.acoustic_zq, &fx["acoustic_zq"]);
    let d_pre = util::max_abs_diff(&out.prenet_out, &fx["prenet_out"]);
    let d_conv = util::max_abs_diff(&out.converter_out, &fx["converter_out"]);
    let d_wav = util::max_abs_diff(&out.wav, &fx["wav_out"]);
    println!(
        "chain step: sem {d_sem:.2e} zq {d_zq:.2e} prenet {d_pre:.2e} \
         conv {d_conv:.2e} wav {d_wav:.2e}"
    );
    assert!(d_sem < 1e-4, "sem_adapter_out diff {d_sem}");
    assert!(d_zq < 1e-4, "acoustic_zq diff {d_zq}");
    assert!(d_pre < 1e-4, "prenet_out diff {d_pre}");
    assert!(d_conv < 1e-4, "converter_out diff {d_conv}");
    assert!(d_wav < 5e-3, "wav_out diff {d_wav}");

    // The driver bookkeeping slices (history 280 ms / current 240 ms /
    // smooth 20 ms).
    let cfg = StreamConfig::default();
    let wav = out.wav.i((0, 0)).unwrap();
    let current = wav.narrow(0, cfg.history_len(), cfg.current_len()).unwrap();
    let tail = wav
        .narrow(0, cfg.history_len() + cfg.current_len(), cfg.smooth_len())
        .unwrap();
    let d_cur = util::max_abs_diff(&current, &fx["wav_current"].i((0, 0)).unwrap());
    let d_tail = util::max_abs_diff(&tail, &fx["wav_tail"].i((0, 0)).unwrap());
    assert!(d_cur < 5e-3, "wav_current diff {d_cur}");
    assert!(d_tail < 5e-3, "wav_tail diff {d_tail}");
}

#[test]
fn offline_matches_e2e_fixture() {
    let (Some(fx), Some(eng)) = (util::ckpt_fixture("xvc_e2e_fixture.safetensors"), engine())
    else {
        return;
    };
    let reference = Reference {
        speaker_condition: fx["speaker_condition"].clone(),
        frame_condition: fx["frame_condition"].clone(),
    };
    // `source_wav` is already preprocessed: run the window forward
    // directly (identical to `convert_offline` after `preprocess`).
    let out = eng
        .forward_window(&samples(&fx["source_wav"]), &reference)
        .unwrap();
    let (got, want) = (samples(&out.wav), samples(&fx["offline_out"]));
    assert_eq!(got.len(), want.len());
    let max = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let c = corr(&got, &want);
    let ec = envelope_corr(&got, &want);
    println!("offline: max abs {max:.2e}, corr {c:.6}, envelope corr {ec:.6}");
    assert!(max < 5e-2, "offline max abs diff {max}");
    assert!(c > 0.995, "offline corr {c}");
    assert!(ec > 0.995, "offline envelope corr {ec}");
}

/// The pipelined driver must be bit-identical to the sequential one: the
/// stage split only moves the same ops onto different threads, so any
/// difference is a bug (issue #38, stage pipelining).
#[test]
fn pipelined_stream_matches_sequential() {
    let (Some(fx), Some(eng)) = (util::ckpt_fixture("xvc_e2e_fixture.safetensors"), engine())
    else {
        return;
    };
    let reference = Reference {
        speaker_condition: fx["speaker_condition"].clone(),
        frame_condition: fx["frame_condition"].clone(),
    };
    let source = samples(&fx["source_wav"]);
    let cfg = StreamConfig::default();

    // Sequential reference output.
    let mut seq = eng.stream(reference.clone(), cfg).unwrap();
    let mut want: Vec<f32> = Vec::with_capacity(source.len());
    for chunk in source.chunks(cfg.current_len()) {
        seq.push(chunk);
        while let Some(step) = seq.step().unwrap() {
            want.extend_from_slice(&step.samples);
        }
    }
    want.extend_from_slice(&seq.finish().unwrap());

    // Pipelined output over the same hops.
    let eng = std::sync::Arc::new(eng);
    let mut pipe = xvc::XvcPipelinedStream::new(eng, reference, cfg).unwrap();
    let mut got: Vec<f32> = Vec::with_capacity(source.len());
    for chunk in source.chunks(cfg.current_len()) {
        pipe.push(chunk).unwrap();
        while let Some(step) = pipe.try_next().unwrap() {
            got.extend_from_slice(&step.samples);
        }
    }
    got.extend_from_slice(&pipe.finish().unwrap());

    assert_eq!(got.len(), want.len());
    let max = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("pipelined vs sequential stream: max abs {max:.2e}");
    assert_eq!(max, 0.0, "pipelined stream must be bit-identical");
}

#[test]
fn stream_cpu_preset_matches_e2e_fixture() {
    let (Some(fx), Some(chain), Some(eng)) = (
        util::ckpt_fixture("xvc_e2e_fixture.safetensors"),
        util::ckpt_fixture("xvc_chain_fixture.safetensors"),
        engine(),
    ) else {
        return;
    };
    let reference = Reference {
        speaker_condition: fx["speaker_condition"].clone(),
        frame_condition: fx["frame_condition"].clone(),
    };
    let source = samples(&fx["source_wav"]);
    let cfg = StreamConfig::default();
    let mut stream = eng.stream(reference, cfg).unwrap();

    // Feed in 240 ms hops like the live demo and drain as we go.
    let mut got: Vec<f32> = Vec::with_capacity(source.len());
    let mut steps = Vec::new();
    for chunk in source.chunks(cfg.current_len()) {
        stream.push(chunk);
        while let Some(step) = stream.step().unwrap() {
            got.extend_from_slice(&step.samples);
            steps.push(step);
        }
    }
    got.extend_from_slice(&stream.finish().unwrap());
    assert_eq!(got.len(), source.len());

    // Window #2 is the chain fixture: past the 20 ms crossfade the
    // emitted hop must match `wav_current` tightly.
    let cur = cfg.current_len();
    let win2 = &got[2 * cur..3 * cur];
    let want2 = samples(&chain["wav_current"]);
    let d2 = win2[cfg.smooth_len()..]
        .iter()
        .zip(&want2[cfg.smooth_len()..])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("stream window #2 vs chain fixture (past crossfade): {d2:.2e}");
    assert!(d2 < 5e-3, "window #2 diff {d2}");

    // Full-stream comparison: loose by design (issue #30 — VQ argmin
    // divergence changes the waveform locally).
    let want = samples(&fx["stream_cpu_out"]);
    let max = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let c = corr(&got, &want);
    let ec = envelope_corr(&got, &want);
    println!("stream cpu preset: max abs {max:.2e}, corr {c:.6}, envelope corr {ec:.6}");
    assert!(max < 5e-2, "stream max abs diff {max}");
    assert!(c > 0.995, "stream corr {c}");
    assert!(ec > 0.995, "stream envelope corr {ec}");
}

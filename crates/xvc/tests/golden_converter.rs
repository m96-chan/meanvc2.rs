//! Golden tests for the MMDiT `AcousticConverter` (`xvc::converter`)
//! against the official X-VC PyTorch implementation (issue #30, Phase 1),
//! plus the codec+converter slice of one official streaming step.
//!
//! References come from `tools/gen_xvc_fixtures.py` (`ckpt/`, gitignored),
//! weights from `tools/convert_xvc_generator.py`; tests skip with a
//! message when a file is absent.

mod util;

use candle_core::Device;
use util::{ckpt_fixture, ckpt_path, max_abs_diff};
use xvc::{AcousticConverter, SacCodec};

fn converter() -> Option<AcousticConverter> {
    let path = ckpt_path("xvc_converter.safetensors")?;
    Some(AcousticConverter::load(path, &Device::Cpu).unwrap())
}

/// One converter step on the seed-5 random fixture: joint attention over
/// [x seq (RoPE) || mel-cond seq (own RoPE)], AdaLN-Zero from the 192-d
/// speaker condition. Tolerance < 1e-4 abs (MeanVC DiT precedent).
/// Also reports the wall time of one converter step.
#[test]
fn converter_single_step_matches_official() {
    let (Some(conv), Some(fx)) = (
        converter(),
        ckpt_fixture("xvc_converter_fixture.safetensors"),
    ) else {
        return;
    };
    // Warm-up, then a timed run.
    let out = conv
        .forward(&fx["x"], &fx["frame_cond"], &fx["spk"])
        .unwrap();
    let start = std::time::Instant::now();
    let _ = conv
        .forward(&fx["x"], &fx["frame_cond"], &fx["spk"])
        .unwrap();
    eprintln!(
        "converter step: {:.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    assert_eq!(out.dims(), fx["out"].dims());
    let diff = max_abs_diff(&out, &fx["out"]);
    eprintln!("converter max abs diff: out {diff:.2e}");
    assert!(diff < 1e-4, "converter out max abs diff {diff}");
}

/// The codec+converter slice of one official 640/240/100/20 streaming
/// step (`xvc_chain_fixture.safetensors`): encode the chunk to the
/// quantized acoustic latent, convert the fixture prenet output with the
/// real speaker/mel conditions, decode the converted latent to the
/// waveform, and check the driver's crossfade slices. The semantic and
/// prenet stages come from the fixture (they are owned by the parallel
/// tokenizer/speaker ports).
#[test]
fn chain_streaming_step_codec_converter() {
    let (Some(conv), Some(codec_path), Some(fx)) = (
        converter(),
        ckpt_path("xvc_codec.safetensors"),
        ckpt_fixture("xvc_chain_fixture.safetensors"),
    ) else {
        return;
    };
    let codec = SacCodec::load(codec_path, &Device::Cpu).unwrap();

    // Acoustic branch: chunk wav -> quantized 50 Hz latent.
    let enc = codec.encode(&fx["chunk_wav"]).unwrap();
    let d_zq = max_abs_diff(&enc.zq, &fx["acoustic_zq"]);
    assert!(d_zq < 1e-4, "acoustic_zq max abs diff {d_zq}");

    // Converter on the (fixture) prenet output with the real conditions.
    let out = conv
        .forward(
            &fx["prenet_out"],
            &fx["frame_condition"],
            &fx["speaker_condition"],
        )
        .unwrap();
    let d_conv = max_abs_diff(&out, &fx["converter_out"]);
    assert!(d_conv < 1e-4, "converter_out max abs diff {d_conv}");

    // Decode our converted latent to the waveform of the whole window.
    let wav = codec.decode(&out).unwrap();
    let d_wav = max_abs_diff(&wav, &fx["wav_out"]);
    eprintln!(
        "chain max abs diff: acoustic_zq {d_zq:.2e}, converter_out {d_conv:.2e}, wav_out {d_wav:.2e}"
    );
    assert!(d_wav < 5e-3, "wav_out max abs diff {d_wav}");

    // Driver bookkeeping: the emitted 240 ms slice and the 20 ms
    // crossfade tail (history 280 ms at 16 kHz).
    let (hist, cur, smooth) = (280 * 16, 240 * 16, 20 * 16);
    let current = wav.narrow(2, hist, cur).unwrap();
    let tail = wav.narrow(2, hist + cur, smooth).unwrap();
    assert!(
        max_abs_diff(&current, &fx["wav_current"]) < 5e-3,
        "wav_current"
    );
    assert!(max_abs_diff(&tail, &fx["wav_tail"]) < 5e-3, "wav_tail");
}

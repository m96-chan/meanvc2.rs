//! Golden parity of `FastU2pp::forward_chunk` (incremental streaming with
//! per-layer attention K/V and depthwise-conv caches, issue #9) against the
//! official TorchScript chunked decode (`forward_encoder_chunk` of
//! `ckpt/fastu2pp.pt`, decoding_chunk_size = 5, num_left_chunks = 2).
//!
//! The fixture is generated locally with python (torch + safetensors) from
//! the official checkpoint — see the PR for the generator script — and lives
//! in the gitignored `ckpt/` directory, so the test skips when it is absent.

use candle_core::{Device, Tensor};
use meanvc2::backends::{FastU2pp, FastU2ppConfig};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap()
}

#[test]
fn forward_chunk_matches_official_torchscript() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ckpt");
    let fx_path = dir.join("asr_stream_fixture.safetensors");
    let weights = dir.join("fastu2pp.safetensors");
    if !fx_path.exists() || !weights.exists() {
        eprintln!("skipping: ckpt/asr_stream_fixture.safetensors or weights absent");
        return;
    }
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load(&fx_path, &dev).unwrap();
    let asr = FastU2pp::load(FastU2ppConfig::official_meanvc1(), &weights, &dev).unwrap();

    let fbank = fx["fbank"].unsqueeze(0).unwrap(); // [1, frames, 80]
    let reference = &fx["bnf_ref"]; // [1, 5 * windows, 256]

    // Feed 20 raw fbank frames (one 200 ms audio chunk) at a time, exactly
    // like the real-time demo.
    let mut state = asr.stream();
    let mut outs = Vec::new();
    let n = fbank.dim(1).unwrap();
    let mut cur = 0;
    while cur < n {
        let take = 20.min(n - cur);
        let chunk = fbank.narrow(1, cur, take).unwrap();
        cur += take;
        if let Some(bn) = asr.forward_chunk(&chunk, &mut state).unwrap() {
            outs.push(bn);
        }
    }
    let streamed = Tensor::cat(&outs.iter().collect::<Vec<_>>(), 1).unwrap();

    let t = streamed.dim(1).unwrap().min(reference.dim(1).unwrap());
    assert!(t >= 50, "too few streamed frames to compare: {t}");
    let a = streamed.narrow(1, 0, t).unwrap();
    let b = reference.narrow(1, 0, t).unwrap();
    let diff = max_abs_diff(&a, &b);
    let scale: f32 = b.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
    eprintln!("streamed {t} frames, max |diff| = {diff:.6} (ref |max| {scale:.3})");
    assert!(diff < 1e-3, "mismatch vs official chunked decode: {diff}");
}

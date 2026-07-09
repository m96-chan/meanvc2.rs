//! Checks `prepare_reference`'s prompt-token/prompt-mel self-consistency:
//! `feat_len()` should equal `tokens_len() * TOKEN_MEL_RATIO` exactly,
//! since `Flow::cfm` locates the prompt/source boundary in `mu` at that
//! offset. A field report ("output doesn't sound like the reference")
//! led here — the two are independently resampled/STFT'd, so a frame or
//! two of drift was possible on short references before
//! `prepare_reference` started snapping `feat` to the expected length.
//!
//! ```sh
//! cargo run --release -p cosyvoice --features cuda --example ref_align_probe -- \
//!     <wav> [wav...]
//! ```
use candle_core::Device;
use cosyvoice::CosyVoiceEngine;
use vc_core::profile::resample_analysis;

fn read16(p: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let sc = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.unwrap() as f32 / sc)
                .collect()
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.unwrap())
            .collect(),
    };
    if spec.sample_rate == 16_000 {
        audio
    } else {
        resample_analysis(&audio, spec.sample_rate as usize, 16_000)
    }
}

fn main() {
    let dev = Device::cuda_if_available(0).unwrap();
    let eng = CosyVoiceEngine::load("ckpt", &dev).unwrap();
    for path in std::env::args().skip(1) {
        let audio = read16(&path);
        let r = eng.prepare_reference(&audio, 16_000).unwrap();
        let want = r.tokens_len() * 2;
        println!(
            "{path}: dur={:.2}s tokens={} want_feat={} feat={} diff={}",
            audio.len() as f32 / 16_000.0,
            r.tokens_len(),
            want,
            r.feat_len(),
            r.feat_len() as i64 - want as i64,
        );
    }
}

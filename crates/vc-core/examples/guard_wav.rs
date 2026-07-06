//! Applies [`vc_core::declick::NeedleGuard`] to a wav file offline —
//! for validating detector changes against field recordings.
//!
//! ```sh
//! cargo run --release -p vc-core --example guard_wav -- in.wav out.wav
//! ```

use vc_core::declick::NeedleGuard;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let inp = args.next().expect("in.wav");
    let outp = args.next().expect("out.wav");
    let mut r = hound::WavReader::open(&inp)?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let x: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .step_by(ch)
                .map(|v| v.map(|s| s as f32 / scale))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(ch)
            .map(|v| v.map(|s| s as f32))
            .collect::<Result<_, _>>()?,
    };
    let mut g = NeedleGuard::new(spec.sample_rate as f32);
    let mut y = Vec::with_capacity(x.len());
    for c in x.chunks(4096) {
        y.extend(g.process(c));
    }
    let mut w = hound::WavWriter::create(
        &outp,
        hound::WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for s in &y {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    eprintln!("repaired {} runs over {} samples", g.repaired, y.len());
    Ok(())
}

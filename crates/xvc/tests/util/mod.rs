//! Shared helpers of the X-VC golden tests (skip-if-absent pattern from
//! `crates/meanvc/tests/golden.rs`).

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, Tensor};

/// Path of `rel` at the workspace root (this crate lives two levels below
/// it, in `crates/xvc`); `ckpt/` and `tools/` are kept at the root.
fn workspace_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

/// Path to a file under `ckpt/` (official checkpoints + parity fixtures);
/// `None` (skip) when it has not been downloaded/generated.
pub fn ckpt_path(name: &str) -> Option<PathBuf> {
    let path = workspace_path("ckpt").join(name);
    if !path.exists() {
        eprintln!(
            "skipping: ckpt/{name} not found — run tools/convert_xvc_generator.py and \
             tools/gen_xvc_fixtures.py against the official X-VC checkpoints (see tools/README.md)"
        );
        return None;
    }
    Some(path)
}

pub fn ckpt_fixture(name: &str) -> Option<HashMap<String, Tensor>> {
    ckpt_path(name).map(|p| candle_core::safetensors::load(&p, &Device::Cpu).unwrap())
}

pub fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar()
        .unwrap()
}

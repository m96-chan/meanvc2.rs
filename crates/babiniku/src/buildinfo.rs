//! Build metadata for `babiniku --version` (issue #69, folding in the
//! leftover #67 polish item): the version line names the cargo features
//! the binary was compiled with, so field reports can state exactly
//! which build they run, and the GPL notice for `seedvc` builds is
//! printed where a distributor will see it.

/// Names of the cargo features this binary was compiled with.
pub fn active_features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "wavlm") {
        f.push("wavlm");
    }
    if cfg!(feature = "cuda") {
        f.push("cuda");
    }
    if cfg!(feature = "metal") {
        f.push("metal");
    }
    if cfg!(feature = "seedvc") {
        f.push("seedvc");
    }
    if cfg!(feature = "cpal-backend") {
        f.push("cpal-backend");
    }
    f
}

/// The `--version` line, e.g. `babiniku 0.1.0 (features: wavlm, cuda)`
/// or `babiniku 0.1.0 (features: none — CPU baseline)`.
pub fn version_line() -> String {
    let features = active_features();
    let list = if features.is_empty() {
        "none — CPU baseline".to_string()
    } else {
        features.join(", ")
    };
    format!("babiniku {} (features: {list})", env!("CARGO_PKG_VERSION"))
}

/// License notice for GPL builds: `Some` iff the `seedvc` feature is
/// compiled in (see crates/seedvc — distributing such a binary makes the
/// whole binary GPL-3.0).
pub fn gpl_notice() -> Option<&'static str> {
    if cfg!(feature = "seedvc") {
        Some(
            "note: built with the seedvc feature — this binary links GPL-3.0 \
             code and is GPL-3.0 when distributed (see crates/seedvc).",
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_line_names_crate_and_version() {
        let line = version_line();
        assert!(line.starts_with(&format!("babiniku {}", env!("CARGO_PKG_VERSION"))));
        assert!(line.contains("features:"));
    }

    #[test]
    fn version_line_reflects_compiled_features() {
        let line = version_line();
        for feat in ["wavlm", "cuda", "metal", "seedvc", "cpal-backend"] {
            let expected = match feat {
                "wavlm" => cfg!(feature = "wavlm"),
                "cuda" => cfg!(feature = "cuda"),
                "metal" => cfg!(feature = "metal"),
                "seedvc" => cfg!(feature = "seedvc"),
                "cpal-backend" => cfg!(feature = "cpal-backend"),
                _ => unreachable!(),
            };
            assert_eq!(
                line.contains(feat),
                expected,
                "feature {feat} mis-reported in {line:?}"
            );
        }
        if active_features().is_empty() {
            assert!(line.contains("CPU baseline"));
        }
    }

    #[test]
    fn gpl_notice_tracks_the_seedvc_feature() {
        assert_eq!(gpl_notice().is_some(), cfg!(feature = "seedvc"));
    }
}

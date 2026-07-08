//! Checkpoint-directory resolution (issue #69).
//!
//! The `babiniku` binary used to hardcode the repo-relative `ckpt/`
//! directory, which only works from a repository checkout. Installed
//! binaries (`cargo install --git … babiniku`) resolve the checkpoint
//! directory in this order instead:
//!
//! 1. the `--ckpt-dir <dir>` flag,
//! 2. the `BABINIKU_CKPT_DIR` environment variable,
//! 3. `./ckpt` when it exists (the repo-checkout convention),
//! 4. the per-platform data directory:
//!    - Linux/BSD: `$XDG_DATA_HOME/babiniku/ckpt`, falling back to
//!      `~/.local/share/babiniku/ckpt`,
//!    - macOS: `~/Library/Application Support/babiniku/ckpt`,
//!    - Windows: `%APPDATA%\babiniku\ckpt`.
//!
//! The same default is intended to be shared with the future
//! `babiniku-fetch` weight downloader (#65) so that
//! install → fetch → run needs no path plumbing.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Environment lookup used by the resolution logic, injectable for tests.
pub type EnvLookup<'a> = &'a dyn Fn(&str) -> Option<OsString>;

/// Resolves the checkpoint directory from the real process environment.
///
/// `flag` is the value of `--ckpt-dir`, if given.
pub fn resolve(flag: Option<&Path>) -> PathBuf {
    resolve_from(
        flag,
        &|k| std::env::var_os(k),
        Path::new("ckpt").is_dir(),
        std::env::consts::OS,
    )
}

/// Pure resolution core: `flag` → `BABINIKU_CKPT_DIR` → `./ckpt` (when
/// `repo_ckpt_exists`) → [`platform_default_for`] of `os`.
pub fn resolve_from(
    flag: Option<&Path>,
    env: EnvLookup,
    repo_ckpt_exists: bool,
    os: &str,
) -> PathBuf {
    if let Some(dir) = flag {
        return dir.to_path_buf();
    }
    if let Some(dir) = env("BABINIKU_CKPT_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if repo_ckpt_exists {
        return PathBuf::from("ckpt");
    }
    platform_default_for(env, os)
}

/// The per-platform default checkpoint directory (step 4 above), for the
/// platform the binary was built for.
pub fn platform_default(env: EnvLookup) -> PathBuf {
    platform_default_for(env, std::env::consts::OS)
}

/// The per-platform default for an explicit `os` (`std::env::consts::OS`
/// values), so every platform's rule is testable on any host. With no
/// usable `HOME`/`APPDATA` the relative `ckpt` is kept as a last resort.
pub fn platform_default_for(env: EnvLookup, os: &str) -> PathBuf {
    let data_dir = match os {
        "windows" => env("APPDATA").filter(|v| !v.is_empty()).map(PathBuf::from),
        "macos" => env("HOME")
            .filter(|v| !v.is_empty())
            .map(|h| PathBuf::from(h).join("Library").join("Application Support")),
        _ => env("XDG_DATA_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|h| PathBuf::from(h).join(".local").join("share"))
            }),
    };
    match data_dir {
        Some(d) => d.join("babiniku").join("ckpt"),
        None => PathBuf::from("ckpt"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| OsString::from(v))
        }
    }

    #[test]
    fn flag_wins_over_everything() {
        let env = env_of(&[("BABINIKU_CKPT_DIR", "/from/env")]);
        let got = resolve_from(Some(Path::new("/from/flag")), &env, true, "linux");
        assert_eq!(got, PathBuf::from("/from/flag"));
    }

    #[test]
    fn env_var_wins_over_repo_ckpt() {
        let env = env_of(&[("BABINIKU_CKPT_DIR", "/from/env")]);
        let got = resolve_from(None, &env, true, "linux");
        assert_eq!(got, PathBuf::from("/from/env"));
    }

    #[test]
    fn empty_env_var_is_ignored() {
        let env = env_of(&[("BABINIKU_CKPT_DIR", "")]);
        let got = resolve_from(None, &env, true, "linux");
        assert_eq!(got, PathBuf::from("ckpt"));
    }

    #[test]
    fn repo_ckpt_wins_over_platform_default() {
        let env = env_of(&[("HOME", "/home/alice")]);
        let got = resolve_from(None, &env, true, "linux");
        assert_eq!(got, PathBuf::from("ckpt"));
    }

    #[test]
    fn linux_default_is_xdg_data_home() {
        let env = env_of(&[
            ("XDG_DATA_HOME", "/home/alice/xdg"),
            ("HOME", "/home/alice"),
        ]);
        let got = resolve_from(None, &env, false, "linux");
        assert_eq!(got, PathBuf::from("/home/alice/xdg/babiniku/ckpt"));
    }

    #[test]
    fn linux_default_falls_back_to_home_local_share() {
        let env = env_of(&[("HOME", "/home/alice")]);
        let got = resolve_from(None, &env, false, "linux");
        assert_eq!(got, PathBuf::from("/home/alice/.local/share/babiniku/ckpt"));
    }

    #[test]
    fn macos_default_is_application_support() {
        let env = env_of(&[("HOME", "/Users/alice")]);
        let got = resolve_from(None, &env, false, "macos");
        assert_eq!(
            got,
            PathBuf::from("/Users/alice/Library/Application Support/babiniku/ckpt")
        );
    }

    #[test]
    fn windows_default_is_appdata() {
        let env = env_of(&[("APPDATA", r"C:\Users\alice\AppData\Roaming")]);
        let got = resolve_from(None, &env, false, "windows");
        assert_eq!(
            got,
            Path::new(r"C:\Users\alice\AppData\Roaming")
                .join("babiniku")
                .join("ckpt")
        );
    }

    #[test]
    fn bare_environment_keeps_relative_ckpt() {
        let env = env_of(&[]);
        let got = resolve_from(None, &env, false, "linux");
        assert_eq!(got, PathBuf::from("ckpt"));
    }
}

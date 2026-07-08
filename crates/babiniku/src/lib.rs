//! Library side of the `babiniku` crate: the audio backend layer plus
//! the shared plumbing of the installed binary.
//!
//! The real-time pipeline (`src/bin/babiniku.rs`) is platform-independent;
//! the platform surface — microphone capture, playback into the
//! virtual-mic route, and the route's lifecycle — lives behind the traits
//! in [`audio`] so each OS plugs in its own backend (issue #51/#52).
//! [`ckpt`] resolves the checkpoint directory for installed binaries
//! (issue #69, shared with the future `babiniku-fetch`, #65) and
//! [`buildinfo`] describes the compiled feature set for `--version`.

pub mod audio;
pub mod buildinfo;
pub mod ckpt;

//! Library side of the `babiniku` crate: the audio backend layer.
//!
//! The real-time pipeline (`src/bin/demo.rs`) is platform-independent; the
//! platform surface — microphone capture, playback into the virtual-mic
//! route, and the route's lifecycle — lives behind the traits in
//! [`audio`] so each OS plugs in its own backend (issue #51/#52).

pub mod audio;

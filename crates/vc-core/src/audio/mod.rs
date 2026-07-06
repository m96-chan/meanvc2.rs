//! Audio front-end: mel-spectrogram extraction used to build training
//! targets (the decoder generates mel frames; waveform synthesis is
//! delegated to an external vocoder such as Vocos).

mod mel;

pub use mel::MelSpectrogram;

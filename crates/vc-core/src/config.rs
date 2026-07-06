//! Engine-agnostic configuration types.

/// Audio front-end configuration (mel-spectrogram extraction).
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 1024,
            // 10 ms hop => 4 mel frames per 40 ms FRC chunk.
            hop_length: 160,
            win_length: 640,
            n_mels: 80,
            f_min: 0.0,
            f_max: 8_000.0,
        }
    }
}

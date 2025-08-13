//! Entropy-oscillation reformulation

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct EntropyOscillation {
    oscillation_frequency: f64,
    phase: f64,
    amplitude: f64,
}

impl EntropyOscillation {
    pub fn new() -> Self {
        Self {
            oscillation_frequency: 1e14, // Hz
            phase: 0.0,
            amplitude: 1.0,
        }
    }

    pub fn update(&mut self, dt: f64) -> Result<()> {
        // S = f(ω_final, φ_final, A_final)
        self.phase += 2.0 * std::f64::consts::PI * self.oscillation_frequency * dt;
        self.phase %= 2.0 * std::f64::consts::PI;
        Ok(())
    }

    pub fn entropy_value(&self) -> f64 {
        self.amplitude * self.phase.sin()
    }
}
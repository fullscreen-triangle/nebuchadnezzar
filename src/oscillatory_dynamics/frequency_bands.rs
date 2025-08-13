//! Frequency band management

use crate::oscillatory_dynamics::FrequencyBand;
use crate::error::{Error, Result};

#[derive(Debug)]
pub struct FrequencyBands {
    bands: Vec<FrequencyBand>,
    coherence_level: f64,
}

impl FrequencyBands {
    pub fn new(bands: Vec<FrequencyBand>) -> Self {
        Self {
            bands,
            coherence_level: 1.0,
        }
    }

    pub fn update(&mut self, dt: f64) -> Result<()> {
        // Update coherence based on band interactions
        self.coherence_level = self.calculate_coherence();
        Ok(())
    }

    pub fn coherence_level(&self) -> f64 {
        self.coherence_level
    }

    fn calculate_coherence(&self) -> f64 {
        if self.bands.is_empty() { return 1.0; }
        
        let total_amplitude: f64 = self.bands.iter().map(|b| b.amplitude).sum();
        let normalized_variance = self.bands.iter()
            .map(|b| (b.amplitude / total_amplitude - 1.0 / self.bands.len() as f64).powi(2))
            .sum::<f64>() / self.bands.len() as f64;
        
        1.0 / (1.0 + normalized_variance)
    }
}
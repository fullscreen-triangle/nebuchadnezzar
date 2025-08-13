//! Environmental noise optimization

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct NoiseOptimization {
    enabled: bool,
    noise_level: f64,
    optimization_factor: f64,
}

impl NoiseOptimization {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            noise_level: 0.1,
            optimization_factor: 1.5,
        }
    }

    pub fn optimize(&mut self, dt: f64) -> Result<()> {
        if !self.enabled { return Ok(()); }
        
        // Optimize environmental noise for biological realism
        self.noise_level += dt * (rand::random::<f64>() - 0.5) * 0.01;
        self.noise_level = self.noise_level.max(0.0).min(1.0);
        
        Ok(())
    }

    pub fn current_noise_level(&self) -> f64 {
        self.noise_level
    }
}
//! Noise stratification management

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct NoiseStratification {
    gradient_steepness: f64,
    max_concentration: f64,
    current_gradient: Vec<f64>,
}

impl NoiseStratification {
    pub fn new(gradient_steepness: f64, max_concentration: f64) -> Self {
        Self {
            gradient_steepness,
            max_concentration,
            current_gradient: vec![1.0, 0.8, 0.25, 0.001], // Environmental -> Arterial -> Arteriolar -> Capillary
        }
    }

    pub fn update(&mut self, dt: f64) -> Result<()> {
        // Update gradient based on flow dynamics
        for i in 1..self.current_gradient.len() {
            let flow_factor = (-self.gradient_steepness * i as f64).exp();
            self.current_gradient[i] = self.max_concentration * flow_factor;
        }
        Ok(())
    }

    pub fn concentration_at_level(&self, level: usize) -> f64 {
        self.current_gradient.get(level).copied().unwrap_or(0.0)
    }
}
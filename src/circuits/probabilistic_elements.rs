//! Probabilistic circuit elements

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct ProbabilisticElement {
    pub base_value: f64,
    pub variance: f64,
    pub current_value: f64,
}

impl ProbabilisticElement {
    pub fn new(base_value: f64, variance: f64) -> Self {
        Self {
            base_value,
            variance,
            current_value: base_value,
        }
    }

    pub fn update(&mut self, dt: f64) -> Result<()> {
        // Add stochastic variation
        let noise = self.variance * (rand::random::<f64>() - 0.5);
        self.current_value = self.base_value + noise;
        Ok(())
    }

    pub fn resistance(base_r: f64, variance: f64) -> Self {
        Self::new(base_r, variance)
    }

    pub fn capacitance(base_c: f64, variance: f64) -> Self {
        Self::new(base_c, variance)
    }
}
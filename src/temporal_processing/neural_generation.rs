//! High-frequency neural generation

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct NeuralGeneration {
    generation_rate: f64,
    neurons_generated: u64,
}

impl NeuralGeneration {
    pub fn new(generation_rate: f64) -> Result<Self> {
        Ok(Self {
            generation_rate,
            neurons_generated: 0,
        })
    }

    pub fn generate_neurons(&mut self, dt: f64) -> Result<()> {
        let new_neurons = (self.generation_rate * dt) as u64;
        self.neurons_generated += new_neurons;
        Ok(())
    }
}
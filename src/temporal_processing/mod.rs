//! Temporal Processing System
//! 
//! High-frequency neural generation and statistical solution emergence

pub mod neural_generation;
pub mod memory_initialization;
pub mod statistical_emergence;
pub mod noise_driven_compute;

pub use neural_generation::NeuralGeneration;
pub use memory_initialization::MemoryInitialization;
pub use statistical_emergence::StatisticalEmergence;
pub use noise_driven_compute::NoiseDrivenCompute;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct TemporalConfig {
    pub precision: f64,  // seconds
    pub generation_rate: f64,  // neurons per second
    pub statistical_enabled: bool,
    pub noise_driven: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self::high_frequency()
    }
}

impl TemporalConfig {
    pub fn high_frequency() -> Self {
        Self {
            precision: 1e-30,  // Ultra-high precision
            generation_rate: 1e30,  // Massive generation rate
            statistical_enabled: true,
            noise_driven: true,
        }
    }
}

#[derive(Debug)]
pub struct TemporalProcessingSystem {
    config: TemporalConfig,
    neural_generation: NeuralGeneration,
    memory_initialization: MemoryInitialization,
    statistical_emergence: StatisticalEmergence,
    noise_driven_compute: NoiseDrivenCompute,
}

impl TemporalProcessingSystem {
    pub fn new(config: TemporalConfig) -> Result<Self> {
        Ok(Self {
            neural_generation: NeuralGeneration::new(config.generation_rate)?,
            memory_initialization: MemoryInitialization::new(),
            statistical_emergence: StatisticalEmergence::new(config.statistical_enabled),
            noise_driven_compute: NoiseDrivenCompute::new(config.noise_driven),
            config,
        })
    }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.neural_generation.generate_neurons(dt)?;
        self.memory_initialization.initialize_memories(dt)?;
        
        if self.config.statistical_enabled {
            self.statistical_emergence.process(dt)?;
        }
        
        if self.config.noise_driven {
            self.noise_driven_compute.compute(dt)?;
        }
        
        Ok(())
    }
}
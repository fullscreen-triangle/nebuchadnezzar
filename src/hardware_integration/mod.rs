//! Hardware Integration System
//! 
//! Direct coupling with system oscillations and environmental noise

pub mod oscillation_harvesting;
pub mod noise_optimization;

pub use oscillation_harvesting::OscillationHarvesting;
pub use noise_optimization::NoiseOptimization;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct HardwareConfig {
    pub oscillation_harvesting: bool,
    pub noise_optimization: bool,
    pub pwm_integration: bool,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            oscillation_harvesting: true,
            noise_optimization: true,
            pwm_integration: true,
        }
    }
}

#[derive(Debug)]
pub struct HardwareSystem {
    config: HardwareConfig,
    oscillation_harvesting: OscillationHarvesting,
    noise_optimization: NoiseOptimization,
}

impl HardwareSystem {
    pub fn new(config: HardwareConfig) -> Result<Self> {
        Ok(Self {
            oscillation_harvesting: OscillationHarvesting::new(config.oscillation_harvesting),
            noise_optimization: NoiseOptimization::new(config.noise_optimization),
            config,
        })
    }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        if self.config.oscillation_harvesting {
            self.oscillation_harvesting.harvest(dt)?;
        }
        
        if self.config.noise_optimization {
            self.noise_optimization.optimize(dt)?;
        }
        
        Ok(())
    }
}
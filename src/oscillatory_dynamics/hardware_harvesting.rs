//! Hardware oscillation harvesting

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct HardwareHarvesting {
    enabled: bool,
    harvested_oscillations: Vec<f64>,
}

impl HardwareHarvesting {
    pub fn new(enabled: bool) -> Self {
        Self { enabled, harvested_oscillations: Vec::new() }
    }

    pub fn harvest_oscillations(&mut self, dt: f64) -> Result<()> {
        if self.enabled {
            // Harvest system oscillations
            let oscillation = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as f64 * 1e-9).sin();
            self.harvested_oscillations.push(oscillation);
            if self.harvested_oscillations.len() > 1000 {
                self.harvested_oscillations.remove(0);
            }
        }
        Ok(())
    }
}
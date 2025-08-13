//! System oscillation harvesting

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct OscillationHarvesting {
    enabled: bool,
    harvested_frequencies: Vec<f64>,
    harvest_efficiency: f64,
}

impl OscillationHarvesting {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            harvested_frequencies: Vec::new(),
            harvest_efficiency: 0.8,
        }
    }

    pub fn harvest(&mut self, dt: f64) -> Result<()> {
        if !self.enabled { return Ok(()); }
        
        // Harvest system oscillations
        let system_freq = 1000.0 + 100.0 * (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as f64 * 1e-9).sin();
            
        self.harvested_frequencies.push(system_freq);
        
        // Keep only recent frequencies
        if self.harvested_frequencies.len() > 1000 {
            self.harvested_frequencies.remove(0);
        }
        
        Ok(())
    }

    pub fn current_frequency(&self) -> f64 {
        self.harvested_frequencies.last().copied().unwrap_or(1000.0)
    }
}
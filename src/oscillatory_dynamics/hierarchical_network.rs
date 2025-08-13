//! Hierarchical oscillator network

use crate::oscillatory_dynamics::FrequencyBand;
use crate::error::{Error, Result};

#[derive(Debug)]
pub struct HierarchicalNetwork {
    oscillators: Vec<Oscillator>,
    coupling_matrix: Vec<Vec<f64>>,
}

#[derive(Debug)]
struct Oscillator {
    frequency: f64,
    amplitude: f64,
    phase: f64,
    level: usize,
}

impl HierarchicalNetwork {
    pub fn new(frequency_bands: &[FrequencyBand]) -> Result<Self> {
        let mut oscillators = Vec::new();
        
        for (level, band) in frequency_bands.iter().enumerate() {
            oscillators.push(Oscillator {
                frequency: (band.min_freq + band.max_freq) / 2.0,
                amplitude: band.amplitude,
                phase: 0.0,
                level,
            });
        }
        
        let n = oscillators.len();
        let coupling_matrix = vec![vec![0.1; n]; n];
        
        Ok(Self {
            oscillators,
            coupling_matrix,
        })
    }

    pub fn update(&mut self, dt: f64) -> Result<()> {
        // Update oscillator phases
        for osc in &mut self.oscillators {
            osc.phase += 2.0 * std::f64::consts::PI * osc.frequency * dt;
            osc.phase %= 2.0 * std::f64::consts::PI;
        }
        
        // Apply coupling
        self.apply_coupling(dt)?;
        
        Ok(())
    }
    
    fn apply_coupling(&mut self, dt: f64) -> Result<()> {
        let n = self.oscillators.len();
        let mut phase_corrections = vec![0.0; n];
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let coupling = self.coupling_matrix[i][j];
                    let phase_diff = self.oscillators[j].phase - self.oscillators[i].phase;
                    phase_corrections[i] += coupling * phase_diff.sin() * dt;
                }
            }
        }
        
        for (i, correction) in phase_corrections.iter().enumerate() {
            self.oscillators[i].phase += correction;
        }
        
        Ok(())
    }
}
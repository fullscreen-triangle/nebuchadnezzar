//! Conservation Law Validation
//! 
//! Validates energy and mass conservation in ATP system

use crate::atp_system::AtpPool;
use crate::error::{Error, Result};

/// Validator for conservation laws in ATP system
#[derive(Debug)]
pub struct ConservationValidator {
    tolerance: f64,
    initial_total_energy: Option<f64>,
    initial_total_adenylate: Option<f64>,
}

impl ConservationValidator {
    /// Create new conservation validator
    pub fn new() -> Self {
        Self {
            tolerance: 1e-6,
            initial_total_energy: None,
            initial_total_adenylate: None,
        }
    }

    /// Initialize with reference state for conservation checking
    pub fn initialize(&mut self, atp_pool: &AtpPool) {
        self.initial_total_adenylate = Some(atp_pool.total_adenylate());
        self.initial_total_energy = Some(self.calculate_total_energy(atp_pool));
    }

    /// Validate all conservation laws
    pub fn validate(&self, atp_pool: &AtpPool) -> Result<()> {
        self.validate_adenylate_conservation(atp_pool)?;
        self.validate_energy_conservation(atp_pool)?;
        self.validate_mass_balance(atp_pool)?;
        Ok(())
    }

    /// Validate adenylate pool conservation
    /// Total adenylate = [ATP] + [ADP] + [AMP] should be conserved
    pub fn validate_adenylate_conservation(&self, atp_pool: &AtpPool) -> Result<()> {
        if let Some(initial) = self.initial_total_adenylate {
            let current = atp_pool.total_adenylate();
            let difference = (current - initial).abs();
            
            if difference > self.tolerance {
                return Err(Error::InvalidConfiguration {
                    parameter: "adenylate_conservation".to_string(),
                    value: format!("Initial: {}, Current: {}, Difference: {}", 
                                 initial, current, difference),
                });
            }
        }
        Ok(())
    }

    /// Validate energy conservation
    pub fn validate_energy_conservation(&self, atp_pool: &AtpPool) -> Result<()> {
        if let Some(initial) = self.initial_total_energy {
            let current = self.calculate_total_energy(atp_pool);
            let difference = (current - initial).abs();
            
            // Energy can be dissipated but should not be created
            if difference > self.tolerance && current > initial {
                return Err(Error::InvalidConfiguration {
                    parameter: "energy_conservation".to_string(),
                    value: format!("Initial: {}, Current: {}, Difference: {}", 
                                 initial, current, difference),
                });
            }
        }
        Ok(())
    }

    /// Validate mass balance for chemical reactions
    pub fn validate_mass_balance(&self, atp_pool: &AtpPool) -> Result<()> {
        // Check that concentrations are non-negative
        if atp_pool.atp_concentration() < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "atp_concentration".to_string(),
                value: atp_pool.atp_concentration().to_string(),
            });
        }

        if atp_pool.adp_concentration() < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "adp_concentration".to_string(),
                value: atp_pool.adp_concentration().to_string(),
            });
        }

        if atp_pool.amp_concentration() < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "amp_concentration".to_string(),
                value: atp_pool.amp_concentration().to_string(),
            });
        }

        Ok(())
    }

    /// Calculate total chemical energy in the system
    fn calculate_total_energy(&self, atp_pool: &AtpPool) -> f64 {
        // Relative energies (ATP = 2 bonds, ADP = 1 bond, AMP = 0 bonds)
        let atp_energy = 2.0 * atp_pool.atp_concentration();
        let adp_energy = 1.0 * atp_pool.adp_concentration();
        let amp_energy = 0.0 * atp_pool.amp_concentration();
        
        atp_energy + adp_energy + amp_energy
    }

    /// Check for thermodynamic consistency
    pub fn validate_thermodynamics(&self, atp_pool: &AtpPool, temperature: f64) -> Result<()> {
        let energy_charge = atp_pool.energy_charge();
        
        // Energy charge should be physically reasonable
        if energy_charge < 0.0 || energy_charge > 1.0 {
            return Err(Error::EnergyChargeOutOfRange {
                value: energy_charge,
                min: 0.0,
                max: 1.0,
            });
        }

        // Check ATP/ADP ratio thermodynamic feasibility
        let atp_adp_ratio = atp_pool.atp_adp_ratio();
        if atp_adp_ratio.is_infinite() && atp_pool.atp_concentration() > 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "atp_adp_ratio".to_string(),
                value: "infinite".to_string(),
            });
        }

        Ok(())
    }

    /// Validate steady-state conditions
    pub fn validate_steady_state(&self, atp_pool: &AtpPool, synthesis_rate: f64, consumption_rate: f64) -> Result<()> {
        let energy_charge = atp_pool.energy_charge();
        
        // In steady state, synthesis should balance consumption
        let rate_difference = (synthesis_rate - consumption_rate).abs();
        let rate_tolerance = 0.1 * synthesis_rate.max(consumption_rate);
        
        if rate_difference > rate_tolerance {
            return Err(Error::InvalidConfiguration {
                parameter: "steady_state_balance".to_string(),
                value: format!("Synthesis: {}, Consumption: {}, Difference: {}", 
                             synthesis_rate, consumption_rate, rate_difference),
            });
        }

        // Energy charge should be stable in steady state
        if energy_charge < 0.7 || energy_charge > 0.95 {
            return Err(Error::EnergyChargeOutOfRange {
                value: energy_charge,
                min: 0.7,
                max: 0.95,
            });
        }

        Ok(())
    }

    /// Set tolerance for conservation checking
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }
}

impl Default for ConservationValidator {
    fn default() -> Self {
        Self::new()
    }
}
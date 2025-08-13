//! Energy Charge Calculations
//! 
//! Implements adenylate energy charge and related thermodynamic calculations

use crate::atp_system::AtpPool;

/// Calculator for adenylate energy charge and related metrics
#[derive(Debug)]
pub struct EnergyChargeCalculator {
    temperature: f64, // Kelvin
    ph: f64,
}

impl EnergyChargeCalculator {
    /// Create new energy charge calculator with physiological conditions
    pub fn new() -> Self {
        Self {
            temperature: 310.0, // 37°C in Kelvin
            ph: 7.4,
        }
    }

    /// Calculate adenylate energy charge
    /// EC = ([ATP] + 0.5[ADP]) / ([ATP] + [ADP] + [AMP])
    pub fn calculate(&self, atp_pool: &AtpPool) -> f64 {
        let atp = atp_pool.atp_concentration();
        let adp = atp_pool.adp_concentration();
        let amp = atp_pool.amp_concentration();
        
        let total = atp + adp + amp;
        if total > 0.0 {
            (atp + 0.5 * adp) / total
        } else {
            0.0
        }
    }

    /// Calculate phosphorylation potential
    /// PP = [ATP] / ([ADP][Pi])
    pub fn phosphorylation_potential(&self, atp_pool: &AtpPool, phosphate_concentration: f64) -> f64 {
        let atp = atp_pool.atp_concentration();
        let adp = atp_pool.adp_concentration();
        
        if adp > 0.0 && phosphate_concentration > 0.0 {
            atp / (adp * phosphate_concentration)
        } else {
            f64::INFINITY
        }
    }

    /// Calculate free energy of ATP hydrolysis under current conditions
    pub fn atp_hydrolysis_free_energy(&self, atp_pool: &AtpPool, phosphate_concentration: f64) -> f64 {
        // ΔG = ΔG° + RT ln([ADP][Pi]/[ATP])
        let r = 8.314; // J/(mol·K)
        let delta_g_standard = self.standard_free_energy();
        
        let atp = atp_pool.atp_concentration();
        let adp = atp_pool.adp_concentration();
        
        if atp > 0.0 {
            let reaction_quotient = (adp * phosphate_concentration) / atp;
            delta_g_standard + r * self.temperature * reaction_quotient.ln()
        } else {
            delta_g_standard
        }
    }

    /// Standard free energy of ATP hydrolysis at current pH and temperature
    fn standard_free_energy(&self) -> f64 {
        // ATP + H2O -> ADP + Pi
        // ΔG° depends on pH and temperature
        let delta_g_standard_ph7 = -30500.0; // J/mol at pH 7.0, 25°C
        
        // Temperature correction (simplified)
        let temp_correction = -50.0 * (self.temperature - 298.0); // J/mol
        
        // pH correction (simplified)
        let ph_correction = 2300.0 * (7.0 - self.ph); // J/mol
        
        delta_g_standard_ph7 + temp_correction + ph_correction
    }

    /// Calculate ATP turnover rate
    pub fn atp_turnover_rate(&self, atp_pool: &AtpPool, consumption_rate: f64) -> f64 {
        let atp = atp_pool.atp_concentration();
        if atp > 0.0 {
            consumption_rate / atp
        } else {
            0.0
        }
    }

    /// Energy charge classification
    pub fn energy_charge_status(&self, energy_charge: f64) -> EnergyChargeStatus {
        if energy_charge > 0.85 {
            EnergyChargeStatus::High
        } else if energy_charge > 0.75 {
            EnergyChargeStatus::Moderate
        } else if energy_charge > 0.60 {
            EnergyChargeStatus::Low
        } else {
            EnergyChargeStatus::Critical
        }
    }

    /// Calculate optimal energy charge for given metabolic state
    pub fn optimal_energy_charge(&self, metabolic_state: MetabolicState) -> f64 {
        match metabolic_state {
            MetabolicState::Resting => 0.85,
            MetabolicState::Active => 0.90,
            MetabolicState::HighActivity => 0.95,
            MetabolicState::Stressed => 0.80,
        }
    }

    /// Set environmental conditions
    pub fn set_conditions(&mut self, temperature: f64, ph: f64) {
        self.temperature = temperature;
        self.ph = ph;
    }
}

impl Default for EnergyChargeCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Energy charge status classification
#[derive(Debug, Clone, PartialEq)]
pub enum EnergyChargeStatus {
    High,     // > 0.85
    Moderate, // 0.75 - 0.85
    Low,      // 0.60 - 0.75
    Critical, // < 0.60
}

/// Metabolic state for energy charge optimization
#[derive(Debug, Clone, PartialEq)]
pub enum MetabolicState {
    Resting,
    Active,
    HighActivity,
    Stressed,
}
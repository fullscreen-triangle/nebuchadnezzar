//! ATP-based Kinetics Module
//! 
//! This module implements the revolutionary approach of using ATP as the fundamental
//! rate unit (dx/dATP) instead of time (dx/dt). This provides a more biologically
//! meaningful way to model intracellular processes since ATP availability directly
//! controls reaction rates in living cells.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// ATP pool representing the energy currency of the cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpPool {
    /// Current ATP concentration (mM)
    pub atp_concentration: f64,
    
    /// Current ADP concentration (mM)
    pub adp_concentration: f64,
    
    /// Current Pi (inorganic phosphate) concentration (mM)
    pub pi_concentration: f64,
    
    /// ATP/ADP ratio - critical for energy charge
    pub energy_charge: f64,
    
    /// Total adenine nucleotide pool (ATP + ADP + AMP)
    pub total_adenine_pool: f64,
    
    /// ATP synthesis rate (mM/s)
    pub atp_synthesis_rate: f64,
    
    /// ATP hydrolysis rate (mM/s)
    pub atp_hydrolysis_rate: f64,
}

impl AtpPool {
    /// Create a new ATP pool with physiological concentrations
    pub fn new_physiological() -> Self {
        Self {
            atp_concentration: 5.0,      // ~5 mM ATP
            adp_concentration: 0.5,      // ~0.5 mM ADP
            pi_concentration: 5.0,       // ~5 mM Pi
            energy_charge: 0.85,         // Typical cellular energy charge
            total_adenine_pool: 6.0,     // Total adenine nucleotides
            atp_synthesis_rate: 0.0,     // Will be calculated
            atp_hydrolysis_rate: 0.0,    // Will be calculated
        }
    }
    
    /// Calculate the current energy charge: (ATP + 0.5*ADP) / (ATP + ADP + AMP)
    pub fn calculate_energy_charge(&mut self) {
        let amp_concentration = self.total_adenine_pool - self.atp_concentration - self.adp_concentration;
        let total = self.atp_concentration + self.adp_concentration + amp_concentration;
        
        if total > 0.0 {
            self.energy_charge = (self.atp_concentration + 0.5 * self.adp_concentration) / total;
        }
    }
    
    /// Get the free energy of ATP hydrolysis under current conditions
    pub fn atp_hydrolysis_free_energy(&self) -> f64 {
        // ΔG = ΔG° + RT ln([ADP][Pi]/[ATP])
        // Using physiological values: ΔG° = -30.5 kJ/mol
        let r = 8.314; // J/(mol·K)
        let t = 310.0; // 37°C in Kelvin
        let delta_g_standard = -30500.0; // J/mol
        
        let concentration_ratio = (self.adp_concentration * self.pi_concentration) / self.atp_concentration;
        delta_g_standard + r * t * concentration_ratio.ln()
    }
    
    /// Update ATP pool based on net ATP consumption/production
    pub fn update_atp(&mut self, d_atp: f64) {
        let new_atp = self.atp_concentration + d_atp;
        let new_adp = self.adp_concentration - d_atp; // ATP consumption increases ADP
        
        // Ensure concentrations stay positive
        self.atp_concentration = new_atp.max(0.001); // Minimum 1 μM ATP
        self.adp_concentration = new_adp.max(0.001); // Minimum 1 μM ADP
        
        // Recalculate energy charge
        self.calculate_energy_charge();
    }
}

/// ATP-dependent kinetics calculator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpKinetics {
    /// Reference ATP concentration (mM) for normalization
    pub reference_atp: f64,
    
    /// ATP-dependent rate constants for different processes
    pub rate_constants: HashMap<String, AtpRateConstant>,
    
    /// Current ATP pool state
    pub atp_pool: AtpPool,
}

impl AtpKinetics {
    /// Create new ATP kinetics with physiological ATP pool
    pub fn new() -> Self {
        Self {
            reference_atp: 5.0, // 5 mM reference
            rate_constants: HashMap::new(),
            atp_pool: AtpPool::new_physiological(),
        }
    }
    
    /// Add an ATP-dependent rate constant
    pub fn add_rate_constant(&mut self, process_name: String, rate_constant: AtpRateConstant) {
        self.rate_constants.insert(process_name, rate_constant);
    }
    
    /// Calculate the rate of change with respect to ATP (dx/dATP)
    pub fn calculate_datp_rate(&self, process_name: &str, substrate_concentration: f64) -> Option<f64> {
        let rate_constant = self.rate_constants.get(process_name)?;
        
        // Calculate ATP-dependent rate: v = k * [S] * f([ATP])
        let atp_factor = rate_constant.calculate_atp_factor(self.atp_pool.atp_concentration);
        let rate = rate_constant.base_rate * substrate_concentration * atp_factor;
        
        // Convert to dx/dATP by dividing by ATP consumption rate
        // This gives change in substrate per unit ATP consumed
        let atp_stoichiometry = rate_constant.atp_stoichiometry;
        
        Some(rate / atp_stoichiometry)
    }
    
    /// Calculate the energetic efficiency of a process
    pub fn calculate_efficiency(&self, process_name: &str) -> Option<f64> {
        let rate_constant = self.rate_constants.get(process_name)?;
        let current_free_energy = self.atp_pool.atp_hydrolysis_free_energy();
        
        // Efficiency = actual energy utilization / theoretical maximum
        let theoretical_max = rate_constant.atp_stoichiometry * 30500.0; // Standard free energy
        let actual_utilization = rate_constant.atp_stoichiometry * current_free_energy.abs();
        
        Some(actual_utilization / theoretical_max)
    }
    
    /// Update ATP pool and recalculate all rates
    pub fn update_system(&mut self, net_atp_change: f64) {
        self.atp_pool.update_atp(net_atp_change);
        
        // Update synthesis and hydrolysis rates based on energy charge
        self.update_atp_turnover_rates();
    }
    
    /// Update ATP synthesis and hydrolysis rates based on current energy charge
    fn update_atp_turnover_rates(&mut self) {
        let energy_charge = self.atp_pool.energy_charge;
        
        // ATP synthesis increases when energy charge is low (sigmoidal)
        self.atp_pool.atp_synthesis_rate = 10.0 * (1.0 - energy_charge).powi(2);
        
        // ATP hydrolysis decreases when energy charge is low
        self.atp_pool.atp_hydrolysis_rate = 5.0 * energy_charge.powi(2);
    }
}

/// ATP-dependent rate constant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpRateConstant {
    /// Base rate constant (when ATP is at reference concentration)
    pub base_rate: f64,
    
    /// ATP concentration at half-maximal rate (Km for ATP)
    pub atp_km: f64,
    
    /// Hill coefficient for ATP cooperativity
    pub atp_hill_coefficient: f64,
    
    /// Number of ATP molecules consumed per reaction cycle
    pub atp_stoichiometry: f64,
    
    /// Type of ATP dependence
    pub dependence_type: AtpDependenceType,
}

impl AtpRateConstant {
    /// Calculate the ATP-dependent factor for rate calculation
    pub fn calculate_atp_factor(&self, atp_concentration: f64) -> f64 {
        match self.dependence_type {
            AtpDependenceType::Michaelis => {
                // Michaelis-Menten kinetics: v = Vmax * [ATP] / (Km + [ATP])
                atp_concentration / (self.atp_km + atp_concentration)
            }
            
            AtpDependenceType::Hill => {
                // Hill equation: v = Vmax * [ATP]^n / (Km^n + [ATP]^n)
                let atp_n = atp_concentration.powf(self.atp_hill_coefficient);
                let km_n = self.atp_km.powf(self.atp_hill_coefficient);
                atp_n / (km_n + atp_n)
            }
            
            AtpDependenceType::Linear => {
                // Linear dependence: v = k * [ATP]
                atp_concentration / self.atp_km // Normalized to reference
            }
            
            AtpDependenceType::Threshold => {
                // Step function: full rate above threshold, zero below
                if atp_concentration >= self.atp_km { 1.0 } else { 0.0 }
            }
            
            AtpDependenceType::Exponential => {
                // Exponential dependence: v = k * exp([ATP]/Km)
                (atp_concentration / self.atp_km).exp() - 1.0
            }
        }
    }
}

/// Types of ATP dependence for different biological processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AtpDependenceType {
    /// Standard Michaelis-Menten kinetics
    Michaelis,
    
    /// Cooperative binding (Hill equation)
    Hill,
    
    /// Linear dependence on ATP concentration
    Linear,
    
    /// Threshold behavior (all-or-nothing)
    Threshold,
    
    /// Exponential dependence (rare, but found in some processes)
    Exponential,
}

/// Energetic profile of a biological pathway
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergeticProfile {
    /// Total ATP consumption along the pathway
    pub total_atp_cost: f64,
    
    /// ATP yield from the pathway
    pub total_atp_yield: f64,
    
    /// Net ATP balance (yield - cost)
    pub net_atp_balance: f64,
    
    /// Energy efficiency (yield/cost)
    pub energy_efficiency: f64,
    
    /// Rate-limiting step in terms of ATP availability
    pub atp_limiting_step: Option<String>,
    
    /// ATP requirements at each step
    pub step_atp_requirements: HashMap<String, f64>,
}

impl EnergeticProfile {
    /// Create a new energetic profile
    pub fn new() -> Self {
        Self {
            total_atp_cost: 0.0,
            total_atp_yield: 0.0,
            net_atp_balance: 0.0,
            energy_efficiency: 0.0,
            atp_limiting_step: None,
            step_atp_requirements: HashMap::new(),
        }
    }
    
    /// Add ATP cost/yield for a pathway step
    pub fn add_step(&mut self, step_name: String, atp_change: f64) {
        self.step_atp_requirements.insert(step_name.clone(), atp_change);
        
        if atp_change < 0.0 {
            self.total_atp_cost += atp_change.abs();
        } else {
            self.total_atp_yield += atp_change;
        }
        
        self.calculate_metrics();
    }
    
    /// Calculate derived metrics
    fn calculate_metrics(&mut self) {
        self.net_atp_balance = self.total_atp_yield - self.total_atp_cost;
        
        if self.total_atp_cost > 0.0 {
            self.energy_efficiency = self.total_atp_yield / self.total_atp_cost;
        }
        
        // Find the most ATP-consuming step
        self.atp_limiting_step = self.step_atp_requirements
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(name, _)| name.clone());
    }
    
    /// Check if pathway is energetically favorable
    pub fn is_energetically_favorable(&self) -> bool {
        self.net_atp_balance > 0.0
    }
    
    /// Get ATP cost per product molecule
    pub fn atp_cost_per_product(&self, product_yield: f64) -> f64 {
        if product_yield > 0.0 {
            self.total_atp_cost / product_yield
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atp_pool_energy_charge() {
        let mut pool = AtpPool::new_physiological();
        pool.calculate_energy_charge();
        
        // Should be around 0.85 for physiological conditions
        assert!(pool.energy_charge > 0.8 && pool.energy_charge < 0.9);
    }

    #[test]
    fn test_atp_kinetics_rate_calculation() {
        let mut kinetics = AtpKinetics::new();
        
        // Add a simple Michaelis-Menten ATP-dependent process
        kinetics.add_rate_constant(
            "test_process".to_string(),
            AtpRateConstant {
                base_rate: 1.0,
                atp_km: 1.0,
                atp_hill_coefficient: 1.0,
                atp_stoichiometry: 1.0,
                dependence_type: AtpDependenceType::Michaelis,
            }
        );
        
        let rate = kinetics.calculate_datp_rate("test_process", 1.0);
        assert!(rate.is_some());
        assert!(rate.unwrap() > 0.0);
    }

    #[test]
    fn test_energetic_profile() {
        let mut profile = EnergeticProfile::new();
        
        profile.add_step("step1".to_string(), -2.0); // Costs 2 ATP
        profile.add_step("step2".to_string(), 5.0);  // Yields 5 ATP
        
        assert_eq!(profile.total_atp_cost, 2.0);
        assert_eq!(profile.total_atp_yield, 5.0);
        assert_eq!(profile.net_atp_balance, 3.0);
        assert!(profile.is_energetically_favorable());
    }
} 
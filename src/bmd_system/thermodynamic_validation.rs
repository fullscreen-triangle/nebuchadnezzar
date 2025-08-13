//! Thermodynamic Validation Module for BMDs
//! 
//! Ensures BMD operations satisfy thermodynamic constraints

use crate::bmd_system::BMDResponse;
use crate::error::{Error, Result};

/// Thermodynamic validator for BMD operations
#[derive(Debug)]
pub struct ThermodynamicValidator {
    max_entropy_cost: f64,
    temperature: f64, // Kelvin
    boltzmann_constant: f64,
    total_entropy_produced: f64,
    validation_history: Vec<ValidationRecord>,
}

impl ThermodynamicValidator {
    /// Create new thermodynamic validator
    pub fn new(max_entropy_cost: f64) -> Self {
        Self {
            max_entropy_cost,
            temperature: 310.0, // 37°C in Kelvin
            boltzmann_constant: 1.380649e-23, // J/K
            total_entropy_produced: 0.0,
            validation_history: Vec::new(),
        }
    }

    /// Validate BMD operation for thermodynamic consistency
    pub fn validate_operation(&mut self, response: &BMDResponse) -> Result<()> {
        // Check entropy cost
        self.validate_entropy_cost(response)?;
        
        // Check energy conservation
        self.validate_energy_conservation(response)?;
        
        // Check second law of thermodynamics
        self.validate_second_law(response)?;
        
        // Check information-entropy relationship
        self.validate_information_entropy_relationship(response)?;
        
        // Record validation
        self.record_validation(response, true);
        
        // Update total entropy
        self.total_entropy_produced += response.entropy_cost;
        
        Ok(())
    }

    /// Validate entropy cost is within acceptable limits
    fn validate_entropy_cost(&self, response: &BMDResponse) -> Result<()> {
        if response.entropy_cost < 0.0 {
            return Err(Error::BmdThermodynamicViolation {
                entropy_cost: response.entropy_cost,
            });
        }
        
        if response.entropy_cost > self.max_entropy_cost {
            return Err(Error::BmdThermodynamicViolation {
                entropy_cost: response.entropy_cost,
            });
        }
        
        Ok(())
    }

    /// Validate energy conservation
    fn validate_energy_conservation(&self, response: &BMDResponse) -> Result<()> {
        // Calculate energy input and output
        let energy_input = self.calculate_energy_input(response);
        let energy_output = self.calculate_energy_output(response);
        let energy_dissipated = response.entropy_cost * self.boltzmann_constant * self.temperature;
        
        // Energy conservation: Input = Output + Dissipated
        let energy_balance = energy_input - energy_output - energy_dissipated;
        let tolerance = 1e-12; // Energy tolerance in Joules
        
        if energy_balance.abs() > tolerance {
            return Err(Error::BmdThermodynamicViolation {
                entropy_cost: energy_balance,
            });
        }
        
        Ok(())
    }

    /// Validate second law of thermodynamics
    fn validate_second_law(&self, response: &BMDResponse) -> Result<()> {
        // Total entropy change must be non-negative
        let system_entropy_change = -self.calculate_information_gain(response); // Information gain reduces entropy
        let environment_entropy_change = response.entropy_cost;
        let total_entropy_change = system_entropy_change + environment_entropy_change;
        
        if total_entropy_change < -1e-10 { // Small tolerance for numerical errors
            return Err(Error::BmdThermodynamicViolation {
                entropy_cost: total_entropy_change,
            });
        }
        
        Ok(())
    }

    /// Validate information-entropy relationship
    fn validate_information_entropy_relationship(&self, response: &BMDResponse) -> Result<()> {
        // Landauer's principle: Information processing requires minimum entropy cost
        let information_bits = self.calculate_information_gain(response);
        let minimum_entropy_cost = information_bits * (2_f64.ln()); // kT per bit
        
        if response.entropy_cost < minimum_entropy_cost * 0.9 { // 10% tolerance
            return Err(Error::BmdThermodynamicViolation {
                entropy_cost: response.entropy_cost - minimum_entropy_cost,
            });
        }
        
        Ok(())
    }

    /// Calculate energy input to BMD operation
    fn calculate_energy_input(&self, response: &BMDResponse) -> f64 {
        // Energy from signal amplification
        let signal_energy = response.amplified_signal.magnitude.powi(2) / 2.0;
        
        // Energy from ATP (if applicable)
        let atp_energy = 30500.0; // J/mol, approximate
        let molar_concentration = 0.005; // 5mM typical
        let atp_contribution = atp_energy * molar_concentration * 1e-3; // Convert to J
        
        signal_energy + atp_contribution
    }

    /// Calculate energy output from BMD operation
    fn calculate_energy_output(&self, response: &BMDResponse) -> f64 {
        // Energy in catalyst products
        response.catalyst_result.products.iter()
            .map(|&product| product.abs() * 1e-20) // Convert to Joules (simplified)
            .sum()
    }

    /// Calculate information gain from BMD operation
    fn calculate_information_gain(&self, response: &BMDResponse) -> f64 {
        // Information gain based on pattern recognition and amplification
        let pattern_information = -response.target.confidence.log2().max(0.0);
        let amplification_information = response.amplified_signal.magnitude.log2().max(0.0);
        
        (pattern_information + amplification_information) / 2.0
    }

    /// Record validation result
    fn record_validation(&mut self, response: &BMDResponse, success: bool) {
        let record = ValidationRecord {
            entropy_cost: response.entropy_cost,
            information_gain: self.calculate_information_gain(response),
            energy_input: self.calculate_energy_input(response),
            energy_output: self.calculate_energy_output(response),
            success,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        };
        
        self.validation_history.push(record);
        
        // Keep only recent history
        if self.validation_history.len() > 1000 {
            self.validation_history.remove(0);
        }
    }

    /// Calculate thermodynamic efficiency
    pub fn calculate_efficiency(&self, response: &BMDResponse) -> f64 {
        let energy_input = self.calculate_energy_input(response);
        let useful_output = response.catalyst_result.efficiency * self.calculate_energy_output(response);
        
        if energy_input > 0.0 {
            useful_output / energy_input
        } else {
            0.0
        }
    }

    /// Check if system is thermodynamically stable
    pub fn is_stable(&self) -> bool {
        let recent_validations = self.validation_history.iter()
            .rev()
            .take(100)
            .collect::<Vec<_>>();
        
        if recent_validations.is_empty() {
            return true;
        }
        
        let success_rate = recent_validations.iter()
            .filter(|r| r.success)
            .count() as f64 / recent_validations.len() as f64;
        
        success_rate > 0.95 && self.total_entropy_produced < self.max_entropy_cost * 1000.0
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> ValidationStatistics {
        let total_validations = self.validation_history.len();
        let successful_validations = self.validation_history.iter()
            .filter(|r| r.success)
            .count();
        
        let average_entropy_cost = if total_validations > 0 {
            self.validation_history.iter()
                .map(|r| r.entropy_cost)
                .sum::<f64>() / total_validations as f64
        } else {
            0.0
        };
        
        let average_efficiency = if total_validations > 0 {
            self.validation_history.iter()
                .map(|r| if r.energy_input > 0.0 { r.energy_output / r.energy_input } else { 0.0 })
                .sum::<f64>() / total_validations as f64
        } else {
            0.0
        };
        
        ValidationStatistics {
            total_validations,
            successful_validations,
            success_rate: if total_validations > 0 { 
                successful_validations as f64 / total_validations as f64 
            } else { 
                1.0 
            },
            average_entropy_cost,
            total_entropy_produced: self.total_entropy_produced,
            average_efficiency,
        }
    }

    /// Reset validation statistics
    pub fn reset_statistics(&mut self) {
        self.validation_history.clear();
        self.total_entropy_produced = 0.0;
    }

    /// Set temperature for thermodynamic calculations
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature.max(0.1); // Prevent division by zero
    }

    /// Set maximum entropy cost
    pub fn set_max_entropy_cost(&mut self, max_cost: f64) {
        self.max_entropy_cost = max_cost.max(0.0);
    }

    /// Calculate free energy change for BMD operation
    pub fn calculate_free_energy_change(&self, response: &BMDResponse) -> f64 {
        // ΔG = ΔH - TΔS
        let enthalpy_change = self.calculate_energy_output(response) - self.calculate_energy_input(response);
        let entropy_change = response.entropy_cost * self.boltzmann_constant;
        
        enthalpy_change - self.temperature * entropy_change
    }

    /// Check if operation is thermodynamically favorable
    pub fn is_thermodynamically_favorable(&self, response: &BMDResponse) -> bool {
        let free_energy_change = self.calculate_free_energy_change(response);
        free_energy_change < 0.0 // Negative ΔG indicates favorable reaction
    }
}

/// Validation record for statistics
#[derive(Debug, Clone)]
struct ValidationRecord {
    entropy_cost: f64,
    information_gain: f64,
    energy_input: f64,
    energy_output: f64,
    success: bool,
    timestamp: f64,
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    pub total_validations: usize,
    pub successful_validations: usize,
    pub success_rate: f64,
    pub average_entropy_cost: f64,
    pub total_entropy_produced: f64,
    pub average_efficiency: f64,
}
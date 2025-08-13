//! Validation and Testing
//! 
//! Comprehensive validation of all system components

pub mod conservation_laws;
pub mod thermodynamic_consistency;
pub mod numerical_accuracy;
pub mod biological_ranges;

pub use conservation_laws::ConservationValidator;
pub use thermodynamic_consistency::ThermodynamicConsistency;
pub use numerical_accuracy::NumericalAccuracy;
pub use biological_ranges::BiologicalRanges;

use crate::error::{Error, Result};

/// Comprehensive system validator
#[derive(Debug)]
pub struct SystemValidator {
    conservation_validator: ConservationValidator,
    thermodynamic_consistency: ThermodynamicConsistency,
    numerical_accuracy: NumericalAccuracy,
    biological_ranges: BiologicalRanges,
}

impl SystemValidator {
    pub fn new() -> Self {
        Self {
            conservation_validator: ConservationValidator::new(),
            thermodynamic_consistency: ThermodynamicConsistency::new(),
            numerical_accuracy: NumericalAccuracy::new(),
            biological_ranges: BiologicalRanges::new(),
        }
    }

    /// Validate entire system state
    pub fn validate_system(&self, state: &crate::IntracellularState) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        
        // Conservation laws
        if let Err(e) = self.conservation_validator.validate_energy_conservation(state.atp_concentration) {
            report.add_violation("Energy conservation".to_string(), e.to_string());
        }
        
        // Thermodynamic consistency
        if let Err(e) = self.thermodynamic_consistency.validate_entropy_production() {
            report.add_violation("Entropy production".to_string(), e.to_string());
        }
        
        // Biological ranges
        if let Err(e) = self.biological_ranges.validate_concentrations(state.atp_concentration) {
            report.add_violation("Biological ranges".to_string(), e.to_string());
        }
        
        Ok(report)
    }
}

impl Default for SystemValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation report
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub violations: Vec<ValidationViolation>,
    pub passed: bool,
}

#[derive(Debug)]
pub struct ValidationViolation {
    pub category: String,
    pub description: String,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            passed: true,
        }
    }

    pub fn add_violation(&mut self, category: String, description: String) {
        self.violations.push(ValidationViolation { category, description });
        self.passed = false;
    }
}
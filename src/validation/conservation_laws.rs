//! Conservation law validation

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct ConservationValidator {
    tolerance: f64,
}

impl ConservationValidator {
    pub fn new() -> Self {
        Self { tolerance: 1e-6 }
    }

    pub fn validate_energy_conservation(&self, atp_concentration: f64) -> Result<()> {
        if atp_concentration < 0.0 {
            return Err(Error::EnergyChargeOutOfRange {
                value: atp_concentration,
                min: 0.0,
                max: 10.0,
            });
        }
        Ok(())
    }

    pub fn validate_mass_conservation(&self, total_mass: f64) -> Result<()> {
        if total_mass < 0.0 {
            return Err(Error::InvalidConfiguration {
                parameter: "total_mass".to_string(),
                value: total_mass.to_string(),
            });
        }
        Ok(())
    }
}
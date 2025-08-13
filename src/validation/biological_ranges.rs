//! Biological parameter range validation

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct BiologicalRanges {
    atp_min: f64,
    atp_max: f64,
    temperature_min: f64,
    temperature_max: f64,
}

impl BiologicalRanges {
    pub fn new() -> Self {
        Self {
            atp_min: 1.0,   // 1 mM
            atp_max: 10.0,  // 10 mM
            temperature_min: 273.0, // 0°C
            temperature_max: 323.0, // 50°C
        }
    }

    pub fn validate_concentrations(&self, atp_concentration: f64) -> Result<()> {
        if atp_concentration < self.atp_min || atp_concentration > self.atp_max {
            return Err(Error::EnergyChargeOutOfRange {
                value: atp_concentration,
                min: self.atp_min,
                max: self.atp_max,
            });
        }
        Ok(())
    }

    pub fn validate_temperature(&self, temperature: f64) -> Result<()> {
        if temperature < self.temperature_min || temperature > self.temperature_max {
            return Err(Error::InvalidConfiguration {
                parameter: "temperature".to_string(),
                value: temperature.to_string(),
            });
        }
        Ok(())
    }
}
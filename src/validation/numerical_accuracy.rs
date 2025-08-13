//! Numerical accuracy validation

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct NumericalAccuracy {
    tolerance: f64,
}

impl NumericalAccuracy {
    pub fn new() -> Self {
        Self { tolerance: 1e-10 }
    }

    pub fn validate_convergence(&self, error: f64, iterations: usize) -> Result<()> {
        if error > self.tolerance {
            return Err(Error::ConvergenceFailure { iterations });
        }
        Ok(())
    }

    pub fn validate_stability(&self, value: f64) -> Result<()> {
        if value.is_infinite() || value.is_nan() {
            return Err(Error::NumericalInstability);
        }
        Ok(())
    }
}
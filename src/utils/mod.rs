//! # Utilities Module: Mathematical and Statistical Support
//! 
//! This module provides supporting mathematical and statistical functions for the
//! Nebuchadnezzar framework, including probability distributions, statistical analysis,
//! and model validation utilities.

pub mod probability;
pub mod statistics;
pub mod validation;

use crate::error::{NebuchadnezzarError, Result};
use serde::{Deserialize, Serialize};

/// Mathematical constants used throughout the framework
pub mod constants {
    /// Gas constant (J/(mol·K))
    pub const R: f64 = 8.314462618;
    
    /// Faraday constant (C/mol)
    pub const F: f64 = 96485.33289;
    
    /// Standard temperature (K)
    pub const T_STANDARD: f64 = 298.15;
    
    /// Physiological temperature (K)
    pub const T_PHYSIOLOGICAL: f64 = 310.15;
    
    /// Standard ATP hydrolysis free energy (kJ/mol)
    pub const DELTA_G_ATP_STANDARD: f64 = -30.5;
    
    /// Typical physiological ATP free energy (kJ/mol)
    pub const DELTA_G_ATP_PHYSIOLOGICAL: f64 = -54.0;
    
    /// Avogadro's number (1/mol)
    pub const N_A: f64 = 6.02214076e23;
    
    /// Boltzmann constant (J/K)
    pub const K_B: f64 = 1.380649e-23;
}

/// Unit conversion utilities
pub mod units {
    /// Convert mV to V
    pub fn mv_to_v(mv: f64) -> f64 {
        mv / 1000.0
    }
    
    /// Convert V to mV
    pub fn v_to_mv(v: f64) -> f64 {
        v * 1000.0
    }
    
    /// Convert pA to A
    pub fn pa_to_a(pa: f64) -> f64 {
        pa / 1e12
    }
    
    /// Convert A to pA
    pub fn a_to_pa(a: f64) -> f64 {
        a * 1e12
    }
    
    /// Convert pS to S
    pub fn ps_to_s(ps: f64) -> f64 {
        ps / 1e12
    }
    
    /// Convert S to pS
    pub fn s_to_ps(s: f64) -> f64 {
        s * 1e12
    }
    
    /// Convert pF to F
    pub fn pf_to_f(pf: f64) -> f64 {
        pf / 1e12
    }
    
    /// Convert F to pF
    pub fn f_to_pf(f: f64) -> f64 {
        f * 1e12
    }
}

/// Thermodynamic calculations
pub mod thermodynamics {
    use super::constants::*;
    
    /// Calculate Nernst potential for an ion
    /// E = (RT/zF) * ln([ion_out]/[ion_in])
    pub fn nernst_potential(
        ion_out: f64,
        ion_in: f64,
        valence: i32,
        temperature: f64,
    ) -> f64 {
        let rt_over_f = (R * temperature) / F;
        let ln_ratio = (ion_out / ion_in).ln();
        rt_over_f * ln_ratio / (valence as f64) * 1000.0 // Convert to mV
    }
    
    /// Calculate Goldman-Hodgkin-Katz voltage
    pub fn ghk_voltage(
        p_na: f64, na_out: f64, na_in: f64,
        p_k: f64, k_out: f64, k_in: f64,
        p_cl: f64, cl_out: f64, cl_in: f64,
        temperature: f64,
    ) -> f64 {
        let rt_over_f = (R * temperature) / F;
        
        let numerator = p_na * na_out + p_k * k_out + p_cl * cl_in;
        let denominator = p_na * na_in + p_k * k_in + p_cl * cl_out;
        
        rt_over_f * (numerator / denominator).ln() * 1000.0 // Convert to mV
    }
    
    /// Calculate free energy change for ATP hydrolysis
    pub fn atp_free_energy(
        atp_conc: f64,
        adp_conc: f64,
        pi_conc: f64,
        temperature: f64,
    ) -> f64 {
        let rt = R * temperature / 1000.0; // Convert to kJ/(mol·K)
        let ratio = (adp_conc * pi_conc) / atp_conc;
        DELTA_G_ATP_STANDARD + rt * ratio.ln()
    }
    
    /// Calculate energy charge
    pub fn energy_charge(atp: f64, adp: f64, amp: f64) -> f64 {
        (atp + 0.5 * adp) / (atp + adp + amp)
    }
}

/// Numerical utilities
pub mod numerical {
    use super::Result;
    use crate::error::NebuchadnezzarError;
    
    /// Solve quadratic equation ax² + bx + c = 0
    pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Result<(f64, f64)> {
        if a.abs() < f64::EPSILON {
            return Err(NebuchadnezzarError::ComputationError(
                "Not a quadratic equation (a = 0)".to_string()
            ));
        }
        
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return Err(NebuchadnezzarError::ComputationError(
                "No real solutions (negative discriminant)".to_string()
            ));
        }
        
        let sqrt_disc = discriminant.sqrt();
        let x1 = (-b + sqrt_disc) / (2.0 * a);
        let x2 = (-b - sqrt_disc) / (2.0 * a);
        
        Ok((x1, x2))
    }
    
    /// Linear interpolation
    pub fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
    
    /// Bilinear interpolation
    pub fn bilinear_interp(
        x0: f64, y0: f64, x1: f64, y1: f64,
        q00: f64, q01: f64, q10: f64, q11: f64,
        x: f64, y: f64,
    ) -> f64 {
        let r1 = lerp(x0, q00, x1, q10, x);
        let r2 = lerp(x0, q01, x1, q11, x);
        lerp(y0, r1, y1, r2, y)
    }
    
    /// Clamp value to range [min, max]
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        value.max(min).min(max)
    }
    
    /// Sigmoid function
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Hill function
    pub fn hill(x: f64, k: f64, n: f64) -> f64 {
        x.powf(n) / (k.powf(n) + x.powf(n))
    }
    
    /// Michaelis-Menten function
    pub fn michaelis_menten(s: f64, km: f64, vmax: f64) -> f64 {
        vmax * s / (km + s)
    }
}

/// Matrix operations for linear algebra
pub mod matrix {
    use super::Result;
    use crate::error::NebuchadnezzarError;
    
    /// Simple matrix type for small matrices
    #[derive(Debug, Clone, PartialEq)]
    pub struct Matrix {
        pub data: Vec<Vec<f64>>,
        pub rows: usize,
        pub cols: usize,
    }
    
    impl Matrix {
        pub fn new(rows: usize, cols: usize) -> Self {
            Self {
                data: vec![vec![0.0; cols]; rows],
                rows,
                cols,
            }
        }
        
        pub fn identity(size: usize) -> Self {
            let mut matrix = Self::new(size, size);
            for i in 0..size {
                matrix.data[i][i] = 1.0;
            }
            matrix
        }
        
        pub fn from_vec(data: Vec<Vec<f64>>) -> Result<Self> {
            if data.is_empty() {
                return Err(NebuchadnezzarError::ComputationError(
                    "Empty matrix data".to_string()
                ));
            }
            
            let rows = data.len();
            let cols = data[0].len();
            
            // Check that all rows have the same length
            for row in &data {
                if row.len() != cols {
                    return Err(NebuchadnezzarError::ComputationError(
                        "Inconsistent row lengths".to_string()
                    ));
                }
            }
            
            Ok(Self { data, rows, cols })
        }
        
        pub fn get(&self, row: usize, col: usize) -> Result<f64> {
            if row >= self.rows || col >= self.cols {
                return Err(NebuchadnezzarError::ComputationError(
                    "Matrix index out of bounds".to_string()
                ));
            }
            Ok(self.data[row][col])
        }
        
        pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
            if row >= self.rows || col >= self.cols {
                return Err(NebuchadnezzarError::ComputationError(
                    "Matrix index out of bounds".to_string()
                ));
            }
            self.data[row][col] = value;
            Ok(())
        }
        
        /// Matrix multiplication
        pub fn multiply(&self, other: &Matrix) -> Result<Matrix> {
            if self.cols != other.rows {
                return Err(NebuchadnezzarError::ComputationError(
                    "Matrix dimensions incompatible for multiplication".to_string()
                ));
            }
            
            let mut result = Matrix::new(self.rows, other.cols);
            
            for i in 0..self.rows {
                for j in 0..other.cols {
                    let mut sum = 0.0;
                    for k in 0..self.cols {
                        sum += self.data[i][k] * other.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }
            
            Ok(result)
        }
        
        /// Matrix-vector multiplication
        pub fn multiply_vector(&self, vector: &[f64]) -> Result<Vec<f64>> {
            if self.cols != vector.len() {
                return Err(NebuchadnezzarError::ComputationError(
                    "Matrix and vector dimensions incompatible".to_string()
                ));
            }
            
            let mut result = vec![0.0; self.rows];
            
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result[i] += self.data[i][j] * vector[j];
                }
            }
            
            Ok(result)
        }
        
        /// LU decomposition for solving linear systems
        pub fn lu_decomposition(&self) -> Result<(Matrix, Matrix)> {
            if self.rows != self.cols {
                return Err(NebuchadnezzarError::ComputationError(
                    "Matrix must be square for LU decomposition".to_string()
                ));
            }
            
            let n = self.rows;
            let mut l = Matrix::identity(n);
            let mut u = self.clone();
            
            for i in 0..n {
                // Find pivot
                let mut max_row = i;
                for k in i + 1..n {
                    if u.data[k][i].abs() > u.data[max_row][i].abs() {
                        max_row = k;
                    }
                }
                
                // Swap rows if needed
                if max_row != i {
                    u.data.swap(i, max_row);
                }
                
                // Check for zero pivot
                if u.data[i][i].abs() < f64::EPSILON {
                    return Err(NebuchadnezzarError::ComputationError(
                        "Matrix is singular".to_string()
                    ));
                }
                
                // Eliminate column
                for k in i + 1..n {
                    let factor = u.data[k][i] / u.data[i][i];
                    l.data[k][i] = factor;
                    
                    for j in i..n {
                        u.data[k][j] -= factor * u.data[i][j];
                    }
                }
            }
            
            Ok((l, u))
        }
    }
}

// Re-export utilities
pub use probability::*;
pub use statistics::*;
pub use validation::*; 
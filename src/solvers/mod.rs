//! # Solvers Module: ATP-Based Differential Equation Integration
//! 
//! This module provides the core numerical methods for solving ATP-based differential equations,
//! the mathematical foundation of the Nebuchadnezzar framework. It implements both explicit and
//! implicit integration methods specifically designed for ATP-coupled systems.
//!
//! ## Core Innovation: ATP-Based Integration
//!
//! Instead of traditional time-based integration (dx/dt), this module implements:
//! - dx/dATP integration methods
//! - ATP consumption-driven adaptive stepping
//! - Energy-aware error control
//! - Stiff system handling for ATP-coupled reactions
//!
//! ## Mathematical Foundation
//!
//! The fundamental transformation: dx/dATP = (dx/dt) / (dATP/dt)
//! This allows integration with respect to ATP consumption rather than time.

pub mod ode_solvers;
pub mod stochastic;
pub mod adaptive;
pub mod linear_algebra;

use crate::error::{NebuchadnezzarError, Result};
use crate::systems_biology::AtpPool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// System state vector for ATP-based integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Metabolite concentrations (mM)
    pub concentrations: Vec<f64>,
    
    /// ATP pool state
    pub atp_pool: AtpPool,
    
    /// Electrical state variables (mV, pA)
    pub electrical_state: Vec<f64>,
    
    /// Gating variables for ion channels [0,1]
    pub gating_variables: Vec<f64>,
    
    /// System time (s)
    pub time: f64,
    
    /// Cumulative ATP consumption (mM)
    pub cumulative_atp: f64,
    
    /// Additional state variables
    pub additional_variables: HashMap<String, f64>,
}

impl SystemState {
    pub fn new(n_metabolites: usize, n_electrical: usize, n_gating: usize) -> Self {
        Self {
            concentrations: vec![0.0; n_metabolites],
            atp_pool: AtpPool::new_physiological(),
            electrical_state: vec![0.0; n_electrical],
            gating_variables: vec![0.0; n_gating],
            time: 0.0,
            cumulative_atp: 0.0,
            additional_variables: HashMap::new(),
        }
    }

    pub fn zeros(size: usize) -> Self {
        Self::new(size, 0, 0)
    }

    pub fn size(&self) -> usize {
        self.concentrations.len() + 
        5 + // ATP pool components
        self.electrical_state.len() + 
        self.gating_variables.len() +
        self.additional_variables.len()
    }

    /// Convert to flat vector for numerical operations
    pub fn to_vector(&self) -> Vec<f64> {
        let mut vec = Vec::new();
        
        // Metabolite concentrations
        vec.extend_from_slice(&self.concentrations);
        
        // ATP pool state
        vec.push(self.atp_pool.atp_concentration);
        vec.push(self.atp_pool.adp_concentration);
        vec.push(self.atp_pool.amp_concentration);
        vec.push(self.atp_pool.pi_concentration);
        vec.push(self.atp_pool.energy_charge);
        
        // Electrical state
        vec.extend_from_slice(&self.electrical_state);
        
        // Gating variables
        vec.extend_from_slice(&self.gating_variables);
        
        // Additional variables (in deterministic order)
        let mut additional_keys: Vec<_> = self.additional_variables.keys().collect();
        additional_keys.sort();
        for key in additional_keys {
            vec.push(self.additional_variables[key]);
        }
        
        vec
    }

    /// Create from flat vector
    pub fn from_vector(&self, vec: &[f64]) -> Result<Self> {
        let mut index = 0;
        let mut new_state = self.clone();

        // Metabolite concentrations
        let n_metabolites = self.concentrations.len();
        if vec.len() < index + n_metabolites {
            return Err(NebuchadnezzarError::ComputationError(
                "Vector too short for metabolite concentrations".to_string()
            ));
        }
        new_state.concentrations.copy_from_slice(&vec[index..index + n_metabolites]);
        index += n_metabolites;

        // ATP pool
        if vec.len() < index + 5 {
            return Err(NebuchadnezzarError::ComputationError(
                "Vector too short for ATP pool".to_string()
            ));
        }
        new_state.atp_pool.atp_concentration = vec[index];
        new_state.atp_pool.adp_concentration = vec[index + 1];
        new_state.atp_pool.amp_concentration = vec[index + 2];
        new_state.atp_pool.pi_concentration = vec[index + 3];
        new_state.atp_pool.energy_charge = vec[index + 4];
        index += 5;

        // Electrical state
        let n_electrical = self.electrical_state.len();
        if vec.len() < index + n_electrical {
            return Err(NebuchadnezzarError::ComputationError(
                "Vector too short for electrical state".to_string()
            ));
        }
        new_state.electrical_state.copy_from_slice(&vec[index..index + n_electrical]);
        index += n_electrical;

        // Gating variables
        let n_gating = self.gating_variables.len();
        if vec.len() < index + n_gating {
            return Err(NebuchadnezzarError::ComputationError(
                "Vector too short for gating variables".to_string()
            ));
        }
        new_state.gating_variables.copy_from_slice(&vec[index..index + n_gating]);
        index += n_gating;

        // Additional variables
        let mut additional_keys: Vec<_> = self.additional_variables.keys().collect();
        additional_keys.sort();
        for key in additional_keys {
            if vec.len() <= index {
                return Err(NebuchadnezzarError::ComputationError(
                    "Vector too short for additional variables".to_string()
                ));
            }
            new_state.additional_variables.insert(key.clone(), vec[index]);
            index += 1;
        }

        Ok(new_state)
    }
}

/// Trait for ATP-based derivative calculation
pub trait AtpDerivativeFunction {
    fn calculate_derivatives(
        &self,
        state: &SystemState,
        parameters: &SystemParameters,
    ) -> Result<SystemState>;
    
    fn calculate_atp_consumption_rate(
        &self,
        state: &SystemState,
        parameters: &SystemParameters,
    ) -> Result<f64>;
}

/// System parameters for ATP-based differential equations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParameters {
    /// Metabolite names
    pub metabolites: Vec<String>,
    
    /// Reaction parameters
    pub reactions: Vec<ReactionParameters>,
    
    /// Membrane parameters for electrical dynamics
    pub membranes: Vec<MembraneParameters>,
    
    /// ATP synthesis/consumption parameters
    pub atp_parameters: AtpParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionParameters {
    pub name: String,
    pub rate_constant: f64,
    pub atp_stoichiometry: f64, // ATP consumed (+) or produced (-)
    pub substrates: HashMap<String, f64>, // substrate name -> stoichiometry
    pub products: HashMap<String, f64>,   // product name -> stoichiometry
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneParameters {
    pub name: String,
    pub capacitance: f64, // pF
    pub area: f64,        // μm²
    pub ion_channels: Vec<IonChannelParameters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonChannelParameters {
    pub name: String,
    pub max_conductance: f64, // pS
    pub reversal_potential: f64, // mV
    pub gating_variables: Vec<String>,
    pub atp_dependence: Option<AtpDependence>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpDependence {
    pub km: f64,
    pub hill_coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpParameters {
    pub synthesis_rate: f64,  // mM/s
    pub basal_consumption_rate: f64, // mM/s
    pub adenylate_kinase_keq: f64, // Equilibrium constant
}

/// ATP-based integration methods
pub trait AtpIntegrator {
    fn step(
        &mut self,
        state: &SystemState,
        d_atp: f64,
        derivative_fn: &dyn AtpDerivativeFunction,
        parameters: &SystemParameters,
    ) -> Result<SystemState>;
    
    fn step_adaptive(
        &mut self,
        state: &SystemState,
        d_atp_initial: f64,
        tolerance: f64,
        derivative_fn: &dyn AtpDerivativeFunction,
        parameters: &SystemParameters,
    ) -> Result<(SystemState, f64)>;
}

/// Forward Euler method for ATP-based ODEs
pub struct AtpEulerIntegrator;

impl AtpIntegrator for AtpEulerIntegrator {
    fn step(
        &mut self,
        state: &SystemState,
        d_atp: f64,
        derivative_fn: &dyn AtpDerivativeFunction,
        parameters: &SystemParameters,
    ) -> Result<SystemState> {
        let derivatives = derivative_fn.calculate_derivatives(state, parameters)?;
        let state_vec = state.to_vector();
        let deriv_vec = derivatives.to_vector();
        
        let new_vec: Vec<f64> = state_vec.iter()
            .zip(deriv_vec.iter())
            .map(|(s, d)| s + d * d_atp)
            .collect();
        
        let mut new_state = state.from_vector(&new_vec)?;
        new_state.cumulative_atp += d_atp;
        
        // Update time based on ATP consumption rate
        let atp_rate = derivative_fn.calculate_atp_consumption_rate(state, parameters)?;
        if atp_rate.abs() > f64::EPSILON {
            new_state.time += d_atp / atp_rate;
        }
        
        Ok(new_state)
    }
    
    fn step_adaptive(
        &mut self,
        state: &SystemState,
        d_atp_initial: f64,
        tolerance: f64,
        derivative_fn: &dyn AtpDerivativeFunction,
        parameters: &SystemParameters,
    ) -> Result<(SystemState, f64)> {
        // Simple adaptive stepping for Euler method
        let mut d_atp = d_atp_initial;
        
        loop {
            // Calculate with full step
            let result_full = self.step(state, d_atp, derivative_fn, parameters)?;
            
            // Calculate with two half steps
            let intermediate = self.step(state, d_atp / 2.0, derivative_fn, parameters)?;
            let result_half = self.step(&intermediate, d_atp / 2.0, derivative_fn, parameters)?;
            
            // Estimate error
            let error = calculate_error_estimate(&result_full, &result_half)?;
            
            if error < tolerance {
                // Accept step, possibly increase step size
                let new_d_atp = d_atp * (tolerance / error).powf(0.2).min(2.0);
                return Ok((result_half, new_d_atp));
            } else {
                // Reject step, decrease step size
                d_atp *= (tolerance / error).powf(0.25).max(0.1);
                if d_atp < 1e-12 {
                    return Err(NebuchadnezzarError::ConvergenceError(
                        "ATP step size too small".to_string()
                    ));
                }
            }
        }
    }
}

/// 4th-order Runge-Kutta method for ATP-based ODEs
pub struct AtpRk4Integrator;

impl AtpIntegrator for AtpRk4Integrator {
    fn step(
        &mut self,
        state: &SystemState,
        d_atp: f64,
        derivative_fn: &dyn AtpDerivativeFunction,
        parameters: &SystemParameters,
    ) -> Result<SystemState> {
        // k1 = f(y_n)
        let k1 = derivative_fn.calculate_derivatives(state, parameters)?;
        
        // k2 = f(y_n + d_atp/2 * k1)
        let state_k2 = add_scaled_derivative(state, &k1, d_atp / 2.0)?;
        let k2 = derivative_fn.calculate_derivatives(&state_k2, parameters)?;
        
        // k3 = f(y_n + d_atp/2 * k2)
        let state_k3 = add_scaled_derivative(state, &k2, d_atp / 2.0)?;
        let k3 = derivative_fn.calculate_derivatives(&state_k3, parameters)?;
        
        // k4 = f(y_n + d_atp * k3)
        let state_k4 = add_scaled_derivative(state, &k3, d_atp)?;
        let k4 = derivative_fn.calculate_derivatives(&state_k4, parameters)?;
        
        // Combine derivatives with RK4 weights
        let combined = combine_derivatives(&[k1, k2, k3, k4], &[1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])?;
        
        let mut new_state = add_scaled_derivative(state, &combined, d_atp)?;
        new_state.cumulative_atp += d_atp;
        
        // Update time
        let atp_rate = derivative_fn.calculate_atp_consumption_rate(state, parameters)?;
        if atp_rate.abs() > f64::EPSILON {
            new_state.time += d_atp / atp_rate;
        }
        
        Ok(new_state)
    }
    
    fn step_adaptive(
        &mut self,
        state: &SystemState,
        d_atp_initial: f64,
        tolerance: f64,
        derivative_fn: &dyn AtpDerivativeFunction,
        parameters: &SystemParameters,
    ) -> Result<(SystemState, f64)> {
        let mut d_atp = d_atp_initial;
        
        loop {
            // Calculate with full step
            let result_full = self.step(state, d_atp, derivative_fn, parameters)?;
            
            // Calculate with two half steps
            let intermediate = self.step(state, d_atp / 2.0, derivative_fn, parameters)?;
            let result_half = self.step(&intermediate, d_atp / 2.0, derivative_fn, parameters)?;
            
            // Estimate error (RK4 has higher order, so error is O(h^5))
            let error = calculate_error_estimate(&result_full, &result_half)?;
            
            if error < tolerance {
                // Accept step, possibly increase step size
                let new_d_atp = d_atp * (tolerance / error).powf(0.2).min(2.0);
                return Ok((result_half, new_d_atp));
            } else {
                // Reject step, decrease step size
                d_atp *= (tolerance / error).powf(0.25).max(0.1);
                if d_atp < 1e-12 {
                    return Err(NebuchadnezzarError::ConvergenceError(
                        "ATP step size too small".to_string()
                    ));
                }
            }
        }
    }
}

// Helper functions for ATP-based integration

fn add_scaled_derivative(
    state: &SystemState,
    derivative: &SystemState,
    scale: f64,
) -> Result<SystemState> {
    let state_vec = state.to_vector();
    let deriv_vec = derivative.to_vector();
    
    let new_vec: Vec<f64> = state_vec.iter()
        .zip(deriv_vec.iter())
        .map(|(s, d)| s + d * scale)
        .collect();
    
    state.from_vector(&new_vec)
}

fn combine_derivatives(
    derivatives: &[SystemState],
    weights: &[f64],
) -> Result<SystemState> {
    if derivatives.is_empty() {
        return Err(NebuchadnezzarError::ComputationError(
            "No derivatives to combine".to_string()
        ));
    }
    
    if derivatives.len() != weights.len() {
        return Err(NebuchadnezzarError::ComputationError(
            "Derivatives and weights length mismatch".to_string()
        ));
    }
    
    let size = derivatives[0].size();
    let mut combined_vec = vec![0.0; size];
    
    for (derivative, weight) in derivatives.iter().zip(weights.iter()) {
        let deriv_vec = derivative.to_vector();
        for (i, &val) in deriv_vec.iter().enumerate() {
            combined_vec[i] += weight * val;
        }
    }
    
    derivatives[0].from_vector(&combined_vec)
}

fn calculate_error_estimate(
    result_full: &SystemState,
    result_half: &SystemState,
) -> Result<f64> {
    let full_vec = result_full.to_vector();
    let half_vec = result_half.to_vector();
    
    let mut error = 0.0;
    for (full, half) in full_vec.iter().zip(half_vec.iter()) {
        let diff = (full - half).abs();
        let scale = full.abs().max(half.abs()).max(1e-10);
        error = error.max(diff / scale);
    }
    
    Ok(error)
}

// Re-export key components
pub use ode_solvers::*;
pub use stochastic::*;
pub use adaptive::*;
pub use linear_algebra::*; 
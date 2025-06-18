//! # ATP-Based Differential Equation Solvers
//! 
//! This module provides numerical integration methods for solving differential equations
//! with ATP consumption as the independent variable (dx/dATP instead of dx/dt).
//! 
//! ## Solver Architecture
//! 
//! The solver system includes:
//! - **Basic ATP Integrators**: Euler, RK4, Adaptive methods
//! - **Advanced Solvers**: Linear/nonlinear, Laplace domain, stochastic
//! - **Interfacial Process Modeling**: O2 uptake, temperature, pH effects
//! - **Multi-scale Hybrid Solvers**: Combining multiple approaches

use crate::error::{NebuchadnezzarError, Result};
use crate::systems_biology::AtpPool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod advanced_solvers;

pub use advanced_solvers::*;

/// System state representation for ATP-based differential equations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub concentrations: Vec<f64>,
    pub fluxes: Vec<f64>,
    pub voltages: Vec<f64>,
    pub atp_pool: AtpPool,
    pub time: f64,
    pub cumulative_atp: f64,
    pub metabolite_names: Vec<String>,
}

impl SystemState {
    pub fn new(n_concentrations: usize, n_fluxes: usize, n_voltages: usize) -> Self {
        Self {
            concentrations: vec![0.0; n_concentrations],
            fluxes: vec![0.0; n_fluxes],
            voltages: vec![0.0; n_voltages],
            atp_pool: AtpPool::new(5.0, 1.0, 0.5),
            time: 0.0,
            cumulative_atp: 0.0,
            metabolite_names: Vec::new(),
        }
    }

    pub fn set_metabolite_names(&mut self, names: Vec<String>) {
        self.metabolite_names = names;
    }

    pub fn get_concentration(&self, metabolite: &str) -> Option<f64> {
        self.metabolite_names.iter()
            .position(|name| name == metabolite)
            .map(|index| self.concentrations[index])
    }

    pub fn set_concentration(&mut self, metabolite: &str, concentration: f64) -> Result<()> {
        if let Some(index) = self.metabolite_names.iter().position(|name| name == metabolite) {
            self.concentrations[index] = concentration;
            Ok(())
        } else {
            Err(NebuchadnezzarError::ComputationError(
                format!("Metabolite not found: {}", metabolite)
            ))
        }
    }
}

/// Base trait for ATP-based integrators
pub trait AtpIntegrator {
    fn integrate_step(&mut self, state: &SystemState, delta_atp: f64) -> Result<SystemState>;
    fn set_step_size(&mut self, step_size: f64);
    fn get_step_size(&self) -> f64;
}

/// Euler integrator for ATP-based systems
pub struct AtpEulerIntegrator {
    step_size: f64,
}

impl AtpEulerIntegrator {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl AtpIntegrator for AtpEulerIntegrator {
    fn integrate_step(&mut self, state: &SystemState, delta_atp: f64) -> Result<SystemState> {
        let mut new_state = state.clone();
        
        // Simple Euler integration: x_new = x_old + (dx/dATP) * delta_ATP
        for i in 0..state.concentrations.len() {
            let derivative = self.compute_derivative(state, i, delta_atp)?;
            new_state.concentrations[i] += derivative * delta_atp;
            
            // Ensure non-negative concentrations
            if new_state.concentrations[i] < 0.0 {
                new_state.concentrations[i] = 0.0;
            }
        }

        // Update ATP pool
        new_state.atp_pool.consume_atp(delta_atp)?;
        new_state.cumulative_atp += delta_atp;

        Ok(new_state)
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn get_step_size(&self) -> f64 {
        self.step_size
    }
}

impl AtpEulerIntegrator {
    fn compute_derivative(&self, state: &SystemState, metabolite_index: usize, delta_atp: f64) -> Result<f64> {
        // Simplified derivative calculation
        // In practice, this would involve complex biochemical rate laws
        let concentration = state.concentrations[metabolite_index];
        let atp_effect = state.atp_pool.atp_concentration / (1.0 + state.atp_pool.atp_concentration);
        
        // Simple first-order kinetics with ATP dependence
        Ok(-0.1 * concentration * atp_effect)
    }
}

/// Runge-Kutta 4th order integrator for ATP-based systems
pub struct AtpRk4Integrator {
    step_size: f64,
}

impl AtpRk4Integrator {
    pub fn new(step_size: f64) -> Self {
        Self { step_size }
    }
}

impl AtpIntegrator for AtpRk4Integrator {
    fn integrate_step(&mut self, state: &SystemState, delta_atp: f64) -> Result<SystemState> {
        let mut new_state = state.clone();
        
        // RK4 integration for each concentration
        for i in 0..state.concentrations.len() {
            let k1 = self.compute_derivative(state, i, delta_atp)?;
            
            // Create intermediate state for k2
            let mut temp_state = state.clone();
            temp_state.concentrations[i] += k1 * delta_atp * 0.5;
            let k2 = self.compute_derivative(&temp_state, i, delta_atp)?;
            
            // Create intermediate state for k3
            temp_state.concentrations[i] = state.concentrations[i] + k2 * delta_atp * 0.5;
            let k3 = self.compute_derivative(&temp_state, i, delta_atp)?;
            
            // Create intermediate state for k4
            temp_state.concentrations[i] = state.concentrations[i] + k3 * delta_atp;
            let k4 = self.compute_derivative(&temp_state, i, delta_atp)?;
            
            // RK4 formula
            new_state.concentrations[i] += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * delta_atp / 6.0;
            
            // Ensure non-negative concentrations
            if new_state.concentrations[i] < 0.0 {
                new_state.concentrations[i] = 0.0;
            }
        }

        // Update ATP pool
        new_state.atp_pool.consume_atp(delta_atp)?;
        new_state.cumulative_atp += delta_atp;

        Ok(new_state)
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn get_step_size(&self) -> f64 {
        self.step_size
    }
}

impl AtpRk4Integrator {
    fn compute_derivative(&self, state: &SystemState, metabolite_index: usize, delta_atp: f64) -> Result<f64> {
        // More sophisticated derivative calculation for RK4
        let concentration = state.concentrations[metabolite_index];
        let atp_effect = state.atp_pool.atp_concentration / (0.5 + state.atp_pool.atp_concentration);
        
        // Include flux terms if available
        let flux_contribution = if metabolite_index < state.fluxes.len() {
            state.fluxes[metabolite_index]
        } else {
            0.0
        };
        
        // Michaelis-Menten-like kinetics with ATP dependence
        let km = 1.0;
        let vmax = 2.0;
        let rate = vmax * concentration / (km + concentration) * atp_effect;
        
        Ok(flux_contribution - rate)
    }
}

/// Adaptive step size integrator
pub struct AdaptiveStepIntegrator {
    base_integrator: AtpRk4Integrator,
    min_step: f64,
    max_step: f64,
    tolerance: f64,
    safety_factor: f64,
}

impl AdaptiveStepIntegrator {
    pub fn new(initial_step: f64, tolerance: f64) -> Self {
        Self {
            base_integrator: AtpRk4Integrator::new(initial_step),
            min_step: initial_step * 0.01,
            max_step: initial_step * 100.0,
            tolerance,
            safety_factor: 0.9,
        }
    }

    fn estimate_error(&self, state1: &SystemState, state2: &SystemState) -> f64 {
        let mut max_error = 0.0;
        
        for i in 0..state1.concentrations.len() {
            let error = (state1.concentrations[i] - state2.concentrations[i]).abs();
            let scale = state1.concentrations[i].abs().max(state2.concentrations[i].abs()).max(1e-10);
            let relative_error = error / scale;
            max_error = max_error.max(relative_error);
        }
        
        max_error
    }

    fn adjust_step_size(&mut self, error: f64) {
        if error > self.tolerance {
            // Reduce step size
            let factor = self.safety_factor * (self.tolerance / error).powf(0.25);
            let new_step = self.base_integrator.get_step_size() * factor.max(0.1);
            self.base_integrator.set_step_size(new_step.max(self.min_step));
        } else if error < self.tolerance * 0.1 {
            // Increase step size
            let factor = self.safety_factor * (self.tolerance / error).powf(0.2);
            let new_step = self.base_integrator.get_step_size() * factor.min(2.0);
            self.base_integrator.set_step_size(new_step.min(self.max_step));
        }
    }
}

impl AtpIntegrator for AdaptiveStepIntegrator {
    fn integrate_step(&mut self, state: &SystemState, delta_atp: f64) -> Result<SystemState> {
        loop {
            // Take a full step
            let full_step = self.base_integrator.integrate_step(state, delta_atp)?;
            
            // Take two half steps
            let half_step = delta_atp * 0.5;
            let intermediate = self.base_integrator.integrate_step(state, half_step)?;
            let double_step = self.base_integrator.integrate_step(&intermediate, half_step)?;
            
            // Estimate error
            let error = self.estimate_error(&full_step, &double_step);
            
            if error <= self.tolerance {
                // Accept the step and adjust step size for next iteration
                self.adjust_step_size(error);
                return Ok(double_step); // Use the more accurate double step
            } else {
                // Reject the step and try again with smaller step size
                self.adjust_step_size(error);
                
                // Prevent infinite loop
                if self.base_integrator.get_step_size() <= self.min_step {
                    return Err(NebuchadnezzarError::ComputationError(
                        "Step size too small, integration failed".to_string()
                    ));
                }
            }
        }
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.base_integrator.set_step_size(step_size);
    }

    fn get_step_size(&self) -> f64 {
        self.base_integrator.get_step_size()
    }
}

/// Specialized integrator for stiff systems
pub struct StiffAtpIntegrator {
    step_size: f64,
    jacobian_cache: HashMap<String, Vec<Vec<f64>>>,
}

impl StiffAtpIntegrator {
    pub fn new(step_size: f64) -> Self {
        Self {
            step_size,
            jacobian_cache: HashMap::new(),
        }
    }

    fn compute_jacobian(&self, state: &SystemState) -> Result<Vec<Vec<f64>>> {
        let n = state.concentrations.len();
        let mut jacobian = vec![vec![0.0; n]; n];
        
        // Compute numerical Jacobian
        let epsilon = 1e-8;
        
        for i in 0..n {
            for j in 0..n {
                let mut state_plus = state.clone();
                let mut state_minus = state.clone();
                
                state_plus.concentrations[j] += epsilon;
                state_minus.concentrations[j] -= epsilon;
                
                let f_plus = self.compute_derivative_vector(&state_plus)?;
                let f_minus = self.compute_derivative_vector(&state_minus)?;
                
                jacobian[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * epsilon);
            }
        }
        
        Ok(jacobian)
    }

    fn compute_derivative_vector(&self, state: &SystemState) -> Result<Vec<f64>> {
        let mut derivatives = vec![0.0; state.concentrations.len()];
        
        for i in 0..state.concentrations.len() {
            let concentration = state.concentrations[i];
            let atp_effect = state.atp_pool.atp_concentration / (1.0 + state.atp_pool.atp_concentration);
            
            // Stiff system example: fast equilibrium reactions
            derivatives[i] = -10.0 * concentration * atp_effect; // Fast reaction
            
            if i > 0 {
                derivatives[i] += 0.1 * state.concentrations[i-1]; // Slow coupling
            }
        }
        
        Ok(derivatives)
    }
}

impl AtpIntegrator for StiffAtpIntegrator {
    fn integrate_step(&mut self, state: &SystemState, delta_atp: f64) -> Result<SystemState> {
        let mut new_state = state.clone();
        
        // Implicit Euler method for stiff systems
        // (I - h*J) * (x_new - x_old) = h * f(x_old)
        
        let jacobian = self.compute_jacobian(state)?;
        let derivatives = self.compute_derivative_vector(state)?;
        
        // Solve linear system (simplified implementation)
        // In practice, would use proper linear algebra library
        for i in 0..state.concentrations.len() {
            let implicit_term = 1.0 + delta_atp * jacobian[i][i];
            new_state.concentrations[i] = (state.concentrations[i] + delta_atp * derivatives[i]) / implicit_term;
            
            // Ensure non-negative concentrations
            if new_state.concentrations[i] < 0.0 {
                new_state.concentrations[i] = 0.0;
            }
        }

        // Update ATP pool
        new_state.atp_pool.consume_atp(delta_atp)?;
        new_state.cumulative_atp += delta_atp;

        Ok(new_state)
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn get_step_size(&self) -> f64 {
        self.step_size
    }
}

/// Factory for creating integrators
pub struct IntegratorFactory;

impl IntegratorFactory {
    pub fn create_euler(step_size: f64) -> Box<dyn AtpIntegrator> {
        Box::new(AtpEulerIntegrator::new(step_size))
    }

    pub fn create_rk4(step_size: f64) -> Box<dyn AtpIntegrator> {
        Box::new(AtpRk4Integrator::new(step_size))
    }

    pub fn create_adaptive(initial_step: f64, tolerance: f64) -> Box<dyn AtpIntegrator> {
        Box::new(AdaptiveStepIntegrator::new(initial_step, tolerance))
    }

    pub fn create_stiff(step_size: f64) -> Box<dyn AtpIntegrator> {
        Box::new(StiffAtpIntegrator::new(step_size))
    }

    pub fn create_hybrid(config: HybridSolverConfig) -> Box<dyn AtpIntegrator> {
        Box::new(HybridMultiScaleSolver::new(config))
    }

    /// Create solver based on problem characteristics
    pub fn create_auto(problem_type: ProblemType, step_size: f64) -> Box<dyn AtpIntegrator> {
        match problem_type {
            ProblemType::Stiff => Self::create_stiff(step_size),
            ProblemType::Oscillatory => {
                let config = HybridSolverConfig {
                    linear_config: LinearSolverConfig {
                        matrix_solver: MatrixSolverType::LU,
                        tolerance: 1e-6,
                        max_iterations: 100,
                        use_laplace_analysis: true,
                    },
                    nonlinear_config: NonlinearSolverConfig {
                        method: NonlinearMethod::NewtonRaphson,
                        tolerance: 1e-6,
                        max_iterations: 50,
                        step_size_control: true,
                    },
                    coupling_strength: 0.1,
                    scale_separation_threshold: 10.0,
                };
                Self::create_hybrid(config)
            },
            ProblemType::FastSlow => Self::create_adaptive(step_size, 1e-6),
            ProblemType::Standard => Self::create_rk4(step_size),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProblemType {
    Standard,    // Regular biochemical kinetics
    Stiff,       // Fast equilibrium reactions
    Oscillatory, // Circadian rhythms, action potentials
    FastSlow,    // Multi-timescale dynamics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_state() {
        let mut state = SystemState::new(3, 2, 1);
        state.set_metabolite_names(vec!["glucose".to_string(), "atp".to_string(), "adp".to_string()]);
        
        assert!(state.set_concentration("glucose", 5.0).is_ok());
        assert_eq!(state.get_concentration("glucose"), Some(5.0));
        assert_eq!(state.get_concentration("unknown"), None);
    }

    #[test]
    fn test_euler_integrator() {
        let mut integrator = AtpEulerIntegrator::new(0.01);
        let state = SystemState::new(2, 0, 0);
        
        let result = integrator.integrate_step(&state, 0.1);
        assert!(result.is_ok());
        
        let new_state = result.unwrap();
        assert_eq!(new_state.cumulative_atp, 0.1);
    }

    #[test]
    fn test_rk4_integrator() {
        let mut integrator = AtpRk4Integrator::new(0.01);
        let state = SystemState::new(2, 0, 0);
        
        let result = integrator.integrate_step(&state, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_integrator() {
        let mut integrator = AdaptiveStepIntegrator::new(0.01, 1e-6);
        let state = SystemState::new(2, 0, 0);
        
        let result = integrator.integrate_step(&state, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_integrator_factory() {
        let euler = IntegratorFactory::create_euler(0.01);
        assert_eq!(euler.get_step_size(), 0.01);
        
        let rk4 = IntegratorFactory::create_rk4(0.01);
        assert_eq!(rk4.get_step_size(), 0.01);
        
        let adaptive = IntegratorFactory::create_adaptive(0.01, 1e-6);
        assert_eq!(adaptive.get_step_size(), 0.01);

        let auto_solver = IntegratorFactory::create_auto(ProblemType::Oscillatory, 0.01);
        assert_eq!(auto_solver.get_step_size(), 0.001); // Hybrid solver default
    }
} 
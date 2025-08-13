//! ATP-Constrained Differential Equation Solver
//! 
//! Implements dx/d[ATP] equations instead of traditional dx/dt formulations

use crate::atp_system::{AtpPool, AtpConfig};
use crate::error::{Error, Result};

/// ATP-constrained differential equation solver
#[derive(Debug)]
pub struct AtpDifferentialSolver {
    config: AtpConfig,
    integration_method: IntegrationMethod,
    step_size_controller: StepSizeController,
}

/// Integration methods for ATP-constrained equations
#[derive(Debug, Clone)]
pub enum IntegrationMethod {
    AtpRungeKutta4,
    AtpEuler,
    AtpAdaptive,
}

/// Step size control for numerical stability
#[derive(Debug)]
struct StepSizeController {
    max_atp_step: f64,
    min_atp_step: f64,
    error_tolerance: f64,
}

impl AtpDifferentialSolver {
    /// Create new ATP differential solver
    pub fn new(config: AtpConfig) -> Result<Self> {
        let step_size_controller = StepSizeController {
            max_atp_step: 0.1, // Maximum ATP change per step
            min_atp_step: 1e-6,
            error_tolerance: 1e-5,
        };

        Ok(Self {
            config,
            integration_method: IntegrationMethod::AtpRungeKutta4,
            step_size_controller,
        })
    }

    /// Perform integration step for ATP pool
    pub fn step(&mut self, atp_pool: &mut AtpPool, dt: f64) -> Result<()> {
        match self.integration_method {
            IntegrationMethod::AtpRungeKutta4 => self.atp_runge_kutta_4(atp_pool, dt),
            IntegrationMethod::AtpEuler => self.atp_euler(atp_pool, dt),
            IntegrationMethod::AtpAdaptive => self.atp_adaptive(atp_pool, dt),
        }
    }

    /// ATP-constrained Runge-Kutta 4th order integration
    fn atp_runge_kutta_4(&self, atp_pool: &mut AtpPool, dt: f64) -> Result<()> {
        let initial_atp = atp_pool.atp_concentration();
        let atp_step = self.calculate_atp_step(atp_pool, dt)?;

        // State variables: [ATP, ADP, AMP]
        let mut state = [
            atp_pool.atp_concentration(),
            atp_pool.adp_concentration(),
            atp_pool.amp_concentration(),
        ];

        // RK4 integration with respect to ATP consumption
        let k1 = self.evaluate_derivatives(&state, initial_atp)?;
        
        let state_k2 = [
            state[0] + atp_step * k1[0] / 2.0,
            state[1] + atp_step * k1[1] / 2.0,
            state[2] + atp_step * k1[2] / 2.0,
        ];
        let k2 = self.evaluate_derivatives(&state_k2, initial_atp + atp_step / 2.0)?;

        let state_k3 = [
            state[0] + atp_step * k2[0] / 2.0,
            state[1] + atp_step * k2[1] / 2.0,
            state[2] + atp_step * k2[2] / 2.0,
        ];
        let k3 = self.evaluate_derivatives(&state_k3, initial_atp + atp_step / 2.0)?;

        let state_k4 = [
            state[0] + atp_step * k3[0],
            state[1] + atp_step * k3[1],
            state[2] + atp_step * k3[2],
        ];
        let k4 = self.evaluate_derivatives(&state_k4, initial_atp + atp_step)?;

        // Final integration step
        for i in 0..3 {
            state[i] += atp_step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }

        // Update ATP pool with new concentrations
        atp_pool.set_concentrations(state[0], state[1], state[2])?;

        Ok(())
    }

    /// Simple Euler method for ATP-constrained integration
    fn atp_euler(&self, atp_pool: &mut AtpPool, dt: f64) -> Result<()> {
        let atp_step = self.calculate_atp_step(atp_pool, dt)?;
        
        let state = [
            atp_pool.atp_concentration(),
            atp_pool.adp_concentration(),
            atp_pool.amp_concentration(),
        ];

        let derivatives = self.evaluate_derivatives(&state, state[0])?;
        
        let new_state = [
            state[0] + atp_step * derivatives[0],
            state[1] + atp_step * derivatives[1],
            state[2] + atp_step * derivatives[2],
        ];

        atp_pool.set_concentrations(new_state[0], new_state[1], new_state[2])?;

        Ok(())
    }

    /// Adaptive step size ATP integration
    fn atp_adaptive(&self, atp_pool: &mut AtpPool, dt: f64) -> Result<()> {
        let mut current_step = self.calculate_atp_step(atp_pool, dt)?;
        let mut remaining_time = dt;

        while remaining_time > 0.0 {
            let step_dt = remaining_time.min(current_step);
            
            // Try full step
            let backup_pool = atp_pool.clone();
            self.atp_runge_kutta_4(atp_pool, step_dt)?;
            
            // Try half step twice for error estimation
            let mut test_pool = backup_pool.clone();
            self.atp_runge_kutta_4(&mut test_pool, step_dt / 2.0)?;
            self.atp_runge_kutta_4(&mut test_pool, step_dt / 2.0)?;
            
            // Estimate error
            let error = (atp_pool.atp_concentration() - test_pool.atp_concentration()).abs();
            
            if error < self.step_size_controller.error_tolerance {
                // Accept step
                remaining_time -= step_dt;
                current_step = (current_step * 1.2).min(self.step_size_controller.max_atp_step);
            } else {
                // Reject step and reduce step size
                *atp_pool = backup_pool;
                current_step = (current_step * 0.5).max(self.step_size_controller.min_atp_step);
                
                if current_step <= self.step_size_controller.min_atp_step {
                    return Err(Error::IntegrationFailure { step_size: current_step });
                }
            }
        }

        Ok(())
    }

    /// Calculate appropriate ATP step size based on current state
    fn calculate_atp_step(&self, atp_pool: &AtpPool, dt: f64) -> Result<f64> {
        // ATP step size based on consumption rate and time step
        let consumption_rate = self.estimate_consumption_rate(atp_pool);
        let atp_step = (consumption_rate * dt).min(self.step_size_controller.max_atp_step);
        
        if atp_step < self.step_size_controller.min_atp_step {
            return Err(Error::IntegrationFailure { step_size: atp_step });
        }

        Ok(atp_step)
    }

    /// Evaluate dx/d[ATP] derivatives for ATP-constrained system
    fn evaluate_derivatives(&self, state: &[f64; 3], atp_level: f64) -> Result<[f64; 3]> {
        let [atp, adp, amp] = *state;

        // ATP consumption rate based on current state
        let consumption_rate = self.config.synthesis_rate * adp * atp_level.max(0.1);
        
        // ATP synthesis rate (from ADP + Pi)
        let synthesis_rate = self.config.synthesis_rate * adp;
        
        // ATP degradation rate
        let degradation_rate = self.config.degradation_rate * atp;

        // Derivatives with respect to ATP consumption
        let d_atp_d_atp = -1.0 + synthesis_rate / consumption_rate - degradation_rate / consumption_rate;
        let d_adp_d_atp = 1.0 - synthesis_rate / consumption_rate;
        let d_amp_d_atp = degradation_rate / consumption_rate;

        Ok([d_atp_d_atp, d_adp_d_atp, d_amp_d_atp])
    }

    /// Estimate current ATP consumption rate
    fn estimate_consumption_rate(&self, atp_pool: &AtpPool) -> f64 {
        // Base consumption rate proportional to ATP concentration and energy charge
        let energy_charge = atp_pool.energy_charge();
        let base_rate = 0.1; // Base consumption rate
        
        base_rate * atp_pool.atp_concentration() * energy_charge
    }

    /// Set integration method
    pub fn set_integration_method(&mut self, method: IntegrationMethod) {
        self.integration_method = method;
    }

    /// Solve ATP-constrained differential equation
    pub fn solve_atp_differential<F>(&self, initial_value: f64, rate_function: F, atp_consumption: f64) -> Result<f64>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Solve dx/d[ATP] = rate_function(x, [ATP])
        let step_size = 0.01; // Small ATP step
        let mut x = initial_value;
        let mut atp_consumed = 0.0;

        while atp_consumed < atp_consumption {
            let current_atp = atp_consumption - atp_consumed;
            let rate = rate_function(x, current_atp);
            
            x += step_size * rate;
            atp_consumed += step_size;

            if x < 0.0 {
                return Err(Error::NumericalInstability);
            }
        }

        Ok(x)
    }
}
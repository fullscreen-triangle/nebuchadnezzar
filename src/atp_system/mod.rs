//! ATP-Constrained Dynamics System
//! 
//! Implements energy-based rate control using ATP concentration as the fundamental parameter
//! rather than time-based differential equations.

pub mod atp_pool;
pub mod differential_solver;
pub mod energy_charge;
pub mod conservation;

pub use atp_pool::AtpPool;
pub use differential_solver::AtpDifferentialSolver;
pub use energy_charge::EnergyChargeCalculator;
pub use conservation::ConservationValidator;

use crate::error::{Error, Result};

/// Configuration for ATP system
#[derive(Debug, Clone)]
pub struct AtpConfig {
    pub initial_atp_concentration: f64,
    pub initial_adp_concentration: f64, 
    pub initial_amp_concentration: f64,
    pub synthesis_rate: f64,
    pub degradation_rate: f64,
    pub min_energy_charge: f64,
    pub max_energy_charge: f64,
}

impl Default for AtpConfig {
    fn default() -> Self {
        Self::physiological()
    }
}

impl AtpConfig {
    /// Create physiological ATP configuration
    pub fn physiological() -> Self {
        Self {
            initial_atp_concentration: 5.0, // mM
            initial_adp_concentration: 0.5, // mM
            initial_amp_concentration: 0.05, // mM
            synthesis_rate: 1.0,
            degradation_rate: 0.1,
            min_energy_charge: 0.7,
            max_energy_charge: 0.95,
        }
    }

    /// Create high-energy configuration for neural activity
    pub fn neural_optimized() -> Self {
        Self {
            initial_atp_concentration: 8.0,
            initial_adp_concentration: 0.3,
            initial_amp_concentration: 0.02,
            synthesis_rate: 2.0,
            degradation_rate: 0.05,
            min_energy_charge: 0.8,
            max_energy_charge: 0.98,
        }
    }
}

/// ATP system managing energy-constrained dynamics
#[derive(Debug)]
pub struct AtpSystem {
    config: AtpConfig,
    atp_pool: AtpPool,
    differential_solver: AtpDifferentialSolver,
    energy_charge_calculator: EnergyChargeCalculator,
    conservation_validator: ConservationValidator,
    current_time: f64,
}

impl AtpSystem {
    /// Create new ATP system with configuration
    pub fn new(config: AtpConfig) -> Result<Self> {
        let atp_pool = AtpPool::new(
            config.initial_atp_concentration,
            config.initial_adp_concentration,
            config.initial_amp_concentration,
        )?;

        let differential_solver = AtpDifferentialSolver::new(config.clone())?;
        let energy_charge_calculator = EnergyChargeCalculator::new();
        let conservation_validator = ConservationValidator::new();

        Ok(Self {
            config,
            atp_pool,
            differential_solver,
            energy_charge_calculator,
            conservation_validator,
            current_time: 0.0,
        })
    }

    /// Perform simulation step with time increment
    pub fn step(&mut self, dt: f64) -> Result<()> {
        // Validate current energy charge
        let energy_charge = self.energy_charge();
        if energy_charge < self.config.min_energy_charge {
            return Err(Error::EnergyChargeOutOfRange {
                value: energy_charge,
                min: self.config.min_energy_charge,
                max: self.config.max_energy_charge,
            });
        }

        // Update ATP pool through differential solver
        self.differential_solver.step(&mut self.atp_pool, dt)?;

        // Validate energy conservation
        self.conservation_validator.validate(&self.atp_pool)?;

        self.current_time += dt;
        Ok(())
    }

    /// Get current ATP concentration
    pub fn current_concentration(&self) -> f64 {
        self.atp_pool.atp_concentration()
    }

    /// Calculate current energy charge
    pub fn energy_charge(&self) -> f64 {
        self.energy_charge_calculator.calculate(&self.atp_pool)
    }

    /// Check if ATP system is in stable state
    pub fn is_stable(&self) -> bool {
        let energy_charge = self.energy_charge();
        energy_charge >= self.config.min_energy_charge && 
        energy_charge <= self.config.max_energy_charge &&
        self.current_concentration() > 1.0 // Minimum viable concentration
    }

    /// Get ATP pool for external access
    pub fn atp_pool(&self) -> &AtpPool {
        &self.atp_pool
    }

    /// Consume ATP for a process and return available amount
    pub fn consume_atp(&mut self, requested_amount: f64) -> Result<f64> {
        self.atp_pool.consume(requested_amount)
    }

    /// Add ATP from synthesis or external source
    pub fn add_atp(&mut self, amount: f64) -> Result<()> {
        self.atp_pool.add_atp(amount)
    }
}
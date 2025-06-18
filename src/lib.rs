//! # Nebuchadnezzar: Hierarchical Probabilistic Electric Circuit System for Biological Simulation
//! 
//! Nebuchadnezzar is a revolutionary systems biology framework that models intracellular processes 
//! through hierarchical probabilistic electrical circuits, using ATP as the fundamental rate unit 
//! instead of time.
//!
//! ## Core Philosophy
//! 
//! - **ATP-based rates**: Use dx/dATP instead of dx/dt for biologically meaningful modeling
//! - **Hierarchical abstraction**: Start with probabilistic nodes, expand when needed
//! - **Electrical circuit analogs**: Map biochemical processes to circuit elements
//! - **Energy-coupled optimization**: Clear energetic objectives for pathway efficiency
//!
//! ## Quick Start
//!
//! ```rust
//! use nebuchadnezzar::prelude::*;
//! 
//! // Create ATP pool with physiological conditions
//! let atp_pool = AtpPool::new_physiological();
//! 
//! // Define a simple glycolysis pathway
//! let pathway = BiochemicalPathway::builder("glycolysis")
//!     .reaction("hexokinase")
//!         .substrate("glucose", 1.0)
//!         .product("G6P", 1.0)
//!         .atp_cost(1.0)
//!         .rate_datp(AtpRateConstant::michaelis(2.0, 0.5))
//!     .build()?;
//! 
//! // Optimize for maximum ATP efficiency
//! let optimized = PathwayOptimizer::new()
//!     .objective(PathwayObjective::MaximizeEnergyEfficiency)
//!     .optimize(&pathway)?;
//! ```

pub mod circuits;
pub mod systems_biology;
pub mod solvers;
pub mod utils;

// Core error types
pub mod error {
    use std::fmt;

    #[derive(Debug, Clone)]
    pub enum NebuchadnezzarError {
        /// Mathematical computation error
        ComputationError(String),
        /// Invalid parameter values
        ParameterError(String),
        /// Convergence failure in numerical methods
        ConvergenceError(String),
        /// Invalid system configuration
        ConfigurationError(String),
        /// ATP pool inconsistency
        AtpPoolError(String),
        /// Circuit topology error
        CircuitError(String),
    }

    impl fmt::Display for NebuchadnezzarError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::ComputationError(msg) => write!(f, "Computation error: {}", msg),
                Self::ParameterError(msg) => write!(f, "Parameter error: {}", msg),
                Self::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
                Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
                Self::AtpPoolError(msg) => write!(f, "ATP pool error: {}", msg),
                Self::CircuitError(msg) => write!(f, "Circuit error: {}", msg),
            }
        }
    }

    impl std::error::Error for NebuchadnezzarError {}

    pub type Result<T> = std::result::Result<T, NebuchadnezzarError>;
}

// Convenience prelude
pub mod prelude {
    pub use crate::circuits::*;
    pub use crate::systems_biology::*;
    pub use crate::solvers::*;
    pub use crate::error::{NebuchadnezzarError, Result};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_imports() {
        // Basic smoke test to ensure all modules can be imported
        assert!(true);
    }
} 
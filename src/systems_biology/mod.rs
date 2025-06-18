//! Hierarchical Probabilistic Systems Biology Framework
//! 
//! This module provides a hierarchical approach to modeling intracellular pathways
//! using probabilistic electrical circuits. Unlike traditional systems biology 
//! approaches that require modeling all cellular reactions, this framework:
//! 
//! 1. Uses hierarchical abstraction - start with probabilistic nodes, expand to detailed circuits when needed
//! 2. Models reaction rates using ATP as the fundamental unit (dx/dATP instead of dx/dt)
//! 3. Represents cross-reactions and feedback as probabilistic nodes
//! 4. Provides tangible optimization objectives for pathway optimization
//! 5. Builds on the existing electrical circuit foundation

pub mod pathway;
pub mod reaction;
pub mod probabilistic_node;
pub mod atp_kinetics;
pub mod optimization;
pub mod circuit_builder;

// Re-export core types for easy access
pub use pathway::{BiochemicalPathway, PathwayOptimizer};
pub use reaction::{BiochemicalReaction, ReactionRate, AtpDependentRate};
pub use probabilistic_node::{ProbabilisticNode, NodeExpansion, CrossReactionNode};
pub use atp_kinetics::{AtpPool, AtpKinetics, EnergeticProfile};
pub use optimization::{PathwayObjective, OptimizationTarget, ObjectiveFunction};
pub use circuit_builder::{PathwayCircuitBuilder, HierarchicalCircuit};

/// Core trait for ATP-dependent biological processes
pub trait AtpDependent {
    /// Calculate the ATP cost/yield for this process
    fn atp_impact(&self) -> f64;
    
    /// Update process state based on ATP availability
    fn update_with_atp(&mut self, atp_concentration: f64, d_atp: f64);
    
    /// Get the current energetic efficiency
    fn energetic_efficiency(&self) -> f64;
}

/// Trait for processes that can be hierarchically expanded
pub trait HierarchicalExpansion {
    type DetailedCircuit;
    
    /// Check if this node should be expanded to detailed circuit
    fn should_expand(&self, criteria: &ExpansionCriteria) -> bool;
    
    /// Expand probabilistic node into detailed electrical circuit
    fn expand_to_circuit(&self) -> Self::DetailedCircuit;
    
    /// Collapse detailed circuit back to probabilistic node
    fn collapse_from_circuit(circuit: &Self::DetailedCircuit) -> Self;
}

/// Criteria for deciding when to expand probabilistic nodes
#[derive(Debug, Clone)]
pub struct ExpansionCriteria {
    /// Minimum uncertainty threshold - expand if uncertainty is above this
    pub uncertainty_threshold: f64,
    
    /// Minimum optimization impact - expand if this node significantly affects objectives
    pub optimization_impact_threshold: f64,
    
    /// Available computational budget for detailed modeling
    pub computational_budget: f64,
    
    /// Biological significance threshold
    pub biological_significance_threshold: f64,
}

impl Default for ExpansionCriteria {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.3,
            optimization_impact_threshold: 0.1,
            computational_budget: 1000.0,
            biological_significance_threshold: 0.2,
        }
    }
}

/// Error types for systems biology operations
#[derive(Debug, thiserror::Error)]
pub enum SystemsBiologyError {
    #[error("ATP pool depleted: {0}")]
    AtpDepletion(String),
    
    #[error("Pathway construction failed: {0}")]
    PathwayConstructionError(String),
    
    #[error("Circuit expansion failed: {0}")]
    CircuitExpansionError(String),
    
    #[error("Optimization failed: {0}")]
    OptimizationError(String),
    
    #[error("Invalid reaction parameters: {0}")]
    InvalidReactionParameters(String),
}

pub type Result<T> = std::result::Result<T, SystemsBiologyError>; 
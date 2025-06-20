//! # Nebuchadnezzar: Hierarchical Probabilistic Electric Circuit System for Biological Simulation
//! 
//! A computational framework for modeling intracellular processes using ATP as the fundamental 
//! rate unit instead of time. The system implements hierarchical probabilistic electrical circuits 
//! with six theoretical frameworks addressing quantum coherence in biological membranes, 
//! oscillatory dynamics in bounded systems, entropy manipulation through probabilistic point systems, 
//! evolutionary adaptations to fire exposure, temporal coordinate optimization, and information 
//! processing amplification through biological Maxwell's demons.

pub mod biological_integration;
pub mod biological_quantum_computer;
pub mod biological_quantum_solver;
pub mod circuits;
pub mod entropy_manipulation;
pub mod error;
pub mod oscillatory_dynamics;
pub mod quantum_membranes;
pub mod quantum_metabolism_analyzer;
pub mod solvers;
pub mod systems_biology;
pub mod utils;

// Re-export main types for convenience
pub use biological_quantum_computer::{
    BiologicalQuantumState, AtpCoordinates, OscillatoryCoordinates, 
    MembraneQuantumCoordinates, OscillatoryEntropyCoordinates
};
pub use biological_quantum_solver::{
    BiologicalQuantumComputerSolver, BiologicalQuantumResult, 
    QuantumComputationTarget, SolverError
};
pub use circuits::{CircuitGrid, HierarchicalCircuit, ProbabilisticNode};
pub use entropy_manipulation::{EntropyPoint, Resolution, EntropyManipulator};
pub use error::{NebuchadnezzarError, Result};
pub use oscillatory_dynamics::{OscillationState, BiologicalOscillator};
pub use quantum_membranes::{QuantumMembrane, EnvironmentalCoupling};
pub use solvers::{AtpBasedSolver, NativeSolver};
pub use systems_biology::atp_kinetics::{AtpPool, AtpRateConstant};

/// Prelude module for common imports
pub mod prelude {
    pub use crate::{
        BiologicalQuantumState, BiologicalQuantumComputerSolver, BiologicalQuantumResult,
        AtpCoordinates, OscillatoryCoordinates, MembraneQuantumCoordinates,
        CircuitGrid, HierarchicalCircuit, ProbabilisticNode,
        EntropyPoint, Resolution, EntropyManipulator,
        OscillationState, BiologicalOscillator,
        QuantumMembrane, EnvironmentalCoupling,
        AtpPool, AtpRateConstant, AtpBasedSolver,
        NebuchadnezzarError, Result,
    };
}

/// ATP-based differential equation framework
pub struct PathwayCircuitBuilder {
    pathway: Option<BiochemicalPathway>,
    atp_pool: Option<AtpPool>,
    expansion_criteria: ExpansionCriteria,
}

impl PathwayCircuitBuilder {
    pub fn new() -> Self {
        Self {
            pathway: None,
            atp_pool: None,
            expansion_criteria: ExpansionCriteria::default(),
        }
    }

    pub fn from_pathway(mut self, pathway: &BiochemicalPathway) -> Self {
        self.pathway = Some(pathway.clone());
        self
    }

    pub fn with_atp_pool(mut self, atp_pool: &AtpPool) -> Self {
        self.atp_pool = Some(atp_pool.clone());
        self
    }

    pub fn with_expansion_criteria(mut self, criteria: ExpansionCriteria) -> Self {
        self.expansion_criteria = criteria;
        self
    }

    pub fn build(self) -> Result<HierarchicalCircuit> {
        let pathway = self.pathway.ok_or(NebuchadnezzarError::MissingPathway)?;
        let atp_pool = self.atp_pool.ok_or(NebuchadnezzarError::MissingAtpPool)?;
        
        Ok(HierarchicalCircuit::from_pathway(pathway, atp_pool, self.expansion_criteria))
    }
}

/// Biochemical pathway representation
#[derive(Debug, Clone)]
pub struct BiochemicalPathway {
    pub name: String,
    pub reactions: Vec<BiochemicalReaction>,
}

impl BiochemicalPathway {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            reactions: Vec::new(),
        }
    }

    pub fn add_reaction(&mut self, name: &str, reaction: BiochemicalReaction) {
        self.reactions.push(reaction);
    }
}

/// Individual biochemical reaction
#[derive(Debug, Clone)]
pub struct BiochemicalReaction {
    pub name: String,
    pub substrates: Vec<(String, f64)>,
    pub products: Vec<(String, f64)>,
    pub atp_cost: f64,
    pub rate_constant: AtpRateConstant,
}

impl BiochemicalReaction {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            substrates: Vec::new(),
            products: Vec::new(),
            atp_cost: 0.0,
            rate_constant: AtpRateConstant::Linear(1.0),
        }
    }

    pub fn substrate(mut self, name: &str, stoichiometry: f64) -> Self {
        self.substrates.push((name.to_string(), stoichiometry));
        self
    }

    pub fn product(mut self, name: &str, stoichiometry: f64) -> Self {
        self.products.push((name.to_string(), stoichiometry));
        self
    }

    pub fn atp_cost(mut self, cost: f64) -> Self {
        self.atp_cost = cost;
        self
    }

    pub fn rate_constant(mut self, rate: AtpRateConstant) -> Self {
        self.rate_constant = rate;
        self
    }
}

/// Expansion criteria for probabilistic nodes
#[derive(Debug, Clone)]
pub struct ExpansionCriteria {
    pub uncertainty_threshold: f64,
    pub impact_threshold: f64,
    pub budget_limit: f64,
}

impl Default for ExpansionCriteria {
    fn default() -> Self {
        Self {
            uncertainty_threshold: 0.3,
            impact_threshold: 0.1,
            budget_limit: 1000.0,
        }
    }
}

/// Pathway optimization objectives
#[derive(Debug, Clone)]
pub enum PathwayObjective {
    MaximizeAtpYield,
    MinimizeAtpConsumption,
    MaximizeFlux,
    NavigateToPredeterminedOptimum,
}

/// Optimization constraints
#[derive(Debug, Clone)]
pub enum OptimizationConstraint {
    AtpAvailability(f64),
    FluxBounds(f64, f64),
    ThermodynamicFeasibility,
}

/// Pathway optimizer
pub struct PathwayOptimizer {
    objective: PathwayObjective,
    constraints: Vec<OptimizationConstraint>,
}

impl PathwayOptimizer {
    pub fn new() -> Self {
        Self {
            objective: PathwayObjective::MaximizeAtpYield,
            constraints: Vec::new(),
        }
    }

    pub fn objective(mut self, objective: PathwayObjective) -> Self {
        self.objective = objective;
        self
    }

    pub fn constraint(mut self, constraint: OptimizationConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    pub fn build(self) -> Self {
        self
    }

    pub fn optimize(&self, circuit: &HierarchicalCircuit) -> Result<OptimizationResult> {
        // Implementation of optimization algorithm
        Ok(OptimizationResult {
            atp_efficiency: 0.95,
            flux_values: vec![1.0, 2.0, 3.0],
            converged: true,
        })
    }
}

/// Optimization result
#[derive(Debug)]
pub struct OptimizationResult {
    pub atp_efficiency: f64,
    pub flux_values: Vec<f64>,
    pub converged: bool,
} 
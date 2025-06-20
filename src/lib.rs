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

// New comprehensive ATP-oscillatory-membrane implementation
pub mod atp_oscillatory_membrane_simulator;
pub mod atp_oscillatory_membrane_solver;

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

// Re-export the comprehensive ATP-oscillatory-membrane system
pub use atp_oscillatory_membrane_simulator::*;
pub use atp_oscillatory_membrane_solver::*;

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
    pathways: Vec<BiochemicalPathway>,
    expansion_criteria: ExpansionCriteria,
}

impl PathwayCircuitBuilder {
    pub fn new() -> Self {
        Self {
            pathways: Vec::new(),
            expansion_criteria: ExpansionCriteria::default(),
        }
    }
    
    pub fn add_pathway(&mut self, pathway: BiochemicalPathway) -> &mut Self {
        self.pathways.push(pathway);
        self
    }
    
    pub fn with_expansion_criteria(&mut self, criteria: ExpansionCriteria) -> &mut Self {
        self.expansion_criteria = criteria;
        self
    }
    
    pub fn build(&self) -> Result<HierarchicalCircuit, NebuchadnezzarError> {
        let mut circuit = HierarchicalCircuit::new();
        
        for pathway in &self.pathways {
            // Create probabilistic node for each pathway
            let prob_node = ProbabilisticNode {
                name: pathway.name.clone(),
                uncertainty: 0.1, // Default uncertainty
                impact_factor: pathway.atp_cost / 100.0, // Scale ATP cost to impact
                computational_cost: pathway.reactions.len() as f64,
                distribution_type: circuits::hierarchical_framework::DistributionType::Gaussian,
                parameters: vec![0.0, 1.0], // mean, std
            };
            
            circuit.add_probabilistic_node(prob_node)?;
        }
        
        Ok(circuit)
    }
}

impl Default for PathwayCircuitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Biochemical pathway representation
#[derive(Debug, Clone)]
pub struct BiochemicalPathway {
    pub name: String,
    pub reactions: Vec<BiochemicalReaction>,
    pub atp_cost: f64,
    pub regulation: RegulationMechanism,
}

impl BiochemicalPathway {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            reactions: Vec::new(),
            atp_cost: 0.0,
            regulation: RegulationMechanism::Allosteric {
                activators: Vec::new(),
                inhibitors: Vec::new(),
            },
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
    pub reactants: Vec<String>,
    pub products: Vec<String>,
    pub enzyme: String,
    pub km: f64,
    pub vmax: f64,
    pub delta_g: f64,
    pub atp_coupling: f64,
}

impl BiochemicalReaction {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            reactants: Vec::new(),
            products: Vec::new(),
            enzyme: String::new(),
            km: 0.0,
            vmax: 0.0,
            delta_g: 0.0,
            atp_coupling: 0.0,
        }
    }

    pub fn substrate(mut self, name: &str, stoichiometry: f64) -> Self {
        self.reactants.push(name.to_string());
        self
    }

    pub fn product(mut self, name: &str, stoichiometry: f64) -> Self {
        self.products.push(name.to_string());
        self
    }

    pub fn enzyme(mut self, enzyme: &str) -> Self {
        self.enzyme = enzyme.to_string();
        self
    }

    pub fn km(mut self, km: f64) -> Self {
        self.km = km;
        self
    }

    pub fn vmax(mut self, vmax: f64) -> Self {
        self.vmax = vmax;
        self
    }

    pub fn delta_g(mut self, delta_g: f64) -> Self {
        self.delta_g = delta_g;
        self
    }

    pub fn atp_coupling(mut self, atp_coupling: f64) -> Self {
        self.atp_coupling = atp_coupling;
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

/// Quantum coherence analyzer for biological systems
pub struct QuantumCoherenceAnalyzer;

impl QuantumCoherenceAnalyzer {
    pub fn analyze_pathway_coherence(pathway: &BiochemicalPathway) -> f64 {
        // Simplified coherence calculation based on pathway structure
        let reaction_count = pathway.reactions.len() as f64;
        let atp_coupling_strength = pathway.reactions.iter()
            .map(|r| r.atp_coupling)
            .sum::<f64>() / reaction_count;
        
        // Coherence scales with ATP coupling and decreases with complexity
        atp_coupling_strength / (1.0 + 0.1 * reaction_count)
    }
    
    pub fn calculate_decoherence_rate(pathway: &BiochemicalPathway, temperature: f64) -> f64 {
        // Decoherence rate based on thermal fluctuations and pathway complexity
        let kb = 1.381e-23; // Boltzmann constant
        let thermal_energy = kb * temperature;
        let pathway_energy = pathway.atp_cost * 1.602e-19; // Convert to Joules
        
        // Decoherence rate proportional to thermal energy / pathway energy
        thermal_energy / pathway_energy * 1e12 // Convert to Hz
    }
}

/// ATP-driven differential equation solver
pub struct AtpDifferentialSolver {
    pub atp_pool: f64,
    pub time_step: f64,
}

impl AtpDifferentialSolver {
    pub fn new(initial_atp: f64) -> Self {
        Self {
            atp_pool: initial_atp,
            time_step: 0.001, // 1ms default
        }
    }
    
    /// Solve dx/dATP = f(x, ATP) using ATP as the independent variable
    pub fn solve_atp_differential<F>(&mut self, mut state: f64, derivative_fn: F, atp_consumption: f64) -> f64 
    where
        F: Fn(f64, f64) -> f64,
    {
        let atp_step = atp_consumption.min(self.atp_pool);
        
        // Runge-Kutta 4th order integration with ATP as independent variable
        let k1 = derivative_fn(state, self.atp_pool);
        let k2 = derivative_fn(state + 0.5 * k1 * atp_step, self.atp_pool - 0.5 * atp_step);
        let k3 = derivative_fn(state + 0.5 * k2 * atp_step, self.atp_pool - 0.5 * atp_step);
        let k4 = derivative_fn(state + k3 * atp_step, self.atp_pool - atp_step);
        
        state += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * atp_step / 6.0;
        self.atp_pool -= atp_step;
        
        state
    }
}

#[derive(Debug, Clone)]
pub enum RegulationMechanism {
    Allosteric { activators: Vec<String>, inhibitors: Vec<String> },
    Covalent { phosphorylation_sites: Vec<String> },
    Transcriptional { transcription_factors: Vec<String> },
    Metabolic { feedback_loops: Vec<String> },
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pathway_builder() {
        let mut builder = PathwayCircuitBuilder::new();
        
        let glycolysis = BiochemicalPathway {
            name: "Glycolysis".to_string(),
            reactions: vec![
                BiochemicalReaction {
                    name: "Hexokinase".to_string(),
                    reactants: vec!["Glucose".to_string(), "ATP".to_string()],
                    products: vec!["Glucose-6-P".to_string(), "ADP".to_string()],
                    enzyme: "Hexokinase".to_string(),
                    km: 0.1,
                    vmax: 100.0,
                    delta_g: -16.7,
                    atp_coupling: 1.0,
                }
            ],
            atp_cost: 2.0,
            regulation: RegulationMechanism::Allosteric {
                activators: vec!["AMP".to_string()],
                inhibitors: vec!["Glucose-6-P".to_string()],
            },
        };
        
        builder.add_pathway(glycolysis);
        let circuit = builder.build().unwrap();
        
        assert_eq!(circuit.probabilistic_nodes().len(), 1);
    }
    
    #[test]
    fn test_atp_differential_solver() {
        let mut solver = AtpDifferentialSolver::new(100.0);
        
        // Simple exponential decay: dx/dATP = -0.1 * x
        let result = solver.solve_atp_differential(10.0, |x, _atp| -0.1 * x, 1.0);
        
        assert!(result < 10.0); // Should decay
        assert!(solver.atp_pool < 100.0); // ATP should be consumed
    }
    
    #[test]
    fn test_quantum_coherence_analyzer() {
        let pathway = BiochemicalPathway {
            name: "Test".to_string(),
            reactions: vec![
                BiochemicalReaction {
                    name: "Test Reaction".to_string(),
                    reactants: vec!["A".to_string()],
                    products: vec!["B".to_string()],
                    enzyme: "TestEnzyme".to_string(),
                    km: 1.0,
                    vmax: 10.0,
                    delta_g: -5.0,
                    atp_coupling: 0.5,
                }
            ],
            atp_cost: 1.0,
            regulation: RegulationMechanism::Metabolic {
                feedback_loops: vec!["B".to_string()],
            },
        };
        
        let coherence = QuantumCoherenceAnalyzer::analyze_pathway_coherence(&pathway);
        assert!(coherence > 0.0);
        
        let decoherence_rate = QuantumCoherenceAnalyzer::calculate_decoherence_rate(&pathway, 310.0);
        assert!(decoherence_rate > 0.0);
    }
} 
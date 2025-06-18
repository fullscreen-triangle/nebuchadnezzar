//! # Nebuchadnezzar: ATP-Driven Intracellular Circuit Simulation
//! 
//! A comprehensive system for simulating intracellular cellular processes using hierarchical 
//! probabilistic electric circuits with differential equations based on ATP as the rate unit.
//! 
//! ## Key Features
//! 
//! - **ATP-based Rate Modeling**: Uses dx/dATP instead of dx/dt for biologically meaningful rates
//! - **Hierarchical Probabilistic Circuits**: 4-level framework from molecular to tissue level
//! - **Dynamic Circuit Resolution**: Converts probabilistic nodes to detailed circuits adaptively
//! - **Multi-level Voltage Hierarchy**: Cellular, organelle, compartment, and molecular levels
//! - **Temporal Evidence Decay**: Across timescales from milliseconds to days
//! - **Enzyme Logic Gates**: Biochemical transformations as probabilistic logic circuits
//! 
//! ## Architecture
//! 
//! The system is organized into several key modules:
//! 
//! - **circuits**: Electrical circuit foundation with ion channels, enzyme circuits, and grid systems
//! - **solvers**: ATP-based differential equation integration
//! - **systems_biology**: ATP kinetics and cellular energy management
//! - **utils**: Mathematical utilities and helper functions
//! 
//! ## Usage Example
//! 
//! ```rust
//! use nebuchadnezzar::{
//!     circuits::{CircuitFactory, AdaptiveGrid},
//!     systems_biology::AtpPool,
//!     solvers::AtpRk4Integrator,
//! };
//! 
//! // Create a neuron membrane with ion channels
//! let mut membrane = CircuitFactory::create_neuron_membrane();
//! 
//! // Create an ATP pool
//! let mut atp_pool = AtpPool::new(5.0, 1.0, 0.5);
//! 
//! // Create a glycolysis circuit
//! let mut glycolysis = CircuitFactory::create_glycolysis_circuit();
//! 
//! // Solve the system
//! let result = glycolysis.base_grid.solve_circuit_grid(1.0);
//! ```

pub mod error;
pub mod circuits;
pub mod solvers;
pub mod systems_biology;
pub mod utils;

// Re-export commonly used types
pub use error::{NebuchadnezzarError, Result};

pub use circuits::{
    Circuit,
    CircuitFactory,
    CircuitGrid,
    AdaptiveGrid,
    HierarchicalSystem,
    MembraneModel,
    CircuitNetwork,
    ProbabilisticIonChannel,
    EnzymeProbCircuit,
    EnzymeCircuitFactory,
    VoltageHierarchy,
    TemporalEvidence,
};

pub use solvers::{
    SystemState,
    AtpIntegrator,
    AtpEulerIntegrator,
    AtpRk4Integrator,
    AdaptiveStepIntegrator,
};

pub use systems_biology::{
    AtpPool,
    AtpKinetics,
    EnergyCharge,
};

pub use utils::{
    ThermodynamicState,
    UnitConverter,
    NumericalMethods,
    MatrixOperations,
};

/// Main simulation engine that orchestrates all components
pub struct NebuchadnezzarEngine {
    pub hierarchical_system: HierarchicalSystem,
    pub global_atp_pool: AtpPool,
    pub integrator: Box<dyn AtpIntegrator>,
    pub current_time: f64,
    pub simulation_parameters: SimulationParameters,
}

/// Configuration parameters for the simulation
#[derive(Debug, Clone)]
pub struct SimulationParameters {
    pub time_step: f64,
    pub max_time: f64,
    pub atp_threshold: f64,
    pub voltage_tolerance: f64,
    pub adaptive_stepping: bool,
    pub resolution_threshold: f64,
    pub output_frequency: usize,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            time_step: 0.001,
            max_time: 10.0,
            atp_threshold: 0.01,
            voltage_tolerance: 0.001,
            adaptive_stepping: true,
            resolution_threshold: 0.7,
            output_frequency: 100,
        }
    }
}

impl NebuchadnezzarEngine {
    /// Create a new simulation engine
    pub fn new(initial_atp: f64) -> Self {
        Self {
            hierarchical_system: HierarchicalSystem::new(),
            global_atp_pool: AtpPool::new(initial_atp, 1.0, 0.5),
            integrator: Box::new(AtpRk4Integrator::new(0.001)),
            current_time: 0.0,
            simulation_parameters: SimulationParameters::default(),
        }
    }

    /// Run the complete simulation
    pub fn run_simulation(&mut self) -> Result<SimulationResults> {
        let mut results = SimulationResults::new();
        let mut step_count = 0;

        while self.current_time < self.simulation_parameters.max_time &&
              self.global_atp_pool.atp_concentration > self.simulation_parameters.atp_threshold {
            
            // Calculate ATP consumption for this step
            let delta_atp = self.calculate_step_atp_consumption()?;
            
            // Solve hierarchical system
            let hierarchical_state = self.hierarchical_system.solve_hierarchical_system(delta_atp)?;
            
            // Update global ATP pool
            self.global_atp_pool.consume_atp(delta_atp)?;
            
            // Update time
            self.current_time += self.simulation_parameters.time_step;
            self.hierarchical_system.current_time = self.current_time;
            
            // Store results
            if step_count % self.simulation_parameters.output_frequency == 0 {
                results.add_timepoint(self.current_time, hierarchical_state, self.global_atp_pool.clone());
            }
            
            step_count += 1;
        }

        results.finalize(self.current_time, step_count);
        Ok(results)
    }

    /// Calculate ATP consumption for the current step
    fn calculate_step_atp_consumption(&self) -> Result<f64> {
        let base_consumption = self.simulation_parameters.time_step * 0.1; // Base metabolic rate
        
        // Add consumption from active circuits
        let circuit_consumption = self.hierarchical_system.molecular_level.len() as f64 * 0.01;
        
        Ok(base_consumption + circuit_consumption)
    }

    /// Add a pre-configured circuit to the system
    pub fn add_neuron_circuit(&mut self, neuron_id: String) -> Result<()> {
        let membrane = CircuitFactory::create_neuron_membrane();
        
        // Convert to molecular circuit and add to hierarchical system
        let molecular_circuit = circuits::hierarchical_framework::MolecularCircuit {
            circuit_id: neuron_id,
            protein_complexes: Vec::new(),
            enzyme_circuits: Vec::new(),
            ion_channels: membrane.ion_channels,
            molecular_voltage: membrane.current_voltage,
            atp_local: 1.0,
            temporal_state: circuits::hierarchical_framework::TemporalEvidence::new(),
        };
        
        self.hierarchical_system.molecular_level.push(molecular_circuit);
        Ok(())
    }

    /// Add a metabolic pathway circuit
    pub fn add_metabolic_pathway(&mut self, pathway_name: String, pathway_type: MetabolicPathwayType) -> Result<()> {
        let grid = match pathway_type {
            MetabolicPathwayType::Glycolysis => CircuitFactory::create_glycolysis_circuit(),
            MetabolicPathwayType::CitricAcidCycle => self.create_citric_acid_cycle()?,
            MetabolicPathwayType::ElectronTransport => self.create_electron_transport_chain()?,
        };

        // Convert grid to organelle network
        let organelle = circuits::hierarchical_framework::OrganelleNetwork {
            organelle_id: pathway_name,
            organelle_type: circuits::hierarchical_framework::OrganelleType::Mitochondrion { cristae_density: 1.0 },
            molecular_circuits: Vec::new(),
            inter_circuit_connections: Vec::new(),
            organelle_voltage: -180.0,
            local_atp_pool: 5.0,
            metabolic_state: circuits::hierarchical_framework::MetabolicState {
                energy_charge: 0.8,
                nadh_nad_ratio: 0.1,
                calcium_level: 0.0001,
                ph: 7.4,
                oxygen_level: 0.2,
            },
        };

        self.hierarchical_system.organelle_level.push(organelle);
        Ok(())
    }

    fn create_citric_acid_cycle(&self) -> Result<AdaptiveGrid> {
        let mut grid = AdaptiveGrid::new("citric_acid_cycle".to_string(), 8.0);
        
        // Add citric acid cycle enzymes
        let enzymes = vec![
            ("citrate_synthase", "acetyl_CoA", "citrate"),
            ("aconitase", "citrate", "isocitrate"),
            ("isocitrate_dehydrogenase", "isocitrate", "alpha_ketoglutarate"),
            ("alpha_ketoglutarate_dehydrogenase", "alpha_ketoglutarate", "succinyl_CoA"),
            ("succinate_thiokinase", "succinyl_CoA", "succinate"),
            ("succinate_dehydrogenase", "succinate", "fumarate"),
            ("fumarase", "fumarate", "malate"),
            ("malate_dehydrogenase", "malate", "oxaloacetate"),
        ];

        for (i, (enzyme_name, substrate, product)) in enzymes.iter().enumerate() {
            let node = circuits::circuit_grid::ProbabilisticNode {
                node_id: format!("tca_step_{}_{}", i, enzyme_name),
                node_type: circuits::circuit_grid::NodeType::EnzymaticReaction {
                    enzyme_class: enzyme_name.to_string(),
                    substrate_binding_prob: 0.85,
                    product_formation_prob: 0.9,
                },
                probability: 0.9,
                atp_cost: 0.5,
                inputs: vec![substrate.to_string()],
                outputs: vec![product.to_string()],
                resolution_importance: 0.95,
                last_activity: 1.0,
            };
            
            grid.base_grid.add_probabilistic_node(node);
        }

        Ok(grid)
    }

    fn create_electron_transport_chain(&self) -> Result<AdaptiveGrid> {
        let mut grid = AdaptiveGrid::new("electron_transport_chain".to_string(), 12.0);
        
        // Add electron transport complexes
        let complexes = vec![
            ("complex_I", "NADH", "NAD+"),
            ("complex_II", "succinate", "fumarate"),
            ("complex_III", "cytochrome_c_red", "cytochrome_c_ox"),
            ("complex_IV", "cytochrome_c_ox", "H2O"),
            ("ATP_synthase", "ADP", "ATP"),
        ];

        for (i, (complex_name, substrate, product)) in complexes.iter().enumerate() {
            let node = circuits::circuit_grid::ProbabilisticNode {
                node_id: format!("etc_complex_{}_{}", i, complex_name),
                node_type: circuits::circuit_grid::NodeType::EnzymaticReaction {
                    enzyme_class: complex_name.to_string(),
                    substrate_binding_prob: 0.9,
                    product_formation_prob: 0.95,
                },
                probability: 0.95,
                atp_cost: if complex_name == &"ATP_synthase" { -3.0 } else { 0.2 }, // ATP synthase produces ATP
                inputs: vec![substrate.to_string()],
                outputs: vec![product.to_string()],
                resolution_importance: 1.0,
                last_activity: 1.0,
            };
            
            grid.base_grid.add_probabilistic_node(node);
        }

        Ok(grid)
    }
}

/// Types of metabolic pathways that can be added
#[derive(Debug, Clone)]
pub enum MetabolicPathwayType {
    Glycolysis,
    CitricAcidCycle,
    ElectronTransport,
}

/// Results from a complete simulation run
#[derive(Debug, Clone)]
pub struct SimulationResults {
    pub timepoints: Vec<f64>,
    pub hierarchical_states: Vec<circuits::hierarchical_framework::HierarchicalState>,
    pub atp_levels: Vec<f64>,
    pub energy_charges: Vec<f64>,
    pub total_simulation_time: f64,
    pub total_steps: usize,
    pub final_atp_level: f64,
}

impl SimulationResults {
    fn new() -> Self {
        Self {
            timepoints: Vec::new(),
            hierarchical_states: Vec::new(),
            atp_levels: Vec::new(),
            energy_charges: Vec::new(),
            total_simulation_time: 0.0,
            total_steps: 0,
            final_atp_level: 0.0,
        }
    }

    fn add_timepoint(&mut self, time: f64, state: circuits::hierarchical_framework::HierarchicalState, atp_pool: AtpPool) {
        self.timepoints.push(time);
        self.hierarchical_states.push(state);
        self.atp_levels.push(atp_pool.atp_concentration);
        self.energy_charges.push(atp_pool.energy_charge());
    }

    fn finalize(&mut self, final_time: f64, total_steps: usize) {
        self.total_simulation_time = final_time;
        self.total_steps = total_steps;
        self.final_atp_level = self.atp_levels.last().copied().unwrap_or(0.0);
    }

    /// Get summary statistics from the simulation
    pub fn get_summary(&self) -> SimulationSummary {
        let avg_atp = self.atp_levels.iter().sum::<f64>() / self.atp_levels.len() as f64;
        let min_atp = self.atp_levels.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_atp = self.atp_levels.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let avg_energy_charge = self.energy_charges.iter().sum::<f64>() / self.energy_charges.len() as f64;

        SimulationSummary {
            total_time: self.total_simulation_time,
            total_steps: self.total_steps,
            average_atp: avg_atp,
            min_atp,
            max_atp,
            final_atp: self.final_atp_level,
            average_energy_charge,
            atp_depletion_rate: (self.atp_levels[0] - self.final_atp_level) / self.total_simulation_time,
        }
    }
}

/// Summary statistics from a simulation
#[derive(Debug, Clone)]
pub struct SimulationSummary {
    pub total_time: f64,
    pub total_steps: usize,
    pub average_atp: f64,
    pub min_atp: f64,
    pub max_atp: f64,
    pub final_atp: f64,
    pub average_energy_charge: f64,
    pub atp_depletion_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = NebuchadnezzarEngine::new(10.0);
        assert_eq!(engine.global_atp_pool.atp_concentration, 10.0);
        assert_eq!(engine.current_time, 0.0);
    }

    #[test]
    fn test_add_neuron_circuit() {
        let mut engine = NebuchadnezzarEngine::new(10.0);
        let result = engine.add_neuron_circuit("test_neuron".to_string());
        assert!(result.is_ok());
        assert_eq!(engine.hierarchical_system.molecular_level.len(), 1);
    }

    #[test]
    fn test_add_metabolic_pathway() {
        let mut engine = NebuchadnezzarEngine::new(15.0);
        let result = engine.add_metabolic_pathway("glycolysis".to_string(), MetabolicPathwayType::Glycolysis);
        assert!(result.is_ok());
        assert_eq!(engine.hierarchical_system.organelle_level.len(), 1);
    }

    #[test]
    fn test_simulation_parameters() {
        let params = SimulationParameters::default();
        assert_eq!(params.time_step, 0.001);
        assert_eq!(params.max_time, 10.0);
        assert_eq!(params.atp_threshold, 0.01);
    }
} 
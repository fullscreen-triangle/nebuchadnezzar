//! Hierarchical Circuit Architecture
//! 
//! Probabilistic electric circuit simulation for biological systems

pub mod probabilistic_elements;
pub mod nodal_analysis;
pub mod biological_mapping;
pub mod hierarchical_solver;

pub use probabilistic_elements::ProbabilisticElement;
pub use nodal_analysis::NodalAnalysis;
pub use biological_mapping::BiologicalMapping;
pub use hierarchical_solver::HierarchicalSolver;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct CircuitConfig {
    pub num_nodes: usize,
    pub probabilistic_elements: bool,
    pub biological_mapping: bool,
    pub time_step: f64,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            num_nodes: 100,
            probabilistic_elements: true,
            biological_mapping: true,
            time_step: 1e-6, // 1 μs
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitState {
    pub node_voltages: Vec<f64>,
    pub branch_currents: Vec<f64>,
    pub total_power: f64,
}

#[derive(Debug)]
pub struct CircuitSystem {
    config: CircuitConfig,
    nodal_analysis: NodalAnalysis,
    biological_mapping: BiologicalMapping,
    hierarchical_solver: HierarchicalSolver,
    current_state: CircuitState,
}

impl CircuitSystem {
    pub fn new(config: CircuitConfig) -> Result<Self> {
        Ok(Self {
            nodal_analysis: NodalAnalysis::new(config.num_nodes)?,
            biological_mapping: BiologicalMapping::new(config.biological_mapping),
            hierarchical_solver: HierarchicalSolver::new(),
            current_state: CircuitState {
                node_voltages: vec![0.0; config.num_nodes],
                branch_currents: vec![0.0; config.num_nodes],
                total_power: 0.0,
            },
            config,
        })
    }

    pub fn step(&mut self, dt: f64, atp_concentration: f64) -> Result<()> {
        self.nodal_analysis.solve(dt, atp_concentration)?;
        self.current_state.node_voltages = self.nodal_analysis.get_voltages();
        self.current_state.branch_currents = self.nodal_analysis.get_currents();
        self.current_state.total_power = self.nodal_analysis.total_power();
        Ok(())
    }

    pub fn current_state(&self) -> &CircuitState {
        &self.current_state
    }
}
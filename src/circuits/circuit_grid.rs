//! # Circuit Grid System: ATP-Driven Differential Equations
//! 
//! This module implements the circuit grid system that solves dx/dATP differential
//! equations with dynamic circuit resolution and adaptive grid optimization.

use crate::error::{NebuchadnezzarError, Result};
use crate::circuits::enzyme_circuits::{EnzymeProbCircuit, EnzymeCircuitFactory};
use crate::circuits::ion_channel::ProbabilisticIonChannel;
use crate::systems_biology::AtpPool;
use crate::solvers::{SystemState, AtpIntegrator, AtpRk4Integrator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ATP threshold below which circuits cannot fire
pub const ATP_THRESHOLD: f64 = 0.01;

/// Resolution threshold for converting probabilistic nodes to circuits
pub const RESOLUTION_THRESHOLD: f64 = 0.7;

/// Circuit grid containing probabilistic nodes and resolved circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitGrid {
    pub probabilistic_nodes: Vec<ProbabilisticNode>,
    pub resolved_circuits: Vec<ResolvedCircuit>,
    pub metabolite_concentrations: HashMap<String, f64>,
    pub current_atp: f64,
    pub time: f64,
    pub grid_id: String,
}

/// Probabilistic node that can be resolved to detailed circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticNode {
    pub node_id: String,
    pub node_type: NodeType,
    pub probability: f64,
    pub atp_cost: f64,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub resolution_importance: f64,
    pub last_activity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    EnzymaticReaction {
        enzyme_class: String,
        substrate_binding_prob: f64,
        product_formation_prob: f64,
    },
    IonChannelCluster {
        channel_type: String,
        open_probability: f64,
        conductance_distribution: (f64, f64), // (mean, std)
    },
    MetabolicPathway {
        pathway_name: String,
        flux_efficiency: f64,
        regulation_strength: f64,
    },
    SignalingCascade {
        cascade_type: String,
        amplification_factor: f64,
        decay_rate: f64,
    },
}

/// Resolved circuit with detailed kinetics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedCircuit {
    pub circuit_id: String,
    pub circuit_type: CircuitType,
    pub detailed_model: DetailedCircuitModel,
    pub current_state: CircuitState,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitType {
    EnzymeCircuit(Box<dyn EnzymeProbCircuit>),
    IonChannelCircuit(ProbabilisticIonChannel),
    CompoundCircuit {
        sub_circuits: Vec<String>,
        interaction_matrix: Vec<Vec<f64>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedCircuitModel {
    pub state_variables: Vec<String>,
    pub rate_equations: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
    pub conservation_laws: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitState {
    pub current_values: HashMap<String, f64>,
    pub flux_rates: HashMap<String, f64>,
    pub energy_consumption: f64,
    pub last_update: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub efficiency: f64,
    pub robustness: f64,
    pub response_time: f64,
    pub energy_cost: f64,
}

impl CircuitGrid {
    pub fn new(grid_id: String, initial_atp: f64) -> Self {
        Self {
            probabilistic_nodes: Vec::new(),
            resolved_circuits: Vec::new(),
            metabolite_concentrations: HashMap::new(),
            current_atp: initial_atp,
            time: 0.0,
            grid_id,
        }
    }

    /// Add probabilistic node to the grid
    pub fn add_probabilistic_node(&mut self, node: ProbabilisticNode) {
        self.probabilistic_nodes.push(node);
    }

    /// Solve circuit grid with ATP-driven differential equations
    pub fn solve_circuit_grid(&mut self, target_atp_consumption: f64) -> Result<SystemState> {
        let mut current_atp = self.current_atp;
        let mut total_atp_consumed = 0.0;
        let mut metabolite_changes = HashMap::new();

        while current_atp > ATP_THRESHOLD && total_atp_consumed < target_atp_consumption {
            // Find all circuits that can fire with current ATP
            let active_circuits = self.find_active_circuits(current_atp)?;
            
            if active_circuits.is_empty() {
                break; // No more circuits can fire
            }

            // Solve dx/dATP for each active circuit
            for circuit_id in active_circuits {
                let delta_metabolites = self.compute_circuit_flux(&circuit_id, current_atp)?;
                let delta_atp = self.get_circuit_atp_cost(&circuit_id);
                
                // Update metabolite concentrations
                self.update_metabolite_concentrations(&delta_metabolites)?;
                
                // Update ATP
                current_atp -= delta_atp;
                total_atp_consumed += delta_atp;

                // Store changes for dx/dATP calculation
                for (metabolite, change) in delta_metabolites {
                    *metabolite_changes.entry(metabolite).or_insert(0.0) += change / delta_atp;
                }

                if current_atp <= ATP_THRESHOLD {
                    break;
                }
            }
        }

        self.current_atp = current_atp;
        
        // Create system state representing dx/dATP
        Ok(self.create_system_state(metabolite_changes, total_atp_consumed)?)
    }

    /// Find circuits that can fire with current ATP level
    fn find_active_circuits(&self, current_atp: f64) -> Result<Vec<String>> {
        let mut active_circuits = Vec::new();

        // Check probabilistic nodes
        for node in &self.probabilistic_nodes {
            if node.can_fire_with_atp(current_atp) {
                active_circuits.push(node.node_id.clone());
            }
        }

        // Check resolved circuits
        for circuit in &self.resolved_circuits {
            if circuit.can_fire_with_atp(current_atp) {
                active_circuits.push(circuit.circuit_id.clone());
            }
        }

        Ok(active_circuits)
    }

    /// Compute flux for a specific circuit
    fn compute_circuit_flux(&self, circuit_id: &str, current_atp: f64) -> Result<HashMap<String, f64>> {
        // Try to find in probabilistic nodes first
        for node in &self.probabilistic_nodes {
            if node.node_id == circuit_id {
                return node.compute_probabilistic_flux(current_atp, &self.metabolite_concentrations);
            }
        }

        // Try resolved circuits
        for circuit in &self.resolved_circuits {
            if circuit.circuit_id == circuit_id {
                return circuit.compute_detailed_flux(current_atp, &self.metabolite_concentrations);
            }
        }

        Err(NebuchadnezzarError::CircuitError(
            format!("Circuit not found: {}", circuit_id)
        ))
    }

    /// Get ATP cost for a circuit
    fn get_circuit_atp_cost(&self, circuit_id: &str) -> f64 {
        // Check probabilistic nodes
        for node in &self.probabilistic_nodes {
            if node.node_id == circuit_id {
                return node.atp_cost;
            }
        }

        // Check resolved circuits
        for circuit in &self.resolved_circuits {
            if circuit.circuit_id == circuit_id {
                return circuit.current_state.energy_consumption;
            }
        }

        0.0 // Default if not found
    }

    /// Update metabolite concentrations
    fn update_metabolite_concentrations(&mut self, changes: &HashMap<String, f64>) -> Result<()> {
        for (metabolite, change) in changes {
            let current_conc = self.metabolite_concentrations.get(metabolite).unwrap_or(&0.0);
            let new_conc = current_conc + change;
            
            // Ensure non-negative concentrations
            if new_conc < 0.0 {
                return Err(NebuchadnezzarError::ComputationError(
                    format!("Negative concentration for {}: {}", metabolite, new_conc)
                ));
            }
            
            self.metabolite_concentrations.insert(metabolite.clone(), new_conc);
        }
        Ok(())
    }

    /// Create system state from changes
    fn create_system_state(&self, changes: HashMap<String, f64>, atp_consumed: f64) -> Result<SystemState> {
        let mut state = SystemState::new(changes.len(), 0, 0);
        
        // Set metabolite concentrations
        for (i, (_, change_rate)) in changes.iter().enumerate() {
            state.concentrations[i] = *change_rate;
        }
        
        // Update ATP pool
        state.atp_pool.atp_concentration = self.current_atp;
        state.cumulative_atp = atp_consumed;
        state.time = self.time;
        
        Ok(state)
    }
}

/// Adaptive grid that dynamically resolves probabilistic nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveGrid {
    pub base_grid: CircuitGrid,
    pub resolution_criteria: ResolutionCriteria,
    pub optimization_history: Vec<OptimizationStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionCriteria {
    pub probability_threshold: f64,
    pub activity_threshold: f64,
    pub importance_threshold: f64,
    pub computational_budget: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStep {
    pub step_number: usize,
    pub nodes_resolved: Vec<String>,
    pub performance_improvement: f64,
    pub computational_cost: f64,
    pub timestamp: f64,
}

impl AdaptiveGrid {
    pub fn new(grid_id: String, initial_atp: f64) -> Self {
        Self {
            base_grid: CircuitGrid::new(grid_id, initial_atp),
            resolution_criteria: ResolutionCriteria {
                probability_threshold: RESOLUTION_THRESHOLD,
                activity_threshold: 0.5,
                importance_threshold: 0.6,
                computational_budget: 100.0,
            },
            optimization_history: Vec::new(),
        }
    }

    /// Perform optimization step with dynamic resolution
    pub fn optimize_step(&mut self) -> Result<OptimizationStep> {
        let step_number = self.optimization_history.len();
        let mut nodes_resolved = Vec::new();
        let mut total_cost = 0.0;

        // Evaluate which probabilistic nodes are "hot"
        let promising_nodes = self.find_promising_nodes()?;

        // Convert promising nodes to circuits
        for node_id in promising_nodes {
            if total_cost < self.resolution_criteria.computational_budget {
                let resolved_circuit = self.resolve_node_to_circuit(&node_id)?;
                let resolution_cost = self.estimate_resolution_cost(&node_id);
                
                self.base_grid.resolved_circuits.push(resolved_circuit);
                self.remove_probabilistic_node(&node_id)?;
                
                nodes_resolved.push(node_id);
                total_cost += resolution_cost;
            }
        }

        // Calculate performance improvement
        let performance_improvement = self.calculate_performance_improvement(&nodes_resolved)?;

        let optimization_step = OptimizationStep {
            step_number,
            nodes_resolved,
            performance_improvement,
            computational_cost: total_cost,
            timestamp: self.base_grid.time,
        };

        self.optimization_history.push(optimization_step.clone());
        Ok(optimization_step)
    }

    /// Find probabilistic nodes that should be resolved
    fn find_promising_nodes(&self) -> Result<Vec<String>> {
        let mut promising = Vec::new();

        for node in &self.base_grid.probabilistic_nodes {
            if node.probability > self.resolution_criteria.probability_threshold &&
               node.resolution_importance > self.resolution_criteria.importance_threshold &&
               node.last_activity > self.resolution_criteria.activity_threshold {
                promising.push(node.node_id.clone());
            }
        }

        Ok(promising)
    }

    /// Resolve a probabilistic node to a detailed circuit
    fn resolve_node_to_circuit(&self, node_id: &str) -> Result<ResolvedCircuit> {
        let node = self.base_grid.probabilistic_nodes.iter()
            .find(|n| n.node_id == node_id)
            .ok_or_else(|| NebuchadnezzarError::CircuitError(
                format!("Node not found: {}", node_id)
            ))?;

        let detailed_model = match &node.node_type {
            NodeType::EnzymaticReaction { enzyme_class, .. } => {
                self.create_enzyme_circuit_model(enzyme_class)?
            },
            NodeType::IonChannelCluster { channel_type, .. } => {
                self.create_ion_channel_model(channel_type)?
            },
            NodeType::MetabolicPathway { pathway_name, .. } => {
                self.create_pathway_model(pathway_name)?
            },
            NodeType::SignalingCascade { cascade_type, .. } => {
                self.create_signaling_model(cascade_type)?
            },
        };

        Ok(ResolvedCircuit {
            circuit_id: format!("resolved_{}", node_id),
            circuit_type: CircuitType::CompoundCircuit {
                sub_circuits: vec![node_id.to_string()],
                interaction_matrix: vec![vec![1.0]],
            },
            detailed_model,
            current_state: CircuitState {
                current_values: HashMap::new(),
                flux_rates: HashMap::new(),
                energy_consumption: node.atp_cost,
                last_update: 0.0,
            },
            performance_metrics: PerformanceMetrics {
                efficiency: 0.8,
                robustness: 0.7,
                response_time: 0.1,
                energy_cost: node.atp_cost,
            },
        })
    }

    fn create_enzyme_circuit_model(&self, enzyme_class: &str) -> Result<DetailedCircuitModel> {
        Ok(DetailedCircuitModel {
            state_variables: vec![
                "substrate".to_string(),
                "enzyme".to_string(),
                "product".to_string(),
                "ATP".to_string(),
            ],
            rate_equations: vec![
                "dS/dATP = -k1*[S]*[E] / v_ATP".to_string(),
                "dP/dATP = +k1*[S]*[E] / v_ATP".to_string(),
            ],
            parameter_values: HashMap::from([
                ("k1".to_string(), 1.0),
                ("v_ATP".to_string(), 0.1),
            ]),
            conservation_laws: vec![
                "[S] + [P] = constant".to_string(),
            ],
        })
    }

    fn create_ion_channel_model(&self, channel_type: &str) -> Result<DetailedCircuitModel> {
        Ok(DetailedCircuitModel {
            state_variables: vec![
                "voltage".to_string(),
                "current".to_string(),
                "open_probability".to_string(),
            ],
            rate_equations: vec![
                "dV/dATP = -I_total/(C*v_ATP)".to_string(),
                "dP_open/dATP = (alpha*(1-P_open) - beta*P_open)/v_ATP".to_string(),
            ],
            parameter_values: HashMap::from([
                ("alpha".to_string(), 0.1),
                ("beta".to_string(), 0.05),
                ("C".to_string(), 1.0),
            ]),
            conservation_laws: vec![],
        })
    }

    fn create_pathway_model(&self, pathway_name: &str) -> Result<DetailedCircuitModel> {
        Ok(DetailedCircuitModel {
            state_variables: vec!["flux".to_string(), "regulation".to_string()],
            rate_equations: vec!["dFlux/dATP = k_pathway * regulation / v_ATP".to_string()],
            parameter_values: HashMap::from([("k_pathway".to_string(), 1.0)]),
            conservation_laws: vec![],
        })
    }

    fn create_signaling_model(&self, cascade_type: &str) -> Result<DetailedCircuitModel> {
        Ok(DetailedCircuitModel {
            state_variables: vec!["signal_strength".to_string(), "amplification".to_string()],
            rate_equations: vec!["dSignal/dATP = amplification * input / v_ATP".to_string()],
            parameter_values: HashMap::from([("amplification".to_string(), 2.0)]),
            conservation_laws: vec![],
        })
    }

    fn remove_probabilistic_node(&mut self, node_id: &str) -> Result<()> {
        self.base_grid.probabilistic_nodes.retain(|n| n.node_id != node_id);
        Ok(())
    }

    fn estimate_resolution_cost(&self, node_id: &str) -> f64 {
        // Simple cost model - could be more sophisticated
        10.0
    }

    fn calculate_performance_improvement(&self, resolved_nodes: &[String]) -> Result<f64> {
        // Simple performance model
        Ok(resolved_nodes.len() as f64 * 0.1)
    }
}

// Implementation for node and circuit behavior
impl ProbabilisticNode {
    pub fn can_fire_with_atp(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.atp_cost && self.probability > 0.1
    }

    pub fn compute_probabilistic_flux(
        &self,
        atp_concentration: f64,
        metabolite_concentrations: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut flux = HashMap::new();
        
        if self.can_fire_with_atp(atp_concentration) {
            let base_rate = self.probability * atp_concentration;
            
            for input in &self.inputs {
                flux.insert(input.clone(), -base_rate);
            }
            
            for output in &self.outputs {
                flux.insert(output.clone(), base_rate);
            }
        }
        
        Ok(flux)
    }
}

impl ResolvedCircuit {
    pub fn can_fire_with_atp(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.current_state.energy_consumption
    }

    pub fn compute_detailed_flux(
        &self,
        atp_concentration: f64,
        metabolite_concentrations: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        // Detailed flux calculation based on resolved kinetics
        let mut flux = HashMap::new();
        
        // This would involve solving the detailed rate equations
        // For now, simplified calculation
        let efficiency = self.performance_metrics.efficiency;
        let base_flux = efficiency * atp_concentration;
        
        flux.insert("product".to_string(), base_flux);
        flux.insert("substrate".to_string(), -base_flux);
        
        Ok(flux)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_grid_creation() {
        let grid = CircuitGrid::new("test_grid".to_string(), 10.0);
        assert_eq!(grid.current_atp, 10.0);
        assert_eq!(grid.grid_id, "test_grid");
    }

    #[test]
    fn test_adaptive_grid_optimization() {
        let mut adaptive_grid = AdaptiveGrid::new("adaptive_test".to_string(), 15.0);
        
        // Add a test probabilistic node
        let test_node = ProbabilisticNode {
            node_id: "test_node".to_string(),
            node_type: NodeType::EnzymaticReaction {
                enzyme_class: "kinase".to_string(),
                substrate_binding_prob: 0.8,
                product_formation_prob: 0.9,
            },
            probability: 0.8,
            atp_cost: 1.0,
            inputs: vec!["substrate".to_string()],
            outputs: vec!["product".to_string()],
            resolution_importance: 0.8,
            last_activity: 0.9,
        };
        
        adaptive_grid.base_grid.add_probabilistic_node(test_node);
        
        let result = adaptive_grid.optimize_step();
        assert!(result.is_ok());
    }
} 
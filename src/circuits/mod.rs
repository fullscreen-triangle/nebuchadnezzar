//! # Electrical Circuit Foundation
//! 
//! This module provides the electrical circuit foundation for simulating intracellular
//! processes using hierarchical probabilistic circuits with ATP-based differential equations.

use crate::error::{NebuchadnezzarError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod ion_channel;
pub mod enzyme_circuits;
pub mod circuit_grid;
pub mod hierarchical_framework;

pub use ion_channel::*;
pub use enzyme_circuits::*;
pub use circuit_grid::*;
pub use hierarchical_framework::*;

/// Base circuit trait for all circuit elements
pub trait Circuit {
    fn get_id(&self) -> &str;
    fn can_fire(&self, atp_concentration: f64) -> bool;
    fn compute_current(&self, voltage: f64, atp_concentration: f64) -> Result<f64>;
    fn get_conductance(&self, voltage: f64, atp_concentration: f64) -> f64;
    fn update_state(&mut self, dt: f64, voltage: f64, atp_concentration: f64) -> Result<()>;
}

/// ATP-dependent conductance function
pub fn atp_dependent_conductance(base_conductance: f64, atp_concentration: f64, km_atp: f64) -> f64 {
    base_conductance * (atp_concentration / (km_atp + atp_concentration))
}

/// Kirchhoff's current law solver for circuit nodes
pub fn solve_kirchhoff_current_law(
    node_currents: &[f64],
    tolerance: f64,
) -> Result<bool> {
    let total_current: f64 = node_currents.iter().sum();
    
    if total_current.abs() < tolerance {
        Ok(true)
    } else {
        Err(NebuchadnezzarError::CircuitError(
            format!("Kirchhoff's current law violated: total current = {}", total_current)
        ))
    }
}

/// Kirchhoff's voltage law solver for circuit loops
pub fn solve_kirchhoff_voltage_law(
    loop_voltages: &[f64],
    tolerance: f64,
) -> Result<bool> {
    let total_voltage: f64 = loop_voltages.iter().sum();
    
    if total_voltage.abs() < tolerance {
        Ok(true)
    } else {
        Err(NebuchadnezzarError::CircuitError(
            format!("Kirchhoff's voltage law violated: total voltage = {}", total_voltage)
        ))
    }
}

/// Membrane model with multiple ion channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneModel {
    pub membrane_id: String,
    pub capacitance: f64,
    pub resting_potential: f64,
    pub current_voltage: f64,
    pub ion_channels: Vec<ProbabilisticIonChannel>,
    pub leak_conductance: f64,
    pub leak_reversal: f64,
}

impl MembraneModel {
    pub fn new(membrane_id: String, capacitance: f64, resting_potential: f64) -> Self {
        Self {
            membrane_id,
            capacitance,
            resting_potential,
            current_voltage: resting_potential,
            ion_channels: Vec::new(),
            leak_conductance: 0.1,
            leak_reversal: resting_potential,
        }
    }

    pub fn add_ion_channel(&mut self, channel: ProbabilisticIonChannel) {
        self.ion_channels.push(channel);
    }

    pub fn compute_total_current(&self, atp_concentration: f64) -> Result<f64> {
        let mut total_current = 0.0;

        // Ion channel currents
        for channel in &self.ion_channels {
            total_current += channel.compute_current(self.current_voltage, atp_concentration)?;
        }

        // Leak current
        total_current += self.leak_conductance * (self.current_voltage - self.leak_reversal);

        Ok(total_current)
    }

    pub fn update_voltage(&mut self, dt: f64, atp_concentration: f64) -> Result<()> {
        let total_current = self.compute_total_current(atp_concentration)?;
        
        // dV/dt = -I_total / C
        let dv_dt = -total_current / self.capacitance;
        self.current_voltage += dv_dt * dt;

        // Update ion channel states
        for channel in &mut self.ion_channels {
            channel.update_gating_variables(dt, self.current_voltage, atp_concentration)?;
        }

        Ok(())
    }
}

/// Circuit network for connecting multiple circuits
#[derive(Debug, Clone)]
pub struct CircuitNetwork {
    pub network_id: String,
    pub circuits: HashMap<String, Box<dyn Circuit>>,
    pub connections: Vec<CircuitConnection>,
    pub voltage_nodes: HashMap<String, f64>,
    pub current_flows: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConnection {
    pub from_circuit: String,
    pub to_circuit: String,
    pub connection_type: ConnectionType,
    pub conductance: f64,
    pub delay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Resistive,
    Capacitive,
    Inductive,
    NonLinear { function: String },
}

impl CircuitNetwork {
    pub fn new(network_id: String) -> Self {
        Self {
            network_id,
            circuits: HashMap::new(),
            connections: Vec::new(),
            voltage_nodes: HashMap::new(),
            current_flows: HashMap::new(),
        }
    }

    pub fn add_circuit(&mut self, circuit_id: String, circuit: Box<dyn Circuit>) {
        self.circuits.insert(circuit_id, circuit);
    }

    pub fn add_connection(&mut self, connection: CircuitConnection) {
        self.connections.push(connection);
    }

    pub fn solve_network(&mut self, atp_concentration: f64) -> Result<()> {
        // Solve the network using nodal analysis
        let node_equations = self.build_node_equations(atp_concentration)?;
        let voltages = self.solve_linear_system(node_equations)?;
        
        // Update voltage nodes
        for (node_id, voltage) in voltages {
            self.voltage_nodes.insert(node_id, voltage);
        }

        // Calculate current flows
        self.calculate_current_flows(atp_concentration)?;

        Ok(())
    }

    fn build_node_equations(&self, atp_concentration: f64) -> Result<Vec<Vec<f64>>> {
        // Build the conductance matrix and current vector for nodal analysis
        // This is a simplified implementation
        let mut equations = Vec::new();
        
        // For each node, apply Kirchhoff's current law
        for (node_id, _) in &self.voltage_nodes {
            let mut equation = vec![0.0; self.voltage_nodes.len()];
            
            // Add conductances from connections
            for connection in &self.connections {
                match connection.connection_type {
                    ConnectionType::Resistive => {
                        // Add conductance terms
                        equation[0] += connection.conductance; // Simplified
                    },
                    _ => {
                        // Handle other connection types
                    }
                }
            }
            
            equations.push(equation);
        }

        Ok(equations)
    }

    fn solve_linear_system(&self, equations: Vec<Vec<f64>>) -> Result<HashMap<String, f64>> {
        // Simplified linear system solver
        // In practice, would use proper matrix solver
        let mut voltages = HashMap::new();
        
        for (i, (node_id, _)) in self.voltage_nodes.iter().enumerate() {
            voltages.insert(node_id.clone(), 0.0); // Placeholder
        }

        Ok(voltages)
    }

    fn calculate_current_flows(&mut self, atp_concentration: f64) -> Result<()> {
        // Calculate currents through each connection
        for connection in &self.connections {
            let from_voltage = self.voltage_nodes.get(&connection.from_circuit).unwrap_or(&0.0);
            let to_voltage = self.voltage_nodes.get(&connection.to_circuit).unwrap_or(&0.0);
            
            let current = match connection.connection_type {
                ConnectionType::Resistive => {
                    connection.conductance * (from_voltage - to_voltage)
                },
                _ => 0.0, // Simplified
            };
            
            let connection_id = format!("{}_{}", connection.from_circuit, connection.to_circuit);
            self.current_flows.insert(connection_id, current);
        }

        Ok(())
    }
}

/// Factory for creating common circuit configurations
pub struct CircuitFactory;

impl CircuitFactory {
    /// Create a neuron membrane with standard ion channels
    pub fn create_neuron_membrane() -> MembraneModel {
        let mut membrane = MembraneModel::new(
            "neuron_membrane".to_string(),
            1.0, // 1 µF/cm²
            -70.0, // -70 mV resting potential
        );

        // Add sodium channels
        membrane.add_ion_channel(ProbabilisticIonChannel::voltage_gated_sodium());
        
        // Add potassium channels
        membrane.add_ion_channel(ProbabilisticIonChannel::voltage_gated_potassium());
        
        // Add ATP-sensitive potassium channels
        membrane.add_ion_channel(ProbabilisticIonChannel::atp_sensitive_potassium());

        membrane
    }

    /// Create a mitochondrial membrane
    pub fn create_mitochondrial_membrane() -> MembraneModel {
        let mut membrane = MembraneModel::new(
            "mitochondrial_membrane".to_string(),
            0.5, // Lower capacitance
            -180.0, // More negative potential
        );

        // Add specialized mitochondrial channels
        membrane.add_ion_channel(ProbabilisticIonChannel::new(
            "mito_calcium".to_string(),
            ion_channel::ChannelType::VoltageGated {
                activation_threshold: -20.0,
                inactivation_threshold: Some(10.0),
            },
            1.0,
            120.0,
        ));

        membrane
    }

    /// Create a glycolysis circuit grid
    pub fn create_glycolysis_circuit() -> Result<AdaptiveGrid> {
        let mut grid = AdaptiveGrid::new("glycolysis".to_string(), 10.0);

        // Add glycolysis enzyme nodes
        let enzymes = EnzymeCircuitFactory::create_glycolysis_enzymes();
        
        // Convert enzymes to probabilistic nodes (simplified)
        for (i, _enzyme) in enzymes.iter().enumerate() {
            let node = circuit_grid::ProbabilisticNode {
                node_id: format!("glycolysis_step_{}", i),
                node_type: circuit_grid::NodeType::EnzymaticReaction {
                    enzyme_class: "kinase".to_string(),
                    substrate_binding_prob: 0.8,
                    product_formation_prob: 0.9,
                },
                probability: 0.85,
                atp_cost: 1.0,
                inputs: vec!["glucose".to_string()],
                outputs: vec!["pyruvate".to_string()],
                resolution_importance: 0.9,
                last_activity: 1.0,
            };
            
            grid.base_grid.add_probabilistic_node(node)?;
        }

        Ok(grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atp_dependent_conductance() {
        let conductance = atp_dependent_conductance(1.0, 2.0, 1.0);
        assert!((conductance - 0.6667).abs() < 0.001);
    }

    #[test]
    fn test_kirchhoff_current_law() {
        let currents = vec![1.0, -0.5, -0.5];
        assert!(solve_kirchhoff_current_law(&currents, 0.001).is_ok());
        
        let bad_currents = vec![1.0, -0.3, -0.3];
        assert!(solve_kirchhoff_current_law(&bad_currents, 0.001).is_err());
    }

    #[test]
    fn test_membrane_model() {
        let membrane = CircuitFactory::create_neuron_membrane();
        assert_eq!(membrane.current_voltage, -70.0);
        assert!(!membrane.ion_channels.is_empty());
    }

    #[test]
    fn test_circuit_network() {
        let network = CircuitNetwork::new("test_network".to_string());
        assert_eq!(network.network_id, "test_network");
    }
} 
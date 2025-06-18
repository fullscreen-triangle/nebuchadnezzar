//! # Circuit Module: Electrical Circuit Foundation for Biological Modeling
//! 
//! This module provides the core electrical circuit components that serve as the foundation
//! for modeling biochemical processes. It implements ion channels, membranes, and circuit
//! networks using established biophysical principles.
//!
//! ## Core Components
//!
//! - **Ion Channels**: Voltage-gated and ligand-gated channels with Hodgkin-Huxley dynamics
//! - **Membranes**: Capacitive membrane models with multiple ion channel types
//! - **Circuit Networks**: Multi-compartment electrical circuit simulation
//! - **ATP Coupling**: Integration of ATP-dependent electrical processes
//!
//! ## Mathematical Foundation
//!
//! The module implements the mathematical mappings from the documentation:
//! - Ion channels as variable resistors/conductors
//! - Membranes as RC circuits
//! - Biochemical reactions as circuit elements
//! - ATP power consumption calculations

pub mod ion_channel;
pub mod membrane;
pub mod network;
pub mod hodgkin_huxley;

use crate::error::{NebuchadnezzarError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Circuit element identification
pub type ElementId = String;
pub type NodeId = usize;

/// Fundamental electrical quantities
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Voltage(pub f64); // mV

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Current(pub f64); // pA

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Conductance(pub f64); // pS (picoSiemens)

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Capacitance(pub f64); // pF (picoFarads)

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Resistance(pub f64); // GΩ (gigaohms)

impl From<Conductance> for Resistance {
    fn from(g: Conductance) -> Self {
        if g.0.abs() < f64::EPSILON {
            Resistance(f64::INFINITY)
        } else {
            Resistance(1000.0 / g.0) // Convert pS to GΩ
        }
    }
}

impl From<Resistance> for Conductance {
    fn from(r: Resistance) -> Self {
        if r.0.is_infinite() {
            Conductance(0.0)
        } else {
            Conductance(1000.0 / r.0) // Convert GΩ to pS
        }
    }
}

/// Circuit node representing a point in the electrical network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitNode {
    pub id: NodeId,
    pub voltage: Voltage,
    pub capacitance: Capacitance,
    pub name: String,
}

impl CircuitNode {
    pub fn new(id: NodeId, name: String, initial_voltage: f64) -> Self {
        Self {
            id,
            voltage: Voltage(initial_voltage),
            capacitance: Capacitance(0.0),
            name,
        }
    }

    pub fn with_capacitance(mut self, capacitance: f64) -> Self {
        self.capacitance = Capacitance(capacitance);
        self
    }
}

/// Circuit element connecting nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CircuitElement {
    /// Resistor with fixed resistance
    Resistor {
        id: ElementId,
        from: NodeId,
        to: NodeId,
        resistance: Resistance,
    },
    /// Variable conductance (e.g., ion channel)
    VariableConductance {
        id: ElementId,
        from: NodeId,
        to: NodeId,
        conductance_function: ConductanceFunction,
    },
    /// Current source
    CurrentSource {
        id: ElementId,
        from: NodeId,
        to: NodeId,
        current: Current,
    },
    /// Voltage source
    VoltageSource {
        id: ElementId,
        from: NodeId,
        to: NodeId,
        voltage: Voltage,
    },
}

/// Function defining conductance as a function of state variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConductanceFunction {
    /// Constant conductance
    Constant(f64),
    /// Voltage-dependent (Hodgkin-Huxley type)
    VoltageDependentHH {
        max_conductance: f64,
        gating_variables: Vec<String>,
        reversal_potential: f64,
    },
    /// ATP-dependent conductance
    AtpDependent {
        max_conductance: f64,
        atp_km: f64,
        hill_coefficient: f64,
    },
    /// Ligand-gated
    LigandGated {
        max_conductance: f64,
        ligand_km: f64,
        cooperativity: f64,
    },
}

impl ConductanceFunction {
    pub fn evaluate(
        &self,
        voltage: f64,
        state: &HashMap<String, f64>,
    ) -> Result<f64> {
        match self {
            Self::Constant(g) => Ok(*g),
            Self::VoltageDependentHH { 
                max_conductance, 
                gating_variables, 
                reversal_potential: _ 
            } => {
                let mut total_gating = 1.0;
                for var_name in gating_variables {
                    let gating_value = state.get(var_name)
                        .ok_or_else(|| NebuchadnezzarError::ComputationError(
                            format!("Missing gating variable: {}", var_name)
                        ))?;
                    total_gating *= gating_value;
                }
                Ok(max_conductance * total_gating)
            },
            Self::AtpDependent { 
                max_conductance, 
                atp_km, 
                hill_coefficient 
            } => {
                let atp_conc = state.get("ATP")
                    .ok_or_else(|| NebuchadnezzarError::ComputationError(
                        "Missing ATP concentration".to_string()
                    ))?;
                let atp_factor = atp_conc.powf(*hill_coefficient) / 
                    (atp_km.powf(*hill_coefficient) + atp_conc.powf(*hill_coefficient));
                Ok(max_conductance * atp_factor)
            },
            Self::LigandGated { 
                max_conductance, 
                ligand_km, 
                cooperativity 
            } => {
                let ligand_conc = state.get("ligand")
                    .ok_or_else(|| NebuchadnezzarError::ComputationError(
                        "Missing ligand concentration".to_string()
                    ))?;
                let ligand_factor = ligand_conc.powf(*cooperativity) / 
                    (ligand_km.powf(*cooperativity) + ligand_conc.powf(*cooperativity));
                Ok(max_conductance * ligand_factor)
            },
        }
    }
}

/// Circuit network containing nodes and elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitNetwork {
    pub nodes: HashMap<NodeId, CircuitNode>,
    pub elements: Vec<CircuitElement>,
    pub state_variables: HashMap<String, f64>,
    pub time: f64,
}

impl CircuitNetwork {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            elements: Vec::new(),
            state_variables: HashMap::new(),
            time: 0.0,
        }
    }

    pub fn add_node(&mut self, node: CircuitNode) -> Result<()> {
        if self.nodes.contains_key(&node.id) {
            return Err(NebuchadnezzarError::CircuitError(
                format!("Node with id {} already exists", node.id)
            ));
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    pub fn add_element(&mut self, element: CircuitElement) -> Result<()> {
        // Validate that referenced nodes exist
        let (from, to) = match &element {
            CircuitElement::Resistor { from, to, .. } => (*from, *to),
            CircuitElement::VariableConductance { from, to, .. } => (*from, *to),
            CircuitElement::CurrentSource { from, to, .. } => (*from, *to),
            CircuitElement::VoltageSource { from, to, .. } => (*from, *to),
        };

        if !self.nodes.contains_key(&from) {
            return Err(NebuchadnezzarError::CircuitError(
                format!("From node {} does not exist", from)
            ));
        }
        if !self.nodes.contains_key(&to) {
            return Err(NebuchadnezzarError::CircuitError(
                format!("To node {} does not exist", to)
            ));
        }

        self.elements.push(element);
        Ok(())
    }

    pub fn set_state_variable(&mut self, name: String, value: f64) {
        self.state_variables.insert(name, value);
    }

    /// Calculate current through each element
    pub fn calculate_element_currents(&self) -> Result<HashMap<String, Current>> {
        let mut currents = HashMap::new();

        for element in &self.elements {
            let current = match element {
                CircuitElement::Resistor { id, from, to, resistance } => {
                    let v_from = self.nodes.get(from)
                        .ok_or_else(|| NebuchadnezzarError::CircuitError(
                            format!("Node {} not found", from)
                        ))?.voltage.0;
                    let v_to = self.nodes.get(to)
                        .ok_or_else(|| NebuchadnezzarError::CircuitError(
                            format!("Node {} not found", to)
                        ))?.voltage.0;
                    
                    if resistance.0.is_infinite() {
                        Current(0.0)
                    } else {
                        Current((v_from - v_to) / resistance.0 * 1000.0) // Convert to pA
                    }
                },
                CircuitElement::VariableConductance { id, from, to, conductance_function } => {
                    let v_from = self.nodes.get(from)
                        .ok_or_else(|| NebuchadnezzarError::CircuitError(
                            format!("Node {} not found", from)
                        ))?.voltage.0;
                    let v_to = self.nodes.get(to)
                        .ok_or_else(|| NebuchadnezzarError::CircuitError(
                            format!("Node {} not found", to)
                        ))?.voltage.0;
                    
                    let conductance = conductance_function.evaluate(v_from, &self.state_variables)?;
                    Current((v_from - v_to) * conductance)
                },
                CircuitElement::CurrentSource { id, current, .. } => *current,
                CircuitElement::VoltageSource { .. } => {
                    // Voltage sources require special handling in circuit analysis
                    Current(0.0) // Placeholder
                },
            };

            let element_id = match element {
                CircuitElement::Resistor { id, .. } => id,
                CircuitElement::VariableConductance { id, .. } => id,
                CircuitElement::CurrentSource { id, .. } => id,
                CircuitElement::VoltageSource { id, .. } => id,
            };

            currents.insert(element_id.clone(), current);
        }

        Ok(currents)
    }

    /// Calculate total current entering each node (Kirchhoff's current law)
    pub fn calculate_nodal_currents(&self) -> Result<HashMap<NodeId, Current>> {
        let element_currents = self.calculate_element_currents()?;
        let mut nodal_currents: HashMap<NodeId, f64> = HashMap::new();

        // Initialize all nodes with zero current
        for node_id in self.nodes.keys() {
            nodal_currents.insert(*node_id, 0.0);
        }

        // Sum currents for each node
        for element in &self.elements {
            let (from, to, element_id) = match element {
                CircuitElement::Resistor { id, from, to, .. } => (from, to, id),
                CircuitElement::VariableConductance { id, from, to, .. } => (from, to, id),
                CircuitElement::CurrentSource { id, from, to, .. } => (from, to, id),
                CircuitElement::VoltageSource { id, from, to, .. } => (from, to, id),
            };

            if let Some(current) = element_currents.get(element_id) {
                // Current flows from 'from' node to 'to' node
                *nodal_currents.get_mut(from).unwrap() -= current.0; // Current leaving 'from'
                *nodal_currents.get_mut(to).unwrap() += current.0;   // Current entering 'to'
            }
        }

        Ok(nodal_currents.into_iter()
            .map(|(node_id, current)| (node_id, Current(current)))
            .collect())
    }
}

impl Default for CircuitNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export key types for convenience
pub use ion_channel::*;
pub use membrane::*;
pub use network::*;
pub use hodgkin_huxley::*; 
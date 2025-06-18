//! # Hierarchical Framework: Multi-Level Voltage and Temporal Dynamics
//! 
//! This module implements the 4-level hierarchical framework with temporal evidence decay
//! across different timescales from milliseconds to days.

use crate::error::{NebuchadnezzarError, Result};
use crate::circuits::circuit_grid::{CircuitGrid, AdaptiveGrid};
use crate::circuits::enzyme_circuits::EnzymeProbCircuit;
use crate::circuits::ion_channel::ProbabilisticIonChannel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-level voltage hierarchy as specified
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageHierarchy {
    /// Cellular level: -70mV to +40mV (action potentials, membrane potential)
    pub cellular_voltage: f64,
    /// Organelle level: Mitochondrial (-180mV), ER (-60mV)
    pub organelle_voltages: HashMap<String, f64>,
    /// Compartment levels: various subcellular compartments
    pub compartment_voltages: HashMap<String, f64>,
    /// Molecular protein states: individual protein conformational states
    pub molecular_states: HashMap<String, ProteinState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinState {
    pub conformation_energy: f64,
    pub binding_affinity: f64,
    pub activity_level: f64,
    pub last_transition_time: f64,
}

/// Temporal evidence decay across different timescales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvidence {
    /// Millisecond scale: ion channel gating, enzyme conformational changes
    pub millisecond_events: Vec<TemporalEvent>,
    /// Second scale: metabolic reactions, signaling cascades
    pub second_events: Vec<TemporalEvent>,
    /// Minute scale: gene expression changes, protein synthesis
    pub minute_events: Vec<TemporalEvent>,
    /// Hour scale: cell cycle events, metabolic shifts
    pub hour_events: Vec<TemporalEvent>,
    /// Day scale: circadian rhythms, long-term adaptations
    pub day_events: Vec<TemporalEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEvent {
    pub event_id: String,
    pub event_type: EventType,
    pub strength: f64,
    pub decay_rate: f64,
    pub timestamp: f64,
    pub half_life: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    IonChannelGating,
    EnzymeActivation,
    MetabolicFlux,
    GeneExpression,
    ProteinSynthesis,
    CellCycleEvent,
    CircadianSignal,
}

/// Level 1: Molecular Circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularCircuit {
    pub circuit_id: String,
    pub protein_complexes: Vec<ProteinComplex>,
    pub enzyme_circuits: Vec<Box<dyn EnzymeProbCircuit>>,
    pub ion_channels: Vec<ProbabilisticIonChannel>,
    pub molecular_voltage: f64,
    pub atp_local: f64,
    pub temporal_state: TemporalEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinComplex {
    pub complex_id: String,
    pub subunits: Vec<String>,
    pub binding_energy: f64,
    pub conformational_states: Vec<ProteinState>,
    pub current_state_index: usize,
}

/// Level 2: Organelle Networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganelleNetwork {
    pub organelle_id: String,
    pub organelle_type: OrganelleType,
    pub molecular_circuits: Vec<MolecularCircuit>,
    pub inter_circuit_connections: Vec<CircuitConnection>,
    pub organelle_voltage: f64,
    pub local_atp_pool: f64,
    pub metabolic_state: MetabolicState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrganelleType {
    Mitochondrion { cristae_density: f64 },
    EndoplasmicReticulum { membrane_area: f64 },
    Golgi { stack_number: usize },
    Nucleus { chromatin_state: String },
    Peroxisome { enzyme_content: f64 },
    Lysosome { ph_level: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitConnection {
    pub from_circuit: String,
    pub to_circuit: String,
    pub connection_strength: f64,
    pub signal_type: SignalType,
    pub delay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Electrical,
    Chemical,
    Mechanical,
    Thermal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicState {
    pub energy_charge: f64,
    pub nadh_nad_ratio: f64,
    pub calcium_level: f64,
    pub ph: f64,
    pub oxygen_level: f64,
}

/// Level 3: Cellular Integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularIntegration {
    pub cell_id: String,
    pub cell_type: CellType,
    pub organelle_networks: Vec<OrganelleNetwork>,
    pub membrane_potential: f64,
    pub global_atp_pool: f64,
    pub cellular_state: CellularState,
    pub signaling_networks: Vec<SignalingNetwork>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellType {
    Neuron { neuron_type: String },
    Muscle { muscle_type: String },
    Epithelial { barrier_function: f64 },
    Immune { activation_state: String },
    Stem { differentiation_potential: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularState {
    pub cell_cycle_phase: String,
    pub stress_level: f64,
    pub metabolic_activity: f64,
    pub growth_rate: f64,
    pub differentiation_state: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalingNetwork {
    pub pathway_name: String,
    pub active_nodes: Vec<String>,
    pub signal_strength: f64,
    pub crosstalk_connections: Vec<String>,
}

/// Level 4: Tissue/System Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueSystem {
    pub tissue_id: String,
    pub tissue_type: TissueType,
    pub cellular_populations: Vec<CellularPopulation>,
    pub extracellular_matrix: ExtracellularMatrix,
    pub system_voltage: f64,
    pub nutrient_gradients: HashMap<String, Vec<f64>>,
    pub mechanical_properties: MechanicalProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TissueType {
    Nervous { myelination_level: f64 },
    Muscular { fiber_type_ratio: f64 },
    Connective { collagen_density: f64 },
    Epithelial { permeability: f64 },
    Vascular { vessel_density: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularPopulation {
    pub population_id: String,
    pub cell_count: usize,
    pub cells: Vec<CellularIntegration>,
    pub population_behavior: PopulationBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationBehavior {
    pub synchronization_level: f64,
    pub collective_response: f64,
    pub intercellular_communication: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtracellularMatrix {
    pub protein_composition: HashMap<String, f64>,
    pub stiffness: f64,
    pub porosity: f64,
    pub chemical_gradients: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanicalProperties {
    pub elasticity: f64,
    pub viscosity: f64,
    pub contractility: f64,
    pub tension: f64,
}

/// Hierarchical System Manager
pub struct HierarchicalSystem {
    pub molecular_level: Vec<MolecularCircuit>,
    pub organelle_level: Vec<OrganelleNetwork>,
    pub cellular_level: Vec<CellularIntegration>,
    pub tissue_level: Vec<TissueSystem>,
    pub voltage_hierarchy: VoltageHierarchy,
    pub global_temporal_state: TemporalEvidence,
    pub current_time: f64,
}

impl HierarchicalSystem {
    pub fn new() -> Self {
        Self {
            molecular_level: Vec::new(),
            organelle_level: Vec::new(),
            cellular_level: Vec::new(),
            tissue_level: Vec::new(),
            voltage_hierarchy: VoltageHierarchy::default(),
            global_temporal_state: TemporalEvidence::new(),
            current_time: 0.0,
        }
    }

    /// Solve hierarchical system with cross-level interactions
    pub fn solve_hierarchical_system(&mut self, delta_atp: f64) -> Result<HierarchicalState> {
        // Update temporal evidence decay
        self.update_temporal_decay()?;

        // Solve from bottom up: molecular → organelle → cellular → tissue
        let molecular_states = self.solve_molecular_level(delta_atp)?;
        let organelle_states = self.solve_organelle_level(&molecular_states, delta_atp)?;
        let cellular_states = self.solve_cellular_level(&organelle_states, delta_atp)?;
        let tissue_states = self.solve_tissue_level(&cellular_states, delta_atp)?;

        // Update voltage hierarchy
        self.update_voltage_hierarchy(&molecular_states, &organelle_states, &cellular_states)?;

        Ok(HierarchicalState {
            molecular_states,
            organelle_states,
            cellular_states,
            tissue_states,
            voltage_hierarchy: self.voltage_hierarchy.clone(),
            timestamp: self.current_time,
        })
    }

    fn solve_molecular_level(&mut self, delta_atp: f64) -> Result<Vec<MolecularState>> {
        let mut states = Vec::new();

        for circuit in &mut self.molecular_level {
            let mut grid = CircuitGrid::new(circuit.circuit_id.clone(), circuit.atp_local);
            
            // Add enzyme circuits as probabilistic nodes
            for enzyme in &circuit.enzyme_circuits {
                // Convert enzyme to probabilistic node
                // This would involve more complex logic in practice
            }

            let system_state = grid.solve_circuit_grid(delta_atp)?;
            
            states.push(MolecularState {
                circuit_id: circuit.circuit_id.clone(),
                protein_activities: HashMap::new(),
                local_voltage: circuit.molecular_voltage,
                atp_consumption: delta_atp,
                flux_rates: HashMap::new(),
            });
        }

        Ok(states)
    }

    fn solve_organelle_level(&mut self, molecular_states: &[MolecularState], delta_atp: f64) -> Result<Vec<OrganelleState>> {
        let mut states = Vec::new();

        for organelle in &mut self.organelle_level {
            // Integrate molecular circuit outputs
            let total_molecular_flux = self.integrate_molecular_outputs(molecular_states, &organelle.organelle_id)?;
            
            states.push(OrganelleState {
                organelle_id: organelle.organelle_id.clone(),
                organelle_voltage: organelle.organelle_voltage,
                metabolic_flux: total_molecular_flux,
                atp_production: self.calculate_organelle_atp_production(organelle)?,
                calcium_dynamics: self.calculate_calcium_dynamics(organelle)?,
            });
        }

        Ok(states)
    }

    fn solve_cellular_level(&mut self, organelle_states: &[OrganelleState], delta_atp: f64) -> Result<Vec<CellularState>> {
        let mut states = Vec::new();

        for cell in &mut self.cellular_level {
            let membrane_current = self.calculate_membrane_current(organelle_states, &cell.cell_id)?;
            let atp_balance = self.calculate_cellular_atp_balance(organelle_states, &cell.cell_id)?;
            
            states.push(CellularState {
                cell_id: cell.cell_id.clone(),
                membrane_potential: cell.membrane_potential + membrane_current * 0.001, // Simple integration
                atp_level: cell.global_atp_pool + atp_balance,
                signaling_activity: self.calculate_signaling_activity(cell)?,
                metabolic_state: cell.cellular_state.metabolic_activity,
            });
        }

        Ok(states)
    }

    fn solve_tissue_level(&mut self, cellular_states: &[CellularState], delta_atp: f64) -> Result<Vec<TissueState>> {
        let mut states = Vec::new();

        for tissue in &mut self.tissue_level {
            let collective_behavior = self.calculate_collective_behavior(cellular_states, &tissue.tissue_id)?;
            
            states.push(TissueState {
                tissue_id: tissue.tissue_id.clone(),
                collective_voltage: self.calculate_tissue_voltage(cellular_states, &tissue.tissue_id)?,
                mechanical_response: collective_behavior,
                nutrient_consumption: self.calculate_nutrient_consumption(cellular_states)?,
                waste_production: self.calculate_waste_production(cellular_states)?,
            });
        }

        Ok(states)
    }

    fn update_temporal_decay(&mut self) -> Result<()> {
        let current_time = self.current_time;

        // Update millisecond events
        self.global_temporal_state.millisecond_events.retain_mut(|event| {
            let time_elapsed = current_time - event.timestamp;
            event.strength *= (-event.decay_rate * time_elapsed).exp();
            event.strength > 0.01 // Remove very weak events
        });

        // Similar for other timescales
        self.global_temporal_state.second_events.retain_mut(|event| {
            let time_elapsed = current_time - event.timestamp;
            event.strength *= (-event.decay_rate * time_elapsed).exp();
            event.strength > 0.01
        });

        // Continue for minute, hour, and day events...
        
        Ok(())
    }

    fn update_voltage_hierarchy(
        &mut self,
        molecular_states: &[MolecularState],
        organelle_states: &[OrganelleState],
        cellular_states: &[CellularState],
    ) -> Result<()> {
        // Update molecular voltages
        for state in molecular_states {
            self.voltage_hierarchy.molecular_states.insert(
                state.circuit_id.clone(),
                ProteinState {
                    conformation_energy: state.local_voltage,
                    binding_affinity: 1.0,
                    activity_level: 0.8,
                    last_transition_time: self.current_time,
                }
            );
        }

        // Update organelle voltages
        for state in organelle_states {
            self.voltage_hierarchy.organelle_voltages.insert(
                state.organelle_id.clone(),
                state.organelle_voltage,
            );
        }

        // Update cellular voltage (average of all cellular states)
        if !cellular_states.is_empty() {
            self.voltage_hierarchy.cellular_voltage = 
                cellular_states.iter().map(|s| s.membrane_potential).sum::<f64>() / cellular_states.len() as f64;
        }

        Ok(())
    }

    // Helper methods for calculations
    fn integrate_molecular_outputs(&self, molecular_states: &[MolecularState], organelle_id: &str) -> Result<f64> {
        Ok(molecular_states.iter().map(|s| s.atp_consumption).sum())
    }

    fn calculate_organelle_atp_production(&self, organelle: &OrganelleNetwork) -> Result<f64> {
        match organelle.organelle_type {
            OrganelleType::Mitochondrion { cristae_density } => Ok(cristae_density * 10.0),
            _ => Ok(0.0),
        }
    }

    fn calculate_calcium_dynamics(&self, organelle: &OrganelleNetwork) -> Result<f64> {
        Ok(organelle.metabolic_state.calcium_level)
    }

    fn calculate_membrane_current(&self, organelle_states: &[OrganelleState], cell_id: &str) -> Result<f64> {
        Ok(organelle_states.iter().map(|s| s.metabolic_flux * 0.1).sum())
    }

    fn calculate_cellular_atp_balance(&self, organelle_states: &[OrganelleState], cell_id: &str) -> Result<f64> {
        Ok(organelle_states.iter().map(|s| s.atp_production).sum())
    }

    fn calculate_signaling_activity(&self, cell: &CellularIntegration) -> Result<f64> {
        Ok(cell.signaling_networks.iter().map(|s| s.signal_strength).sum())
    }

    fn calculate_collective_behavior(&self, cellular_states: &[CellularState], tissue_id: &str) -> Result<f64> {
        Ok(cellular_states.iter().map(|s| s.signaling_activity).sum())
    }

    fn calculate_tissue_voltage(&self, cellular_states: &[CellularState], tissue_id: &str) -> Result<f64> {
        if cellular_states.is_empty() {
            return Ok(0.0);
        }
        Ok(cellular_states.iter().map(|s| s.membrane_potential).sum::<f64>() / cellular_states.len() as f64)
    }

    fn calculate_nutrient_consumption(&self, cellular_states: &[CellularState]) -> Result<f64> {
        Ok(cellular_states.iter().map(|s| s.metabolic_state * 0.1).sum())
    }

    fn calculate_waste_production(&self, cellular_states: &[CellularState]) -> Result<f64> {
        Ok(cellular_states.iter().map(|s| s.metabolic_state * 0.05).sum())
    }
}

// State structures for each level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularState {
    pub circuit_id: String,
    pub protein_activities: HashMap<String, f64>,
    pub local_voltage: f64,
    pub atp_consumption: f64,
    pub flux_rates: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganelleState {
    pub organelle_id: String,
    pub organelle_voltage: f64,
    pub metabolic_flux: f64,
    pub atp_production: f64,
    pub calcium_dynamics: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularState {
    pub cell_id: String,
    pub membrane_potential: f64,
    pub atp_level: f64,
    pub signaling_activity: f64,
    pub metabolic_state: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueState {
    pub tissue_id: String,
    pub collective_voltage: f64,
    pub mechanical_response: f64,
    pub nutrient_consumption: f64,
    pub waste_production: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalState {
    pub molecular_states: Vec<MolecularState>,
    pub organelle_states: Vec<OrganelleState>,
    pub cellular_states: Vec<CellularState>,
    pub tissue_states: Vec<TissueState>,
    pub voltage_hierarchy: VoltageHierarchy,
    pub timestamp: f64,
}

// Default implementations
impl Default for VoltageHierarchy {
    fn default() -> Self {
        Self {
            cellular_voltage: -70.0, // Resting potential
            organelle_voltages: HashMap::from([
                ("mitochondrion".to_string(), -180.0),
                ("endoplasmic_reticulum".to_string(), -60.0),
            ]),
            compartment_voltages: HashMap::new(),
            molecular_states: HashMap::new(),
        }
    }
}

impl TemporalEvidence {
    pub fn new() -> Self {
        Self {
            millisecond_events: Vec::new(),
            second_events: Vec::new(),
            minute_events: Vec::new(),
            hour_events: Vec::new(),
            day_events: Vec::new(),
        }
    }

    pub fn add_event(&mut self, event: TemporalEvent) {
        match event.half_life {
            t if t < 1.0 => self.millisecond_events.push(event),
            t if t < 60.0 => self.second_events.push(event),
            t if t < 3600.0 => self.minute_events.push(event),
            t if t < 86400.0 => self.hour_events.push(event),
            _ => self.day_events.push(event),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_system_creation() {
        let system = HierarchicalSystem::new();
        assert_eq!(system.voltage_hierarchy.cellular_voltage, -70.0);
        assert_eq!(system.voltage_hierarchy.organelle_voltages.get("mitochondrion"), Some(&-180.0));
    }

    #[test]
    fn test_temporal_evidence() {
        let mut evidence = TemporalEvidence::new();
        let event = TemporalEvent {
            event_id: "test_event".to_string(),
            event_type: EventType::IonChannelGating,
            strength: 1.0,
            decay_rate: 0.1,
            timestamp: 0.0,
            half_life: 0.5, // 500ms
        };
        
        evidence.add_event(event);
        assert_eq!(evidence.millisecond_events.len(), 1);
    }
} 
//! # Biological Integration Module
//! 
//! Integration layer between quantum computation and biological processes

use crate::error::{NebuchadnezzarError, Result};
use crate::biological_quantum_computer::BiologicalQuantumState;
use crate::systems_biology::atp_kinetics::AtpPool;
use std::collections::HashMap;

/// Biological system integrator
#[derive(Debug)]
pub struct BiologicalIntegrator {
    metabolic_networks: Vec<MetabolicNetwork>,
    signaling_pathways: Vec<SignalingPathway>,
    gene_regulation: GeneRegulationNetwork,
    protein_interactions: ProteinInteractionNetwork,
}

impl BiologicalIntegrator {
    pub fn new() -> Self {
        Self {
            metabolic_networks: Vec::new(),
            signaling_pathways: Vec::new(),
            gene_regulation: GeneRegulationNetwork::new(),
            protein_interactions: ProteinInteractionNetwork::new(),
        }
    }

    pub fn integrate_quantum_state(&self, quantum_state: &BiologicalQuantumState) -> Result<BiologicalResponse> {
        let metabolic_response = self.process_metabolic_state(quantum_state)?;
        let signaling_response = self.process_signaling_state(quantum_state)?;
        let regulatory_response = self.process_regulatory_state(quantum_state)?;

        Ok(BiologicalResponse {
            metabolic_changes: metabolic_response,
            signaling_changes: signaling_response,
            regulatory_changes: regulatory_response,
            overall_fitness: self.calculate_fitness(&quantum_state)?,
        })
    }

    fn process_metabolic_state(&self, state: &BiologicalQuantumState) -> Result<MetabolicResponse> {
        let energy_charge = state.atp_coords.energy_charge;
        let flux_rates = self.calculate_metabolic_fluxes(state)?;
        
        Ok(MetabolicResponse {
            energy_charge,
            flux_rates,
            atp_production_rate: state.atp_coords.atp_concentration * 0.1,
            metabolic_efficiency: energy_charge * 0.8,
        })
    }

    fn process_signaling_state(&self, state: &BiologicalQuantumState) -> Result<SignalingResponse> {
        let signal_strength = state.oscillatory_coords.oscillations.iter()
            .map(|osc| osc.amplitude * osc.frequency)
            .sum::<f64>();

        Ok(SignalingResponse {
            signal_strength,
            pathway_activation: self.calculate_pathway_activation(state)?,
            response_time: 1.0 / signal_strength.max(0.1),
        })
    }

    fn process_regulatory_state(&self, state: &BiologicalQuantumState) -> Result<RegulatoryResponse> {
        let coherence = state.membrane_coords.quantum_states.iter()
            .map(|qs| qs.amplitude.norm_sqr())
            .sum::<f64>();

        Ok(RegulatoryResponse {
            gene_expression_changes: HashMap::new(),
            protein_modifications: HashMap::new(),
            regulatory_coherence: coherence,
        })
    }

    fn calculate_metabolic_fluxes(&self, _state: &BiologicalQuantumState) -> Result<HashMap<String, f64>> {
        // Simplified flux calculation
        let mut fluxes = HashMap::new();
        fluxes.insert("Glycolysis".to_string(), 1.0);
        fluxes.insert("TCA_Cycle".to_string(), 0.8);
        fluxes.insert("Electron_Transport".to_string(), 1.2);
        Ok(fluxes)
    }

    fn calculate_pathway_activation(&self, _state: &BiologicalQuantumState) -> Result<HashMap<String, f64>> {
        let mut activation = HashMap::new();
        activation.insert("MAPK".to_string(), 0.5);
        activation.insert("PI3K_AKT".to_string(), 0.7);
        activation.insert("p53".to_string(), 0.3);
        Ok(activation)
    }

    fn calculate_fitness(&self, state: &BiologicalQuantumState) -> Result<f64> {
        let energy_fitness = state.atp_coords.energy_charge;
        let oscillatory_fitness = state.oscillatory_coords.oscillations.iter()
            .map(|osc| (osc.amplitude * osc.frequency).min(1.0))
            .sum::<f64>() / state.oscillatory_coords.oscillations.len() as f64;
        let quantum_fitness = state.membrane_coords.quantum_states.iter()
            .map(|qs| qs.amplitude.norm_sqr())
            .sum::<f64>();

        Ok((energy_fitness + oscillatory_fitness + quantum_fitness) / 3.0)
    }
}

/// Metabolic network representation
#[derive(Debug, Clone)]
pub struct MetabolicNetwork {
    pub name: String,
    pub reactions: Vec<String>,
    pub metabolites: Vec<String>,
    pub flux_constraints: HashMap<String, (f64, f64)>,
}

/// Signaling pathway representation
#[derive(Debug, Clone)]
pub struct SignalingPathway {
    pub name: String,
    pub components: Vec<String>,
    pub interactions: Vec<(String, String, f64)>,
    pub activation_threshold: f64,
}

/// Gene regulation network
#[derive(Debug)]
pub struct GeneRegulationNetwork {
    genes: HashMap<String, GeneNode>,
    regulatory_interactions: Vec<RegulatoryInteraction>,
}

impl GeneRegulationNetwork {
    pub fn new() -> Self {
        Self {
            genes: HashMap::new(),
            regulatory_interactions: Vec::new(),
        }
    }

    pub fn add_gene(&mut self, name: String, node: GeneNode) {
        self.genes.insert(name, node);
    }

    pub fn add_regulation(&mut self, interaction: RegulatoryInteraction) {
        self.regulatory_interactions.push(interaction);
    }
}

/// Gene node in regulation network
#[derive(Debug, Clone)]
pub struct GeneNode {
    pub name: String,
    pub expression_level: f64,
    pub basal_expression: f64,
    pub regulation_strength: f64,
}

/// Regulatory interaction between genes
#[derive(Debug, Clone)]
pub struct RegulatoryInteraction {
    pub regulator: String,
    pub target: String,
    pub interaction_type: RegulationType,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub enum RegulationType {
    Activation,
    Repression,
    Dual,
}

/// Protein interaction network
#[derive(Debug)]
pub struct ProteinInteractionNetwork {
    proteins: HashMap<String, ProteinNode>,
    interactions: Vec<ProteinInteraction>,
}

impl ProteinInteractionNetwork {
    pub fn new() -> Self {
        Self {
            proteins: HashMap::new(),
            interactions: Vec::new(),
        }
    }

    pub fn add_protein(&mut self, name: String, node: ProteinNode) {
        self.proteins.insert(name, node);
    }

    pub fn add_interaction(&mut self, interaction: ProteinInteraction) {
        self.interactions.push(interaction);
    }
}

#[derive(Debug, Clone)]
pub struct ProteinNode {
    pub name: String,
    pub concentration: f64,
    pub activity: f64,
    pub modifications: Vec<ProteinModification>,
}

#[derive(Debug, Clone)]
pub struct ProteinInteraction {
    pub protein_a: String,
    pub protein_b: String,
    pub interaction_type: InteractionType,
    pub binding_affinity: f64,
}

#[derive(Debug, Clone)]
pub enum InteractionType {
    Binding,
    Phosphorylation,
    Ubiquitination,
    Methylation,
    Acetylation,
}

#[derive(Debug, Clone)]
pub struct ProteinModification {
    pub modification_type: InteractionType,
    pub position: usize,
    pub level: f64,
}

/// Biological response to quantum state
#[derive(Debug)]
pub struct BiologicalResponse {
    pub metabolic_changes: MetabolicResponse,
    pub signaling_changes: SignalingResponse,
    pub regulatory_changes: RegulatoryResponse,
    pub overall_fitness: f64,
}

#[derive(Debug)]
pub struct MetabolicResponse {
    pub energy_charge: f64,
    pub flux_rates: HashMap<String, f64>,
    pub atp_production_rate: f64,
    pub metabolic_efficiency: f64,
}

#[derive(Debug)]
pub struct SignalingResponse {
    pub signal_strength: f64,
    pub pathway_activation: HashMap<String, f64>,
    pub response_time: f64,
}

#[derive(Debug)]
pub struct RegulatoryResponse {
    pub gene_expression_changes: HashMap<String, f64>,
    pub protein_modifications: HashMap<String, f64>,
    pub regulatory_coherence: f64,
}

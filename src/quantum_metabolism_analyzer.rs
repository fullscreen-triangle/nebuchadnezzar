//! # Quantum Metabolism Analyzer
//! 
//! Analyzes metabolic pathways using quantum computational principles

use crate::error::{NebuchadnezzarError, Result};
use crate::biological_quantum_computer::BiologicalQuantumState;
use crate::systems_biology::atp_kinetics::AtpPool;
use std::collections::HashMap;

/// Quantum metabolism analyzer
#[derive(Debug)]
pub struct QuantumMetabolismAnalyzer {
    pathways: HashMap<String, MetabolicPathway>,
    quantum_effects: QuantumEffectsModel,
    analysis_parameters: AnalysisParameters,
}

impl QuantumMetabolismAnalyzer {
    pub fn new() -> Self {
        Self {
            pathways: HashMap::new(),
            quantum_effects: QuantumEffectsModel::new(),
            analysis_parameters: AnalysisParameters::default(),
        }
    }

    pub fn add_pathway(&mut self, name: String, pathway: MetabolicPathway) {
        self.pathways.insert(name, pathway);
    }

    pub fn analyze_quantum_metabolism(
        &self,
        quantum_state: &BiologicalQuantumState,
        atp_pool: &AtpPool,
    ) -> Result<MetabolismAnalysisResult> {
        let mut pathway_analyses = HashMap::new();

        for (name, pathway) in &self.pathways {
            let analysis = self.analyze_pathway_quantum_effects(pathway, quantum_state, atp_pool)?;
            pathway_analyses.insert(name.clone(), analysis);
        }

        let overall_efficiency = self.calculate_overall_efficiency(&pathway_analyses);
        let quantum_advantage = self.calculate_quantum_advantage(&pathway_analyses);
        let metabolic_coherence = self.calculate_metabolic_coherence(quantum_state);

        Ok(MetabolismAnalysisResult {
            pathway_analyses,
            overall_efficiency,
            quantum_advantage,
            metabolic_coherence,
            atp_utilization_efficiency: self.calculate_atp_efficiency(atp_pool),
        })
    }

    fn analyze_pathway_quantum_effects(
        &self,
        pathway: &MetabolicPathway,
        quantum_state: &BiologicalQuantumState,
        atp_pool: &AtpPool,
    ) -> Result<PathwayAnalysis> {
        let mut reaction_analyses = Vec::new();

        for reaction in &pathway.reactions {
            let quantum_tunneling_rate = self.calculate_tunneling_rate(reaction, quantum_state)?;
            let coherence_enhancement = self.calculate_coherence_enhancement(reaction, quantum_state)?;
            let atp_coupling_efficiency = self.calculate_atp_coupling_efficiency(reaction, atp_pool)?;

            reaction_analyses.push(ReactionAnalysis {
                reaction_name: reaction.name.clone(),
                quantum_tunneling_rate,
                coherence_enhancement,
                atp_coupling_efficiency,
                classical_rate: reaction.classical_rate,
                quantum_enhanced_rate: reaction.classical_rate * (1.0 + quantum_tunneling_rate + coherence_enhancement),
            });
        }

        let pathway_flux = self.calculate_pathway_flux(&reaction_analyses);
        let bottleneck_reactions = self.identify_bottlenecks(&reaction_analyses);

        Ok(PathwayAnalysis {
            pathway_name: pathway.name.clone(),
            reaction_analyses,
            pathway_flux,
            bottleneck_reactions,
            quantum_enhancement_factor: self.calculate_enhancement_factor(&reaction_analyses),
        })
    }

    fn calculate_tunneling_rate(
        &self,
        reaction: &MetabolicReaction,
        quantum_state: &BiologicalQuantumState,
    ) -> Result<f64> {
        let barrier_height = reaction.activation_energy;
        let temperature = 310.0; // Body temperature in K
        let kb = 1.38e-23; // Boltzmann constant

        // Quantum tunneling probability
        let tunneling_prob = (-2.0 * (2.0 * 9.109e-31 * barrier_height * 1.602e-19).sqrt() * 
                              reaction.tunnel_distance / 1.055e-34).exp();

        // Enhancement from quantum coherence
        let coherence_factor = quantum_state.membrane_coords.quantum_states.iter()
            .map(|qs| qs.amplitude.norm_sqr())
            .sum::<f64>();

        Ok(tunneling_prob * coherence_factor * self.quantum_effects.tunneling_enhancement)
    }

    fn calculate_coherence_enhancement(
        &self,
        _reaction: &MetabolicReaction,
        quantum_state: &BiologicalQuantumState,
    ) -> Result<f64> {
        let oscillatory_coherence = quantum_state.oscillatory_coords.oscillations.iter()
            .map(|osc| (osc.phase.cos() + 1.0) / 2.0)
            .sum::<f64>() / quantum_state.oscillatory_coords.oscillations.len() as f64;

        let membrane_coherence = quantum_state.membrane_coords.quantum_states.iter()
            .map(|qs| qs.amplitude.norm_sqr())
            .sum::<f64>();

        Ok((oscillatory_coherence + membrane_coherence) / 2.0 * self.quantum_effects.coherence_enhancement)
    }

    fn calculate_atp_coupling_efficiency(
        &self,
        reaction: &MetabolicReaction,
        atp_pool: &AtpPool,
    ) -> Result<f64> {
        if reaction.atp_requirement == 0.0 {
            return Ok(1.0);
        }

        let energy_charge = atp_pool.energy_charge();
        let atp_availability = atp_pool.atp_concentration / (atp_pool.atp_concentration + atp_pool.adp_concentration);

        Ok(energy_charge * atp_availability * (1.0 - reaction.atp_requirement.abs() / 10.0).max(0.1))
    }

    fn calculate_pathway_flux(&self, reactions: &[ReactionAnalysis]) -> f64 {
        // Flux is limited by the slowest reaction
        reactions.iter()
            .map(|r| r.quantum_enhanced_rate)
            .fold(f64::INFINITY, f64::min)
    }

    fn identify_bottlenecks(&self, reactions: &[ReactionAnalysis]) -> Vec<String> {
        let min_rate = reactions.iter()
            .map(|r| r.quantum_enhanced_rate)
            .fold(f64::INFINITY, f64::min);

        reactions.iter()
            .filter(|r| r.quantum_enhanced_rate <= min_rate * 1.1)
            .map(|r| r.reaction_name.clone())
            .collect()
    }

    fn calculate_enhancement_factor(&self, reactions: &[ReactionAnalysis]) -> f64 {
        let classical_flux = reactions.iter()
            .map(|r| r.classical_rate)
            .fold(f64::INFINITY, f64::min);

        let quantum_flux = reactions.iter()
            .map(|r| r.quantum_enhanced_rate)
            .fold(f64::INFINITY, f64::min);

        if classical_flux > 0.0 {
            quantum_flux / classical_flux
        } else {
            1.0
        }
    }

    fn calculate_overall_efficiency(&self, analyses: &HashMap<String, PathwayAnalysis>) -> f64 {
        if analyses.is_empty() {
            return 0.0;
        }

        analyses.values()
            .map(|analysis| analysis.quantum_enhancement_factor)
            .sum::<f64>() / analyses.len() as f64
    }

    fn calculate_quantum_advantage(&self, analyses: &HashMap<String, PathwayAnalysis>) -> f64 {
        let total_enhancement: f64 = analyses.values()
            .map(|analysis| analysis.quantum_enhancement_factor - 1.0)
            .sum();

        total_enhancement / analyses.len() as f64
    }

    fn calculate_metabolic_coherence(&self, quantum_state: &BiologicalQuantumState) -> f64 {
        let oscillatory_coherence = if quantum_state.oscillatory_coords.oscillations.is_empty() {
            0.0
        } else {
            quantum_state.oscillatory_coords.oscillations.iter()
                .map(|osc| (osc.phase.cos().abs() + osc.phase.sin().abs()) / 2.0)
                .sum::<f64>() / quantum_state.oscillatory_coords.oscillations.len() as f64
        };

        let membrane_coherence = quantum_state.membrane_coords.quantum_states.iter()
            .map(|qs| qs.amplitude.norm_sqr())
            .sum::<f64>();

        (oscillatory_coherence + membrane_coherence) / 2.0
    }

    fn calculate_atp_efficiency(&self, atp_pool: &AtpPool) -> f64 {
        let energy_charge = atp_pool.energy_charge();
        let utilization_ratio = atp_pool.atp_concentration / 
                               (atp_pool.atp_concentration + atp_pool.adp_concentration + atp_pool.pi_concentration);
        
        energy_charge * utilization_ratio
    }
}

/// Metabolic pathway representation
#[derive(Debug, Clone)]
pub struct MetabolicPathway {
    pub name: String,
    pub reactions: Vec<MetabolicReaction>,
    pub regulation: PathwayRegulation,
}

/// Individual metabolic reaction
#[derive(Debug, Clone)]
pub struct MetabolicReaction {
    pub name: String,
    pub classical_rate: f64,
    pub activation_energy: f64, // in eV
    pub tunnel_distance: f64,   // in meters
    pub atp_requirement: f64,
    pub substrates: Vec<String>,
    pub products: Vec<String>,
}

/// Pathway regulation mechanisms
#[derive(Debug, Clone)]
pub struct PathwayRegulation {
    pub allosteric_regulators: Vec<AllostericRegulator>,
    pub feedback_loops: Vec<FeedbackLoop>,
    pub transcriptional_control: Vec<TranscriptionalControl>,
}

#[derive(Debug, Clone)]
pub struct AllostericRegulator {
    pub molecule: String,
    pub target_enzyme: String,
    pub effect_type: RegulationEffect,
    pub binding_affinity: f64,
}

#[derive(Debug, Clone)]
pub struct FeedbackLoop {
    pub product: String,
    pub target_enzyme: String,
    pub loop_type: FeedbackType,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub struct TranscriptionalControl {
    pub transcription_factor: String,
    pub target_gene: String,
    pub effect_type: RegulationEffect,
}

#[derive(Debug, Clone)]
pub enum RegulationEffect {
    Activation,
    Inhibition,
    Competitive,
    NonCompetitive,
}

#[derive(Debug, Clone)]
pub enum FeedbackType {
    Negative,
    Positive,
    Cooperative,
}

/// Quantum effects model
#[derive(Debug)]
pub struct QuantumEffectsModel {
    pub tunneling_enhancement: f64,
    pub coherence_enhancement: f64,
    pub environmental_decoherence: f64,
    pub temperature_dependence: f64,
}

impl QuantumEffectsModel {
    pub fn new() -> Self {
        Self {
            tunneling_enhancement: 1.5,
            coherence_enhancement: 1.2,
            environmental_decoherence: 0.1,
            temperature_dependence: 0.05,
        }
    }
}

/// Analysis parameters
#[derive(Debug, Clone)]
pub struct AnalysisParameters {
    pub temperature: f64,
    pub ph: f64,
    pub ionic_strength: f64,
    pub membrane_potential: f64,
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
            temperature: 310.0, // Body temperature
            ph: 7.4,           // Physiological pH
            ionic_strength: 0.15, // Physiological ionic strength
            membrane_potential: -70e-3, // Resting membrane potential in V
        }
    }
}

/// Result of metabolism analysis
#[derive(Debug)]
pub struct MetabolismAnalysisResult {
    pub pathway_analyses: HashMap<String, PathwayAnalysis>,
    pub overall_efficiency: f64,
    pub quantum_advantage: f64,
    pub metabolic_coherence: f64,
    pub atp_utilization_efficiency: f64,
}

/// Analysis of a single pathway
#[derive(Debug)]
pub struct PathwayAnalysis {
    pub pathway_name: String,
    pub reaction_analyses: Vec<ReactionAnalysis>,
    pub pathway_flux: f64,
    pub bottleneck_reactions: Vec<String>,
    pub quantum_enhancement_factor: f64,
}

/// Analysis of a single reaction
#[derive(Debug)]
pub struct ReactionAnalysis {
    pub reaction_name: String,
    pub quantum_tunneling_rate: f64,
    pub coherence_enhancement: f64,
    pub atp_coupling_efficiency: f64,
    pub classical_rate: f64,
    pub quantum_enhanced_rate: f64,
} 
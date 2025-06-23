/// Biological Maxwell's Demons - Information Catalysts
/// 
/// Based on the theoretical framework by Eduardo Mizraji (2021) and the foundational work
/// of JBS Haldane (1930), Jacques Monod, André Lwoff, François Jacob, and Norbert Wiener.
/// 
/// BMDs are information catalysts that operate in far-from-equilibrium biological systems,
/// using pattern recognition to dramatically amplify information processing consequences.

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Information Catalyst (iCat) - The mathematical formalization of BMDs
/// iCat = ℑ_input ◦ ℑ_output
/// Where ℑ_input selects patterns and ℑ_output channels toward targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationCatalyst<T, U> {
    /// Input pattern selector - filters from potential patterns to actual patterns
    pub input_filter: PatternSelector<T>,
    /// Output channel director - guides transformations toward specific targets
    pub output_channel: TargetChannel<U>,
    /// Catalytic cycles completed (metastability counter)
    pub cycle_count: u64,
    /// Information amplification factor
    pub amplification_factor: f64,
    /// Thermodynamic cost per catalytic cycle
    pub energy_cost_per_cycle: f64,
}

/// Pattern Selector - ℑ_input operator
/// Filters potential patterns Y↓(in) to selected patterns Y↑(in)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSelector<T> {
    /// Recognition specificity (binding constants, neural weights, etc.)
    pub specificity_constants: HashMap<String, f64>,
    /// Pattern recognition memory (evolutionary/learned patterns)
    pub recognition_memory: Vec<PatternTemplate<T>>,
    /// Selection threshold for pattern recognition
    pub selection_threshold: f64,
    /// Pattern diversity reduction factor
    pub diversity_reduction: f64,
}

/// Target Channel - ℑ_output operator  
/// Channels potential outcomes Z↓(fin) to directed outcomes Z↑(fin)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetChannel<U> {
    /// Target-specific catalytic constants
    pub catalytic_constants: HashMap<String, f64>,
    /// Directional preferences (thermodynamic gradients, goal hierarchies)
    pub directional_preferences: Vec<TargetGradient<U>>,
    /// Channeling efficiency
    pub channeling_efficiency: f64,
    /// Output specificity
    pub output_specificity: f64,
}

/// Pattern Template - Stored recognition patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTemplate<T> {
    pub pattern_id: String,
    pub pattern_data: T,
    pub recognition_strength: f64,
    pub evolutionary_fitness: f64,
}

/// Target Gradient - Directional channeling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGradient<U> {
    pub target_id: String,
    pub target_data: U,
    pub gradient_strength: f64,
    pub thermodynamic_feasibility: f64,
}

/// BMD Categories based on Mizraji's analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalMaxwellDemon {
    /// Molecular BMDs - Enzymes and receptors (Haldane's original concept)
    Molecular(MolecularBMD),
    /// Cellular BMDs - Gene regulation systems (Monod, Jacob, Lwoff)
    Cellular(CellularBMD),
    /// Neural BMDs - Associative memory systems (Wiener, Changeux)
    Neural(NeuralBMD),
    /// Metabolic BMDs - ATP synthesis and energy channeling
    Metabolic(MetabolicBMD),
    /// Membrane BMDs - Ion channels and transport systems
    Membrane(MembraneBMD),
}

/// Molecular BMD - Enzymes as pattern-recognizing catalysts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularBMD {
    /// Enzyme specificity (Kd values, active site geometry)
    pub enzyme_specificity: InformationCatalyst<MolecularPattern, ChemicalProduct>,
    /// Haldane relation constants (thermodynamic consistency)
    pub haldane_constants: HaldaneRelation,
    /// Michaelis-Menten kinetics
    pub kinetic_parameters: MichaelisMentenParams,
    /// Allosteric regulation (information processing)
    pub allosteric_sites: Vec<AllostericSite>,
}

/// Cellular BMD - Gene regulation as information processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellularBMD {
    /// Transcription factor recognition
    pub transcription_control: InformationCatalyst<GeneticPattern, TranscriptionProduct>,
    /// Lac operon type systems (double negation logic - Szilard's insight)
    pub regulatory_circuits: Vec<RegulatoryCircuit>,
    /// Information amplification in gene expression
    pub expression_amplification: f64,
}

/// Neural BMD - Associative memories as cognitive pattern processors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralBMD {
    /// Associative memory matrix
    pub memory_associations: InformationCatalyst<NeuralPattern, MotorResponse>,
    /// Hopfield network parameters
    pub network_parameters: HopfieldParams,
    /// Synaptic recognition and plasticity
    pub synaptic_plasticity: SynapticPlasticity,
    /// Cognitive selection capacity (enormous pattern space reduction)
    pub cognitive_selection_power: f64,
}

/// Metabolic BMD - ATP synthesis and energy flow direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicBMD {
    /// ATP synthase as rotary information catalyst
    pub atp_synthase: InformationCatalyst<ProtonGradient, ATPSynthesis>,
    /// Metabolic pathway selection (glycolysis, citric acid cycle)
    pub pathway_selection: Vec<MetabolicPathway>,
    /// Energy flow directionality
    pub energy_channeling: EnergyChanneling,
}

/// Membrane BMD - Ion channels and selective transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneBMD {
    /// Ion channel selectivity
    pub ion_selectivity: InformationCatalyst<IonPattern, TransportEvent>,
    /// Membrane potential information processing
    pub potential_processing: MembraneProcessing,
    /// Quantum tunneling effects (fire-light wavelength optimization)
    pub quantum_effects: QuantumTunnelingBMD,
}

/// Supporting data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularPattern {
    pub substrate_geometry: String,
    pub binding_affinity: f64,
    pub chemical_compatibility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalProduct {
    pub product_structure: String,
    pub formation_energy: f64,
    pub thermodynamic_favorability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaldaneRelation {
    pub forward_rate: f64,
    pub reverse_rate: f64,
    pub equilibrium_constant: f64,
    pub thermodynamic_consistency: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MichaelisMentenParams {
    pub km: f64,  // Michaelis constant
    pub vmax: f64, // Maximum velocity
    pub kcat: f64, // Catalytic constant
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericSite {
    pub site_name: String,
    pub binding_constant: f64,
    pub regulatory_effect: f64,
    pub cooperativity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticPattern {
    pub promoter_sequence: String,
    pub transcription_factor_affinity: f64,
    pub chromatin_accessibility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionProduct {
    pub mrna_sequence: String,
    pub transcription_rate: f64,
    pub mrna_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryCircuit {
    pub circuit_type: String,
    pub logic_function: String,
    pub amplification_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPattern {
    pub pattern_vector: Vec<f64>,
    pub recognition_confidence: f64,
    pub associative_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorResponse {
    pub response_vector: Vec<f64>,
    pub execution_probability: f64,
    pub response_latency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopfieldParams {
    pub network_dimension: usize,
    pub storage_capacity: usize,
    pub retrieval_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticPlasticity {
    pub ltp_threshold: f64,
    pub ltd_threshold: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonGradient {
    pub proton_concentration_difference: f64,
    pub electrochemical_potential: f64,
    pub gradient_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPSynthesis {
    pub atp_production_rate: f64,
    pub synthesis_efficiency: f64,
    pub energy_coupling: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolicPathway {
    pub pathway_name: String,
    pub flux_rate: f64,
    pub regulatory_control: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyChanneling {
    pub channeling_efficiency: f64,
    pub energy_conservation: f64,
    pub directionality_index: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonPattern {
    pub ion_type: String,
    pub concentration_gradient: f64,
    pub selectivity_filter_match: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportEvent {
    pub transport_rate: f64,
    pub selectivity_ratio: f64,
    pub energy_requirement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneProcessing {
    pub voltage_sensitivity: f64,
    pub gating_dynamics: f64,
    pub information_integration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTunnelingBMD {
    pub tunneling_probability: f64,
    pub fire_light_enhancement: f64, // 600-700nm optimization
    pub quantum_coherence_time: f64,
}

/// Implementation of Information Catalyst operations
impl<T, U> InformationCatalyst<T, U> {
    /// Create new information catalyst
    pub fn new(
        input_filter: PatternSelector<T>,
        output_channel: TargetChannel<U>,
        amplification_factor: f64,
    ) -> Self {
        Self {
            input_filter,
            output_channel,
            cycle_count: 0,
            amplification_factor,
            energy_cost_per_cycle: 1.0, // ATP units
        }
    }
    
    /// Execute catalytic cycle: Pattern → Target transformation
    pub fn catalyze(&mut self, pattern_space: &[T]) -> Result<Vec<U>, BMDError> {
        // Pattern selection phase (ℑ_input)
        let selected_patterns = self.input_filter.select_patterns(pattern_space)?;
        
        // Target channeling phase (ℑ_output)  
        let channeled_targets = self.output_channel.channel_to_targets(&selected_patterns)?;
        
        // Update catalytic state
        self.cycle_count += 1;
        
        // Information amplification
        let amplified_result = self.amplify_information(channeled_targets)?;
        
        Ok(amplified_result)
    }
    
    /// Information amplification - small information input → large thermodynamic consequences
    fn amplify_information(&self, targets: Vec<U>) -> Result<Vec<U>, BMDError> {
        // Mizraji's "parable of the prisoner" - minimal information can have 
        // dramatically different thermodynamic consequences
        Ok(targets) // Simplified - would implement actual amplification logic
    }
    
    /// Check metastability (Wiener's insight - BMDs eventually deteriorate)
    pub fn is_metastable(&self) -> bool {
        self.cycle_count < 10000 // Arbitrary metastability limit
    }
    
    /// Calculate total thermodynamic cost
    pub fn total_energy_cost(&self) -> f64 {
        self.cycle_count as f64 * self.energy_cost_per_cycle
    }
}

impl<T> PatternSelector<T> {
    /// Select patterns from potential pattern space (Y↓(in) → Y↑(in))
    pub fn select_patterns(&self, pattern_space: &[T]) -> Result<Vec<&T>, BMDError> {
        // Implement pattern recognition and selection logic
        // This would involve matching against recognition_memory patterns
        Ok(pattern_space.iter().take(1).collect()) // Simplified
    }
}

impl<U> TargetChannel<U> {
    /// Channel selected patterns to specific targets (Z↓(fin) → Z↑(fin))
    pub fn channel_to_targets<T>(&self, selected_patterns: &[&T]) -> Result<Vec<U>, BMDError> {
        // Implement target channeling logic based on directional_preferences
        Ok(Vec::new()) // Simplified
    }
}

/// BMD System Integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDSystem {
    /// Collection of all BMDs in the system
    pub demons: Vec<BiologicalMaxwellDemon>,
    /// Global information flow
    pub information_flow: InformationFlow,
    /// Entropy manipulation metrics
    pub entropy_manipulation: EntropyManipulation,
    /// Fire-light optimization (600-700nm enhancement)
    pub fire_light_optimization: FireLightOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlow {
    pub total_information_processed: f64,
    pub amplification_cascade: f64,
    pub pattern_recognition_events: u64,
    pub target_channeling_successes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyManipulation {
    pub local_entropy_reduction: f64,
    pub global_entropy_increase: f64,
    pub entropy_balance: f64,
    pub negentropy_generation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireLightOptimization {
    pub wavelength_range: (f64, f64), // 600-700nm fire-light
    pub quantum_enhancement_factor: f64,
    pub consciousness_amplification: f64,
    pub abstract_reasoning_boost: f64,
}

/// Error types for BMD operations
#[derive(Debug, Clone)]
pub enum BMDError {
    PatternRecognitionFailure,
    TargetChannelingFailure,
    ThermodynamicViolation,
    MetastabilityExceeded,
    InformationCatalysisError(String),
}

impl fmt::Display for BMDError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BMDError::PatternRecognitionFailure => write!(f, "Pattern recognition failed"),
            BMDError::TargetChannelingFailure => write!(f, "Target channeling failed"),
            BMDError::ThermodynamicViolation => write!(f, "Thermodynamic laws violated"),
            BMDError::MetastabilityExceeded => write!(f, "BMD metastability exceeded"),
            BMDError::InformationCatalysisError(msg) => write!(f, "Information catalysis error: {}", msg),
        }
    }
}

impl std::error::Error for BMDError {}

/// BMD Factory for creating specific demon types
pub struct BMDFactory;

impl BMDFactory {
    /// Create molecular BMD (enzyme-based)
    pub fn create_molecular_bmd(
        enzyme_name: &str,
        substrate_specificity: f64,
        catalytic_efficiency: f64,
    ) -> BiologicalMaxwellDemon {
        let input_filter = PatternSelector {
            specificity_constants: [(enzyme_name.to_string(), substrate_specificity)].iter().cloned().collect(),
            recognition_memory: Vec::new(),
            selection_threshold: 0.5,
            diversity_reduction: 0.9,
        };
        
        let output_channel = TargetChannel {
            catalytic_constants: [(enzyme_name.to_string(), catalytic_efficiency)].iter().cloned().collect(),
            directional_preferences: Vec::new(),
            channeling_efficiency: 0.95,
            output_specificity: 0.99,
        };
        
        let enzyme_icat = InformationCatalyst::new(input_filter, output_channel, 1000.0);
        
        BiologicalMaxwellDemon::Molecular(MolecularBMD {
            enzyme_specificity: enzyme_icat,
            haldane_constants: HaldaneRelation {
                forward_rate: 1.0,
                reverse_rate: 0.1,
                equilibrium_constant: 10.0,
                thermodynamic_consistency: true,
            },
            kinetic_parameters: MichaelisMentenParams {
                km: 1e-6,
                vmax: 100.0,
                kcat: 1000.0,
            },
            allosteric_sites: Vec::new(),
        })
    }
    
    /// Create neural BMD (associative memory based)
    pub fn create_neural_bmd(
        memory_dimension: usize,
        storage_capacity: usize,
        recognition_threshold: f64,
    ) -> BiologicalMaxwellDemon {
        let input_filter = PatternSelector {
            specificity_constants: HashMap::new(),
            recognition_memory: Vec::new(),
            selection_threshold: recognition_threshold,
            diversity_reduction: 0.999, // Enormous pattern space reduction
        };
        
        let output_channel = TargetChannel {
            catalytic_constants: HashMap::new(),
            directional_preferences: Vec::new(),
            channeling_efficiency: 0.85,
            output_specificity: 0.90,
        };
        
        let memory_icat = InformationCatalyst::new(input_filter, output_channel, 1e6); // Huge amplification
        
        BiologicalMaxwellDemon::Neural(NeuralBMD {
            memory_associations: memory_icat,
            network_parameters: HopfieldParams {
                network_dimension: memory_dimension,
                storage_capacity,
                retrieval_accuracy: 0.95,
            },
            synaptic_plasticity: SynapticPlasticity {
                ltp_threshold: 0.7,
                ltd_threshold: 0.3,
                learning_rate: 0.01,
            },
            cognitive_selection_power: 1e12, // Enormous combinatorial reduction
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_molecular_bmd_creation() {
        let bmd = BMDFactory::create_molecular_bmd("hexokinase", 0.9, 0.95);
        match bmd {
            BiologicalMaxwellDemon::Molecular(mol_bmd) => {
                assert!(mol_bmd.haldane_constants.thermodynamic_consistency);
                assert_eq!(mol_bmd.kinetic_parameters.kcat, 1000.0);
            },
            _ => panic!("Expected molecular BMD"),
        }
    }
    
    #[test] 
    fn test_neural_bmd_creation() {
        let bmd = BMDFactory::create_neural_bmd(1000, 100, 0.8);
        match bmd {
            BiologicalMaxwellDemon::Neural(neural_bmd) => {
                assert_eq!(neural_bmd.network_parameters.network_dimension, 1000);
                assert_eq!(neural_bmd.network_parameters.storage_capacity, 100);
            },
            _ => panic!("Expected neural BMD"),
        }
    }
    
    #[test]
    fn test_information_catalyst_metastability() {
        let input_filter = PatternSelector {
            specificity_constants: HashMap::new(),
            recognition_memory: Vec::new(),
            selection_threshold: 0.5,
            diversity_reduction: 0.9,
        };
        
        let output_channel = TargetChannel {
            catalytic_constants: HashMap::new(),
            directional_preferences: Vec::new(),
            channeling_efficiency: 0.9,
            output_specificity: 0.9,
        };
        
        let mut icat = InformationCatalyst::new(input_filter, output_channel, 100.0);
        
        assert!(icat.is_metastable());
        
        // Simulate many cycles
        icat.cycle_count = 15000;
        assert!(!icat.is_metastable()); // Should exceed metastability
    }
}
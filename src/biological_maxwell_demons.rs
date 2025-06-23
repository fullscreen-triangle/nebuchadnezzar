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
use crate::systems_biology::atp_kinetics::{AtpPool, AtpKinetics, AtpRateConstant};
use crate::circuits::{Circuit, ProbabilisticNode};
use crate::entropy_manipulation::{EntropyPoint, Resolution};
use crate::oscillatory_dynamics::OscillationState;

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

    /// Implement the "Prisoner Parable" as a BMD system test
    /// Tests information amplification through pattern recognition
    pub fn simulate_prisoner_parable(&mut self, signals: &[LightSignal]) -> ThermodynamicOutcome {
        // Check if the BMD has pattern recognition capability (Morse code knowledge)
        let has_pattern_recognition = self.input_filter.recognition_memory.len() > 0;
        
        if has_pattern_recognition {
            // Try to decode signals using pattern recognition
            match self.decode_morse_signals(signals) {
                Ok(decoded_combination) => {
                    // Success: Pattern recognition leads to survival
                    let life_energy = 90.0 * 86400.0; // 90 days of life energy (in seconds)
                    ThermodynamicOutcome::Survival(life_energy)
                }
                Err(_) => {
                    // Pattern recognition failed
                    ThermodynamicOutcome::Death(self.calculate_entropy_increase())
                }
            }
        } else {
            // No pattern recognition capability - cannot decode signals
            ThermodynamicOutcome::Death(self.calculate_entropy_increase())
        }
    }

    /// Decode Morse code signals (simplified implementation)
    fn decode_morse_signals(&self, signals: &[LightSignal]) -> Result<String, BMDError> {
        // Check for Morse code pattern template
        let morse_template = self.input_filter.recognition_memory
            .iter()
            .find(|template| template.pattern_id == "morse_code");
            
        match morse_template {
            Some(_) => {
                // Simplified Morse decoding - in reality this would be much more complex
                let decoded = signals.iter()
                    .map(|signal| if signal.duration > 0.5 { "dash" } else { "dot" })
                    .collect::<Vec<_>>()
                    .join("");
                Ok(decoded)
            }
            None => Err(BMDError::PatternRecognitionFailure)
        }
    }

    /// Calculate entropy increase when pattern recognition fails
    fn calculate_entropy_increase(&self) -> f64 {
        // Simplified entropy calculation - loss of organization leads to entropy increase
        let information_content = self.amplification_factor * self.input_filter.recognition_memory.len() as f64;
        information_content * 1.38e-23 * 310.0 // kT at body temperature
    }

    /// Check thermodynamic consistency using Haldane relations
    pub fn verify_thermodynamic_consistency(&self) -> bool {
        // Ensure that the BMD doesn't violate thermodynamic laws
        // This is a simplified check - real implementation would be more complex
        let energy_input = self.energy_cost_per_cycle * self.cycle_count as f64;
        let information_output = self.amplification_factor;
        
        // Information processing must have energetic cost
        energy_input > 0.0 && information_output / energy_input < 1e6 // Reasonable amplification limit
    }

    /// Calculate current metastability based on cycle count and energy consumption
    pub fn calculate_metastability_status(&self) -> MetastabilityStatus {
        let degradation_rate = 0.001; // Simplified degradation per cycle
        let degradation_factor = 1.0 - (degradation_rate * self.cycle_count as f64);
        
        if degradation_factor > 0.8 {
            MetastabilityStatus::Stable
        } else if degradation_factor > 0.3 {
            MetastabilityStatus::Degrading(degradation_factor)
        } else {
            MetastabilityStatus::RequiresReplacement
        }
    }

    /// Integrate with ATP-based rate system
    pub fn calculate_atp_consumption_rate(&self, atp_pool: &AtpPool) -> f64 {
        // BMD ATP consumption depends on current ATP availability and amplification factor
        let base_consumption = self.energy_cost_per_cycle;
        let atp_factor = atp_pool.atp_concentration / (2.0 + atp_pool.atp_concentration); // Michaelis-Menten like
        base_consumption * atp_factor * self.amplification_factor
    }

    /// Enhance performance with fire-light optimization (600-700nm)
    pub fn apply_fire_light_enhancement(&mut self, wavelength: f64) -> Result<(), BMDError> {
        if wavelength >= 600.0 && wavelength <= 700.0 {
            // Fire-light wavelength enhances BMD performance
            let enhancement_factor = 1.0 + 0.5 * ((wavelength - 650.0) / 50.0).cos().abs();
            self.amplification_factor *= enhancement_factor;
            
            // Improve pattern recognition sensitivity
            for template in &mut self.input_filter.recognition_memory {
                template.recognition_strength *= enhancement_factor;
            }
            
            Ok(())
        } else {
            Err(BMDError::InformationCatalysisError(
                "Wavelength outside fire-light optimization range (600-700nm)".to_string()
            ))
        }
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

/// Light signal for prisoner parable simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightSignal {
    pub intensity: f64,
    pub duration: f64,
    pub wavelength: f64,
    pub timestamp: f64,
}

/// Thermodynamic outcome of information processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermodynamicOutcome {
    Survival(f64), // Life energy gained
    Death(f64),    // Entropy increase
}

/// Metastability status of BMD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetastabilityStatus {
    Stable,
    Degrading(f64), // Degradation factor (0-1)
    RequiresReplacement,
}

/// Enhanced BMD system with integration capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedBMDSystem {
    /// Core BMD system
    pub base_system: BMDSystem,
    /// ATP integration
    pub atp_kinetics: AtpKinetics,
    /// Circuit integration
    pub circuit_nodes: Vec<BMDCircuitNode>,
    /// Metastability tracker
    pub metastability_tracker: MetastabilityTracker,
    /// Thermodynamic consistency checker
    pub thermodynamic_checker: ThermodynamicChecker,
}

/// BMD-enhanced circuit node for probabilistic circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDCircuitNode {
    pub node_id: String,
    pub information_catalyst: InformationCatalyst<BiologicalPattern, BiologicalTarget>,
    pub probability_weights: Vec<f64>,
    pub atp_cost: f64,
    pub amplification_factor: f64,
    pub fire_light_enhancement: f64,
}

/// Metastability tracking for BMDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetastabilityTracker {
    pub cycle_count: u64,
    pub energy_consumed: f64,
    pub degradation_rate: f64,
    pub replacement_threshold: u64,
    pub repair_mechanisms: Vec<RepairMechanism>,
}

/// Thermodynamic consistency checker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicChecker {
    pub haldane_relations: Vec<HaldaneRelation>,
    pub entropy_balance: f64,
    pub energy_conservation_violations: u32,
    pub consistency_threshold: f64,
}

/// Repair mechanism for BMD maintenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairMechanism {
    pub mechanism_name: String,
    pub repair_efficiency: f64,
    pub atp_cost_per_repair: f64,
    pub activation_threshold: f64,
}

/// Generic biological pattern for BMD processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalPattern {
    pub pattern_type: String,
    pub complexity: f64,
    pub recognition_features: Vec<f64>,
    pub evolutionary_age: f64,
}

/// Generic biological target for BMD output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalTarget {
    pub target_type: String,
    pub specificity: f64,
    pub biological_impact: f64,
    pub energy_requirement: f64,
}

impl MetastabilityTracker {
    pub fn new() -> Self {
        Self {
            cycle_count: 0,
            energy_consumed: 0.0,
            degradation_rate: 0.001,
            replacement_threshold: 10000,
            repair_mechanisms: Vec::new(),
        }
    }

    pub fn is_functional(&self) -> bool {
        self.cycle_count < self.replacement_threshold
    }
    
    pub fn requires_replacement(&self) -> bool {
        !self.is_functional()
    }

    pub fn update_cycle(&mut self, energy_cost: f64) {
        self.cycle_count += 1;
        self.energy_consumed += energy_cost;
    }

    pub fn attempt_repair(&mut self, atp_available: f64) -> bool {
        for mechanism in &self.repair_mechanisms {
            if atp_available >= mechanism.atp_cost_per_repair && 
               self.degradation_rate > mechanism.activation_threshold {
                // Successful repair
                self.degradation_rate *= (1.0 - mechanism.repair_efficiency);
                return true;
            }
        }
        false
    }
}

impl ThermodynamicChecker {
    pub fn new() -> Self {
        Self {
            haldane_relations: Vec::new(),
            entropy_balance: 0.0,
            energy_conservation_violations: 0,
            consistency_threshold: 1e-6,
        }
    }

    pub fn check_haldane_consistency(&mut self, relation: &HaldaneRelation) -> bool {
        let theoretical_ratio = relation.forward_rate / relation.reverse_rate;
        let experimental_ratio = relation.equilibrium_constant;
        
        let deviation = (theoretical_ratio - experimental_ratio).abs() / experimental_ratio;
        
        if deviation > self.consistency_threshold {
            self.energy_conservation_violations += 1;
            false
        } else {
            true
        }
    }

    pub fn update_entropy_balance(&mut self, local_decrease: f64, global_increase: f64) {
        self.entropy_balance = global_increase - local_decrease;
        
        // Second law of thermodynamics: global entropy must increase
        if self.entropy_balance < 0.0 {
            self.energy_conservation_violations += 1;
        }
    }
}

impl EnhancedBMDSystem {
    pub fn new() -> Self {
        Self {
            base_system: BMDSystem {
                demons: Vec::new(),
                information_flow: InformationFlow {
                    total_information_processed: 0.0,
                    amplification_cascade: 1.0,
                    pattern_recognition_events: 0,
                    target_channeling_successes: 0,
                },
                entropy_manipulation: EntropyManipulation {
                    local_entropy_reduction: 0.0,
                    global_entropy_increase: 0.0,
                    entropy_balance: 0.0,
                    negentropy_generation: 0.0,
                },
                fire_light_optimization: FireLightOptimization {
                    wavelength_range: (600.0, 700.0),
                    quantum_enhancement_factor: 1.0,
                    consciousness_amplification: 1.0,
                    abstract_reasoning_boost: 1.0,
                },
            },
            atp_kinetics: AtpKinetics::new(),
            circuit_nodes: Vec::new(),
            metastability_tracker: MetastabilityTracker::new(),
            thermodynamic_checker: ThermodynamicChecker::new(),
        }
    }

    /// Simulate the complete BMD system with all integrations
    pub fn simulate_complete_cycle(&mut self, time_step: f64) -> Result<BMDSimulationResult, BMDError> {
        // Check metastability
        if !self.metastability_tracker.is_functional() {
            return Err(BMDError::MetastabilityExceeded);
        }

        // Calculate ATP consumption
        let total_atp_consumption = self.calculate_total_atp_consumption();
        
        // Update ATP pool
        self.atp_kinetics.update_system(-total_atp_consumption);
        
        // Process each BMD
        let mut total_information_processed = 0.0;
        for demon in &mut self.base_system.demons {
            match demon {
                BiologicalMaxwellDemon::Molecular(ref mut molecular) => {
                    total_information_processed += self.process_molecular_bmd(molecular)?;
                }
                BiologicalMaxwellDemon::Neural(ref mut neural) => {
                    total_information_processed += self.process_neural_bmd(neural)?;
                }
                BiologicalMaxwellDemon::Metabolic(ref mut metabolic) => {
                    total_information_processed += self.process_metabolic_bmd(metabolic)?;
                }
                // Add other BMD types as needed
                _ => {}
            }
        }

        // Update metastability
        self.metastability_tracker.update_cycle(total_atp_consumption);

        // Check thermodynamic consistency
        let entropy_increase = total_information_processed * 1.38e-23 * 310.0; // kT
        self.thermodynamic_checker.update_entropy_balance(
            total_information_processed * 0.1, // Local entropy decrease
            entropy_increase
        );

        // Update information flow
        self.base_system.information_flow.total_information_processed += total_information_processed;
        self.base_system.information_flow.pattern_recognition_events += 1;

        Ok(BMDSimulationResult {
            information_processed: total_information_processed,
            atp_consumed: total_atp_consumption,
            entropy_generated: entropy_increase,
            metastability_status: self.metastability_tracker.is_functional(),
            thermodynamic_consistent: self.thermodynamic_checker.energy_conservation_violations == 0,
        })
    }

    fn calculate_total_atp_consumption(&self) -> f64 {
        // Sum ATP consumption from all BMDs
        self.base_system.demons.iter().map(|demon| {
            match demon {
                BiologicalMaxwellDemon::Molecular(molecular) => {
                    molecular.enzyme_specificity.calculate_atp_consumption_rate(&self.atp_kinetics.atp_pool)
                }
                BiologicalMaxwellDemon::Neural(neural) => {
                    neural.memory_associations.calculate_atp_consumption_rate(&self.atp_kinetics.atp_pool)
                }
                BiologicalMaxwellDemon::Metabolic(metabolic) => {
                    metabolic.atp_synthase.calculate_atp_consumption_rate(&self.atp_kinetics.atp_pool)
                }
                _ => 0.1 // Default consumption
            }
        }).sum()
    }

    fn process_molecular_bmd(&mut self, molecular: &mut MolecularBMD) -> Result<f64, BMDError> {
        // Simplified molecular BMD processing
        let substrate_patterns = vec![MolecularPattern {
            substrate_geometry: "enzyme_specific".to_string(),
            binding_affinity: 0.8,
            chemical_compatibility: 0.9,
        }];

        // Process through information catalyst
        let products = molecular.enzyme_specificity.catalyze(&substrate_patterns)?;
        Ok(products.len() as f64 * molecular.enzyme_specificity.amplification_factor)
    }

    fn process_neural_bmd(&mut self, neural: &mut NeuralBMD) -> Result<f64, BMDError> {
        // Simplified neural BMD processing
        let neural_patterns = vec![NeuralPattern {
            pattern_vector: vec![0.5, 0.8, 0.3, 0.9],
            recognition_confidence: 0.85,
            associative_strength: 0.7,
        }];

        let responses = neural.memory_associations.catalyze(&neural_patterns)?;
        Ok(responses.len() as f64 * neural.memory_associations.amplification_factor * neural.cognitive_selection_power)
    }

    fn process_metabolic_bmd(&mut self, metabolic: &mut MetabolicBMD) -> Result<f64, BMDError> {
        // Simplified metabolic BMD processing
        let proton_gradients = vec![ProtonGradient {
            proton_concentration_difference: 1.0,
            electrochemical_potential: -200.0, // mV
            gradient_stability: 0.9,
        }];

        let atp_synthesis = metabolic.atp_synthase.catalyze(&proton_gradients)?;
        Ok(atp_synthesis.len() as f64 * metabolic.atp_synthase.amplification_factor)
    }
}

/// Result of BMD system simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDSimulationResult {
    pub information_processed: f64,
    pub atp_consumed: f64,
    pub entropy_generated: f64,
    pub metastability_status: bool,
    pub thermodynamic_consistent: bool,
}
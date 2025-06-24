//! # Integration API for Neuron Construction
//!
//! This module provides clean, builder-pattern APIs for constructing intracellular environments
//! that can be easily integrated with Autobahn, Bene Gesserit, and Imhotep to build complete neurons.

use crate::{
    systems_biology::atp_kinetics::{AtpPool, AtpKinetics},
    biological_maxwell_demons::{BMDSystem, EnhancedBMDSystem},
    oscillatory_dynamics::{UniversalOscillator, OscillatorNetwork},
    quantum_membranes::QuantumMembrane,
    hardware_integration::{HardwareOscillatorSystem, EnvironmentalNoiseSystem},
    circuits::{HierarchicalCircuit, CircuitGrid},
    turbulance::TurbulanceEngine,
    error::{NebuchadnezzarError, Result},
};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Complete intracellular environment for neuron construction
#[derive(Debug, Clone)]
pub struct IntracellularEnvironment {
    /// ATP energy management system
    pub atp_system: AtpSystem,
    
    /// Biological Maxwell's Demons for information processing
    pub bmd_system: EnhancedBMDSystem,
    
    /// Oscillatory dynamics across multiple scales
    pub oscillatory_system: OscillatorySystem,
    
    /// Quantum membrane transport
    pub membrane_system: MembraneSystem,
    
    /// Hardware integration for environmental coupling
    pub hardware_system: HardwareSystem,
    
    /// Circuit representation of cellular processes
    pub circuit_system: CircuitSystem,
    
    /// Current state of the intracellular environment
    pub state: IntracellularState,
    
    /// Configuration parameters
    pub config: IntracellularConfig,
}

/// ATP energy management system
#[derive(Debug, Clone)]
pub struct AtpSystem {
    pub pool: AtpPool,
    pub kinetics: AtpKinetics,
    pub consumption_rate: f64,
    pub synthesis_rate: f64,
    pub energy_charge: f64,
}

/// Oscillatory dynamics system
#[derive(Debug, Clone)]
pub struct OscillatorySystem {
    pub primary_oscillator: UniversalOscillator,
    pub network: OscillatorNetwork,
    pub frequency_bands: HashMap<String, f64>,
    pub coupling_strength: f64,
}

/// Membrane transport system
#[derive(Debug, Clone)]
pub struct MembraneSystem {
    pub quantum_membrane: QuantumMembrane,
    pub transport_efficiency: f64,
    pub coherence_time: f64,
    pub environmental_coupling: f64,
}

/// Hardware integration system
#[derive(Debug, Clone)]
pub struct HardwareSystem {
    pub oscillator_system: HardwareOscillatorSystem,
    pub noise_system: EnvironmentalNoiseSystem,
    pub coupling_enabled: bool,
    pub harvesting_efficiency: f64,
}

/// Circuit representation system
#[derive(Debug, Clone)]
pub struct CircuitSystem {
    pub hierarchical_circuit: HierarchicalCircuit,
    pub circuit_grid: CircuitGrid,
    pub topology_dynamic: bool,
    pub resolution_level: usize,
}

/// Current state of the intracellular environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntracellularState {
    /// Current ATP concentration (mM)
    pub atp_concentration: f64,
    
    /// Energy charge (0.0 to 1.0)
    pub energy_charge: f64,
    
    /// Overall oscillatory phase
    pub oscillatory_phase: f64,
    
    /// Membrane potential (mV)
    pub membrane_potential: f64,
    
    /// Quantum coherence level (0.0 to 1.0)
    pub quantum_coherence: f64,
    
    /// Information processing rate (bits/s)
    pub information_processing_rate: f64,
    
    /// Environmental coupling strength
    pub environmental_coupling: f64,
    
    /// Timestamp of last update
    pub timestamp: f64,
}

/// Configuration for intracellular environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntracellularConfig {
    /// Target ATP concentration (mM)
    pub target_atp: f64,
    
    /// Oscillatory frequency range (Hz)
    pub frequency_range: (f64, f64),
    
    /// Quantum coherence threshold
    pub coherence_threshold: f64,
    
    /// Hardware integration enabled
    pub hardware_integration: bool,
    
    /// BMD information processing mode
    pub bmd_mode: BMDMode,
    
    /// Temperature (Kelvin)
    pub temperature: f64,
    
    /// pH level
    pub ph: f64,
    
    /// Ionic strength (mM)
    pub ionic_strength: f64,
}

/// BMD processing modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BMDMode {
    /// Optimized for neural processing
    Neural,
    /// Optimized for metabolic efficiency
    Metabolic,
    /// Optimized for information catalysis
    Catalytic,
    /// Custom configuration
    Custom(HashMap<String, f64>),
}

/// Oscillatory configuration presets
#[derive(Debug, Clone)]
pub struct OscillatoryConfig {
    pub base_frequency: f64,
    pub coupling_strength: f64,
    pub hierarchy_levels: usize,
    pub coherence_target: f64,
}

impl OscillatoryConfig {
    /// Biological oscillatory configuration
    pub fn biological() -> Self {
        Self {
            base_frequency: 40.0, // Gamma band for neural activity
            coupling_strength: 0.1,
            hierarchy_levels: 5,
            coherence_target: 0.8,
        }
    }
    
    /// High-frequency configuration for rapid processing
    pub fn high_frequency() -> Self {
        Self {
            base_frequency: 100.0,
            coupling_strength: 0.2,
            hierarchy_levels: 7,
            coherence_target: 0.9,
        }
    }
    
    /// Low-energy configuration for efficiency
    pub fn low_energy() -> Self {
        Self {
            base_frequency: 10.0,
            coupling_strength: 0.05,
            hierarchy_levels: 3,
            coherence_target: 0.6,
        }
    }
}

/// BMD configuration presets
#[derive(Debug, Clone)]
pub struct BMDConfig {
    pub mode: BMDMode,
    pub information_threshold: f64,
    pub catalysis_efficiency: f64,
    pub pattern_recognition_depth: usize,
}

impl BMDConfig {
    /// Neural-optimized BMD configuration
    pub fn neural_optimized() -> Self {
        Self {
            mode: BMDMode::Neural,
            information_threshold: 0.7,
            catalysis_efficiency: 0.85,
            pattern_recognition_depth: 5,
        }
    }
    
    /// Metabolic-optimized BMD configuration
    pub fn metabolic_optimized() -> Self {
        Self {
            mode: BMDMode::Metabolic,
            information_threshold: 0.5,
            catalysis_efficiency: 0.95,
            pattern_recognition_depth: 3,
        }
    }
    
    /// High-performance catalytic configuration
    pub fn catalytic_performance() -> Self {
        Self {
            mode: BMDMode::Catalytic,
            information_threshold: 0.8,
            catalysis_efficiency: 0.9,
            pattern_recognition_depth: 7,
        }
    }
}

/// Builder for constructing intracellular environments
pub struct IntracellularBuilder {
    atp_pool: Option<AtpPool>,
    oscillatory_config: Option<OscillatoryConfig>,
    bmd_config: Option<BMDConfig>,
    membrane_quantum_transport: bool,
    hardware_integration: bool,
    config: IntracellularConfig,
}

impl IntracellularBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            atp_pool: None,
            oscillatory_config: None,
            bmd_config: None,
            membrane_quantum_transport: false,
            hardware_integration: false,
            config: IntracellularConfig::default(),
        }
    }
    
    /// Set the ATP pool for energy management
    pub fn with_atp_pool(mut self, pool: AtpPool) -> Self {
        self.atp_pool = Some(pool);
        self
    }
    
    /// Configure oscillatory dynamics
    pub fn with_oscillatory_dynamics(mut self, config: OscillatoryConfig) -> Self {
        self.oscillatory_config = Some(config);
        self
    }
    
    /// Configure Biological Maxwell's Demons
    pub fn with_maxwell_demons(mut self, config: BMDConfig) -> Self {
        self.bmd_config = Some(config);
        self
    }
    
    /// Enable quantum membrane transport
    pub fn with_membrane_quantum_transport(mut self, enabled: bool) -> Self {
        self.membrane_quantum_transport = enabled;
        self
    }
    
    /// Enable hardware integration
    pub fn with_hardware_integration(mut self, enabled: bool) -> Self {
        self.hardware_integration = enabled;
        self.config.hardware_integration = enabled;
        self
    }
    
    /// Set target ATP concentration
    pub fn with_target_atp(mut self, concentration: f64) -> Self {
        self.config.target_atp = concentration;
        self
    }
    
    /// Set temperature
    pub fn with_temperature(mut self, temp_k: f64) -> Self {
        self.config.temperature = temp_k;
        self
    }
    
    /// Set pH level
    pub fn with_ph(mut self, ph: f64) -> Self {
        self.config.ph = ph;
        self
    }
    
    /// Build the intracellular environment
    pub fn build(self) -> Result<IntracellularEnvironment> {
        // Use provided ATP pool or create physiological default
        let atp_pool = self.atp_pool.unwrap_or_else(|| AtpPool::new_physiological());
        
        // Create ATP system
        let atp_system = AtpSystem {
            energy_charge: atp_pool.energy_charge,
            consumption_rate: 0.0,
            synthesis_rate: 0.0,
            pool: atp_pool.clone(),
            kinetics: AtpKinetics::new(),
        };
        
        // Create BMD system
        let bmd_config = self.bmd_config.unwrap_or_else(|| BMDConfig::neural_optimized());
        let bmd_system = EnhancedBMDSystem::new();
        
        // Create oscillatory system
        let osc_config = self.oscillatory_config.unwrap_or_else(|| OscillatoryConfig::biological());
        let oscillatory_system = OscillatorySystem {
            primary_oscillator: UniversalOscillator::new(
                "primary".to_string(),
                crate::oscillatory_dynamics::OscillatorParameters {
                    natural_frequency: osc_config.base_frequency,
                    damping_coefficient: 0.1,
                    driving_amplitude: 1.0,
                    driving_frequency: osc_config.base_frequency,
                    nonlinearity_strength: 0.1,
                    mass: 1.0,
                    spring_constant: 1.0,
                }
            ),
            network: OscillatorNetwork::new(),
            frequency_bands: HashMap::new(),
            coupling_strength: osc_config.coupling_strength,
        };
        
        // Create membrane system
        let membrane_system = MembraneSystem {
            quantum_membrane: QuantumMembrane::new(100)?,
            transport_efficiency: 0.8,
            coherence_time: 1e-12, // picoseconds
            environmental_coupling: 0.1,
        };
        
        // Create hardware system
        let hardware_system = HardwareSystem {
            oscillator_system: HardwareOscillatorSystem::new(),
            noise_system: EnvironmentalNoiseSystem::new(),
            coupling_enabled: self.hardware_integration,
            harvesting_efficiency: 0.6,
        };
        
        // Create circuit system
        let circuit_system = CircuitSystem {
            hierarchical_circuit: HierarchicalCircuit::new(),
            circuit_grid: CircuitGrid::new(10, 10), // Default 10x10 grid
            topology_dynamic: true,
            resolution_level: 1,
        };
        
        // Initial state
        let state = IntracellularState {
            atp_concentration: atp_pool.atp_concentration,
            energy_charge: atp_pool.energy_charge,
            oscillatory_phase: 0.0,
            membrane_potential: -70.0, // Typical resting potential
            quantum_coherence: 0.5,
            information_processing_rate: 1000.0, // bits/s
            environmental_coupling: 0.1,
            timestamp: 0.0,
        };
        
        Ok(IntracellularEnvironment {
            atp_system,
            bmd_system,
            oscillatory_system,
            membrane_system,
            hardware_system,
            circuit_system,
            state,
            config: self.config,
        })
    }
}

impl Default for IntracellularBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IntracellularConfig {
    fn default() -> Self {
        Self {
            target_atp: 5.0, // mM
            frequency_range: (1.0, 100.0), // Hz
            coherence_threshold: 0.7,
            hardware_integration: false,
            bmd_mode: BMDMode::Neural,
            temperature: 310.0, // 37°C in Kelvin
            ph: 7.4,
            ionic_strength: 150.0, // mM
        }
    }
}

impl IntracellularEnvironment {
    /// Create a new builder for constructing intracellular environments
    pub fn builder() -> IntracellularBuilder {
        IntracellularBuilder::new()
    }
    
    /// Update the intracellular state based on ATP consumption
    pub fn update(&mut self, dt_atp: f64) -> Result<()> {
        // Update ATP system
        self.atp_system.pool.update_atp(dt_atp);
        self.state.atp_concentration = self.atp_system.pool.atp_concentration;
        self.state.energy_charge = self.atp_system.pool.energy_charge;
        
        // Update oscillatory phase
        self.state.oscillatory_phase += self.oscillatory_system.primary_oscillator.state.frequency * dt_atp;
        
        // Update quantum coherence based on energy availability
        let coherence_factor = (self.state.energy_charge - 0.5).max(0.0) * 2.0;
        self.state.quantum_coherence = (self.state.quantum_coherence * 0.9 + coherence_factor * 0.1).min(1.0);
        
        // Update information processing rate based on ATP availability
        self.state.information_processing_rate = 1000.0 * self.state.energy_charge;
        
        self.state.timestamp += dt_atp;
        
        Ok(())
    }
    
    /// Get current energy status
    pub fn energy_status(&self) -> EnergyStatus {
        match self.state.energy_charge {
            x if x > 0.8 => EnergyStatus::High,
            x if x > 0.6 => EnergyStatus::Moderate,
            x if x > 0.4 => EnergyStatus::Low,
            _ => EnergyStatus::Critical,
        }
    }
    
    /// Check if the environment is ready for neuron integration
    pub fn integration_ready(&self) -> bool {
        self.state.energy_charge > 0.5 &&
        self.state.quantum_coherence > self.config.coherence_threshold &&
        self.state.atp_concentration > 1.0
    }
}

/// Energy status of the intracellular environment
#[derive(Debug, Clone, PartialEq)]
pub enum EnergyStatus {
    High,
    Moderate,
    Low,
    Critical,
}

/// Neuron construction kit for integration with other packages
pub struct NeuronConstructionKit {
    pub intracellular: IntracellularEnvironment,
    pub integration_points: IntegrationPoints,
    pub components: NeuronComponents,
}

/// Integration points for connecting with other packages
#[derive(Debug, Clone)]
pub struct IntegrationPoints {
    /// Connection point for Autobahn RAG system
    pub autobahn_interface: Option<AutobahnInterface>,
    
    /// Connection point for Bene Gesserit membrane dynamics
    pub bene_gesserit_interface: Option<BeneGesseritInterface>,
    
    /// Connection point for Imhotep neural interface
    pub imhotep_interface: Option<ImhotepInterface>,
}

/// Autobahn integration interface
#[derive(Debug, Clone)]
pub struct AutobahnInterface {
    pub knowledge_processing_rate: f64,
    pub retrieval_efficiency: f64,
    pub generation_quality: f64,
}

/// Bene Gesserit integration interface
#[derive(Debug, Clone)]
pub struct BeneGesseritInterface {
    pub membrane_dynamics_coupling: f64,
    pub hardware_oscillation_harvesting: bool,
    pub pixel_noise_optimization: bool,
}

/// Imhotep integration interface
#[derive(Debug, Clone)]
pub struct ImhotepInterface {
    pub consciousness_emergence_threshold: f64,
    pub neural_interface_active: bool,
    pub bmd_neural_processing: bool,
}

/// Components available for neuron construction
#[derive(Debug, Clone)]
pub struct NeuronComponents {
    pub dendrites: Vec<DendriteComponent>,
    pub soma: SomaComponent,
    pub axon: Option<AxonComponent>,
    pub synapses: Vec<SynapseComponent>,
}

/// Dendrite component for neuron construction
#[derive(Debug, Clone)]
pub struct DendriteComponent {
    pub branch_id: String,
    pub surface_area: f64,
    pub receptor_density: f64,
    pub integration_time_constant: f64,
}

/// Soma component for neuron construction
#[derive(Debug, Clone)]
pub struct SomaComponent {
    pub diameter: f64,
    pub membrane_capacitance: f64,
    pub threshold_potential: f64,
    pub refractory_period: f64,
}

/// Axon component for neuron construction
#[derive(Debug, Clone)]
pub struct AxonComponent {
    pub length: f64,
    pub diameter: f64,
    pub myelination: bool,
    pub conduction_velocity: f64,
}

/// Synapse component for neuron construction
#[derive(Debug, Clone)]
pub struct SynapseComponent {
    pub synapse_id: String,
    pub synapse_type: SynapseType,
    pub strength: f64,
    pub plasticity: f64,
}

/// Types of synapses
#[derive(Debug, Clone)]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Electrical,
}

impl Default for IntegrationPoints {
    fn default() -> Self {
        Self {
            autobahn_interface: None,
            bene_gesserit_interface: None,
            imhotep_interface: None,
        }
    }
}

impl Default for NeuronComponents {
    fn default() -> Self {
        Self {
            dendrites: Vec::new(),
            soma: SomaComponent {
                diameter: 20.0, // micrometers
                membrane_capacitance: 1.0, // microfarads/cm²
                threshold_potential: -55.0, // mV
                refractory_period: 2.0, // ms
            },
            axon: None,
            synapses: Vec::new(),
        }
    }
}

impl NeuronConstructionKit {
    /// Create a new neuron construction kit
    pub fn new(intracellular: IntracellularEnvironment) -> Self {
        Self {
            intracellular,
            integration_points: IntegrationPoints::default(),
            components: NeuronComponents::default(),
        }
    }
    
    /// Add Autobahn integration
    pub fn with_autobahn(mut self, interface: AutobahnInterface) -> Self {
        self.integration_points.autobahn_interface = Some(interface);
        self
    }
    
    /// Add Bene Gesserit integration
    pub fn with_bene_gesserit(mut self, interface: BeneGesseritInterface) -> Self {
        self.integration_points.bene_gesserit_interface = Some(interface);
        self
    }
    
    /// Add Imhotep integration
    pub fn with_imhotep(mut self, interface: ImhotepInterface) -> Self {
        self.integration_points.imhotep_interface = Some(interface);
        self
    }
    
    /// Check if all required integrations are available
    pub fn integration_complete(&self) -> bool {
        self.integration_points.autobahn_interface.is_some() &&
        self.integration_points.bene_gesserit_interface.is_some() &&
        self.integration_points.imhotep_interface.is_some()
    }
} 
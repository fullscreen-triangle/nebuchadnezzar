//! Common imports and types for Nebuchadnezzar

pub use crate::error::{Error, Result};
pub use crate::{IntracellularEnvironment, IntracellularEnvironmentBuilder, IntracellularState};

// ATP System exports
pub use crate::atp_system::{AtpPool, AtpSystem, AtpConfig};

// BMD System exports
pub use crate::bmd_system::{
    BiologicalMaxwellDemon, BmdSystem, BMDConfig,
    PatternRecognition, TargetSelection, Amplification
};

// Quantum Membrane exports
pub use crate::quantum_membranes::{
    QuantumMembrane, QuantumMembraneSystem, MembraneConfig,
    IonChannel, CoherenceManager
};

// Oscillatory Dynamics exports
pub use crate::oscillatory_dynamics::{
    OscillatorySystem, OscillatoryConfig,
    HierarchicalNetwork, FrequencyBands, PhaseCoupling
};

// Virtual Circulation exports
pub use crate::virtual_circulation::{
    VirtualCirculationSystem, CirculationConfig,
    NoiseStratification, VesselNetwork, FlowDynamics
};

// Circuit exports
pub use crate::circuits::{
    CircuitSystem, CircuitConfig, CircuitState,
    ProbabilisticElement, BiologicalMapping
};

// Temporal Processing exports
pub use crate::temporal_processing::{
    TemporalProcessingSystem, TemporalConfig,
    NeuralGeneration, MemoryInitialization, StatisticalEmergence
};

// Atmospheric Processing exports
pub use crate::atmospheric_processing::{
    AtmosphericProcessingSystem, AtmosphericConfig,
    EntropyOscillation, MolecularNetwork, VirtualProcessors
};

// Integration exports
pub use crate::integration::{
    AutobahnInterface, BeneGesseritInterface, ImhotepInterface,
    ConsciousnessBridge
};

// Hardware Integration exports
pub use crate::hardware_integration::{
    HardwareSystem, HardwareConfig,
    OscillationHarvesting, NoiseOptimization
};

// Common mathematical types
pub type AtpConcentration = f64;
pub type EnergyCharge = f64;
pub type QuantumCoherence = f64;
pub type OscillatoryPhase = f64;
pub type NoiseLevel = f64;
pub type AmplificationFactor = f64;
pub type ProcessingRate = f64;
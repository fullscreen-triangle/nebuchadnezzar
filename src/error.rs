//! Error types and handling for Nebuchadnezzar

use std::fmt;

/// Result type alias for Nebuchadnezzar operations
pub type Result<T> = std::result::Result<T, Error>;

/// Comprehensive error types for the Nebuchadnezzar system
#[derive(Debug, Clone)]
pub enum Error {
    /// ATP system errors
    AtpDepletion,
    AtpOverflow,
    EnergyChargeOutOfRange { value: f64, min: f64, max: f64 },
    
    /// BMD system errors
    BmdThermodynamicViolation { entropy_cost: f64 },
    BmdPatternRecognitionFailure,
    BmdAmplificationOverflow,
    
    /// Quantum membrane errors
    QuantumDecoherence { coherence_time: f64 },
    IonChannelFailure,
    TunnelingCalculationError,
    
    /// Oscillatory dynamics errors
    FrequencyMismatch { expected: f64, actual: f64 },
    PhaseCouplingFailure,
    HardwareHarvestingError,
    
    /// Virtual circulation errors
    CirculationBackflow,
    NoiseGradientViolation { position: f64, expected_concentration: f64, actual: f64 },
    VesselNetworkError,
    
    /// Circuit system errors
    CircuitOverflow,
    NodalAnalysisFailure,
    ProbabilisticElementError,
    
    /// Temporal processing errors
    TemporalPrecisionLoss { precision: f64 },
    NeuralGenerationFailure,
    StatisticalEmergenceTimeout,
    
    /// Atmospheric processing errors
    MolecularNetworkFailure,
    EntropyOscillationError,
    VirtualProcessorOverflow,
    
    /// Integration errors
    IntegrationFailure { system: String },
    ConsciousnessIntegrationError,
    
    /// Hardware integration errors
    HardwareConnectionLost,
    OscillationHarvestingFailure,
    
    /// Numerical errors
    IntegrationFailure { step_size: f64 },
    ConvergenceFailure { iterations: usize },
    NumericalInstability,
    
    /// Configuration errors
    InvalidConfiguration { parameter: String, value: String },
    MissingDependency { dependency: String },
    
    /// System errors
    SystemNotInitialized,
    ResourceExhaustion,
    ConcurrencyError,
    
    /// I/O errors
    SerializationError(String),
    DeserializationError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AtpDepletion => write!(f, "ATP pool depleted below minimum threshold"),
            Error::AtpOverflow => write!(f, "ATP concentration exceeded maximum physiological range"),
            Error::EnergyChargeOutOfRange { value, min, max } => 
                write!(f, "Energy charge {} outside range [{}, {}]", value, min, max),
                
            Error::BmdThermodynamicViolation { entropy_cost } => 
                write!(f, "BMD operation violates thermodynamics with entropy cost {}", entropy_cost),
            Error::BmdPatternRecognitionFailure => 
                write!(f, "BMD pattern recognition failed to identify valid patterns"),
            Error::BmdAmplificationOverflow => 
                write!(f, "BMD amplification exceeded safe operating limits"),
                
            Error::QuantumDecoherence { coherence_time } => 
                write!(f, "Quantum coherence lost after {} seconds", coherence_time),
            Error::IonChannelFailure => 
                write!(f, "Ion channel state transition failed"),
            Error::TunnelingCalculationError => 
                write!(f, "Quantum tunneling probability calculation failed"),
                
            Error::FrequencyMismatch { expected, actual } => 
                write!(f, "Frequency mismatch: expected {} Hz, got {} Hz", expected, actual),
            Error::PhaseCouplingFailure => 
                write!(f, "Cross-frequency phase coupling failed"),
            Error::HardwareHarvestingError => 
                write!(f, "Hardware oscillation harvesting failed"),
                
            Error::CirculationBackflow => 
                write!(f, "Virtual circulation backflow detected"),
            Error::NoiseGradientViolation { position, expected_concentration, actual } => 
                write!(f, "Noise gradient violation at position {}: expected {}, got {}", 
                       position, expected_concentration, actual),
            Error::VesselNetworkError => 
                write!(f, "Virtual vessel network topology error"),
                
            Error::CircuitOverflow => 
                write!(f, "Circuit analysis overflow"),
            Error::NodalAnalysisFailure => 
                write!(f, "Modified nodal analysis failed to converge"),
            Error::ProbabilisticElementError => 
                write!(f, "Probabilistic circuit element calculation failed"),
                
            Error::TemporalPrecisionLoss { precision } => 
                write!(f, "Temporal precision degraded to {} seconds", precision),
            Error::NeuralGenerationFailure => 
                write!(f, "High-frequency neural generation failed"),
            Error::StatisticalEmergenceTimeout => 
                write!(f, "Statistical solution emergence timed out"),
                
            Error::MolecularNetworkFailure => 
                write!(f, "Atmospheric molecular network failed"),
            Error::EntropyOscillationError => 
                write!(f, "Entropy-oscillation reformulation error"),
            Error::VirtualProcessorOverflow => 
                write!(f, "Virtual processor generation overflow"),
                
            Error::IntegrationFailure { system } => 
                write!(f, "Integration with {} failed", system),
            Error::ConsciousnessIntegrationError => 
                write!(f, "Consciousness-computation integration failed"),
                
            Error::HardwareConnectionLost => 
                write!(f, "Hardware connection lost"),
            Error::OscillationHarvestingFailure => 
                write!(f, "System oscillation harvesting failed"),
                
            Error::IntegrationFailure { step_size } => 
                write!(f, "Numerical integration failed with step size {}", step_size),
            Error::ConvergenceFailure { iterations } => 
                write!(f, "Failed to converge after {} iterations", iterations),
            Error::NumericalInstability => 
                write!(f, "Numerical instability detected"),
                
            Error::InvalidConfiguration { parameter, value } => 
                write!(f, "Invalid configuration: {}={}", parameter, value),
            Error::MissingDependency { dependency } => 
                write!(f, "Missing required dependency: {}", dependency),
                
            Error::SystemNotInitialized => 
                write!(f, "System not properly initialized"),
            Error::ResourceExhaustion => 
                write!(f, "System resources exhausted"),
            Error::ConcurrencyError => 
                write!(f, "Concurrency error in parallel processing"),
                
            Error::SerializationError(msg) => 
                write!(f, "Serialization error: {}", msg),
            Error::DeserializationError(msg) => 
                write!(f, "Deserialization error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

/// Recovery strategies for different error types
impl Error {
    pub fn recovery_strategy(&self) -> RecoveryAction {
        match self {
            Error::AtpDepletion => RecoveryAction::EmergencyMetabolism,
            Error::QuantumDecoherence { .. } => RecoveryAction::CoherenceRestore,
            Error::CircuitOverflow => RecoveryAction::LoadBalance,
            Error::IntegrationFailure { .. } => RecoveryAction::StepSizeReduction,
            Error::NumericalInstability => RecoveryAction::PrecisionIncrease,
            Error::HardwareConnectionLost => RecoveryAction::HardwareReconnect,
            Error::StatisticalEmergenceTimeout => RecoveryAction::ParameterAdjustment,
            _ => RecoveryAction::SystemReset,
        }
    }
}

/// Recovery actions for error handling
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    EmergencyMetabolism,
    CoherenceRestore,
    LoadBalance,
    StepSizeReduction,
    PrecisionIncrease,
    HardwareReconnect,
    ParameterAdjustment,
    SystemReset,
}
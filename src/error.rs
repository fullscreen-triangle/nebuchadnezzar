//! # Error Handling for Nebuchadnezzar
//! 
//! Comprehensive error types for quantum-enhanced biological circuit simulation.

use std::fmt;

/// Main error type for the Nebuchadnezzar library
#[derive(Debug, Clone)]
pub enum NebuchadnezzarError {
    /// Quantum membrane computation errors
    QuantumError {
        error_type: QuantumErrorType,
        message: String,
    },
    
    /// Oscillatory dynamics errors
    OscillatorError {
        error_type: OscillatorErrorType,
        message: String,
    },
    
    /// Entropy manipulation errors
    EntropyError {
        error_type: EntropyErrorType,
        message: String,
    },
    
    /// Biological integration errors
    BiologicalError {
        error_type: BiologicalErrorType,
        message: String,
    },
    
    /// Circuit simulation errors
    CircuitError {
        error_type: CircuitErrorType,
        message: String,
    },
    
    /// Solver and integration errors
    SolverError {
        error_type: SolverErrorType,
        message: String,
    },
    
    /// ATP-related errors
    AtpError {
        error_type: AtpErrorType,
        message: String,
    },
    
    /// Mathematical computation errors
    MathError {
        error_type: MathErrorType,
        message: String,
    },
    
    /// I/O and data errors
    IoError {
        message: String,
        source: Option<std::io::Error>,
    },
    
    /// Configuration and parameter errors
    ConfigError {
        parameter: String,
        value: String,
        message: String,
    },
}

/// Quantum membrane computation error types
#[derive(Debug, Clone)]
pub enum QuantumErrorType {
    CoherenceLoss,
    EnvironmentalDecoherence,
    TunnelingFailure,
    EnaqtProcessingError,
    QuantumStateInvalid,
    TemperatureOutOfRange,
    NoiseExceedsThreshold,
}

/// Oscillatory dynamics error types
#[derive(Debug, Clone)]
pub enum OscillatorErrorType {
    FrequencyOutOfRange,
    AmplitudeTooLarge,
    PhaseIncoherence,
    CouplingInstability,
    CausalSelectionFailure,
    NetworkSynchronizationError,
    ResonanceOverflow,
}

/// Entropy manipulation error types
#[derive(Debug, Clone)]
pub enum EntropyErrorType {
    ProbabilityOutOfRange,
    ConfidenceInvalid,
    ResolutionThresholdError,
    PerturbationFailed,
    ValidationError,
    ConvergenceFailure,
    EntropyInconsistency,
}

/// Biological integration error types
#[derive(Debug, Clone)]
pub enum BiologicalErrorType {
    ComponentMissing,
    MetabolicNetworkInvalid,
    SignalingPathwayError,
    GeneRegulationFailure,
    ProteinInteractionError,
    TransportProcessError,
    SystemIntegrationFailure,
    BiochemicalReactionError,
}

/// Circuit simulation error types
#[derive(Debug, Clone)]
pub enum CircuitErrorType {
    VoltageOutOfRange,
    CurrentOverflow,
    ResistanceInvalid,
    CapacitanceNegative,
    InductanceInvalid,
    NodeNotFound,
    CircuitNotConnected,
    GridResolutionFailed,
    HierarchicalInconsistency,
    IonChannelError,
    EnzymeCircuitError,
}

/// Solver and integration error types
#[derive(Debug, Clone)]
pub enum SolverErrorType {
    StepSizeTooLarge,
    StepSizeTooSmall,
    MaxIterationsExceeded,
    ConvergenceFailure,
    NumericalInstability,
    StateVectorInvalid,
    IntegrationError,
    AdaptiveStepFailed,
}

/// ATP-related error types
#[derive(Debug, Clone)]
pub enum AtpErrorType {
    ConcentrationNegative,
    ConcentrationTooHigh,
    InsufficientAtp,
    KineticsInvalid,
    EnergyChargeOutOfRange,
    ConsumptionRateInvalid,
    ProductionRateInvalid,
}

/// Mathematical computation error types
#[derive(Debug, Clone)]
pub enum MathErrorType {
    DivisionByZero,
    SquareRootNegative,
    LogarithmNonPositive,
    MatrixSingular,
    MatrixDimensionMismatch,
    VectorLengthMismatch,
    NumericalOverflow,
    NumericalUnderflow,
    InvalidFunction,
}

/// Result type for Nebuchadnezzar operations
pub type Result<T> = std::result::Result<T, NebuchadnezzarError>;

impl fmt::Display for NebuchadnezzarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NebuchadnezzarError::QuantumError { error_type, message } => {
                write!(f, "Quantum Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::OscillatorError { error_type, message } => {
                write!(f, "Oscillator Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::EntropyError { error_type, message } => {
                write!(f, "Entropy Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::BiologicalError { error_type, message } => {
                write!(f, "Biological Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::CircuitError { error_type, message } => {
                write!(f, "Circuit Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::SolverError { error_type, message } => {
                write!(f, "Solver Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::AtpError { error_type, message } => {
                write!(f, "ATP Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::MathError { error_type, message } => {
                write!(f, "Math Error ({:?}): {}", error_type, message)
            }
            NebuchadnezzarError::IoError { message, .. } => {
                write!(f, "I/O Error: {}", message)
            }
            NebuchadnezzarError::ConfigError { parameter, value, message } => {
                write!(f, "Config Error: Parameter '{}' with value '{}': {}", parameter, value, message)
            }
        }
    }
}

impl std::error::Error for NebuchadnezzarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NebuchadnezzarError::IoError { source: Some(err), .. } => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NebuchadnezzarError {
    fn from(err: std::io::Error) -> Self {
        NebuchadnezzarError::IoError {
            message: err.to_string(),
            source: Some(err),
        }
    }
}

// Convenience constructors for common error types
impl NebuchadnezzarError {
    /// Create a quantum error
    pub fn quantum_error(error_type: QuantumErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::QuantumError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create an oscillator error
    pub fn oscillator_error(error_type: OscillatorErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::OscillatorError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create an entropy error
    pub fn entropy_error(error_type: EntropyErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::EntropyError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create a biological error
    pub fn biological_error(error_type: BiologicalErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::BiologicalError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create a circuit error
    pub fn circuit_error(error_type: CircuitErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::CircuitError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create a solver error
    pub fn solver_error(error_type: SolverErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::SolverError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create an ATP error
    pub fn atp_error(error_type: AtpErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::AtpError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create a math error
    pub fn math_error(error_type: MathErrorType, message: impl Into<String>) -> Self {
        NebuchadnezzarError::MathError {
            error_type,
            message: message.into(),
        }
    }
    
    /// Create a config error
    pub fn config_error(parameter: impl Into<String>, value: impl Into<String>, message: impl Into<String>) -> Self {
        NebuchadnezzarError::ConfigError {
            parameter: parameter.into(),
            value: value.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let quantum_err = NebuchadnezzarError::quantum_error(
            QuantumErrorType::CoherenceLoss,
            "Quantum coherence lost due to environmental noise"
        );
        
        assert!(matches!(quantum_err, NebuchadnezzarError::QuantumError { .. }));
    }

    #[test]
    fn test_error_display() {
        let circuit_err = NebuchadnezzarError::circuit_error(
            CircuitErrorType::VoltageOutOfRange,
            "Voltage exceeds membrane threshold"
        );
        
        let display_str = format!("{}", circuit_err);
        assert!(display_str.contains("Circuit Error"));
        assert!(display_str.contains("VoltageOutOfRange"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let neb_err: NebuchadnezzarError = io_err.into();
        
        assert!(matches!(neb_err, NebuchadnezzarError::IoError { .. }));
    }
} 
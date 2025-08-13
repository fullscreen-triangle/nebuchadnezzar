//! # Nebuchadnezzar: Intracellular Dynamics Engine
//! 
//! A comprehensive framework for modeling intracellular processes using ATP as the fundamental 
//! rate unit. Designed as the foundational intracellular dynamics package for constructing
//! biologically authentic neurons in the Imhotep neural interface framework.
//!
//! ## Core Features
//! 
//! - **ATP-Constrained Dynamics**: Uses `dx/dATP` equations for metabolically realistic computation
//! - **Biological Maxwell's Demons**: Information catalysts for selective processing and amplification
//! - **Quantum Membrane Transport**: Environment-assisted quantum coherence at biological temperatures
//! - **Virtual Circulatory Infrastructure**: Biologically-constrained noise distribution through concentration gradients
//! - **Temporal Precision Enhancement**: High-frequency neural generation with statistical solution emergence
//! - **Atmospheric Molecular Processing**: Entropy-oscillation reformulation for distributed computation
//! - **Consciousness-Computation Integration**: Multi-modal environmental sensing and meta-language interfaces
//! - **Hierarchical Circuit Architecture**: Multi-scale probabilistic electric circuit simulation
//! - **Hardware Integration**: Direct coupling with system oscillations and environmental noise
//!
//! ## Integration Architecture
//!
//! Nebuchadnezzar serves as the **intracellular dynamics foundation** for:
//! - **Autobahn**: RAG system integration for knowledge processing
//! - **Bene Gesserit**: Membrane dynamics and quantum transport
//! - **Imhotep**: Neural interface and consciousness emergence
//!
//! ## Quick Start
//!
//! ```rust
//! use nebuchadnezzar::prelude::*;
//!
//! // Create intracellular environment for neuron construction
//! let intracellular = IntracellularEnvironment::builder()
//!     .with_atp_pool(AtpPool::new_physiological())
//!     .with_oscillatory_dynamics(OscillatoryConfig::biological())
//!     .with_membrane_quantum_transport(true)
//!     .with_maxwell_demons(BMDConfig::neural_optimized())
//!     .with_virtual_circulation(CirculationConfig::gradient_optimized())
//!     .with_temporal_precision(TemporalConfig::high_frequency())
//!     .build()?;
//!
//! // Ready for integration with Autobahn, Bene Gesserit, and Imhotep
//! ```

// Core modules - organized for maximum portability
pub mod prelude;
pub mod error;

// Primary subsystems
pub mod atp_system;
pub mod bmd_system;
pub mod quantum_membranes;
pub mod oscillatory_dynamics;
pub mod virtual_circulation;
pub mod circuits;
pub mod temporal_processing;
pub mod atmospheric_processing;

// Integration and hardware
pub mod integration;
pub mod hardware_integration;
pub mod validation;

use error::Result;
use prelude::*;

/// Main intracellular environment containing all subsystems
#[derive(Debug)]
pub struct IntracellularEnvironment {
    atp_system: atp_system::AtpSystem,
    bmd_system: bmd_system::BmdSystem,
    quantum_membranes: quantum_membranes::QuantumMembraneSystem,
    oscillatory_dynamics: oscillatory_dynamics::OscillatorySystem,
    virtual_circulation: virtual_circulation::VirtualCirculationSystem,
    circuits: circuits::CircuitSystem,
    temporal_processing: temporal_processing::TemporalProcessingSystem,
    atmospheric_processing: atmospheric_processing::AtmosphericProcessingSystem,
    hardware_integration: hardware_integration::HardwareSystem,
}

impl IntracellularEnvironment {
    /// Create a builder for configuring the intracellular environment
    pub fn builder() -> IntracellularEnvironmentBuilder {
        IntracellularEnvironmentBuilder::new()
    }

    /// Perform a single simulation step
    pub fn step(&mut self, dt: f64) -> Result<()> {
        // Update ATP system first (fundamental rate parameter)
        self.atp_system.step(dt)?;
        
        // Get current ATP concentration for other systems
        let atp_concentration = self.atp_system.current_concentration();
        
        // Update all other systems in parallel where possible
        self.bmd_system.step(dt, atp_concentration)?;
        self.quantum_membranes.step(dt, atp_concentration)?;
        self.oscillatory_dynamics.step(dt)?;
        self.virtual_circulation.step(dt)?;
        self.circuits.step(dt, atp_concentration)?;
        self.temporal_processing.step(dt)?;
        self.atmospheric_processing.step(dt)?;
        self.hardware_integration.step(dt)?;
        
        Ok(())
    }

    /// Check if system is ready for integration with external systems
    pub fn integration_ready(&self) -> bool {
        self.atp_system.is_stable() &&
        self.bmd_system.is_operational() &&
        self.quantum_membranes.is_coherent() &&
        self.oscillatory_dynamics.is_synchronized()
    }

    /// Get current system state for external integration
    pub fn current_state(&self) -> IntracellularState {
        IntracellularState {
            atp_concentration: self.atp_system.current_concentration(),
            energy_charge: self.atp_system.energy_charge(),
            bmd_activity: self.bmd_system.current_activity(),
            quantum_coherence: self.quantum_membranes.coherence_level(),
            oscillatory_phase: self.oscillatory_dynamics.current_phase(),
            circulation_flow: self.virtual_circulation.current_flow(),
            circuit_state: self.circuits.current_state(),
        }
    }
}

/// Builder for IntracellularEnvironment
#[derive(Debug, Default)]
pub struct IntracellularEnvironmentBuilder {
    atp_config: Option<atp_system::AtpConfig>,
    bmd_config: Option<bmd_system::BMDConfig>,
    membrane_config: Option<quantum_membranes::MembraneConfig>,
    oscillatory_config: Option<oscillatory_dynamics::OscillatoryConfig>,
    circulation_config: Option<virtual_circulation::CirculationConfig>,
    circuit_config: Option<circuits::CircuitConfig>,
    temporal_config: Option<temporal_processing::TemporalConfig>,
    atmospheric_config: Option<atmospheric_processing::AtmosphericConfig>,
    hardware_config: Option<hardware_integration::HardwareConfig>,
}

impl IntracellularEnvironmentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_atp_pool(mut self, config: atp_system::AtpConfig) -> Self {
        self.atp_config = Some(config);
        self
    }

    pub fn with_oscillatory_dynamics(mut self, config: oscillatory_dynamics::OscillatoryConfig) -> Self {
        self.oscillatory_config = Some(config);
        self
    }

    pub fn with_membrane_quantum_transport(mut self, enabled: bool) -> Self {
        let config = if enabled {
            quantum_membranes::MembraneConfig::quantum_enabled()
        } else {
            quantum_membranes::MembraneConfig::classical()
        };
        self.membrane_config = Some(config);
        self
    }

    pub fn with_maxwell_demons(mut self, config: bmd_system::BMDConfig) -> Self {
        self.bmd_config = Some(config);
        self
    }

    pub fn with_virtual_circulation(mut self, config: virtual_circulation::CirculationConfig) -> Self {
        self.circulation_config = Some(config);
        self
    }

    pub fn with_temporal_precision(mut self, config: temporal_processing::TemporalConfig) -> Self {
        self.temporal_config = Some(config);
        self
    }

    pub fn build(self) -> Result<IntracellularEnvironment> {
        let atp_system = atp_system::AtpSystem::new(
            self.atp_config.unwrap_or_default()
        )?;
        
        let bmd_system = bmd_system::BmdSystem::new(
            self.bmd_config.unwrap_or_default()
        )?;
        
        let quantum_membranes = quantum_membranes::QuantumMembraneSystem::new(
            self.membrane_config.unwrap_or_default()
        )?;
        
        let oscillatory_dynamics = oscillatory_dynamics::OscillatorySystem::new(
            self.oscillatory_config.unwrap_or_default()
        )?;
        
        let virtual_circulation = virtual_circulation::VirtualCirculationSystem::new(
            self.circulation_config.unwrap_or_default()
        )?;
        
        let circuits = circuits::CircuitSystem::new(
            self.circuit_config.unwrap_or_default()
        )?;
        
        let temporal_processing = temporal_processing::TemporalProcessingSystem::new(
            self.temporal_config.unwrap_or_default()
        )?;
        
        let atmospheric_processing = atmospheric_processing::AtmosphericProcessingSystem::new(
            self.atmospheric_config.unwrap_or_default()
        )?;
        
        let hardware_integration = hardware_integration::HardwareSystem::new(
            self.hardware_config.unwrap_or_default()
        )?;

        Ok(IntracellularEnvironment {
            atp_system,
            bmd_system,
            quantum_membranes,
            oscillatory_dynamics,
            virtual_circulation,
            circuits,
            temporal_processing,
            atmospheric_processing,
            hardware_integration,
        })
    }
}

/// Current state of the intracellular environment
#[derive(Debug, Clone)]
pub struct IntracellularState {
    pub atp_concentration: f64,
    pub energy_charge: f64,
    pub bmd_activity: f64,
    pub quantum_coherence: f64,
    pub oscillatory_phase: f64,
    pub circulation_flow: f64,
    pub circuit_state: circuits::CircuitState,
}
//! Quantum Membrane Transport System
//! 
//! Environment-assisted quantum coherence at biological temperatures

pub mod coherence_manager;
pub mod ion_channels;
pub mod tunneling_transport;
pub mod decoherence_dynamics;

pub use coherence_manager::CoherenceManager;
pub use ion_channels::IonChannel;
pub use tunneling_transport::TunnelingTransport;
pub use decoherence_dynamics::DecoherenceDynamics;

use crate::error::{Error, Result};

/// Configuration for quantum membrane system
#[derive(Debug, Clone)]
pub struct MembraneConfig {
    pub quantum_enabled: bool,
    pub coherence_time: f64,
    pub temperature: f64,
    pub ion_channels: Vec<IonChannelConfig>,
    pub tunneling_enabled: bool,
}

impl Default for MembraneConfig {
    fn default() -> Self {
        Self::quantum_enabled()
    }
}

impl MembraneConfig {
    pub fn quantum_enabled() -> Self {
        Self {
            quantum_enabled: true,
            coherence_time: 1e-12, // 1 picosecond
            temperature: 310.0,
            ion_channels: vec![
                IonChannelConfig::sodium(),
                IonChannelConfig::potassium(),
                IonChannelConfig::calcium(),
            ],
            tunneling_enabled: true,
        }
    }

    pub fn classical() -> Self {
        Self {
            quantum_enabled: false,
            coherence_time: 0.0,
            temperature: 310.0,
            ion_channels: vec![],
            tunneling_enabled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IonChannelConfig {
    pub ion_type: String,
    pub conductance: f64,
    pub gating_voltage: f64,
}

impl IonChannelConfig {
    pub fn sodium() -> Self {
        Self {
            ion_type: "Na+".to_string(),
            conductance: 120.0, // mS/cm²
            gating_voltage: -55.0, // mV
        }
    }

    pub fn potassium() -> Self {
        Self {
            ion_type: "K+".to_string(),
            conductance: 36.0,
            gating_voltage: -12.0,
        }
    }

    pub fn calcium() -> Self {
        Self {
            ion_type: "Ca2+".to_string(),
            conductance: 1.0,
            gating_voltage: -30.0,
        }
    }
}

/// Quantum membrane transport system
#[derive(Debug)]
pub struct QuantumMembraneSystem {
    config: MembraneConfig,
    coherence_manager: CoherenceManager,
    ion_channels: Vec<IonChannel>,
    tunneling_transport: TunnelingTransport,
    decoherence_dynamics: DecoherenceDynamics,
    coherence_level: f64,
}

impl QuantumMembraneSystem {
    pub fn new(config: MembraneConfig) -> Result<Self> {
        let coherence_manager = CoherenceManager::new(config.coherence_time, config.temperature)?;
        
        let mut ion_channels = Vec::new();
        for channel_config in &config.ion_channels {
            ion_channels.push(IonChannel::new(channel_config.clone())?);
        }
        
        let tunneling_transport = TunnelingTransport::new(config.tunneling_enabled);
        let decoherence_dynamics = DecoherenceDynamics::new(config.temperature);

        Ok(Self {
            config,
            coherence_manager,
            ion_channels,
            tunneling_transport,
            decoherence_dynamics,
            coherence_level: 1.0,
        })
    }

    pub fn step(&mut self, dt: f64, atp_concentration: f64) -> Result<()> {
        if self.config.quantum_enabled {
            self.coherence_manager.update(dt)?;
            self.coherence_level = self.coherence_manager.coherence_level();
        }

        for channel in &mut self.ion_channels {
            channel.update(dt, atp_concentration)?;
        }

        self.tunneling_transport.update(dt, self.coherence_level)?;
        self.decoherence_dynamics.update(dt)?;

        Ok(())
    }

    pub fn is_coherent(&self) -> bool {
        self.coherence_level > 0.5
    }

    pub fn coherence_level(&self) -> f64 {
        self.coherence_level
    }

    pub fn transport_rate(&self, ion_type: &str) -> f64 {
        self.ion_channels.iter()
            .find(|ch| ch.ion_type() == ion_type)
            .map(|ch| ch.current_rate())
            .unwrap_or(0.0)
    }
}
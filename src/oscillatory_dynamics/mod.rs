//! Oscillatory Dynamics System
//! 
//! Multi-scale temporal dynamics and hardware coupling

pub mod hierarchical_network;
pub mod frequency_bands;
pub mod phase_coupling;
pub mod hardware_harvesting;

pub use hierarchical_network::HierarchicalNetwork;
pub use frequency_bands::FrequencyBands;
pub use phase_coupling::PhaseCoupling;
pub use hardware_harvesting::HardwareHarvesting;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct OscillatoryConfig {
    pub frequency_bands: Vec<FrequencyBand>,
    pub hardware_harvesting: bool,
    pub phase_coupling_enabled: bool,
    pub base_frequency: f64,
}

#[derive(Debug, Clone)]
pub struct FrequencyBand {
    pub name: String,
    pub min_freq: f64,
    pub max_freq: f64,
    pub amplitude: f64,
}

impl Default for OscillatoryConfig {
    fn default() -> Self {
        Self::biological()
    }
}

impl OscillatoryConfig {
    pub fn biological() -> Self {
        Self {
            frequency_bands: vec![
                FrequencyBand { name: "delta".to_string(), min_freq: 0.5, max_freq: 4.0, amplitude: 1.0 },
                FrequencyBand { name: "theta".to_string(), min_freq: 4.0, max_freq: 8.0, amplitude: 0.8 },
                FrequencyBand { name: "alpha".to_string(), min_freq: 8.0, max_freq: 13.0, amplitude: 0.6 },
                FrequencyBand { name: "beta".to_string(), min_freq: 13.0, max_freq: 30.0, amplitude: 0.4 },
                FrequencyBand { name: "gamma".to_string(), min_freq: 30.0, max_freq: 100.0, amplitude: 0.2 },
            ],
            hardware_harvesting: true,
            phase_coupling_enabled: true,
            base_frequency: 1000.0, // 1 kHz
        }
    }
}

#[derive(Debug)]
pub struct OscillatorySystem {
    config: OscillatoryConfig,
    hierarchical_network: HierarchicalNetwork,
    frequency_bands: FrequencyBands,
    phase_coupling: PhaseCoupling,
    hardware_harvesting: HardwareHarvesting,
    current_phase: f64,
    is_synchronized: bool,
}

impl OscillatorySystem {
    pub fn new(config: OscillatoryConfig) -> Result<Self> {
        let hierarchical_network = HierarchicalNetwork::new(&config.frequency_bands)?;
        let frequency_bands = FrequencyBands::new(config.frequency_bands.clone());
        let phase_coupling = PhaseCoupling::new(config.phase_coupling_enabled);
        let hardware_harvesting = HardwareHarvesting::new(config.hardware_harvesting);

        Ok(Self {
            config,
            hierarchical_network,
            frequency_bands,
            phase_coupling,
            hardware_harvesting,
            current_phase: 0.0,
            is_synchronized: false,
        })
    }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.hierarchical_network.update(dt)?;
        self.frequency_bands.update(dt)?;
        
        if self.config.phase_coupling_enabled {
            self.phase_coupling.update(dt, &mut self.frequency_bands)?;
        }
        
        if self.config.hardware_harvesting {
            self.hardware_harvesting.harvest_oscillations(dt)?;
        }
        
        self.current_phase += 2.0 * std::f64::consts::PI * self.config.base_frequency * dt;
        self.current_phase %= 2.0 * std::f64::consts::PI;
        
        self.is_synchronized = self.check_synchronization();
        
        Ok(())
    }

    pub fn current_phase(&self) -> f64 {
        self.current_phase
    }

    pub fn is_synchronized(&self) -> bool {
        self.is_synchronized
    }

    fn check_synchronization(&self) -> bool {
        self.frequency_bands.coherence_level() > 0.8
    }
}
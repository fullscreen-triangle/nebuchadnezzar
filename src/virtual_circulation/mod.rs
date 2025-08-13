//! Virtual Circulation System
//! 
//! Biologically-constrained noise distribution through concentration gradients

pub mod noise_stratification;
pub mod vessel_network;
pub mod flow_dynamics;
pub mod boundary_exchange;

pub use noise_stratification::NoiseStratification;
pub use vessel_network::VesselNetwork;
pub use flow_dynamics::FlowDynamics;
pub use boundary_exchange::BoundaryExchange;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct CirculationConfig {
    pub gradient_steepness: f64,
    pub max_concentration: f64,
    pub vessel_branching: bool,
    pub boundary_permeability: f64,
}

impl Default for CirculationConfig {
    fn default() -> Self {
        Self::gradient_optimized()
    }
}

impl CirculationConfig {
    pub fn gradient_optimized() -> Self {
        Self {
            gradient_steepness: 2.0,
            max_concentration: 1.0,
            vessel_branching: true,
            boundary_permeability: 0.1,
        }
    }
}

#[derive(Debug)]
pub struct VirtualCirculationSystem {
    config: CirculationConfig,
    noise_stratification: NoiseStratification,
    vessel_network: VesselNetwork,
    flow_dynamics: FlowDynamics,
    boundary_exchange: BoundaryExchange,
    current_flow: f64,
}

impl VirtualCirculationSystem {
    pub fn new(config: CirculationConfig) -> Result<Self> {
        Ok(Self {
            noise_stratification: NoiseStratification::new(config.gradient_steepness, config.max_concentration),
            vessel_network: VesselNetwork::new(config.vessel_branching),
            flow_dynamics: FlowDynamics::new(),
            boundary_exchange: BoundaryExchange::new(config.boundary_permeability),
            current_flow: 0.0,
            config,
        })
    }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        self.noise_stratification.update(dt)?;
        self.vessel_network.update(dt)?;
        self.flow_dynamics.update(dt)?;
        self.boundary_exchange.update(dt)?;
        
        self.current_flow = self.flow_dynamics.current_flow();
        Ok(())
    }

    pub fn current_flow(&self) -> f64 {
        self.current_flow
    }
}
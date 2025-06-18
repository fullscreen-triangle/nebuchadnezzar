//! # Ion Channel Circuits: Molecular-Level Probabilistic Switches
//! 
//! This module implements ion channels as probabilistic circuit elements that can
//! switch between states based on voltage, ligand binding, and ATP availability.

use crate::error::{NebuchadnezzarError, Result};
use crate::circuits::{Voltage, Current, Conductance};
use crate::utils::numerical::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Probabilistic ion channel - the fundamental circuit element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticIonChannel {
    pub name: String,
    pub channel_type: ChannelType,
    pub max_conductance: f64,  // pS
    pub reversal_potential: f64, // mV
    pub gating_variables: Vec<GatingVariable>,
    pub atp_dependence: Option<AtpDependence>,
    pub current_state: ChannelState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    /// Voltage-gated channels (Na+, K+, Ca2+)
    VoltageGated {
        activation_threshold: f64,  // mV
        inactivation_threshold: Option<f64>, // mV
    },
    /// Ligand-gated channels (GABA, glutamate, etc.)
    LigandGated {
        ligand_name: String,
        binding_sites: usize,
        cooperativity: f64,
    },
    /// Mechanosensitive channels
    Mechanosensitive {
        pressure_threshold: f64,
        adaptation_rate: f64,
    },
    /// ATP-sensitive channels (KATP)
    AtpSensitive {
        atp_inhibition: bool,
        sensitivity: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingVariable {
    pub name: String,
    pub value: f64,  // [0,1]
    pub alpha: f64,  // Opening rate
    pub beta: f64,   // Closing rate
    pub power: f64,  // Exponent in conductance calculation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpDependence {
    pub km_atp: f64,
    pub hill_coefficient: f64,
    pub inhibitory: bool,  // true if ATP inhibits, false if ATP activates
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelState {
    pub open_probability: f64,
    pub conductance: f64,  // Current conductance in pS
    pub last_update_time: f64,
    pub state_history: Vec<(f64, f64)>, // (time, open_probability)
}

impl ProbabilisticIonChannel {
    pub fn new(name: String, channel_type: ChannelType, max_conductance: f64, reversal_potential: f64) -> Self {
        Self {
            name,
            channel_type,
            max_conductance,
            reversal_potential,
            gating_variables: Vec::new(),
            atp_dependence: None,
            current_state: ChannelState {
                open_probability: 0.0,
                conductance: 0.0,
                last_update_time: 0.0,
                state_history: Vec::new(),
            },
        }
    }

    /// Create a voltage-gated sodium channel
    pub fn sodium_channel() -> Self {
        let mut channel = Self::new(
            "Nav1.1".to_string(),
            ChannelType::VoltageGated {
                activation_threshold: -55.0,
                inactivation_threshold: Some(-30.0),
            },
            20.0,  // 20 pS max conductance
            50.0,  // +50 mV reversal potential
        );

        // Add m gating variable (activation)
        channel.gating_variables.push(GatingVariable {
            name: "m".to_string(),
            value: 0.0,
            alpha: 0.0,
            beta: 0.0,
            power: 3.0,
        });

        // Add h gating variable (inactivation)
        channel.gating_variables.push(GatingVariable {
            name: "h".to_string(),
            value: 1.0,
            alpha: 0.0,
            beta: 0.0,
            power: 1.0,
        });

        channel
    }

    /// Create a voltage-gated potassium channel
    pub fn potassium_channel() -> Self {
        let mut channel = Self::new(
            "Kv1.1".to_string(),
            ChannelType::VoltageGated {
                activation_threshold: -50.0,
                inactivation_threshold: None,
            },
            36.0,  // 36 pS max conductance
            -77.0, // -77 mV reversal potential
        );

        // Add n gating variable
        channel.gating_variables.push(GatingVariable {
            name: "n".to_string(),
            value: 0.0,
            alpha: 0.0,
            beta: 0.0,
            power: 4.0,
        });

        channel
    }

    /// Create an ATP-sensitive potassium channel
    pub fn katp_channel() -> Self {
        let mut channel = Self::new(
            "KATP".to_string(),
            ChannelType::AtpSensitive {
                atp_inhibition: true,
                sensitivity: 0.1,
            },
            80.0,  // High conductance when open
            -85.0, // K+ reversal potential
        );

        channel.atp_dependence = Some(AtpDependence {
            km_atp: 0.01,  // 10 μM
            hill_coefficient: 1.0,
            inhibitory: true,
        });

        channel
    }

    /// Update channel state based on voltage and ATP
    pub fn update_state(&mut self, voltage: f64, atp_concentration: f64, dt: f64) -> Result<()> {
        // Update gating variables
        for gating_var in &mut self.gating_variables {
            let (alpha, beta) = self.calculate_rate_constants(gating_var, voltage)?;
            gating_var.alpha = alpha;
            gating_var.beta = beta;

            // Update gating variable: dm/dt = alpha*(1-m) - beta*m
            let dm_dt = alpha * (1.0 - gating_var.value) - beta * gating_var.value;
            gating_var.value += dm_dt * dt;
            gating_var.value = clamp(gating_var.value, 0.0, 1.0);
        }

        // Calculate open probability from gating variables
        let mut open_prob = 1.0;
        for gating_var in &self.gating_variables {
            open_prob *= gating_var.value.powf(gating_var.power);
        }

        // Apply ATP dependence if present
        if let Some(atp_dep) = &self.atp_dependence {
            let atp_factor = self.calculate_atp_factor(atp_concentration, atp_dep);
            if atp_dep.inhibitory {
                open_prob *= (1.0 - atp_factor);
            } else {
                open_prob *= atp_factor;
            }
        }

        // Update state
        self.current_state.open_probability = open_prob;
        self.current_state.conductance = self.max_conductance * open_prob;
        self.current_state.last_update_time += dt;

        // Store history
        self.current_state.state_history.push((
            self.current_state.last_update_time,
            open_prob,
        ));

        // Limit history size
        if self.current_state.state_history.len() > 1000 {
            self.current_state.state_history.remove(0);
        }

        Ok(())
    }

    /// Calculate current through the channel
    pub fn calculate_current(&self, voltage: f64) -> Current {
        let driving_force = voltage - self.reversal_potential;
        let current = self.current_state.conductance * driving_force;
        Current(current)
    }

    /// Check if channel can fire (probabilistic switching)
    pub fn can_fire(&self, atp_concentration: f64) -> bool {
        match &self.channel_type {
            ChannelType::AtpSensitive { sensitivity, .. } => {
                atp_concentration > *sensitivity
            },
            _ => self.current_state.open_probability > 0.1,
        }
    }

    /// Calculate ATP cost for channel operation
    pub fn atp_cost(&self) -> f64 {
        match &self.channel_type {
            ChannelType::AtpSensitive { .. } => 0.0, // Passive
            ChannelType::VoltageGated { .. } => 0.001, // Small cost for gating
            ChannelType::LigandGated { .. } => 0.005, // Cost for ligand binding
            ChannelType::Mechanosensitive { .. } => 0.002, // Mechanical work
        }
    }

    /// Resolve to detailed circuit when needed
    pub fn resolve_to_circuit(&self) -> DetailedChannelCircuit {
        DetailedChannelCircuit {
            base_channel: self.clone(),
            subunit_states: self.generate_subunit_states(),
            kinetic_model: self.build_kinetic_model(),
        }
    }

    // Private helper methods
    fn calculate_rate_constants(&self, gating_var: &GatingVariable, voltage: f64) -> Result<(f64, f64)> {
        match gating_var.name.as_str() {
            "m" => {
                // Sodium activation
                let alpha = 0.1 * (voltage + 40.0) / (1.0 - (-0.1 * (voltage + 40.0)).exp());
                let beta = 4.0 * (-(voltage + 65.0) / 18.0).exp();
                Ok((alpha, beta))
            },
            "h" => {
                // Sodium inactivation
                let alpha = 0.07 * (-(voltage + 65.0) / 20.0).exp();
                let beta = 1.0 / (1.0 + (-(voltage + 35.0) / 10.0).exp());
                Ok((alpha, beta))
            },
            "n" => {
                // Potassium activation
                let alpha = 0.01 * (voltage + 55.0) / (1.0 - (-0.1 * (voltage + 55.0)).exp());
                let beta = 0.125 * (-(voltage + 65.0) / 80.0).exp();
                Ok((alpha, beta))
            },
            _ => Err(NebuchadnezzarError::ComputationError(
                format!("Unknown gating variable: {}", gating_var.name)
            )),
        }
    }

    fn calculate_atp_factor(&self, atp_conc: f64, atp_dep: &AtpDependence) -> f64 {
        hill(atp_conc, atp_dep.km_atp, atp_dep.hill_coefficient)
    }

    fn generate_subunit_states(&self) -> Vec<SubunitState> {
        // Generate detailed subunit states for high-resolution simulation
        vec![
            SubunitState {
                subunit_id: 0,
                conformation: "closed".to_string(),
                energy_state: 0.0,
                transition_rates: HashMap::new(),
            }
        ]
    }

    fn build_kinetic_model(&self) -> KineticModel {
        KineticModel {
            states: vec!["closed".to_string(), "open".to_string()],
            transitions: HashMap::new(),
            current_state: 0,
        }
    }
}

/// Detailed circuit representation for high-resolution simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedChannelCircuit {
    pub base_channel: ProbabilisticIonChannel,
    pub subunit_states: Vec<SubunitState>,
    pub kinetic_model: KineticModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubunitState {
    pub subunit_id: usize,
    pub conformation: String,
    pub energy_state: f64,
    pub transition_rates: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KineticModel {
    pub states: Vec<String>,
    pub transitions: HashMap<(usize, usize), f64>,
    pub current_state: usize,
}

/// Ion channel factory for creating common channel types
pub struct IonChannelFactory;

impl IonChannelFactory {
    pub fn create_voltage_gated_sodium() -> ProbabilisticIonChannel {
        ProbabilisticIonChannel::sodium_channel()
    }

    pub fn create_voltage_gated_potassium() -> ProbabilisticIonChannel {
        ProbabilisticIonChannel::potassium_channel()
    }

    pub fn create_atp_sensitive_potassium() -> ProbabilisticIonChannel {
        ProbabilisticIonChannel::katp_channel()
    }

    pub fn create_calcium_channel() -> ProbabilisticIonChannel {
        ProbabilisticIonChannel::new(
            "Cav1.2".to_string(),
            ChannelType::VoltageGated {
                activation_threshold: -40.0,
                inactivation_threshold: Some(-20.0),
            },
            25.0,  // 25 pS
            120.0, // +120 mV Ca2+ reversal potential
        )
    }

    pub fn create_ligand_gated_channel(ligand: String, binding_sites: usize) -> ProbabilisticIonChannel {
        ProbabilisticIonChannel::new(
            format!("LGC_{}", ligand),
            ChannelType::LigandGated {
                ligand_name: ligand,
                binding_sites,
                cooperativity: 2.0,
            },
            50.0,  // Variable conductance
            0.0,   // Non-selective
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sodium_channel_creation() {
        let channel = IonChannelFactory::create_voltage_gated_sodium();
        assert_eq!(channel.name, "Nav1.1");
        assert_eq!(channel.gating_variables.len(), 2);
    }

    #[test]
    fn test_channel_state_update() {
        let mut channel = IonChannelFactory::create_voltage_gated_sodium();
        let result = channel.update_state(-70.0, 5.0, 0.1);
        assert!(result.is_ok());
        assert!(channel.current_state.open_probability >= 0.0);
        assert!(channel.current_state.open_probability <= 1.0);
    }

    #[test]
    fn test_current_calculation() {
        let channel = IonChannelFactory::create_voltage_gated_sodium();
        let current = channel.calculate_current(-70.0);
        assert!(current.0.is_finite());
    }
} 
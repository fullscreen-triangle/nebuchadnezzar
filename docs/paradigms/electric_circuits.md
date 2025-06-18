`// src/circuits/ion_channel.rs
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct IonChannel {
    pub channel_type: ChannelType,
    pub conductance: f64,        // Siemens (S)
    pub reversal_potential: f64, // mV
    pub open_probability: f64,   // 0.0 to 1.0
    pub gating_variables: HashMap<String, f64>,
    pub membrane_voltage: f64,   // mV
}

#[derive(Debug, Clone)]
pub enum ChannelType {
    Sodium,
    Potassium,
    Calcium,
    Chloride,
    Leak,
}

impl IonChannel {
    pub fn new(channel_type: ChannelType, max_conductance: f64, reversal_potential: f64) -> Self {
        Self {
            channel_type,
            conductance: max_conductance,
            reversal_potential,
            open_probability: 0.0,
            gating_variables: HashMap::new(),
            membrane_voltage: -70.0, // resting potential
        }
    }

    pub fn update_gating(&mut self, voltage: f64, dt: f64) {
        self.membrane_voltage = voltage;
        
        match self.channel_type {
            ChannelType::Sodium => self.update_sodium_gating(voltage, dt),
            ChannelType::Potassium => self.update_potassium_gating(voltage, dt),
            ChannelType::Calcium => self.update_calcium_gating(voltage, dt),
            ChannelType::Chloride => self.update_chloride_gating(voltage, dt),
            ChannelType::Leak => self.open_probability = 1.0, // Always open
        }
    }

    fn update_sodium_gating(&mut self, v: f64, dt: f64) {
        // Hodgkin-Huxley sodium channel kinetics
        let alpha_m = 0.1 * (v + 40.0) / (1.0 - (-0.1 * (v + 40.0)).exp());
        let beta_m = 4.0 * (-0.0556 * (v + 65.0)).exp();
        let alpha_h = 0.07 * (-0.05 * (v + 65.0)).exp();
        let beta_h = 1.0 / (1.0 + (-0.1 * (v + 35.0)).exp());

        let m = self.gating_variables.get("m").unwrap_or(&0.0);
        let h = self.gating_variables.get("h").unwrap_or(&1.0);

        let dm_dt = alpha_m * (1.0 - m) - beta_m * m;
        let dh_dt = alpha_h * (1.0 - h) - beta_h * h;

        let new_m = m + dm_dt * dt;
        let new_h = h + dh_dt * dt;

        self.gating_variables.insert("m".to_string(), new_m.clamp(0.0, 1.0));
        self.gating_variables.insert("h".to_string(), new_h.clamp(0.0, 1.0));

        self.open_probability = new_m.powi(3) * new_h;
    }

    fn update_potassium_gating(&mut self, v: f64, dt: f64) {
        // Hodgkin-Huxley potassium channel kinetics
        let alpha_n = 0.01 * (v + 55.0) / (1.0 - (-0.1 * (v + 55.0)).exp());
        let beta_n = 0.125 * (-0.0125 * (v + 65.0)).exp();

        let n = self.gating_variables.get("n").unwrap_or(&0.0);
        let dn_dt = alpha_n * (1.0 - n) - beta_n * n;
        let new_n = n + dn_dt * dt;

        self.gating_variables.insert("n".to_string(), new_n.clamp(0.0, 1.0));
        self.open_probability = new_n.powi(4);
    }

    fn update_calcium_gating(&mut self, v: f64, dt: f64) {
        // Simplified L-type calcium channel
        let steady_state = 1.0 / (1.0 + (-(v + 10.0) / 10.0).exp());
        let tau = 5.0; // ms
        
        let current_state = self.gating_variables.get("d").unwrap_or(&0.0);
        let new_state = current_state + (steady_state - current_state) * dt / tau;
        
        self.gating_variables.insert("d".to_string(), new_state.clamp(0.0, 1.0));
        self.open_probability = new_state;
    }

    fn update_chloride_gating(&mut self, v: f64, _dt: f64) {
        // Voltage-independent chloride channel
        self.open_probability = 1.0 / (1.0 + ((v + 50.0) / 20.0).exp());
    }

    pub fn calculate_current(&self, voltage: f64) -> f64 {
        // I = g * P_open * (V - E_rev)
        self.conductance * self.open_probability * (voltage - self.reversal_potential)
    }
}
`

`// src/circuits/membrane.rs
use crate::circuits::ion_channel::{IonChannel, ChannelType};
use std::collections::HashMap;

#[derive(Debug)]
pub struct MembraneCircuit {
    pub capacitance: f64,           // μF/cm²
    pub voltage: f64,               // mV
    pub channels: Vec<IonChannel>,
    pub external_current: f64,      // μA/cm²
    pub temperature: f64,           // Kelvin
}

impl MembraneCircuit {
    pub fn new(capacitance: f64) -> Self {
        let mut circuit = Self {
            capacitance,
            voltage: -70.0, // resting potential
            channels: Vec::new(),
            external_current: 0.0,
            temperature: 310.0, // 37°C in Kelvin
        };

        // Add standard channels
        circuit.add_channel(IonChannel::new(ChannelType::Sodium, 120.0, 50.0));
        circuit.add_channel(IonChannel::new(ChannelType::Potassium, 36.0, -77.0));
        circuit.add_channel(IonChannel::new(ChannelType::Leak, 0.3, -54.4));

        circuit
    }

    pub fn add_channel(&mut self, channel: IonChannel) {
        self.channels.push(channel);
    }

    pub fn step(&mut self, dt: f64) {
        // Update all channel gating variables
        for channel in &mut self.channels {
            channel.update_gating(self.voltage, dt);
        }

        // Calculate total membrane current
        let total_current = self.calculate_total_current();

        // Update membrane voltage using capacitor equation: C * dV/dt = -I
        let dv_dt = -(total_current + self.external_current) / self.capacitance;
        self.voltage += dv_dt * dt;
    }

    fn calculate_total_current(&self) -> f64 {
        self.channels
            .iter()
            .map(|channel| channel.calculate_current(self.voltage))
            .sum()
    }

    pub fn set_external_current(&mut self, current: f64) {
        self.external_current = current;
    }

    pub fn get_channel_currents(&self) -> HashMap<String, f64> {
        let mut currents = HashMap::new();
        for (i, channel) in self.channels.iter().enumerate() {
            let current = channel.calculate_current(self.voltage);
            currents.insert(format!("{:?}_{}", channel.channel_type, i), current);
        }
        currents
    }
}
`

`// src/circuits/network.rs
use crate::circuits::membrane::MembraneCircuit;
use std::collections::HashMap;

#[derive(Debug)]
pub struct CircuitNetwork {
    pub membranes: HashMap<String, MembraneCircuit>,
    pub connections: Vec<Connection>,
    pub time: f64,
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub from: String,
    pub to: String,
    pub conductance: f64,    // Gap junction conductance
    pub delay: f64,          // Propagation delay (ms)
}

impl CircuitNetwork {
    pub fn new() -> Self {
        Self {
            membranes: HashMap::new(),
            connections: Vec::new(),
            time: 0.0,
        }
    }

    pub fn add_membrane(&mut self, id: String, membrane: MembraneCircuit) {
        self.membranes.insert(id, membrane);
    }

    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.push(connection);
    }

    pub fn step(&mut self, dt: f64) {
        // Calculate coupling currents
        let coupling_currents = self.calculate_coupling_currents();

        // Update each membrane with coupling current
        for (id, membrane) in &mut self.membranes {
            if let Some(coupling_current) = coupling_currents.get(id) {
                membrane.set_external_current(*coupling_current);
            }
            membrane.step(dt);
        }

        self.time += dt;
    }

    fn calculate_coupling_currents(&self) -> HashMap<String, f64> {
        let mut currents: HashMap<String, f64> = HashMap::new();

        for connection in &self.connections {
            if let (Some(from_membrane), Some(to_membrane)) = (
                self.membranes.get(&connection.from),
                self.membranes.get(&connection.to),
            ) {
                let voltage_diff = from_membrane.voltage - to_membrane.voltage;
                let coupling_current = connection.conductance * voltage_diff;

                // Add current to 'to' membrane, subtract from 'from' membrane
                *currents.entry(connection.to.clone()).or_insert(0.0) += coupling_current;
                *currents.entry(connection.from.clone()).or_insert(0.0) -= coupling_current;
            }
        }

        currents
    }

    pub fn get_network_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        for (id, membrane) in &self.membranes {
            state.insert(format!("{}_voltage", id), membrane.voltage);
        }
        state.insert("time".to_string(), self.time);
        state
    }
}
`
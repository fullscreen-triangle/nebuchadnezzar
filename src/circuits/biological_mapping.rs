//! Biological to circuit element mapping

use crate::error::{Error, Result};

#[derive(Debug)]
pub struct BiologicalMapping {
    enabled: bool,
    mappings: Vec<ElementMapping>,
}

#[derive(Debug)]
struct ElementMapping {
    biological_component: String,
    circuit_element: String,
    conversion_factor: f64,
}

impl BiologicalMapping {
    pub fn new(enabled: bool) -> Self {
        let mappings = vec![
            ElementMapping {
                biological_component: "molecular_transport".to_string(),
                circuit_element: "resistor".to_string(),
                conversion_factor: 1e6, // Ω
            },
            ElementMapping {
                biological_component: "enzymatic_reaction".to_string(),
                circuit_element: "capacitor".to_string(),
                conversion_factor: 1e-12, // F
            },
        ];
        Self { enabled, mappings }
    }

    pub fn map_resistance(&self, molecular_transport_rate: f64) -> f64 {
        if !self.enabled { return 1e6; }
        1.0 / molecular_transport_rate.max(1e-12)
    }

    pub fn map_capacitance(&self, enzymatic_rate: f64) -> f64 {
        if !self.enabled { return 1e-12; }
        enzymatic_rate * 1e-12
    }
}
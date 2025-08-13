//! Integration with External Systems
//! 
//! Interfaces for Autobahn, Bene Gesserit, Imhotep, and consciousness integration

pub mod autobahn_interface;
pub mod bene_gesserit_interface;
pub mod imhotep_interface;
pub mod consciousness_bridge;

pub use autobahn_interface::AutobahnInterface;
pub use bene_gesserit_interface::BeneGesseritInterface;
pub use imhotep_interface::ImhotepInterface;
pub use consciousness_bridge::ConsciousnessBridge;

use crate::error::{Error, Result};

/// Knowledge query for Autobahn integration
#[derive(Debug, Clone)]
pub struct KnowledgeQuery {
    pub query_text: String,
    pub complexity: f64,
    pub context: Vec<String>,
}

/// Knowledge response from Autobahn
#[derive(Debug, Clone)]
pub struct KnowledgeResponse {
    pub content: String,
    pub confidence: f64,
    pub processing_time: f64,
}

/// Membrane state for Bene Gesserit integration
#[derive(Debug, Clone)]
pub struct MembraneState {
    pub transport_efficiency: f64,
    pub quantum_coherence: f64,
    pub ion_gradients: Vec<f64>,
}

/// Neural state for Imhotep integration
#[derive(Debug, Clone)]
pub struct NeuralState {
    pub activation_level: f64,
    pub connection_strength: f64,
    pub plasticity: f64,
}

/// Consciousness level assessment
#[derive(Debug, Clone)]
pub struct ConsciousnessLevel {
    pub emergence_detected: bool,
    pub complexity_score: f64,
    pub integration_level: f64,
}
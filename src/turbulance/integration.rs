//! # Turbulance-Nebuchadnezzar Integration
//!
//! This module provides integration between Turbulance language constructs
//! and Nebuchadnezzar's biological simulation systems.

use crate::{
    AtpPool, OscillationState, QuantumMembrane, CircuitGrid, BiologicalQuantumState,
    BiologicalQuantumComputerSolver, BiologicalQuantumResult, AtpCoordinates,
    OscillatoryCoordinates, MembraneQuantumCoordinates, HierarchicalCircuit,
    EntropyManipulator, BiologicalOscillator, BiologicalMaxwellDemons,
    MaxwellDemonNetwork, NebuchadnezzarError, Result,
    oscillatory_dynamics::{UniversalOscillator, OscillatorParameters},
    error::BiologicalErrorType,
};
use crate::turbulance::{TurbulanceValue, BiologicalDataValue, PatternData, PatternMatch};
use std::collections::HashMap;

/// Integration bridge between Turbulance and Nebuchadnezzar
pub struct NebuIntegration {
    /// Current biological quantum state
    quantum_state: Option<BiologicalQuantumState>,
    
    /// ATP pool for energy calculations
    atp_pool: AtpPool,
    
    /// Current oscillatory state
    oscillation_state: OscillationState,
    
    /// Quantum membrane system
    quantum_membrane: QuantumMembrane,
    
    /// Circuit grid for biological computations
    circuit_grid: CircuitGrid,
    
    /// Biological oscillator
    oscillator: UniversalOscillator,
    
    /// Maxwell demons network
    maxwell_demons: MaxwellDemonNetwork,
    
    /// Pattern recognition system
    pattern_recognizer: PatternRecognizer,
    
    /// Experiment tracking
    experiments: HashMap<String, ExperimentState>,
    
    /// Variables exposed to Turbulance
    variables: HashMap<String, TurbulanceValue>,
}

impl NebuIntegration {
    /// Create a new integration instance
    pub fn new() -> Self {
        let oscillator_params = OscillatorParameters {
            natural_frequency: 1.0,
            damping_coefficient: 0.1,
            driving_amplitude: 0.5,
            driving_frequency: 1.0,
            nonlinearity_strength: 0.1,
            mass: 1.0,
            spring_constant: 1.0,
        };

        Self {
            quantum_state: None,
            atp_pool: AtpPool::new(100.0), // Initial ATP
            oscillation_state: OscillationState::new("main_osc", 1.0, 0.0, 1.0),
            quantum_membrane: QuantumMembrane::new(310.0), // Body temperature
            circuit_grid: CircuitGrid::new("main_grid".to_string(), 100.0),
            oscillator: UniversalOscillator::new("main_oscillator".to_string(), oscillator_params),
            maxwell_demons: MaxwellDemonNetwork::new(),
            pattern_recognizer: PatternRecognizer::new(),
            experiments: HashMap::new(),
            variables: HashMap::new(),
        }
    }

    /// Initialize a biological experiment from Turbulance proposition
    pub fn initialize_experiment(&mut self, name: &str, description: &str) -> Result<String> {
        let experiment_id = format!("exp_{}", self.experiments.len());
        
        let experiment = ExperimentState {
            id: experiment_id.clone(),
            name: name.to_string(),
            description: description.to_string(),
            start_time: std::time::SystemTime::now(),
            quantum_state: BiologicalQuantumState::new(),
            measurements: Vec::new(),
            hypotheses: Vec::new(),
            evidence: Vec::new(),
            status: ExperimentStatus::Running,
        };
        
        self.experiments.insert(experiment_id.clone(), experiment);
        
        // Initialize quantum state for this experiment
        self.quantum_state = Some(BiologicalQuantumState::new());
        
        // Set up initial ATP coordinates
        let atp_coords = AtpCoordinates::new(
            self.atp_pool.available_atp(),
            self.atp_pool.available_atp() * 0.1,
            5.0
        );
        
        // Set up oscillatory coordinates
        let osc_coords = OscillatoryCoordinates::new(5);
        
        // Set up membrane coordinates
        let membrane_coords = MembraneQuantumCoordinates::new(10);
        
        // Update quantum state
        if let Some(ref mut state) = self.quantum_state {
            state.set_atp_coordinates(atp_coords);
            state.set_oscillatory_coordinates(osc_coords);
            state.set_membrane_coordinates(membrane_coords);
        }
        
        Ok(experiment_id)
    }

    /// Execute a Turbulance proposition as a biological hypothesis test
    pub fn test_proposition(&mut self, name: &str, motions: &[String]) -> Result<PropositionResult> {
        let experiment_id = self.get_current_experiment_id()?;
        
        let mut motion_results = Vec::new();
        let mut total_support = 0.0;
        
        for motion in motions {
            let result = self.evaluate_motion(motion)?;
            total_support += result.support_level;
            motion_results.push(result);
        }
        
        let overall_support = total_support / motions.len() as f64;
        
        // Record the proposition test
        if let Some(experiment) = self.experiments.get_mut(&experiment_id) {
            experiment.hypotheses.push(HypothesisTest {
                proposition_name: name.to_string(),
                motions: motions.to_vec(),
                support_level: overall_support,
                timestamp: std::time::SystemTime::now(),
            });
        }
        
        Ok(PropositionResult {
            name: name.to_string(),
            motions: motion_results,
            overall_support,
            confidence: self.calculate_confidence(overall_support),
            evidence_count: self.count_evidence_for_proposition(name),
        })
    }

    /// Evaluate a motion within the biological context
    pub fn evaluate_motion(&mut self, motion: &str) -> Result<MotionResult> {
        // Analyze the motion in terms of biological patterns
        let pattern_matches = self.pattern_recognizer.find_patterns(motion)?;
        
        // Calculate support based on biological evidence
        let support_level = self.calculate_biological_support(&pattern_matches)?;
        
        // Determine status based on support level
        let status = if support_level > 0.7 {
            MotionStatus::Supported
        } else if support_level < 0.3 {
            MotionStatus::Contradicted
        } else if support_level > 0.4 {
            MotionStatus::Inconclusive
        } else {
            MotionStatus::InsufficientEvidence
        };
        
        // Collect evidence strings
        let evidence: Vec<String> = pattern_matches.iter()
            .map(|m| format!("Pattern match: {} (score: {:.2})", m.context, m.score))
            .collect();
        
        Ok(MotionResult {
            name: motion.to_string(),
            description: format!("Biological evaluation of motion: {}", motion),
            support_level,
            evidence,
            status,
        })
    }

    /// Collect evidence from biological data
    pub fn collect_evidence(&mut self, source: &str, _data_type: &str) -> Result<EvidenceResult> {
        let evidence = match source {
            "atp_pool" => self.collect_atp_evidence()?,
            "oscillation_state" => self.collect_oscillation_evidence()?,
            "quantum_membrane" => self.collect_membrane_evidence()?,
            "circuit_grid" => self.collect_circuit_evidence()?,
            "maxwell_demons" => self.collect_demon_evidence()?,
            _ => return Err(NebuchadnezzarError::invalid_input(format!("Unknown evidence source: {}", source))),
        };
        
        Ok(evidence)
    }

    /// Set a variable value from Turbulance
    pub fn set_variable(&mut self, name: &str, value: TurbulanceValue) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable value for Turbulance
    pub fn get_variable(&self, name: &str) -> Option<&TurbulanceValue> {
        self.variables.get(name)
    }

    /// Execute a biological function call from Turbulance
    pub fn call_biological_function(&mut self, function: &str, args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        match function {
            "analyze_atp_dynamics" => self.analyze_atp_dynamics(args),
            "simulate_oscillation" => self.simulate_oscillation(args),
            "quantum_membrane_transport" => self.quantum_membrane_transport(args),
            "run_maxwell_demon" => self.run_maxwell_demon(args),
            "calculate_entropy" => self.calculate_entropy(args),
            "optimize_circuit" => self.optimize_circuit(args),
            "measure_coherence" => self.measure_coherence(args),
            "track_pattern" => self.track_pattern(args),
            _ => Err(NebuchadnezzarError::invalid_input(format!("Unknown function: {}", function))),
        }
    }

    /// Convert Nebuchadnezzar data to Turbulance values
    pub fn to_turbulance_value(&self, data: &BiologicalData) -> TurbulanceValue {
        match data {
            BiologicalData::AtpPool(pool) => {
                TurbulanceValue::BiologicalData(BiologicalDataValue::AtpPool(pool.clone()))
            }
            BiologicalData::OscillationState(state) => {
                TurbulanceValue::BiologicalData(BiologicalDataValue::OscillationState(state.clone()))
            }
            BiologicalData::QuantumMembrane(membrane) => {
                TurbulanceValue::BiologicalData(BiologicalDataValue::QuantumMembrane(membrane.clone()))
            }
            BiologicalData::CircuitGrid(grid) => {
                TurbulanceValue::BiologicalData(BiologicalDataValue::CircuitGrid(grid.clone()))
            }
            BiologicalData::TimeSeries(data) => {
                TurbulanceValue::BiologicalData(BiologicalDataValue::TimeSeries(data.clone()))
            }
            BiologicalData::Sequence(seq) => {
                TurbulanceValue::BiologicalData(BiologicalDataValue::Sequence(seq.clone()))
            }
        }
    }

    // Private helper methods

    fn get_current_experiment_id(&self) -> Result<String> {
        self.experiments.keys()
            .last()
            .cloned()
            .ok_or_else(|| NebuchadnezzarError::invalid_input("No active experiment".to_string()))
    }

    fn calculate_confidence(&self, support_level: f64) -> f64 {
        // Use sigmoid function to map support to confidence
        1.0 / (1.0 + (-5.0 * (support_level - 0.5)).exp())
    }

    fn count_evidence_for_proposition(&self, _proposition: &str) -> usize {
        // Count evidence collected for this proposition
        // In a real implementation, this would track evidence by proposition
        self.experiments.values()
            .map(|exp| exp.evidence.len())
            .sum()
    }

    fn calculate_biological_support(&mut self, patterns: &[PatternMatch]) -> Result<f64> {
        if patterns.is_empty() {
            return Ok(0.0);
        }
        
        let total_score: f64 = patterns.iter().map(|p| p.score).sum();
        let avg_score = total_score / patterns.len() as f64;
        
        // Weight by ATP availability (biological relevance)
        let atp_weight = (self.atp_pool.available_atp() / 100.0).min(1.0);
        
        Ok(avg_score * atp_weight)
    }

    fn collect_atp_evidence(&self) -> Result<EvidenceResult> {
        let patterns = vec![
            format!("ATP concentration: {:.2} mM", self.atp_pool.available_atp()),
            format!("Energy efficiency: {:.2}%", self.atp_pool.efficiency() * 100.0),
        ];
        
        Ok(EvidenceResult {
            source: "atp_pool".to_string(),
            data_type: "biochemical".to_string(),
            quality: 0.9,
            relevance: 0.95,
            patterns,
        })
    }

    fn collect_oscillation_evidence(&self) -> Result<EvidenceResult> {
        let patterns = vec![
            format!("Oscillation frequency: {:.2} Hz", self.oscillator.frequency()),
            format!("Amplitude: {:.2}", self.oscillator.amplitude()),
            format!("Phase: {:.2} rad", self.oscillator.phase()),
        ];
        
        Ok(EvidenceResult {
            source: "oscillation_state".to_string(),
            data_type: "temporal".to_string(),
            quality: 0.85,
            relevance: 0.8,
            patterns,
        })
    }

    fn collect_membrane_evidence(&self) -> Result<EvidenceResult> {
        let patterns = vec![
            format!("Membrane potential: {:.2} mV", self.quantum_membrane.membrane_potential),
            format!("Permeability: {:.2e} cm/s", self.quantum_membrane.permeability()),
        ];
        
        Ok(EvidenceResult {
            source: "quantum_membrane".to_string(),
            data_type: "quantum".to_string(),
            quality: 0.8,
            relevance: 0.9,
            patterns,
        })
    }

    fn collect_circuit_evidence(&self) -> Result<EvidenceResult> {
        let patterns = vec![
            format!("Circuit nodes: {}", self.circuit_grid.node_count()),
            format!("Active connections: {}", self.circuit_grid.connection_count()),
        ];
        
        Ok(EvidenceResult {
            source: "circuit_grid".to_string(),
            data_type: "computational".to_string(),
            quality: 0.9,
            relevance: 0.85,
            patterns,
        })
    }

    fn collect_demon_evidence(&self) -> Result<EvidenceResult> {
        let patterns = vec![
            format!("Active demons: {}", self.maxwell_demons.active_count()),
            format!("Entropy reduction: {:.2} J/K", self.maxwell_demons.total_entropy_reduction()),
        ];
        
        Ok(EvidenceResult {
            source: "maxwell_demons".to_string(),
            data_type: "thermodynamic".to_string(),
            quality: 0.75,
            relevance: 0.9,
            patterns,
        })
    }

    // Biological function implementations

    fn analyze_atp_dynamics(&mut self, args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let time_span = match args.get(0) {
            Some(TurbulanceValue::Float(t)) => *t,
            _ => 1.0, // Default 1 second
        };
        
        // Simulate ATP dynamics over time
        let mut time_series = Vec::new();
        let dt = 0.1;
        let steps = (time_span / dt) as usize;
        
        for i in 0..steps {
            let t = i as f64 * dt;
            let atp_level = self.atp_pool.available_atp() * (1.0 + 0.1 * (2.0 * std::f64::consts::PI * t).sin());
            time_series.push((t, atp_level));
        }
        
        Ok(TurbulanceValue::BiologicalData(
            BiologicalDataValue::TimeSeries(time_series)
        ))
    }

    fn simulate_oscillation(&mut self, args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let duration = match args.get(0) {
            Some(TurbulanceValue::Float(d)) => *d,
            _ => 1.0, // Default 1 second
        };
        
        // Evolve oscillator
        self.oscillator.evolve(duration)?;
        
        Ok(TurbulanceValue::BiologicalData(
            BiologicalDataValue::OscillationState(self.oscillation_state.clone())
        ))
    }

    fn quantum_membrane_transport(&mut self, args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let _transport_amount = match args.get(0) {
            Some(TurbulanceValue::Float(amount)) => *amount,
            _ => 1.0, // Default transport amount
        };
        
        let transport_rate = self.quantum_membrane.calculate_transport_rate();
        
        Ok(TurbulanceValue::Float(transport_rate))
    }

    fn run_maxwell_demon(&mut self, _args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        // Simplified Maxwell demon operation
        let entropy_reduction = self.maxwell_demons.total_entropy_reduction();
        
        Ok(TurbulanceValue::Float(entropy_reduction))
    }

    fn calculate_entropy(&mut self, _args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let total_entropy = if let Some(state) = &self.quantum_state {
            state.calculate_entropy()
        } else {
            0.0
        };
        
        Ok(TurbulanceValue::Float(total_entropy))
    }

    fn optimize_circuit(&mut self, args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let optimization_type = match args.get(0) {
            Some(TurbulanceValue::String(opt_type)) => opt_type.as_str(),
            _ => "efficiency", // Default optimization
        };
        
        match optimization_type {
            "efficiency" => self.circuit_grid.optimize_for_efficiency()?,
            "speed" => self.circuit_grid.optimize_for_speed()?,
            "stability" => self.circuit_grid.optimize_for_stability()?,
            _ => return Err(NebuchadnezzarError::invalid_input("Unknown optimization type".to_string())),
        }
        
        Ok(TurbulanceValue::String("Optimization complete".to_string()))
    }

    fn measure_coherence(&mut self, _args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let coherence = if let Some(state) = &self.quantum_state {
            state.measure_coherence()
        } else {
            0.0
        };
        
        Ok(TurbulanceValue::Float(coherence))
    }

    fn track_pattern(&mut self, args: &[TurbulanceValue]) -> Result<TurbulanceValue> {
        let pattern_name = match args.get(0) {
            Some(TurbulanceValue::String(name)) => name,
            _ => return Err(NebuchadnezzarError::invalid_input("Pattern name required".to_string())),
        };
        
        let pattern_data = self.pattern_recognizer.track_pattern(pattern_name)?;
        
        Ok(TurbulanceValue::BiologicalData(
            BiologicalDataValue::PatternData(pattern_data)
        ))
    }
}

impl Default for NebuIntegration {
    fn default() -> Self {
        Self::new()
    }
}

// Supporting types and structures

/// Represents different types of biological data
#[derive(Debug, Clone)]
pub enum BiologicalData {
    AtpPool(AtpPool),
    OscillationState(OscillationState),
    QuantumMembrane(QuantumMembrane),
    CircuitGrid(CircuitGrid),
    TimeSeries(Vec<(f64, f64)>),
    Sequence(String),
}

/// Pattern recognition system for biological data
pub struct PatternRecognizer {
    patterns: HashMap<String, PatternDefinition>,
}

impl PatternRecognizer {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // Add common biological patterns
        patterns.insert("oscillatory".to_string(), PatternDefinition {
            name: "oscillatory".to_string(),
            pattern_type: "temporal".to_string(),
            recognition_threshold: 0.7,
        });
        
        patterns.insert("exponential_growth".to_string(), PatternDefinition {
            name: "exponential_growth".to_string(),
            pattern_type: "growth".to_string(),
            recognition_threshold: 0.8,
        });
        
        patterns.insert("periodic".to_string(), PatternDefinition {
            name: "periodic".to_string(),
            pattern_type: "temporal".to_string(),
            recognition_threshold: 0.75,
        });
        
        Self { patterns }
    }
    
    pub fn find_patterns(&self, data: &str) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        
        // Simple pattern matching - in reality this would be much more sophisticated
        for (name, pattern) in &self.patterns {
            if data.to_lowercase().contains(&name.to_lowercase()) {
                matches.push(PatternMatch {
                    location: data.find(&name.to_lowercase()).unwrap_or(0),
                    length: name.len(),
                    score: pattern.recognition_threshold,
                    context: format!("Found {} pattern in biological data", name),
                });
            }
        }
        
        Ok(matches)
    }
    
    pub fn track_pattern(&self, pattern_name: &str) -> Result<PatternData> {
        let pattern_def = self.patterns.get(pattern_name)
            .ok_or_else(|| NebuchadnezzarError::invalid_input(format!("Unknown pattern: {}", pattern_name)))?;
        
        Ok(PatternData {
            pattern_type: pattern_def.pattern_type.clone(),
            confidence: pattern_def.recognition_threshold,
            matches: Vec::new(), // Would be populated with actual matches
        })
    }
}

impl Default for PatternRecognizer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PatternDefinition {
    pub name: String,
    pub pattern_type: String,
    pub recognition_threshold: f64,
}

/// State of a biological experiment
#[derive(Debug, Clone)]
pub struct ExperimentState {
    pub id: String,
    pub name: String,
    pub description: String,
    pub start_time: std::time::SystemTime,
    pub quantum_state: BiologicalQuantumState,
    pub measurements: Vec<Measurement>,
    pub hypotheses: Vec<HypothesisTest>,
    pub evidence: Vec<Evidence>,
    pub status: ExperimentStatus,
}

#[derive(Debug, Clone)]
pub enum ExperimentStatus {
    Running,
    Paused,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Measurement {
    pub timestamp: std::time::SystemTime,
    pub measurement_type: String,
    pub value: f64,
    pub units: String,
}

#[derive(Debug, Clone)]
pub struct HypothesisTest {
    pub proposition_name: String,
    pub motions: Vec<String>,
    pub support_level: f64,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct Evidence {
    pub source: String,
    pub data_type: String,
    pub content: String,
    pub quality_score: f64,
    pub timestamp: std::time::SystemTime,
}

// Re-export types for easier access from the parent module
pub use crate::turbulance::{
    PropositionResult, MotionResult, MotionStatus, EvidenceResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_creation() {
        let integration = NebuIntegration::new();
        assert!(integration.experiments.is_empty());
        assert!(integration.variables.is_empty());
    }

    #[test]
    fn test_experiment_initialization() {
        let mut integration = NebuIntegration::new();
        let result = integration.initialize_experiment("test_exp", "Test experiment");
        assert!(result.is_ok());
        assert_eq!(integration.experiments.len(), 1);
    }

    #[test]
    fn test_pattern_recognition() {
        let recognizer = PatternRecognizer::new();
        let matches = recognizer.find_patterns("oscillatory behavior in ATP").unwrap();
        assert!(!matches.is_empty());
        assert_eq!(matches[0].context, "Found oscillatory pattern in biological data");
    }

    #[test]
    fn test_variable_storage() {
        let mut integration = NebuIntegration::new();
        let value = TurbulanceValue::Float(42.0);
        integration.set_variable("test_var", value.clone());
        
        let retrieved = integration.get_variable("test_var");
        assert!(retrieved.is_some());
        
        match retrieved.unwrap() {
            TurbulanceValue::Float(f) => assert_eq!(*f, 42.0),
            _ => panic!("Expected float value"),
        }
    }

    #[test]
    fn test_atp_analysis() {
        let mut integration = NebuIntegration::new();
        let args = vec![TurbulanceValue::Float(5.0)];
        let result = integration.analyze_atp_dynamics(&args).unwrap();
        
        match result {
            TurbulanceValue::BiologicalData(BiologicalDataValue::TimeSeries(data)) => {
                assert!(!data.is_empty());
                assert!(data.len() > 10); // Should have multiple time points
            }
            _ => panic!("Expected time series data"),
        }
    }
} 
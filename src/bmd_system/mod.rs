//! Biological Maxwell's Demons System
//! 
//! Information catalysts for selective processing, pattern recognition, and amplification

pub mod pattern_recognition;
pub mod target_selection;
pub mod amplification;
pub mod information_catalyst;
pub mod thermodynamic_validation;

pub use pattern_recognition::PatternRecognition;
pub use target_selection::TargetSelection;
pub use amplification::Amplification;
pub use information_catalyst::InformationCatalyst;
pub use thermodynamic_validation::ThermodynamicValidator;

use crate::error::{Error, Result};
use crate::prelude::*;

/// Configuration for BMD system
#[derive(Debug, Clone)]
pub struct BMDConfig {
    pub pattern_threshold: f64,
    pub amplification_gain: f64,
    pub max_entropy_cost: f64,
    pub bmd_types: Vec<BMDType>,
    pub thermodynamic_validation: bool,
}

impl Default for BMDConfig {
    fn default() -> Self {
        Self::neural_optimized()
    }
}

impl BMDConfig {
    /// Create neural-optimized BMD configuration
    pub fn neural_optimized() -> Self {
        Self {
            pattern_threshold: 0.7,
            amplification_gain: 1000.0,
            max_entropy_cost: 10.0, // kT units
            bmd_types: vec![
                BMDType::Neural,
                BMDType::Molecular,
                BMDType::Metabolic,
            ],
            thermodynamic_validation: true,
        }
    }

    /// Create general cellular configuration
    pub fn cellular() -> Self {
        Self {
            pattern_threshold: 0.5,
            amplification_gain: 100.0,
            max_entropy_cost: 5.0,
            bmd_types: vec![
                BMDType::Molecular,
                BMDType::Cellular,
                BMDType::Membrane,
            ],
            thermodynamic_validation: true,
        }
    }
}

/// Types of Biological Maxwell's Demons
#[derive(Debug, Clone, PartialEq)]
pub enum BMDType {
    Molecular,  // Substrate recognition and binding
    Cellular,   // Process coordination
    Neural,     // Enhanced signal processing
    Metabolic,  // Pathway optimization
    Membrane,   // Transport and signaling
}

/// BMD system managing information processing
#[derive(Debug)]
pub struct BmdSystem {
    config: BMDConfig,
    bmds: Vec<BiologicalMaxwellDemon>,
    thermodynamic_validator: ThermodynamicValidator,
    total_entropy_cost: f64,
    activity_level: f64,
}

impl BmdSystem {
    /// Create new BMD system
    pub fn new(config: BMDConfig) -> Result<Self> {
        let mut bmds = Vec::new();
        
        // Create BMDs for each specified type
        for bmd_type in &config.bmd_types {
            let bmd = BiologicalMaxwellDemon::new(bmd_type.clone(), &config)?;
            bmds.push(bmd);
        }

        let thermodynamic_validator = ThermodynamicValidator::new(config.max_entropy_cost);

        Ok(Self {
            config,
            bmds,
            thermodynamic_validator,
            total_entropy_cost: 0.0,
            activity_level: 0.0,
        })
    }

    /// Process patterns through BMD network
    pub fn process_patterns(&mut self, patterns: &[Pattern]) -> Result<Vec<BMDResponse>> {
        let mut responses = Vec::new();
        let mut total_entropy = 0.0;

        for pattern in patterns {
            // Process through each BMD in parallel
            let mut best_response = None;
            let mut min_entropy_cost = f64::INFINITY;

            for bmd in &mut self.bmds {
                if bmd.can_process(pattern) {
                    let response = bmd.process(pattern)?;
                    
                    if response.entropy_cost < min_entropy_cost {
                        min_entropy_cost = response.entropy_cost;
                        best_response = Some(response);
                    }
                }
            }

            if let Some(response) = best_response {
                // Validate thermodynamic consistency
                if self.config.thermodynamic_validation {
                    self.thermodynamic_validator.validate_operation(&response)?;
                }

                total_entropy += response.entropy_cost;
                responses.push(response);
            }
        }

        self.total_entropy_cost += total_entropy;
        self.update_activity_level();

        Ok(responses)
    }

    /// Perform simulation step
    pub fn step(&mut self, dt: f64, atp_concentration: f64) -> Result<()> {
        // Update BMDs based on ATP availability
        for bmd in &mut self.bmds {
            bmd.update(dt, atp_concentration)?;
        }

        // Decay entropy cost over time
        self.total_entropy_cost *= (-dt / 10.0).exp(); // 10s decay time

        self.update_activity_level();
        Ok(())
    }

    /// Check if BMD system is operational
    pub fn is_operational(&self) -> bool {
        !self.bmds.is_empty() && 
        self.total_entropy_cost < self.config.max_entropy_cost &&
        self.bmds.iter().any(|bmd| bmd.is_active())
    }

    /// Get current activity level
    pub fn current_activity(&self) -> f64 {
        self.activity_level
    }

    /// Update activity level based on BMD states
    fn update_activity_level(&mut self) {
        let active_bmds = self.bmds.iter().filter(|bmd| bmd.is_active()).count();
        let total_bmds = self.bmds.len();
        
        if total_bmds > 0 {
            self.activity_level = active_bmds as f64 / total_bmds as f64;
        } else {
            self.activity_level = 0.0;
        }
    }

    /// Add new BMD to system
    pub fn add_bmd(&mut self, bmd_type: BMDType) -> Result<()> {
        let bmd = BiologicalMaxwellDemon::new(bmd_type, &self.config)?;
        self.bmds.push(bmd);
        Ok(())
    }

    /// Get BMD statistics
    pub fn get_statistics(&self) -> BMDStatistics {
        BMDStatistics {
            total_bmds: self.bmds.len(),
            active_bmds: self.bmds.iter().filter(|bmd| bmd.is_active()).count(),
            total_entropy_cost: self.total_entropy_cost,
            average_amplification: self.bmds.iter()
                .map(|bmd| bmd.current_amplification())
                .sum::<f64>() / self.bmds.len() as f64,
            activity_level: self.activity_level,
        }
    }
}

/// Individual Biological Maxwell's Demon
#[derive(Debug)]
pub struct BiologicalMaxwellDemon {
    bmd_type: BMDType,
    pattern_recognition: PatternRecognition,
    target_selection: TargetSelection,
    amplification: Amplification,
    information_catalyst: InformationCatalyst,
    is_active: bool,
    current_amplification: f64,
}

impl BiologicalMaxwellDemon {
    /// Create new BMD of specified type
    pub fn new(bmd_type: BMDType, config: &BMDConfig) -> Result<Self> {
        let pattern_recognition = PatternRecognition::new(config.pattern_threshold);
        let target_selection = TargetSelection::new();
        let amplification = Amplification::new(config.amplification_gain);
        let information_catalyst = InformationCatalyst::new(bmd_type.clone());

        Ok(Self {
            bmd_type,
            pattern_recognition,
            target_selection,
            amplification,
            information_catalyst,
            is_active: true,
            current_amplification: 1.0,
        })
    }

    /// Check if BMD can process given pattern
    pub fn can_process(&self, pattern: &Pattern) -> bool {
        self.is_active && self.pattern_recognition.matches(pattern)
    }

    /// Process pattern through BMD pipeline
    pub fn process(&mut self, pattern: &Pattern) -> Result<BMDResponse> {
        // Pattern recognition
        let recognition_score = self.pattern_recognition.recognize(pattern)?;
        
        // Target selection
        let target = self.target_selection.select_target(pattern, recognition_score)?;
        
        // Amplification
        let amplified_signal = self.amplification.amplify(&target)?;
        
        // Information catalysis
        let catalyst_result = self.information_catalyst.catalyze(&amplified_signal)?;
        
        // Calculate entropy cost
        let entropy_cost = self.calculate_entropy_cost(recognition_score, amplified_signal.magnitude);

        Ok(BMDResponse {
            bmd_type: self.bmd_type.clone(),
            target,
            amplified_signal,
            catalyst_result,
            entropy_cost,
            processing_time: 0.001, // 1ms typical processing time
        })
    }

    /// Update BMD state
    pub fn update(&mut self, dt: f64, atp_concentration: f64) -> Result<()> {
        // BMD activity depends on ATP availability
        self.is_active = atp_concentration > 1.0; // Minimum 1mM ATP

        // Update amplification based on energy availability
        let energy_factor = (atp_concentration / 5.0).min(1.0); // Normalized to 5mM ATP
        self.current_amplification = energy_factor * self.amplification.max_gain();

        // Update sub-components
        self.pattern_recognition.update(dt);
        self.amplification.update(dt, energy_factor);

        Ok(())
    }

    /// Calculate thermodynamic entropy cost
    fn calculate_entropy_cost(&self, recognition_score: f64, amplification: f64) -> f64 {
        // Entropy cost based on information gain
        let information_bits = -recognition_score.log2();
        let amplification_cost = amplification.ln();
        
        // Convert to kT units (thermal energy units)
        (information_bits + amplification_cost) * 1.0 // Simplified conversion
    }

    /// Check if BMD is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Get current amplification factor
    pub fn current_amplification(&self) -> f64 {
        self.current_amplification
    }
}

/// Pattern for BMD processing
#[derive(Debug, Clone)]
pub struct Pattern {
    pub data: Vec<f64>,
    pub pattern_type: PatternType,
    pub timestamp: f64,
}

/// Types of patterns BMDs can recognize
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Molecular,
    Electrical,
    Chemical,
    Temporal,
    Spatial,
}

/// BMD processing response
#[derive(Debug, Clone)]
pub struct BMDResponse {
    pub bmd_type: BMDType,
    pub target: Target,
    pub amplified_signal: AmplifiedSignal,
    pub catalyst_result: CatalystResult,
    pub entropy_cost: f64,
    pub processing_time: f64,
}

/// Target selected by BMD
#[derive(Debug, Clone)]
pub struct Target {
    pub target_type: TargetType,
    pub confidence: f64,
    pub parameters: Vec<f64>,
}

/// Types of targets
#[derive(Debug, Clone, PartialEq)]
pub enum TargetType {
    Substrate,
    Process,
    Signal,
    Pathway,
}

/// Amplified signal from BMD
#[derive(Debug, Clone)]
pub struct AmplifiedSignal {
    pub magnitude: f64,
    pub frequency: f64,
    pub phase: f64,
}

/// Result of information catalysis
#[derive(Debug, Clone)]
pub struct CatalystResult {
    pub catalysis_type: CatalysisType,
    pub efficiency: f64,
    pub products: Vec<f64>,
}

/// Types of catalysis
#[derive(Debug, Clone, PartialEq)]
pub enum CatalysisType {
    PatternAmplification,
    SelectiveBinding,
    SignalTransduction,
    ProcessOptimization,
}

/// BMD system statistics
#[derive(Debug, Clone)]
pub struct BMDStatistics {
    pub total_bmds: usize,
    pub active_bmds: usize,
    pub total_entropy_cost: f64,
    pub average_amplification: f64,
    pub activity_level: f64,
}
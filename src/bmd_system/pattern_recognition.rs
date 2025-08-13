//! Pattern Recognition Module for BMDs
//! 
//! Implements fuzzy pattern matching with adjustable thresholds

use crate::bmd_system::{Pattern, PatternType};
use crate::error::{Error, Result};

/// Pattern recognition system for BMDs
#[derive(Debug)]
pub struct PatternRecognition {
    threshold: f64,
    pattern_templates: Vec<PatternTemplate>,
    learning_rate: f64,
    adaptation_enabled: bool,
}

impl PatternRecognition {
    /// Create new pattern recognition system
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            pattern_templates: Vec::new(),
            learning_rate: 0.01,
            adaptation_enabled: true,
        }
    }

    /// Initialize with default pattern templates
    pub fn with_default_templates(mut self) -> Self {
        self.pattern_templates = vec![
            PatternTemplate::molecular_binding(),
            PatternTemplate::electrical_spike(),
            PatternTemplate::chemical_gradient(),
            PatternTemplate::temporal_oscillation(),
            PatternTemplate::spatial_distribution(),
        ];
        self
    }

    /// Check if pattern matches any template above threshold
    pub fn matches(&self, pattern: &Pattern) -> bool {
        self.recognize(pattern).unwrap_or(0.0) > self.threshold
    }

    /// Recognize pattern and return confidence score
    pub fn recognize(&self, pattern: &Pattern) -> Result<f64> {
        if pattern.data.is_empty() {
            return Err(Error::BmdPatternRecognitionFailure);
        }

        let mut max_score = 0.0;

        // Check against all pattern templates
        for template in &self.pattern_templates {
            if template.pattern_type == pattern.pattern_type {
                let score = self.calculate_similarity(&pattern.data, &template.template_data)?;
                max_score = max_score.max(score);
            }
        }

        // Fuzzy matching with sigmoid function
        let fuzzy_score = self.fuzzy_match(max_score);
        
        Ok(fuzzy_score)
    }

    /// Calculate similarity between pattern and template
    fn calculate_similarity(&self, pattern_data: &[f64], template_data: &[f64]) -> Result<f64> {
        if pattern_data.is_empty() || template_data.is_empty() {
            return Ok(0.0);
        }

        // Normalize to same length for comparison
        let normalized_pattern = self.normalize_length(pattern_data, template_data.len());
        let normalized_template = template_data;

        // Calculate correlation coefficient
        let correlation = self.correlation_coefficient(&normalized_pattern, normalized_template)?;
        
        // Convert to similarity score [0, 1]
        Ok((correlation + 1.0) / 2.0)
    }

    /// Calculate correlation coefficient between two signals
    fn correlation_coefficient(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(Error::BmdPatternRecognitionFailure);
        }

        let n = x.len() as f64;
        if n == 0.0 {
            return Ok(0.0);
        }

        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Normalize pattern length to match template
    fn normalize_length(&self, data: &[f64], target_length: usize) -> Vec<f64> {
        if data.len() == target_length {
            return data.to_vec();
        }

        let mut normalized = Vec::with_capacity(target_length);
        let scale = data.len() as f64 / target_length as f64;

        for i in 0..target_length {
            let source_index = (i as f64 * scale) as usize;
            let source_index = source_index.min(data.len() - 1);
            normalized.push(data[source_index]);
        }

        normalized
    }

    /// Fuzzy pattern matching with sigmoid function
    fn fuzzy_match(&self, raw_score: f64) -> f64 {
        // Sigmoid function: 1 / (1 + e^(-k(x - θ)))
        let k = 10.0; // Steepness parameter
        1.0 / (1.0 + (-k * (raw_score - self.threshold)).exp())
    }

    /// Add new pattern template
    pub fn add_template(&mut self, template: PatternTemplate) {
        self.pattern_templates.push(template);
    }

    /// Learn from successful pattern recognition
    pub fn learn_pattern(&mut self, pattern: &Pattern, success: bool) -> Result<()> {
        if !self.adaptation_enabled {
            return Ok(());
        }

        // Adapt threshold based on success/failure
        if success {
            // Successful recognition - slightly lower threshold for similar patterns
            self.threshold *= 1.0 - self.learning_rate;
        } else {
            // Failed recognition - slightly raise threshold
            self.threshold *= 1.0 + self.learning_rate;
        }

        // Keep threshold in reasonable bounds
        self.threshold = self.threshold.max(0.1).min(0.9);

        // Optionally create new template from successful patterns
        if success && pattern.data.len() > 10 {
            let new_template = PatternTemplate {
                pattern_type: pattern.pattern_type.clone(),
                template_data: pattern.data.clone(),
                confidence: 0.8,
                adaptation_weight: 0.1,
            };
            self.add_template(new_template);
        }

        Ok(())
    }

    /// Update pattern recognition system
    pub fn update(&mut self, dt: f64) {
        // Decay adaptation weights over time
        for template in &mut self.pattern_templates {
            template.adaptation_weight *= (-dt / 100.0).exp(); // 100s decay
        }

        // Remove templates with very low weights
        self.pattern_templates.retain(|t| t.adaptation_weight > 0.01);
    }

    /// Set recognition threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold.max(0.0).min(1.0);
    }

    /// Enable/disable adaptive learning
    pub fn set_adaptation(&mut self, enabled: bool) {
        self.adaptation_enabled = enabled;
    }
}

/// Pattern template for recognition
#[derive(Debug, Clone)]
pub struct PatternTemplate {
    pub pattern_type: PatternType,
    pub template_data: Vec<f64>,
    pub confidence: f64,
    pub adaptation_weight: f64,
}

impl PatternTemplate {
    /// Create molecular binding pattern template
    pub fn molecular_binding() -> Self {
        // Exponential binding curve
        let mut data = Vec::new();
        for i in 0..50 {
            let x = i as f64 / 10.0;
            data.push(1.0 - (-x).exp());
        }

        Self {
            pattern_type: PatternType::Molecular,
            template_data: data,
            confidence: 0.9,
            adaptation_weight: 1.0,
        }
    }

    /// Create electrical spike pattern template
    pub fn electrical_spike() -> Self {
        // Action potential-like shape
        let mut data = Vec::new();
        for i in 0..100 {
            let x = (i as f64 - 25.0) / 10.0;
            let spike = (-x * x / 2.0).exp();
            data.push(spike);
        }

        Self {
            pattern_type: PatternType::Electrical,
            template_data: data,
            confidence: 0.95,
            adaptation_weight: 1.0,
        }
    }

    /// Create chemical gradient pattern template
    pub fn chemical_gradient() -> Self {
        // Linear concentration gradient
        let mut data = Vec::new();
        for i in 0..50 {
            data.push(1.0 - i as f64 / 49.0);
        }

        Self {
            pattern_type: PatternType::Chemical,
            template_data: data,
            confidence: 0.8,
            adaptation_weight: 1.0,
        }
    }

    /// Create temporal oscillation pattern template
    pub fn temporal_oscillation() -> Self {
        // Sinusoidal oscillation
        let mut data = Vec::new();
        for i in 0..100 {
            let x = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
            data.push(x.sin());
        }

        Self {
            pattern_type: PatternType::Temporal,
            template_data: data,
            confidence: 0.85,
            adaptation_weight: 1.0,
        }
    }

    /// Create spatial distribution pattern template
    pub fn spatial_distribution() -> Self {
        // Gaussian distribution
        let mut data = Vec::new();
        for i in 0..50 {
            let x = (i as f64 - 25.0) / 10.0;
            data.push((-x * x / 2.0).exp());
        }

        Self {
            pattern_type: PatternType::Spatial,
            template_data: data,
            confidence: 0.9,
            adaptation_weight: 1.0,
        }
    }

    /// Create custom pattern template
    pub fn custom(pattern_type: PatternType, data: Vec<f64>) -> Self {
        Self {
            pattern_type,
            template_data: data,
            confidence: 0.7,
            adaptation_weight: 0.5,
        }
    }
}
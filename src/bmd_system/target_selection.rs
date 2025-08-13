//! Target Selection Module for BMDs
//! 
//! Implements maximum likelihood target selection

use crate::bmd_system::{Pattern, Target, TargetType};
use crate::error::{Error, Result};

/// Target selection system for BMDs
#[derive(Debug)]
pub struct TargetSelection {
    selection_criteria: Vec<SelectionCriterion>,
    confidence_threshold: f64,
    multi_target_enabled: bool,
}

impl TargetSelection {
    /// Create new target selection system
    pub fn new() -> Self {
        Self {
            selection_criteria: Self::default_criteria(),
            confidence_threshold: 0.6,
            multi_target_enabled: true,
        }
    }

    /// Select optimal target based on pattern and recognition score
    pub fn select_target(&self, pattern: &Pattern, recognition_score: f64) -> Result<Target> {
        if recognition_score < self.confidence_threshold {
            return Err(Error::BmdPatternRecognitionFailure);
        }

        // Evaluate all possible targets
        let candidates = self.generate_target_candidates(pattern)?;
        
        if candidates.is_empty() {
            return Err(Error::BmdPatternRecognitionFailure);
        }

        // Apply maximum likelihood estimation
        let selected = self.maximum_likelihood_selection(&candidates, pattern, recognition_score)?;
        
        Ok(selected)
    }

    /// Generate potential target candidates
    fn generate_target_candidates(&self, pattern: &Pattern) -> Result<Vec<TargetCandidate>> {
        let mut candidates = Vec::new();

        // Generate candidates based on pattern type
        match pattern.pattern_type {
            crate::bmd_system::PatternType::Molecular => {
                candidates.push(TargetCandidate {
                    target_type: TargetType::Substrate,
                    base_probability: 0.8,
                    parameters: self.extract_molecular_parameters(&pattern.data)?,
                });
                candidates.push(TargetCandidate {
                    target_type: TargetType::Process,
                    base_probability: 0.6,
                    parameters: self.extract_process_parameters(&pattern.data)?,
                });
            },
            crate::bmd_system::PatternType::Electrical => {
                candidates.push(TargetCandidate {
                    target_type: TargetType::Signal,
                    base_probability: 0.9,
                    parameters: self.extract_signal_parameters(&pattern.data)?,
                });
            },
            crate::bmd_system::PatternType::Chemical => {
                candidates.push(TargetCandidate {
                    target_type: TargetType::Pathway,
                    base_probability: 0.7,
                    parameters: self.extract_pathway_parameters(&pattern.data)?,
                });
                candidates.push(TargetCandidate {
                    target_type: TargetType::Substrate,
                    base_probability: 0.5,
                    parameters: self.extract_molecular_parameters(&pattern.data)?,
                });
            },
            _ => {
                // Generic target for other pattern types
                candidates.push(TargetCandidate {
                    target_type: TargetType::Process,
                    base_probability: 0.5,
                    parameters: vec![pattern.data.iter().sum::<f64>() / pattern.data.len() as f64],
                });
            }
        }

        Ok(candidates)
    }

    /// Apply maximum likelihood estimation for target selection
    fn maximum_likelihood_selection(
        &self, 
        candidates: &[TargetCandidate],
        pattern: &Pattern,
        recognition_score: f64
    ) -> Result<Target> {
        let mut best_candidate = None;
        let mut max_likelihood = 0.0;

        for candidate in candidates {
            let likelihood = self.calculate_likelihood(candidate, pattern, recognition_score)?;
            
            if likelihood > max_likelihood {
                max_likelihood = likelihood;
                best_candidate = Some(candidate);
            }
        }

        match best_candidate {
            Some(candidate) => Ok(Target {
                target_type: candidate.target_type.clone(),
                confidence: max_likelihood,
                parameters: candidate.parameters.clone(),
            }),
            None => Err(Error::BmdPatternRecognitionFailure),
        }
    }

    /// Calculate likelihood for target candidate
    fn calculate_likelihood(
        &self,
        candidate: &TargetCandidate,
        pattern: &Pattern,
        recognition_score: f64
    ) -> Result<f64> {
        let mut likelihood = candidate.base_probability;
        
        // Apply selection criteria
        for criterion in &self.selection_criteria {
            let criterion_score = criterion.evaluate(candidate, pattern)?;
            likelihood *= criterion_score;
        }

        // Weight by recognition score
        likelihood *= recognition_score;

        // Apply temporal weighting (more recent patterns weighted higher)
        let temporal_weight = self.calculate_temporal_weight(pattern.timestamp);
        likelihood *= temporal_weight;

        Ok(likelihood)
    }

    /// Calculate temporal weight based on pattern timestamp
    fn calculate_temporal_weight(&self, timestamp: f64) -> f64 {
        // More recent patterns get higher weight
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        let age = current_time - timestamp;
        let decay_constant = 10.0; // 10 second half-life
        
        (-age / decay_constant).exp()
    }

    /// Extract molecular-specific parameters
    fn extract_molecular_parameters(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(vec![0.0]);
        }

        let binding_affinity = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let specificity = self.calculate_specificity(data);
        let kinetics = self.calculate_kinetics(data);

        Ok(vec![*binding_affinity, specificity, kinetics])
    }

    /// Extract process-specific parameters
    fn extract_process_parameters(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(vec![0.0]);
        }

        let efficiency = data.iter().sum::<f64>() / data.len() as f64;
        let stability = 1.0 / (self.calculate_variance(data) + 1e-6);
        let throughput = data.len() as f64;

        Ok(vec![efficiency, stability, throughput])
    }

    /// Extract signal-specific parameters
    fn extract_signal_parameters(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(vec![0.0]);
        }

        let amplitude = data.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let frequency = self.estimate_frequency(data);
        let signal_to_noise = self.calculate_signal_to_noise(data);

        Ok(vec![amplitude, frequency, signal_to_noise])
    }

    /// Extract pathway-specific parameters
    fn extract_pathway_parameters(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(vec![0.0]);
        }

        let flux = data.iter().sum::<f64>();
        let regulation = self.calculate_regulation_strength(data);
        let feedback = self.calculate_feedback_strength(data);

        Ok(vec![flux, regulation, feedback])
    }

    /// Calculate specificity from data
    fn calculate_specificity(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.5;
        }

        let max_val = data.iter().fold(0.0, |a, &b| a.max(b));
        let mean_val = data.iter().sum::<f64>() / data.len() as f64;
        
        if max_val > 0.0 {
            1.0 - (mean_val / max_val)
        } else {
            0.0
        }
    }

    /// Calculate kinetics parameter
    fn calculate_kinetics(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 1.0;
        }

        // Estimate rate from data slope
        let n = data.len();
        let mut sum_slope = 0.0;
        
        for i in 1..n {
            sum_slope += (data[i] - data[i-1]).abs();
        }
        
        sum_slope / (n - 1) as f64
    }

    /// Calculate variance
    fn calculate_variance(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        variance
    }

    /// Estimate dominant frequency
    fn estimate_frequency(&self, data: &[f64]) -> f64 {
        // Simple zero-crossing frequency estimation
        if data.len() < 3 {
            return 0.0;
        }

        let mut zero_crossings = 0;
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        
        for i in 1..data.len() {
            if (data[i-1] - mean) * (data[i] - mean) < 0.0 {
                zero_crossings += 1;
            }
        }
        
        zero_crossings as f64 / (2.0 * data.len() as f64)
    }

    /// Calculate signal-to-noise ratio
    fn calculate_signal_to_noise(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let signal_power = data.iter().map(|x| x.powi(2)).sum::<f64>() / data.len() as f64;
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let noise_power = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        
        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            100.0 // High SNR when no noise
        }
    }

    /// Calculate regulation strength
    fn calculate_regulation_strength(&self, data: &[f64]) -> f64 {
        // Measure how much the signal varies from baseline
        if data.is_empty() {
            return 0.0;
        }

        let baseline = data[0];
        let max_deviation = data.iter()
            .map(|x| (x - baseline).abs())
            .fold(0.0, f64::max);
        
        max_deviation / (baseline.abs() + 1e-6)
    }

    /// Calculate feedback strength
    fn calculate_feedback_strength(&self, data: &[f64]) -> f64 {
        // Measure correlation between early and late parts of signal
        if data.len() < 4 {
            return 0.0;
        }

        let mid = data.len() / 2;
        let early = &data[0..mid];
        let late = &data[mid..];
        
        // Simple correlation calculation
        let early_mean = early.iter().sum::<f64>() / early.len() as f64;
        let late_mean = late.iter().sum::<f64>() / late.len() as f64;
        
        let mut correlation = 0.0;
        let min_len = early.len().min(late.len());
        
        for i in 0..min_len {
            correlation += (early[i] - early_mean) * (late[i] - late_mean);
        }
        
        correlation / min_len as f64
    }

    /// Default selection criteria
    fn default_criteria() -> Vec<SelectionCriterion> {
        vec![
            SelectionCriterion::new("specificity", 0.8),
            SelectionCriterion::new("efficiency", 0.7),
            SelectionCriterion::new("stability", 0.6),
        ]
    }
}

impl Default for TargetSelection {
    fn default() -> Self {
        Self::new()
    }
}

/// Target candidate for selection
#[derive(Debug, Clone)]
struct TargetCandidate {
    target_type: TargetType,
    base_probability: f64,
    parameters: Vec<f64>,
}

/// Selection criterion for target evaluation
#[derive(Debug, Clone)]
struct SelectionCriterion {
    name: String,
    weight: f64,
}

impl SelectionCriterion {
    fn new(name: &str, weight: f64) -> Self {
        Self {
            name: name.to_string(),
            weight,
        }
    }

    fn evaluate(&self, candidate: &TargetCandidate, _pattern: &Pattern) -> Result<f64> {
        // Simplified criterion evaluation
        match self.name.as_str() {
            "specificity" => Ok(candidate.parameters.get(1).unwrap_or(&0.5) * self.weight),
            "efficiency" => Ok(candidate.parameters.get(0).unwrap_or(&0.5) * self.weight),
            "stability" => Ok((1.0 / (candidate.parameters.get(2).unwrap_or(&1.0) + 1.0)) * self.weight),
            _ => Ok(0.5 * self.weight),
        }
    }
}
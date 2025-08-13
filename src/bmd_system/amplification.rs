//! Amplification Module for BMDs
//! 
//! Implements controlled positive feedback amplification

use crate::bmd_system::{Target, AmplifiedSignal};
use crate::error::{Error, Result};

/// Amplification system for BMDs
#[derive(Debug)]
pub struct Amplification {
    max_gain: f64,
    current_gain: f64,
    feedback_strength: f64,
    stability_margin: f64,
    saturation_threshold: f64,
}

impl Amplification {
    /// Create new amplification system
    pub fn new(max_gain: f64) -> Self {
        Self {
            max_gain,
            current_gain: 1.0,
            feedback_strength: 0.1,
            stability_margin: 0.1,
            saturation_threshold: 0.95,
        }
    }

    /// Amplify target signal with controlled feedback
    pub fn amplify(&mut self, target: &Target) -> Result<AmplifiedSignal> {
        // Calculate amplification based on target confidence
        let confidence_factor = target.confidence;
        let gain = self.calculate_adaptive_gain(confidence_factor)?;
        
        // Extract signal parameters from target
        let base_magnitude = target.parameters.get(0).unwrap_or(&1.0);
        let base_frequency = target.parameters.get(1).unwrap_or(&1.0);
        let phase = target.parameters.get(2).unwrap_or(&0.0);

        // Apply amplification with saturation protection
        let amplified_magnitude = self.apply_gain_with_saturation(*base_magnitude, gain)?;
        
        // Frequency modulation based on amplification
        let frequency_modulation = 1.0 + 0.1 * gain.ln();
        let amplified_frequency = base_frequency * frequency_modulation;

        // Phase shift due to amplification delay
        let phase_shift = self.calculate_phase_shift(gain);
        let final_phase = phase + phase_shift;

        self.current_gain = gain;

        Ok(AmplifiedSignal {
            magnitude: amplified_magnitude,
            frequency: amplified_frequency,
            phase: final_phase,
        })
    }

    /// Calculate adaptive gain based on target properties
    fn calculate_adaptive_gain(&self, confidence: f64) -> Result<f64> {
        // Base gain from confidence
        let base_gain = confidence * self.max_gain;
        
        // Apply feedback control
        let feedback_gain = self.apply_feedback_control(base_gain)?;
        
        // Ensure stability
        let stable_gain = self.ensure_stability(feedback_gain)?;
        
        Ok(stable_gain.max(1.0).min(self.max_gain))
    }

    /// Apply feedback control to gain
    fn apply_feedback_control(&self, base_gain: f64) -> Result<f64> {
        // Positive feedback with stability constraints
        let feedback_term = self.feedback_strength * base_gain.ln();
        let controlled_gain = base_gain * (1.0 + feedback_term);
        
        // Prevent runaway amplification
        if controlled_gain > self.max_gain * 2.0 {
            return Err(Error::BmdAmplificationOverflow);
        }
        
        Ok(controlled_gain)
    }

    /// Ensure amplification stability
    fn ensure_stability(&self, gain: f64) -> Result<f64> {
        // Check Nyquist stability criterion (simplified)
        let stability_factor = 1.0 / (1.0 + gain * self.feedback_strength);
        
        if stability_factor < self.stability_margin {
            // Reduce gain to maintain stability
            let stable_gain = (1.0 / self.feedback_strength - 1.0) * self.stability_margin;
            Ok(stable_gain.min(gain))
        } else {
            Ok(gain)
        }
    }

    /// Apply gain with saturation protection
    fn apply_gain_with_saturation(&self, input: f64, gain: f64) -> Result<f64> {
        let amplified = input * gain;
        
        // Soft saturation using hyperbolic tangent
        let normalized = amplified / self.max_gain;
        let saturated = normalized.tanh() * self.max_gain;
        
        // Check for overflow
        if saturated.is_infinite() || saturated.is_nan() {
            return Err(Error::BmdAmplificationOverflow);
        }
        
        Ok(saturated)
    }

    /// Calculate phase shift due to amplification
    fn calculate_phase_shift(&self, gain: f64) -> f64 {
        // Phase shift proportional to log of gain
        let base_shift = 0.1 * gain.ln();
        
        // Additional shift from feedback delay
        let feedback_delay = self.feedback_strength * 0.01; // 10ms per unit feedback
        
        base_shift + feedback_delay
    }

    /// Update amplification system state
    pub fn update(&mut self, dt: f64, energy_factor: f64) -> Result<()> {
        // Adjust max gain based on available energy
        let energy_limited_gain = self.max_gain * energy_factor;
        
        // Gradually adjust current gain towards energy-limited value
        let gain_time_constant = 0.1; // 100ms time constant
        let gain_change = (energy_limited_gain - self.current_gain) * dt / gain_time_constant;
        self.current_gain += gain_change;
        
        // Ensure current gain doesn't exceed limits
        self.current_gain = self.current_gain.max(1.0).min(self.max_gain);
        
        Ok(())
    }

    /// Get maximum gain
    pub fn max_gain(&self) -> f64 {
        self.max_gain
    }

    /// Get current gain
    pub fn current_gain(&self) -> f64 {
        self.current_gain
    }

    /// Set feedback strength
    pub fn set_feedback_strength(&mut self, strength: f64) {
        self.feedback_strength = strength.max(0.0).min(1.0);
    }

    /// Set stability margin
    pub fn set_stability_margin(&mut self, margin: f64) {
        self.stability_margin = margin.max(0.01).min(0.5);
    }

    /// Check if amplifier is stable
    pub fn is_stable(&self) -> bool {
        self.current_gain < self.max_gain * self.saturation_threshold &&
        self.current_gain * self.feedback_strength < 1.0 / self.stability_margin
    }

    /// Calculate amplification efficiency
    pub fn efficiency(&self) -> f64 {
        if self.max_gain > 0.0 {
            self.current_gain / self.max_gain
        } else {
            0.0
        }
    }

    /// Amplify multiple signals simultaneously
    pub fn amplify_multiple(&mut self, targets: &[Target]) -> Result<Vec<AmplifiedSignal>> {
        let mut amplified_signals = Vec::new();
        
        for target in targets {
            let amplified = self.amplify(target)?;
            amplified_signals.push(amplified);
        }
        
        // Check for crosstalk and interference
        self.compensate_crosstalk(&mut amplified_signals)?;
        
        Ok(amplified_signals)
    }

    /// Compensate for crosstalk between amplified signals
    fn compensate_crosstalk(&self, signals: &mut [AmplifiedSignal]) -> Result<()> {
        if signals.len() < 2 {
            return Ok(());
        }

        // Simple crosstalk compensation
        for i in 0..signals.len() {
            let mut interference = 0.0;
            
            for j in 0..signals.len() {
                if i != j {
                    // Calculate interference based on frequency proximity
                    let freq_diff = (signals[i].frequency - signals[j].frequency).abs();
                    let crosstalk_factor = (-freq_diff / 10.0).exp(); // 10 Hz characteristic frequency
                    interference += signals[j].magnitude * crosstalk_factor * 0.01; // 1% crosstalk
                }
            }
            
            // Subtract interference from signal
            signals[i].magnitude = (signals[i].magnitude - interference).max(0.0);
        }
        
        Ok(())
    }

    /// Calculate signal-to-noise ratio improvement
    pub fn snr_improvement(&self, input_snr: f64) -> f64 {
        // SNR improvement with noise figure
        let noise_figure = 1.0 + 0.1 * self.current_gain.ln(); // Noise increases with gain
        let ideal_improvement = self.current_gain;
        
        ideal_improvement / noise_figure
    }

    /// Set amplification mode
    pub fn set_mode(&mut self, mode: AmplificationMode) -> Result<()> {
        match mode {
            AmplificationMode::Linear => {
                self.feedback_strength = 0.0;
                self.saturation_threshold = 0.99;
            },
            AmplificationMode::Logarithmic => {
                self.feedback_strength = 0.1;
                self.saturation_threshold = 0.9;
            },
            AmplificationMode::Saturating => {
                self.feedback_strength = 0.2;
                self.saturation_threshold = 0.7;
            },
            AmplificationMode::Adaptive => {
                self.feedback_strength = 0.05;
                self.saturation_threshold = 0.85;
            },
        }
        
        Ok(())
    }
}

/// Amplification modes
#[derive(Debug, Clone, PartialEq)]
pub enum AmplificationMode {
    Linear,      // Constant gain
    Logarithmic, // Logarithmic compression
    Saturating,  // Hard saturation
    Adaptive,    // Adaptive gain control
}
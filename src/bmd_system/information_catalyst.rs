//! Information Catalyst Module for BMDs
//! 
//! Implements information processing and catalysis functions

use crate::bmd_system::{BMDType, AmplifiedSignal, CatalystResult, CatalysisType};
use crate::error::{Error, Result};

/// Information catalyst for BMD processing
#[derive(Debug)]
pub struct InformationCatalyst {
    bmd_type: BMDType,
    catalysis_efficiency: f64,
    information_gain: f64,
    processing_history: Vec<ProcessingRecord>,
    adaptation_rate: f64,
}

impl InformationCatalyst {
    /// Create new information catalyst for specific BMD type
    pub fn new(bmd_type: BMDType) -> Self {
        let catalysis_efficiency = match bmd_type {
            BMDType::Molecular => 0.9,  // High efficiency for molecular recognition
            BMDType::Neural => 0.95,    // Highest for neural processing
            BMDType::Metabolic => 0.8,  // Good for pathway optimization
            BMDType::Cellular => 0.7,   // Moderate for general processes
            BMDType::Membrane => 0.85,  // High for transport processes
        };

        Self {
            bmd_type,
            catalysis_efficiency,
            information_gain: 0.0,
            processing_history: Vec::new(),
            adaptation_rate: 0.01,
        }
    }

    /// Perform information catalysis on amplified signal
    pub fn catalyze(&mut self, signal: &AmplifiedSignal) -> Result<CatalystResult> {
        // Determine catalysis type based on signal characteristics
        let catalysis_type = self.determine_catalysis_type(signal)?;
        
        // Apply catalytic transformation
        let products = self.apply_catalysis(signal, &catalysis_type)?;
        
        // Calculate efficiency based on signal quality and BMD type
        let efficiency = self.calculate_efficiency(signal, &catalysis_type)?;
        
        // Update information gain
        self.update_information_gain(signal, efficiency);
        
        // Record processing for adaptation
        self.record_processing(signal, &catalysis_type, efficiency);

        Ok(CatalystResult {
            catalysis_type,
            efficiency,
            products,
        })
    }

    /// Determine appropriate catalysis type for signal
    fn determine_catalysis_type(&self, signal: &AmplifiedSignal) -> Result<CatalysisType> {
        let magnitude = signal.magnitude;
        let frequency = signal.frequency;
        
        // Classification based on signal characteristics and BMD type
        match self.bmd_type {
            BMDType::Molecular => {
                if frequency > 10.0 {
                    Ok(CatalysisType::SelectiveBinding)
                } else {
                    Ok(CatalysisType::PatternAmplification)
                }
            },
            BMDType::Neural => {
                if magnitude > 0.8 {
                    Ok(CatalysisType::SignalTransduction)
                } else {
                    Ok(CatalysisType::PatternAmplification)
                }
            },
            BMDType::Metabolic => {
                Ok(CatalysisType::ProcessOptimization)
            },
            BMDType::Cellular => {
                if frequency < 1.0 {
                    Ok(CatalysisType::ProcessOptimization)
                } else {
                    Ok(CatalysisType::SignalTransduction)
                }
            },
            BMDType::Membrane => {
                Ok(CatalysisType::SelectiveBinding)
            },
        }
    }

    /// Apply catalytic transformation to signal
    fn apply_catalysis(&self, signal: &AmplifiedSignal, catalysis_type: &CatalysisType) -> Result<Vec<f64>> {
        let mut products = Vec::new();
        
        match catalysis_type {
            CatalysisType::PatternAmplification => {
                // Enhance pattern recognition through selective amplification
                products.push(signal.magnitude * self.catalysis_efficiency);
                products.push(signal.frequency); // Preserve frequency
                products.push(self.calculate_pattern_specificity(signal));
            },
            
            CatalysisType::SelectiveBinding => {
                // Simulate selective molecular binding
                let binding_affinity = self.calculate_binding_affinity(signal)?;
                let selectivity = self.calculate_selectivity(signal);
                let kinetics = self.calculate_binding_kinetics(signal);
                
                products.push(binding_affinity);
                products.push(selectivity);
                products.push(kinetics);
            },
            
            CatalysisType::SignalTransduction => {
                // Convert signal to downstream effects
                let transduced_amplitude = signal.magnitude * self.catalysis_efficiency;
                let cascade_amplification = self.calculate_cascade_amplification(signal);
                let response_time = self.calculate_response_time(signal);
                
                products.push(transduced_amplitude);
                products.push(cascade_amplification);
                products.push(response_time);
            },
            
            CatalysisType::ProcessOptimization => {
                // Optimize process efficiency
                let efficiency_gain = self.calculate_efficiency_gain(signal);
                let resource_utilization = self.calculate_resource_utilization(signal);
                let throughput_improvement = self.calculate_throughput_improvement(signal);
                
                products.push(efficiency_gain);
                products.push(resource_utilization);
                products.push(throughput_improvement);
            },
        }
        
        Ok(products)
    }

    /// Calculate catalysis efficiency
    fn calculate_efficiency(&self, signal: &AmplifiedSignal, catalysis_type: &CatalysisType) -> Result<f64> {
        let base_efficiency = self.catalysis_efficiency;
        
        // Signal quality factor
        let signal_quality = self.assess_signal_quality(signal);
        
        // Type-specific efficiency modifiers
        let type_modifier = match catalysis_type {
            CatalysisType::PatternAmplification => 0.9,
            CatalysisType::SelectiveBinding => 0.95,
            CatalysisType::SignalTransduction => 0.85,
            CatalysisType::ProcessOptimization => 0.8,
        };
        
        // Historical performance factor
        let historical_factor = self.calculate_historical_performance();
        
        let final_efficiency = base_efficiency * signal_quality * type_modifier * historical_factor;
        
        Ok(final_efficiency.min(1.0))
    }

    /// Assess signal quality for catalysis
    fn assess_signal_quality(&self, signal: &AmplifiedSignal) -> f64 {
        // Quality based on magnitude, frequency stability, and phase coherence
        let magnitude_quality = if signal.magnitude > 0.1 { 
            (signal.magnitude / (signal.magnitude + 1.0)).min(1.0) 
        } else { 
            0.1 
        };
        
        let frequency_quality = if signal.frequency > 0.0 {
            1.0 / (1.0 + signal.frequency.abs().ln())
        } else {
            0.5
        };
        
        let phase_quality = 1.0 - (signal.phase % (2.0 * std::f64::consts::PI)).abs() / (2.0 * std::f64::consts::PI);
        
        (magnitude_quality + frequency_quality + phase_quality) / 3.0
    }

    /// Calculate pattern specificity
    fn calculate_pattern_specificity(&self, signal: &AmplifiedSignal) -> f64 {
        // Higher magnitude and lower frequency variance indicate higher specificity
        let magnitude_specificity = signal.magnitude / (signal.magnitude + 1.0);
        let frequency_specificity = 1.0 / (1.0 + signal.frequency.abs());
        
        (magnitude_specificity + frequency_specificity) / 2.0
    }

    /// Calculate binding affinity
    fn calculate_binding_affinity(&self, signal: &AmplifiedSignal) -> Result<f64> {
        // Affinity based on signal strength and frequency match
        let affinity = signal.magnitude * self.catalysis_efficiency;
        
        // Frequency-dependent binding
        let frequency_factor = (-((signal.frequency - 1.0).powi(2)) / 2.0).exp();
        
        Ok(affinity * frequency_factor)
    }

    /// Calculate selectivity
    fn calculate_selectivity(&self, signal: &AmplifiedSignal) -> f64 {
        // Selectivity inversely related to signal bandwidth
        let bandwidth = signal.frequency.abs() + 0.1;
        1.0 / bandwidth
    }

    /// Calculate binding kinetics
    fn calculate_binding_kinetics(&self, signal: &AmplifiedSignal) -> f64 {
        // Kinetics based on signal dynamics
        signal.magnitude * signal.frequency.abs()
    }

    /// Calculate cascade amplification
    fn calculate_cascade_amplification(&self, signal: &AmplifiedSignal) -> f64 {
        // Exponential amplification for strong signals
        let base_amplification = 2.0;
        base_amplification.powf(signal.magnitude)
    }

    /// Calculate response time
    fn calculate_response_time(&self, signal: &AmplifiedSignal) -> f64 {
        // Faster response for stronger signals
        let base_time = 0.001; // 1ms base response time
        base_time / (signal.magnitude + 0.1)
    }

    /// Calculate efficiency gain
    fn calculate_efficiency_gain(&self, signal: &AmplifiedSignal) -> f64 {
        signal.magnitude * self.catalysis_efficiency * 1.5
    }

    /// Calculate resource utilization
    fn calculate_resource_utilization(&self, signal: &AmplifiedSignal) -> f64 {
        // Higher utilization for optimal signal ranges
        let optimal_magnitude = 0.7;
        1.0 - (signal.magnitude - optimal_magnitude).abs()
    }

    /// Calculate throughput improvement
    fn calculate_throughput_improvement(&self, signal: &AmplifiedSignal) -> f64 {
        signal.frequency * self.catalysis_efficiency
    }

    /// Update information gain
    fn update_information_gain(&mut self, signal: &AmplifiedSignal, efficiency: f64) {
        let new_gain = efficiency * signal.magnitude.log2().max(0.0);
        self.information_gain = self.information_gain * 0.99 + new_gain * 0.01; // Exponential averaging
    }

    /// Record processing for adaptation
    fn record_processing(&mut self, signal: &AmplifiedSignal, catalysis_type: &CatalysisType, efficiency: f64) {
        let record = ProcessingRecord {
            signal_magnitude: signal.magnitude,
            signal_frequency: signal.frequency,
            catalysis_type: catalysis_type.clone(),
            efficiency,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        };
        
        self.processing_history.push(record);
        
        // Keep only recent history (last 1000 records)
        if self.processing_history.len() > 1000 {
            self.processing_history.remove(0);
        }
        
        // Adapt efficiency based on recent performance
        self.adapt_efficiency();
    }

    /// Adapt catalysis efficiency based on performance history
    fn adapt_efficiency(&mut self) {
        if self.processing_history.len() < 10 {
            return;
        }
        
        // Calculate recent average efficiency
        let recent_records = &self.processing_history[self.processing_history.len()-10..];
        let average_efficiency = recent_records.iter()
            .map(|r| r.efficiency)
            .sum::<f64>() / recent_records.len() as f64;
        
        // Adapt towards better performance
        let target_efficiency = 0.9;
        let efficiency_error = target_efficiency - average_efficiency;
        
        self.catalysis_efficiency += self.adaptation_rate * efficiency_error;
        self.catalysis_efficiency = self.catalysis_efficiency.max(0.1).min(1.0);
    }

    /// Calculate historical performance factor
    fn calculate_historical_performance(&self) -> f64 {
        if self.processing_history.is_empty() {
            return 1.0;
        }
        
        let recent_count = 50.min(self.processing_history.len());
        let recent_efficiency = self.processing_history
            .iter()
            .rev()
            .take(recent_count)
            .map(|r| r.efficiency)
            .sum::<f64>() / recent_count as f64;
        
        recent_efficiency
    }

    /// Get current information gain
    pub fn information_gain(&self) -> f64 {
        self.information_gain
    }

    /// Get catalysis efficiency
    pub fn efficiency(&self) -> f64 {
        self.catalysis_efficiency
    }

    /// Get processing statistics
    pub fn get_statistics(&self) -> CatalystStatistics {
        CatalystStatistics {
            total_processed: self.processing_history.len(),
            average_efficiency: self.calculate_historical_performance(),
            information_gain: self.information_gain,
            catalysis_efficiency: self.catalysis_efficiency,
        }
    }
}

/// Processing record for adaptation
#[derive(Debug, Clone)]
struct ProcessingRecord {
    signal_magnitude: f64,
    signal_frequency: f64,
    catalysis_type: CatalysisType,
    efficiency: f64,
    timestamp: f64,
}

/// Catalyst statistics
#[derive(Debug, Clone)]
pub struct CatalystStatistics {
    pub total_processed: usize,
    pub average_efficiency: f64,
    pub information_gain: f64,
    pub catalysis_efficiency: f64,
}
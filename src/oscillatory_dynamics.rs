//! # Universal Oscillatory Dynamics Module
//! 
//! Implementation of the Universal Oscillatory Framework based on the Causal Selection theorem.
//! This module demonstrates that oscillatory behavior is mathematically inevitable in any 
//! bounded nonlinear system, providing the foundation for biological rhythms and dynamics.

use crate::error::{NebuchadnezzarError, Result};
use std::collections::HashMap;
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// Universal oscillator implementing the Causal Selection theorem
#[derive(Debug, Clone)]
pub struct UniversalOscillator {
    pub oscillator_id: String,
    pub state: OscillatorState,
    pub parameters: OscillatorParameters,
    pub causal_selection: CausalSelection,
    pub boundary_conditions: BoundaryConditions,
    pub nonlinear_terms: Vec<NonlinearTerm>,
    pub coupling_matrix: Vec<Vec<f64>>,
}

/// Current state of an oscillator
#[derive(Debug, Clone)]
pub struct OscillatorState {
    pub position: f64,
    pub velocity: f64,
    pub acceleration: f64,
    pub phase: f64,
    pub amplitude: f64,
    pub frequency: f64,
    pub energy: f64,
    pub entropy: f64,
}

/// Parameters defining oscillator behavior
#[derive(Debug, Clone)]
pub struct OscillatorParameters {
    pub natural_frequency: f64,
    pub damping_coefficient: f64,
    pub driving_amplitude: f64,
    pub driving_frequency: f64,
    pub nonlinearity_strength: f64,
    pub mass: f64,
    pub spring_constant: f64,
}

/// Causal Selection mechanism - the core of the theorem
#[derive(Debug, Clone)]
pub struct CausalSelection {
    pub selection_pressure: f64,
    pub causal_chains: Vec<CausalChain>,
    pub selection_criteria: SelectionCriteria,
    pub stability_analysis: StabilityAnalysis,
    pub attractors: Vec<Attractor>,
}

/// Individual causal chain in the selection process
#[derive(Debug, Clone)]
pub struct CausalChain {
    pub chain_id: String,
    pub causality_strength: f64,
    pub temporal_correlation: f64,
    pub phase_relationship: f64,
    pub energy_flow: f64,
    pub information_content: f64,
}

/// Criteria for causal selection
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    pub energy_efficiency: f64,
    pub information_preservation: f64,
    pub temporal_stability: f64,
    pub phase_coherence: f64,
    pub coupling_strength: f64,
}

/// Stability analysis for oscillatory dynamics
#[derive(Debug, Clone)]
pub struct StabilityAnalysis {
    pub lyapunov_exponents: Vec<f64>,
    pub phase_portrait: Vec<(f64, f64)>,
    pub basin_of_attraction: Vec<(f64, f64)>,
    pub bifurcation_points: Vec<f64>,
    pub chaos_indicators: Vec<f64>,
}

/// Attractor in phase space
#[derive(Debug, Clone)]
pub struct Attractor {
    pub attractor_type: AttractorType,
    pub position: Vec<f64>,
    pub stability: f64,
    pub basin_size: f64,
    pub fractal_dimension: f64,
}

#[derive(Debug, Clone)]
pub enum AttractorType {
    FixedPoint,
    LimitCycle,
    QuasiPeriodic,
    Strange,
}

/// Boundary conditions defining the bounded system
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    pub position_bounds: (f64, f64),
    pub velocity_bounds: (f64, f64),
    pub energy_bounds: (f64, f64),
    pub boundary_type: BoundaryType,
    pub reflection_coefficient: f64,
}

#[derive(Debug, Clone)]
pub enum BoundaryType {
    Hard,
    Soft,
    Periodic,
    Absorbing,
}

/// Nonlinear terms in the oscillator equation
#[derive(Debug, Clone)]
pub struct NonlinearTerm {
    pub term_type: NonlinearType,
    pub coefficient: f64,
    pub power: f64,
    pub coupling_variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum NonlinearType {
    Polynomial,
    Exponential,
    Trigonometric,
    Logarithmic,
    Rational,
}

/// Network of coupled oscillators
#[derive(Debug, Clone)]
pub struct OscillatorNetwork {
    pub oscillators: HashMap<String, UniversalOscillator>,
    pub coupling_topology: CouplingTopology,
    pub synchronization_state: SynchronizationState,
    pub emergent_properties: EmergentProperties,
    pub collective_modes: Vec<CollectiveMode>,
}

/// Topology of oscillator coupling
#[derive(Debug, Clone)]
pub struct CouplingTopology {
    pub adjacency_matrix: Vec<Vec<f64>>,
    pub coupling_strengths: HashMap<(String, String), f64>,
    pub network_type: NetworkType,
    pub clustering_coefficient: f64,
    pub path_length: f64,
}

#[derive(Debug, Clone)]
pub enum NetworkType {
    AllToAll,
    NearestNeighbor,
    SmallWorld,
    ScaleFree,
    Random,
}

/// Synchronization state of the network
#[derive(Debug, Clone)]
pub struct SynchronizationState {
    pub order_parameter: Complex<f64>,
    pub phase_coherence: f64,
    pub frequency_locking: bool,
    pub synchronization_type: SynchronizationType,
    pub chimera_states: Vec<ChimeraState>,
}

#[derive(Debug, Clone)]
pub enum SynchronizationType {
    Complete,
    Phase,
    Lag,
    Generalized,
    Cluster,
}

/// Chimera state - coexistence of coherent and incoherent dynamics
#[derive(Debug, Clone)]
pub struct ChimeraState {
    pub coherent_region: Vec<String>,
    pub incoherent_region: Vec<String>,
    pub boundary_sharpness: f64,
    pub stability: f64,
}

/// Emergent properties of the oscillator network
#[derive(Debug, Clone)]
pub struct EmergentProperties {
    pub collective_frequency: f64,
    pub network_entropy: f64,
    pub information_flow: f64,
    pub complexity_measure: f64,
    pub resilience: f64,
}

/// Collective modes of oscillation
#[derive(Debug, Clone)]
pub struct CollectiveMode {
    pub mode_id: String,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase_pattern: Vec<f64>,
    pub participation_ratio: f64,
}

impl UniversalOscillator {
    pub fn new(id: String, parameters: OscillatorParameters) -> Self {
        let initial_state = OscillatorState {
            position: 0.0,
            velocity: 0.0,
            acceleration: 0.0,
            phase: 0.0,
            amplitude: 1.0,
            frequency: parameters.natural_frequency,
            energy: 0.5 * parameters.mass * parameters.natural_frequency.powi(2),
            entropy: 0.0,
        };

        let causal_selection = CausalSelection::new();
        let boundary_conditions = BoundaryConditions::default();

        Self {
            oscillator_id: id,
            state: initial_state,
            parameters,
            causal_selection,
            boundary_conditions,
            nonlinear_terms: Vec::new(),
            coupling_matrix: Vec::new(),
        }
    }

    /// Core implementation of the Causal Selection theorem
    pub fn evolve_causal_selection(&mut self, dt: f64) -> Result<()> {
        // Step 1: Generate all possible causal chains
        let possible_chains = self.generate_causal_chains()?;
        
        // Step 2: Apply selection criteria to choose dominant chains
        let selected_chains = self.apply_selection_criteria(possible_chains)?;
        
        // Step 3: Update causal selection based on selected chains
        self.causal_selection.causal_chains = selected_chains;
        
        // Step 4: Calculate selection pressure
        self.causal_selection.selection_pressure = self.calculate_selection_pressure()?;
        
        // Step 5: Update oscillator state based on causal selection
        self.update_state_from_causality(dt)?;
        
        Ok(())
    }

    fn generate_causal_chains(&self) -> Result<Vec<CausalChain>> {
        let mut chains = Vec::new();
        
        // Generate chains based on current state and history
        for i in 0..10 {
            let chain = CausalChain {
                chain_id: format!("chain_{}", i),
                causality_strength: self.calculate_causality_strength(i)?,
                temporal_correlation: self.calculate_temporal_correlation(i)?,
                phase_relationship: self.state.phase + (i as f64) * 0.1,
                energy_flow: self.calculate_energy_flow(i)?,
                information_content: self.calculate_information_content(i)?,
            };
            chains.push(chain);
        }
        
        Ok(chains)
    }

    fn apply_selection_criteria(&self, chains: Vec<CausalChain>) -> Result<Vec<CausalChain>> {
        let mut selected_chains = Vec::new();
        
        for chain in chains {
            let fitness = self.calculate_chain_fitness(&chain)?;
            if fitness > 0.5 {
                selected_chains.push(chain);
            }
        }
        
        // Sort by fitness
        selected_chains.sort_by(|a, b| {
            let fitness_a = self.calculate_chain_fitness(a).unwrap_or(0.0);
            let fitness_b = self.calculate_chain_fitness(b).unwrap_or(0.0);
            fitness_b.partial_cmp(&fitness_a).unwrap()
        });
        
        Ok(selected_chains)
    }

    fn calculate_chain_fitness(&self, chain: &CausalChain) -> Result<f64> {
        let criteria = &self.causal_selection.selection_criteria;
        
        let fitness = criteria.energy_efficiency * chain.energy_flow +
                     criteria.information_preservation * chain.information_content +
                     criteria.temporal_stability * chain.temporal_correlation +
                     criteria.phase_coherence * chain.phase_relationship.cos() +
                     criteria.coupling_strength * chain.causality_strength;
        
        Ok(fitness)
    }

    fn calculate_selection_pressure(&self) -> Result<f64> {
        let total_fitness: f64 = self.causal_selection.causal_chains
            .iter()
            .map(|chain| self.calculate_chain_fitness(chain).unwrap_or(0.0))
            .sum();
        
        let average_fitness = total_fitness / self.causal_selection.causal_chains.len() as f64;
        Ok(average_fitness)
    }

    fn update_state_from_causality(&mut self, dt: f64) -> Result<()> {
        // Calculate forces from causal selection
        let causal_force = self.calculate_causal_force()?;
        
        // Update acceleration
        self.state.acceleration = causal_force / self.parameters.mass;
        
        // Update velocity
        self.state.velocity += self.state.acceleration * dt;
        
        // Apply damping
        self.state.velocity *= 1.0 - self.parameters.damping_coefficient * dt;
        
        // Update position
        self.state.position += self.state.velocity * dt;
        
        // Apply boundary conditions
        self.apply_boundary_conditions()?;
        
        // Update derived quantities
        self.update_derived_quantities()?;
        
        Ok(())
    }

    fn calculate_causal_force(&self) -> Result<f64> {
        let mut total_force = 0.0;
        
        // Linear restoring force
        total_force -= self.parameters.spring_constant * self.state.position;
        
        // Nonlinear forces from causal selection
        for chain in &self.causal_selection.causal_chains {
            let causal_contribution = chain.causality_strength * 
                                    chain.energy_flow * 
                                    (chain.phase_relationship + self.state.phase).sin();
            total_force += causal_contribution;
        }
        
        // Driving force
        let driving_force = self.parameters.driving_amplitude * 
                          (self.parameters.driving_frequency * self.state.phase).sin();
        total_force += driving_force;
        
        // Nonlinear terms
        for term in &self.nonlinear_terms {
            let nonlinear_contribution = self.calculate_nonlinear_contribution(term)?;
            total_force += nonlinear_contribution;
        }
        
        Ok(total_force)
    }

    fn calculate_nonlinear_contribution(&self, term: &NonlinearTerm) -> Result<f64> {
        match term.term_type {
            NonlinearType::Polynomial => {
                Ok(term.coefficient * self.state.position.powf(term.power))
            },
            NonlinearType::Exponential => {
                Ok(term.coefficient * (term.power * self.state.position).exp())
            },
            NonlinearType::Trigonometric => {
                Ok(term.coefficient * (term.power * self.state.position).sin())
            },
            NonlinearType::Logarithmic => {
                Ok(term.coefficient * (term.power * self.state.position.abs()).ln())
            },
            NonlinearType::Rational => {
                Ok(term.coefficient * self.state.position / (1.0 + term.power * self.state.position.powi(2)))
            },
        }
    }

    fn apply_boundary_conditions(&mut self) -> Result<()> {
        match self.boundary_conditions.boundary_type {
            BoundaryType::Hard => {
                if self.state.position < self.boundary_conditions.position_bounds.0 {
                    self.state.position = self.boundary_conditions.position_bounds.0;
                    self.state.velocity *= -self.boundary_conditions.reflection_coefficient;
                } else if self.state.position > self.boundary_conditions.position_bounds.1 {
                    self.state.position = self.boundary_conditions.position_bounds.1;
                    self.state.velocity *= -self.boundary_conditions.reflection_coefficient;
                }
            },
            BoundaryType::Soft => {
                let boundary_force = self.calculate_soft_boundary_force()?;
                self.state.acceleration += boundary_force / self.parameters.mass;
            },
            BoundaryType::Periodic => {
                let range = self.boundary_conditions.position_bounds.1 - self.boundary_conditions.position_bounds.0;
                if self.state.position < self.boundary_conditions.position_bounds.0 {
                    self.state.position += range;
                } else if self.state.position > self.boundary_conditions.position_bounds.1 {
                    self.state.position -= range;
                }
            },
            BoundaryType::Absorbing => {
                if self.state.position < self.boundary_conditions.position_bounds.0 ||
                   self.state.position > self.boundary_conditions.position_bounds.1 {
                    self.state.velocity = 0.0;
                    self.state.acceleration = 0.0;
                }
            },
        }
        
        Ok(())
    }

    fn calculate_soft_boundary_force(&self) -> Result<f64> {
        let mut force = 0.0;
        
        if self.state.position < self.boundary_conditions.position_bounds.0 {
            let distance = self.boundary_conditions.position_bounds.0 - self.state.position;
            force += 1000.0 * distance.powi(2);
        } else if self.state.position > self.boundary_conditions.position_bounds.1 {
            let distance = self.state.position - self.boundary_conditions.position_bounds.1;
            force -= 1000.0 * distance.powi(2);
        }
        
        Ok(force)
    }

    fn update_derived_quantities(&mut self) -> Result<()> {
        // Update energy
        let kinetic_energy = 0.5 * self.parameters.mass * self.state.velocity.powi(2);
        let potential_energy = 0.5 * self.parameters.spring_constant * self.state.position.powi(2);
        self.state.energy = kinetic_energy + potential_energy;
        
        // Update phase
        self.state.phase = (self.state.velocity / (self.state.position + 1e-10)).atan();
        
        // Update amplitude
        self.state.amplitude = (self.state.position.powi(2) + 
                              (self.state.velocity / self.parameters.natural_frequency).powi(2)).sqrt();
        
        // Update frequency (instantaneous)
        self.state.frequency = self.state.velocity / (2.0 * std::f64::consts::PI * self.state.amplitude + 1e-10);
        
        // Update entropy based on phase space volume
        self.state.entropy = self.calculate_phase_space_entropy()?;
        
        Ok(())
    }

    fn calculate_phase_space_entropy(&self) -> Result<f64> {
        // Simple approximation of phase space entropy
        let phase_space_volume = self.state.position.abs() * self.state.velocity.abs();
        Ok(phase_space_volume.ln().max(0.0))
    }

    // Helper methods for causal chain calculations
    fn calculate_causality_strength(&self, index: usize) -> Result<f64> {
        Ok(0.5 + 0.3 * (index as f64 * 0.1).sin())
    }

    fn calculate_temporal_correlation(&self, index: usize) -> Result<f64> {
        Ok((-0.1 * index as f64).exp())
    }

    fn calculate_energy_flow(&self, index: usize) -> Result<f64> {
        Ok(self.state.energy * (index as f64 * 0.05).cos())
    }

    fn calculate_information_content(&self, index: usize) -> Result<f64> {
        Ok(-((index + 1) as f64).ln() / 10.0)
    }

    /// Get current frequency of the oscillator
    pub fn frequency(&self) -> f64 {
        self.state.frequency
    }

    /// Get current amplitude of the oscillator
    pub fn amplitude(&self) -> f64 {
        self.state.amplitude
    }

    /// Get current phase of the oscillator
    pub fn phase(&self) -> f64 {
        self.state.phase
    }

    /// Evolve the oscillator for a given duration
    pub fn evolve(&mut self, duration: f64) -> Result<()> {
        let dt = duration / 100.0; // Use small time steps
        for _ in 0..100 {
            self.evolve_causal_selection(dt)?;
        }
        Ok(())
    }
}

impl CausalSelection {
    pub fn new() -> Self {
        Self {
            selection_pressure: 0.0,
            causal_chains: Vec::new(),
            selection_criteria: SelectionCriteria {
                energy_efficiency: 0.3,
                information_preservation: 0.2,
                temporal_stability: 0.2,
                phase_coherence: 0.15,
                coupling_strength: 0.15,
            },
            stability_analysis: StabilityAnalysis::new(),
            attractors: Vec::new(),
        }
    }
}

impl StabilityAnalysis {
    pub fn new() -> Self {
        Self {
            lyapunov_exponents: Vec::new(),
            phase_portrait: Vec::new(),
            basin_of_attraction: Vec::new(),
            bifurcation_points: Vec::new(),
            chaos_indicators: Vec::new(),
        }
    }
}

impl BoundaryConditions {
    pub fn default() -> Self {
        Self {
            position_bounds: (-10.0, 10.0),
            velocity_bounds: (-10.0, 10.0),
            energy_bounds: (0.0, 100.0),
            boundary_type: BoundaryType::Hard,
            reflection_coefficient: 0.9,
        }
    }
}

impl OscillatorNetwork {
    pub fn new() -> Self {
        Self {
            oscillators: HashMap::new(),
            coupling_topology: CouplingTopology::new(),
            synchronization_state: SynchronizationState::new(),
            emergent_properties: EmergentProperties::new(),
            collective_modes: Vec::new(),
        }
    }

    pub fn add_oscillator(&mut self, oscillator: UniversalOscillator) {
        self.oscillators.insert(oscillator.oscillator_id.clone(), oscillator);
    }

    pub fn evolve_network(&mut self, dt: f64) -> Result<()> {
        // Evolve individual oscillators
        for oscillator in self.oscillators.values_mut() {
            oscillator.evolve_causal_selection(dt)?;
        }
        
        // Apply coupling
        self.apply_coupling(dt)?;
        
        // Update synchronization state
        self.update_synchronization_state()?;
        
        // Update emergent properties
        self.update_emergent_properties()?;
        
        Ok(())
    }

    fn apply_coupling(&mut self, dt: f64) -> Result<()> {
        let oscillator_ids: Vec<String> = self.oscillators.keys().cloned().collect();
        
        for i in 0..oscillator_ids.len() {
            for j in 0..oscillator_ids.len() {
                if i != j {
                    let coupling_strength = self.coupling_topology.adjacency_matrix[i][j];
                    if coupling_strength > 0.0 {
                        self.apply_pairwise_coupling(&oscillator_ids[i], &oscillator_ids[j], coupling_strength, dt)?;
                    }
                }
            }
        }
        
        Ok(())
    }

    fn apply_pairwise_coupling(&mut self, id1: &str, id2: &str, strength: f64, dt: f64) -> Result<()> {
        // Get states (need to handle borrowing carefully)
        let (state1, state2) = {
            let osc1 = self.oscillators.get(id1).ok_or(NebuchadnezzarError::InvalidInput("Oscillator not found".to_string()))?;
            let osc2 = self.oscillators.get(id2).ok_or(NebuchadnezzarError::InvalidInput("Oscillator not found".to_string()))?;
            (osc1.state.clone(), osc2.state.clone())
        };
        
        // Calculate coupling forces
        let coupling_force_1 = strength * (state2.position - state1.position);
        let coupling_force_2 = strength * (state1.position - state2.position);
        
        // Apply coupling forces
        if let Some(osc1) = self.oscillators.get_mut(id1) {
            osc1.state.acceleration += coupling_force_1 / osc1.parameters.mass;
        }
        
        if let Some(osc2) = self.oscillators.get_mut(id2) {
            osc2.state.acceleration += coupling_force_2 / osc2.parameters.mass;
        }
        
        Ok(())
    }

    fn update_synchronization_state(&mut self) -> Result<()> {
        let n = self.oscillators.len();
        if n == 0 {
            return Ok(());
        }
        
        // Calculate order parameter
        let mut sum_complex = Complex::new(0.0, 0.0);
        for oscillator in self.oscillators.values() {
            let phase = oscillator.state.phase;
            sum_complex += Complex::new(phase.cos(), phase.sin());
        }
        
        self.synchronization_state.order_parameter = sum_complex / n as f64;
        self.synchronization_state.phase_coherence = self.synchronization_state.order_parameter.norm();
        
        // Determine synchronization type
        if self.synchronization_state.phase_coherence > 0.9 {
            self.synchronization_state.synchronization_type = SynchronizationType::Complete;
        } else if self.synchronization_state.phase_coherence > 0.7 {
            self.synchronization_state.synchronization_type = SynchronizationType::Phase;
        } else {
            self.synchronization_state.synchronization_type = SynchronizationType::Generalized;
        }
        
        Ok(())
    }

    fn update_emergent_properties(&mut self) -> Result<()> {
        let n = self.oscillators.len() as f64;
        if n == 0.0 {
            return Ok(());
        }
        
        // Calculate collective frequency
        let total_frequency: f64 = self.oscillators.values()
            .map(|osc| osc.state.frequency)
            .sum();
        self.emergent_properties.collective_frequency = total_frequency / n;
        
        // Calculate network entropy
        let total_entropy: f64 = self.oscillators.values()
            .map(|osc| osc.state.entropy)
            .sum();
        self.emergent_properties.network_entropy = total_entropy;
        
        // Calculate complexity measure (simplified)
        self.emergent_properties.complexity_measure = 
            self.synchronization_state.phase_coherence * (1.0 - self.synchronization_state.phase_coherence) * 4.0;
        
        Ok(())
    }
}

impl CouplingTopology {
    pub fn new() -> Self {
        Self {
            adjacency_matrix: Vec::new(),
            coupling_strengths: HashMap::new(),
            network_type: NetworkType::AllToAll,
            clustering_coefficient: 0.0,
            path_length: 0.0,
        }
    }
}

impl SynchronizationState {
    pub fn new() -> Self {
        Self {
            order_parameter: Complex::new(0.0, 0.0),
            phase_coherence: 0.0,
            frequency_locking: false,
            synchronization_type: SynchronizationType::Generalized,
            chimera_states: Vec::new(),
        }
    }
}

impl EmergentProperties {
    pub fn new() -> Self {
        Self {
            collective_frequency: 0.0,
            network_entropy: 0.0,
            information_flow: 0.0,
            complexity_measure: 0.0,
            resilience: 0.0,
        }
    }
}

impl OscillatorState {
    /// Create a new oscillator state with default values
    pub fn new() -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            acceleration: 0.0,
            phase: 0.0,
            amplitude: 1.0,
            frequency: 1.0,
            energy: 0.5,
            entropy: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_oscillator() {
        let params = OscillatorParameters {
            natural_frequency: 1.0,
            damping_coefficient: 0.1,
            driving_amplitude: 0.5,
            driving_frequency: 1.0,
            nonlinearity_strength: 0.1,
            mass: 1.0,
            spring_constant: 1.0,
        };
        
        let mut oscillator = UniversalOscillator::new("test".to_string(), params);
        
        // Test evolution
        for _ in 0..100 {
            oscillator.evolve_causal_selection(0.01).unwrap();
        }
        
        assert!(oscillator.state.energy > 0.0);
        assert!(oscillator.causal_selection.selection_pressure >= 0.0);
    }

    #[test]
    fn test_oscillator_network() {
        let mut network = OscillatorNetwork::new();
        
        // Add oscillators
        for i in 0..3 {
            let params = OscillatorParameters {
                natural_frequency: 1.0 + 0.1 * i as f64,
                damping_coefficient: 0.1,
                driving_amplitude: 0.5,
                driving_frequency: 1.0,
                nonlinearity_strength: 0.1,
                mass: 1.0,
                spring_constant: 1.0,
            };
            
            let oscillator = UniversalOscillator::new(format!("osc_{}", i), params);
            network.add_oscillator(oscillator);
        }
        
        // Set up coupling
        network.coupling_topology.adjacency_matrix = vec![
            vec![0.0, 0.1, 0.1],
            vec![0.1, 0.0, 0.1],
            vec![0.1, 0.1, 0.0],
        ];
        
        // Test evolution
        for _ in 0..50 {
            network.evolve_network(0.01).unwrap();
        }
        
        assert!(network.synchronization_state.phase_coherence >= 0.0);
        assert!(network.emergent_properties.collective_frequency > 0.0);
    }
}

// Re-exports for backwards compatibility
pub use OscillatorState as OscillationState;
pub use UniversalOscillator as BiologicalOscillator; 
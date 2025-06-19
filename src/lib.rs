//! # Nebuchadnezzar: Quantum-Enhanced ATP-Driven Biological Circuit Simulation
//! 
//! A comprehensive system for simulating biological processes using quantum-enhanced
//! hierarchical probabilistic electric circuits with foundational theorems for
//! membrane quantum computation, universal oscillatory dynamics, and entropy manipulation.
//! 
//! ## Revolutionary ATP-Oscillatory-Membrane Quantum Framework
//! 
//! This framework implements three revolutionary insights:
//! 1. **ATP as Universal Energy Currency**: Using dx/dATP instead of dx/dt for biologically meaningful rates
//! 2. **Oscillatory Entropy**: Statistical distributions of oscillation endpoints as fundamental entropy measure
//! 3. **Membrane Quantum Computation**: Environment-Assisted Quantum Transport (ENAQT) for room-temperature quantum computation
//! 
//! ## Foundational Framework
//! 
//! - **Membrane Quantum Computation**: ENAQT-based room-temperature quantum computation in biological membranes
//! - **Universal Oscillatory Dynamics**: Causal Selection theorem governing oscillatory behavior in bounded nonlinear systems  
//! - **Entropy Reformulation**: Probabilistic points and resolutions for direct entropy manipulation
//! - **ATP-based Rate Modeling**: Uses dx/dATP instead of dx/dt for biologically meaningful rates
//! - **Hierarchical Probabilistic Circuits**: 4-level framework from molecular to tissue level
//! - **Biological Quantum Computer**: Complete implementation of ATP-driven quantum biological computation
//! 
//! ## Architecture
//! 
//! The system is organized into several key modules:
//! 
//! - **biological_quantum_computer**: Core ATP-Oscillatory-Membrane quantum biological simulator
//! - **biological_quantum_solver**: Advanced solver for biological quantum computation
//! - **quantum_metabolism_analyzer**: AI-driven analysis and optimization of quantum metabolic networks
//! - **quantum_membranes**: Environment-Assisted Quantum Transport (ENAQT) implementation
//! - **oscillatory_dynamics**: Universal oscillatory framework with Causal Selection
//! - **entropy_manipulation**: Probabilistic points and resolutions system
//! - **circuits**: Electrical circuit foundation with ion channels, enzyme circuits, and grid systems
//! - **solvers**: ATP-based differential equation integration
//! - **systems_biology**: ATP kinetics and cellular energy management
//! - **utils**: Mathematical utilities and helper functions
//! 
//! ## Usage Example
//! 
//! ```rust
//! use nebuchadnezzar::{
//!     biological_quantum_computer::{BiologicalQuantumState, AtpCoordinates, OscillatoryCoordinates, MembraneQuantumCoordinates, OscillatoryEntropyCoordinates},
//!     biological_quantum_solver::{BiologicalQuantumComputerSolver, QuantumComputationTarget, ComputationType},
//!     quantum_metabolism_analyzer::QuantumMetabolismAnalyzer,
//!     circuits::{CircuitFactory, AdaptiveGrid},
//!     systems_biology::AtpPool,
//!     solvers::AtpRk4Integrator,
//! };
//! 
//! // Create initial biological quantum state
//! let atp_coords = AtpCoordinates::new(5.0, 1.0, 0.5); // 5mM ATP, 1mM ADP, 0.5mM Pi
//! let oscillatory_coords = OscillatoryCoordinates::new(10); // 10 oscillators
//! let membrane_coords = MembraneQuantumCoordinates::new(5); // 5 membrane proteins
//! let oscillator_names: Vec<String> = (0..10).map(|i| format!("osc_{}", i)).collect();
//! let entropy_coords = OscillatoryEntropyCoordinates::new(&oscillator_names);
//! 
//! let initial_state = BiologicalQuantumState {
//!     atp_coords,
//!     oscillatory_coords,
//!     membrane_coords,
//!     entropy_coords,
//! };
//! 
//! // Create quantum computation target
//! let target = QuantumComputationTarget {
//!     required_coherence: 0.8,
//!     target_states: vec!["protein_0".to_string(), "protein_1".to_string()],
//!     computation_type: ComputationType::ProteinFolding,
//! };
//! 
//! // Run biological quantum computation
//! let mut solver = BiologicalQuantumComputerSolver::new();
//! let result = solver.solve_biological_quantum_computation(
//!     &initial_state,
//!     10.0,  // ATP budget: 10 mM
//!     1.0,   // Time horizon: 1 second
//!     &target
//! );
//! 
//! // Analyze results with quantum metabolism analyzer
//! let analyzer = QuantumMetabolismAnalyzer::new();
//! if let Ok(computation_result) = result {
//!     let coherence_analysis = analyzer.analyze_quantum_metabolic_coherence(&computation_result.trajectory);
//!     println!("Quantum advantage factor: {:.2}", coherence_analysis.quantum_advantage_factor);
//! }
//! ```

pub mod error;
pub mod quantum_membranes;
pub mod oscillatory_dynamics;
pub mod entropy_manipulation;
pub mod biological_integration;
pub mod circuits;
pub mod solvers;
pub mod systems_biology;
pub mod utils;

// Revolutionary ATP-Oscillatory-Membrane Quantum Framework
pub mod biological_quantum_computer;
pub mod biological_quantum_solver;
pub mod quantum_metabolism_analyzer;

// Re-export commonly used types
pub use error::{NebuchadnezzarError, Result};

// Foundational framework exports
pub use quantum_membranes::{
    EnaqtProcessor,
    QuantumMembrane,
    CoherenceController,
    TunnelingJunction,
    EnvironmentalCoupling,
};

pub use oscillatory_dynamics::{
    CausalSelector,
    UniversalOscillator,
    PhaseLocker,
    FrequencyEntrainer,
    OscillatorNetwork,
};

pub use entropy_manipulation::{
    EntropyPoint,
    Resolution,
    PerturbationEngine,
    ValidationNetwork,
    EntropyManipulator,
};

pub use circuits::{
    Circuit,
    CircuitFactory,
    CircuitGrid,
    AdaptiveGrid,
    HierarchicalSystem,
    MembraneModel,
    CircuitNetwork,
    ProbabilisticIonChannel,
    EnzymeProbCircuit,
    EnzymeCircuitFactory,
    VoltageHierarchy,
    TemporalEvidence,
};

pub use solvers::{
    SystemState,
    AtpIntegrator,
    AtpEulerIntegrator,
    AtpRk4Integrator,
    AdaptiveStepIntegrator,
};

pub use systems_biology::{
    AtpPool,
    AtpKinetics,
    EnergyCharge,
};

pub use utils::{
    ThermodynamicState,
    UnitConverter,
    NumericalMethods,
    MatrixOperations,
};

pub use biological_integration::{
    BiologicalSystem,
    CellularComponent,
    MetabolicNetwork,
    SignalingPathway,
    GeneRegulation,
    ProteinInteraction,
    BiochemicalReaction,
    TransportProcess,
    CellularEnvironment,
    SystemIntegrator,
};

// Revolutionary ATP-Oscillatory-Membrane Quantum Framework exports
pub use biological_quantum_computer::{
    BiologicalQuantumState,
    AtpCoordinates,
    OscillatoryCoordinates,
    MembraneQuantumCoordinates,
    OscillatoryEntropyCoordinates,
    OscillationState,
    MembraneOscillation,
    QuantumStateAmplitude,
    TunnelingState,
    MembraneProperties,
    LipidComposition,
    EndpointDistribution,
    OscillationEndpoint,
    MembraneQuantumEndpoint,
    RadicalEndpoint,
    RadicalType,
    BiologicalQuantumHamiltonian,
    AtpEnergyFunction,
    OscillatoryEnergyFunction,
    MembraneQuantumEnergyFunction,
    TripleCouplingFunction,
};

pub use biological_quantum_solver::{
    BiologicalQuantumComputerSolver,
    IntegrationMethod,
    StepController,
    EntropyConstraintEnforcer,
    QuantumComputationTarget,
    ComputationType,
    BiologicalQuantumResult,
    BiologicalQuantumTrajectory,
    BiologicalQuantumTrajectoryPoint,
};

pub use quantum_metabolism_analyzer::{
    QuantumMetabolismAnalyzer,
    QuantumCoherenceAnalysis,
    OscillationCouplingPatterns,
    QuantumEfficiencyMetrics,
    CoherencePredictions,
    AtpQuantumCoupling,
    OptimalFrequency,
    MetabolicOptimizationResult,
    TissueLevelAnalysis,
    RadicalDamageAnalysis,
    TissueType,
    MetabolicPathway,
};

/// Main simulation engine that orchestrates all components
pub struct NebuchadnezzarEngine {
    pub hierarchical_system: HierarchicalSystem,
    pub global_atp_pool: AtpPool,
    pub integrator: Box<dyn AtpIntegrator>,
    pub current_time: f64,
    pub simulation_parameters: SimulationParameters,
    /// NEW: Biological quantum computer for quantum metabolic processes
    pub quantum_computer: Option<BiologicalQuantumComputerSolver>,
    /// NEW: Quantum metabolism analyzer for advanced analysis
    pub metabolism_analyzer: Option<QuantumMetabolismAnalyzer>,
}

/// Configuration parameters for the simulation
#[derive(Debug, Clone)]
pub struct SimulationParameters {
    pub time_step: f64,
    pub max_time: f64,
    pub atp_threshold: f64,
    pub voltage_tolerance: f64,
    pub adaptive_stepping: bool,
    pub resolution_threshold: f64,
    pub output_frequency: usize,
    /// NEW: Enable quantum computation features
    pub enable_quantum_computation: bool,
    /// NEW: Quantum coherence threshold
    pub quantum_coherence_threshold: f64,
    /// NEW: Maximum radical damage tolerance
    pub max_radical_damage: f64,
}

impl Default for SimulationParameters {
    fn default() -> Self {
        Self {
            time_step: 0.001,
            max_time: 10.0,
            atp_threshold: 0.01,
            voltage_tolerance: 0.001,
            adaptive_stepping: true,
            resolution_threshold: 0.7,
            output_frequency: 100,
            enable_quantum_computation: true,
            quantum_coherence_threshold: 0.1,
            max_radical_damage: 0.5,
        }
    }
}

impl NebuchadnezzarEngine {
    /// Create a new simulation engine
    pub fn new(initial_atp: f64) -> Self {
        Self {
            hierarchical_system: HierarchicalSystem::new(),
            global_atp_pool: AtpPool::new(initial_atp, 1.0, 0.5),
            integrator: Box::new(AtpRk4Integrator::new(0.001)),
            current_time: 0.0,
            simulation_parameters: SimulationParameters::default(),
            quantum_computer: Some(BiologicalQuantumComputerSolver::new()),
            metabolism_analyzer: Some(QuantumMetabolismAnalyzer::new()),
        }
    }

    /// Create a new simulation engine with quantum computation enabled
    pub fn new_with_quantum_computation(initial_atp: f64) -> Self {
        let mut engine = Self::new(initial_atp);
        engine.simulation_parameters.enable_quantum_computation = true;
        engine
    }

    /// Run the complete simulation with quantum computation
    pub fn run_quantum_enhanced_simulation(&mut self) -> Result<QuantumEnhancedSimulationResults> {
        let mut results = QuantumEnhancedSimulationResults::new();
        let mut step_count = 0;

        // Initialize quantum computation state if enabled
        let mut quantum_state = if self.simulation_parameters.enable_quantum_computation {
            Some(self.initialize_quantum_state()?)
        } else {
            None
        };

        while self.current_time < self.simulation_parameters.max_time &&
              self.global_atp_pool.atp_concentration > self.simulation_parameters.atp_threshold {
            
            // Calculate ATP consumption for this step
            let delta_atp = self.calculate_step_atp_consumption()?;
            
            // Solve hierarchical system
            let hierarchical_state = self.hierarchical_system.solve_hierarchical_system(delta_atp)?;
            
            // Update global ATP pool
            self.global_atp_pool.consume_atp(delta_atp)?;
            
            // Run quantum computation step if enabled
            if let (Some(ref mut quantum_computer), Some(ref mut q_state)) = 
               (&mut self.quantum_computer, &mut quantum_state) {
                
                // Create quantum computation target
                let target = QuantumComputationTarget {
                    required_coherence: self.simulation_parameters.quantum_coherence_threshold,
                    target_states: vec!["metabolic_efficiency".to_string()],
                    computation_type: ComputationType::MetabolicOptimization,
                };
                
                // Run single quantum step
                let quantum_result = quantum_computer.solve_biological_quantum_computation(
                    q_state,
                    delta_atp,
                    self.simulation_parameters.time_step,
                    &target,
                )?;
                
                // Update quantum state
                *q_state = quantum_result.final_state;
                
                // Store quantum trajectory
                results.quantum_trajectories.push(quantum_result.trajectory);
            }
            
            // Update time
            self.current_time += self.simulation_parameters.time_step;
            step_count += 1;
            
            // Record results
            if step_count % self.simulation_parameters.output_frequency == 0 {
                results.add_timepoint(
                    self.current_time,
                    hierarchical_state,
                    self.global_atp_pool.clone(),
                    quantum_state.clone(),
                );
            }
        }
        
        // Perform final quantum metabolism analysis
        if let Some(ref analyzer) = self.metabolism_analyzer {
            if !results.quantum_trajectories.is_empty() {
                let coherence_analysis = analyzer.analyze_quantum_metabolic_coherence(
                    &results.quantum_trajectories[0]
                );
                results.coherence_analysis = Some(coherence_analysis);
                
                // Analyze tissue-level effects
                let tissue_analysis = analyzer.analyze_tissue_level_effects(
                    &results.quantum_trajectories[0],
                    TissueType::Neural, // Default to neural tissue
                );
                results.tissue_analysis = Some(tissue_analysis);
                
                // Analyze radical damage
                let damage_analysis = analyzer.analyze_radical_damage_patterns(
                    &results.quantum_trajectories[0]
                );
                results.damage_analysis = Some(damage_analysis);
            }
        }
        
        results.finalize(self.current_time, step_count);
        
        Ok(results)
    }

    fn initialize_quantum_state(&self) -> Result<BiologicalQuantumState> {
        let atp_coords = AtpCoordinates::new(
            self.global_atp_pool.atp_concentration,
            self.global_atp_pool.adp_concentration,
            0.5, // Pi concentration
        );
        
        let oscillatory_coords = OscillatoryCoordinates::new(20); // 20 oscillators
        let membrane_coords = MembraneQuantumCoordinates::new(10); // 10 membrane proteins
        
        let oscillator_names: Vec<String> = (0..20)
            .map(|i| format!("osc_{}", i))
            .collect();
        let entropy_coords = OscillatoryEntropyCoordinates::new(&oscillator_names);
        
        Ok(BiologicalQuantumState {
            atp_coords,
            oscillatory_coords,
            membrane_coords,
            entropy_coords,
        })
    }

    /// Run the complete simulation (original method)
    pub fn run_simulation(&mut self) -> Result<SimulationResults> {
        let mut results = SimulationResults::new();
        let mut step_count = 0;

        while self.current_time < self.simulation_parameters.max_time &&
              self.global_atp_pool.atp_concentration > self.simulation_parameters.atp_threshold {
            
            // Calculate ATP consumption for this step
            let delta_atp = self.calculate_step_atp_consumption()?;
            
            // Solve hierarchical system
            let hierarchical_state = self.hierarchical_system.solve_hierarchical_system(delta_atp)?;
            
            // Update global ATP pool
            self.global_atp_pool.consume_atp(delta_atp)?;
            
            // Update time
            self.current_time += self.simulation_parameters.time_step;
            step_count += 1;
            
            // Record results
            if step_count % self.simulation_parameters.output_frequency == 0 {
                results.add_timepoint(self.current_time, hierarchical_state, self.global_atp_pool.clone());
            }
        }
        
        results.finalize(self.current_time, step_count);
        
        Ok(results)
    }

    fn calculate_step_atp_consumption(&self) -> Result<f64> {
        let base_consumption = 0.1; // Base ATP consumption rate
        let hierarchical_load = self.hierarchical_system.calculate_total_load()?;
        Ok(base_consumption + hierarchical_load * 0.01)
    }

    pub fn add_neuron_circuit(&mut self, neuron_id: String) -> Result<()> {
        let neuron_circuit = CircuitFactory::create_neuron_membrane();
        self.hierarchical_system.add_circuit(
            neuron_id,
            circuits::CircuitType::Neuron,
            neuron_circuit,
        )
    }

    pub fn add_metabolic_pathway(&mut self, pathway_name: String, pathway_type: MetabolicPathwayType) -> Result<()> {
        let pathway_circuit = match pathway_type {
            MetabolicPathwayType::Glycolysis => self.create_glycolysis_circuit()?,
            MetabolicPathwayType::CitricAcidCycle => self.create_citric_acid_cycle()?,
            MetabolicPathwayType::ElectronTransport => self.create_electron_transport_chain()?,
        };
        
        self.hierarchical_system.add_circuit(
            pathway_name,
            circuits::CircuitType::Metabolic,
            pathway_circuit,
        )
    }

    fn create_glycolysis_circuit(&self) -> Result<AdaptiveGrid> {
        let mut grid = AdaptiveGrid::new(10, 10);
        
        // Add glucose input
        grid.add_voltage_source(0, 0, 5.0)?; // 5V representing glucose
        
        // Add enzyme reactions as circuit elements
        for i in 1..10 {
            grid.add_resistor(i-1, 0, i, 0, 1.0)?; // Each enzyme as 1Ω resistor
        }
        
        // Add ATP generation points
        grid.add_voltage_source(3, 0, 2.0)?; // ATP from 1,3-bisphosphoglycerate
        grid.add_voltage_source(9, 0, 2.0)?; // ATP from pyruvate kinase
        
        Ok(grid)
    }

    fn create_citric_acid_cycle(&self) -> Result<AdaptiveGrid> {
        let mut grid = AdaptiveGrid::new(8, 8);
        
        // Circular topology for TCA cycle
        let positions = [
            (0, 0), (1, 0), (2, 0), (2, 1),
            (2, 2), (1, 2), (0, 2), (0, 1)
        ];
        
        // Add enzymes as resistors in circular pattern
        for i in 0..8 {
            let next = (i + 1) % 8;
            let (x1, y1) = positions[i];
            let (x2, y2) = positions[next];
            grid.add_resistor(x1, y1, x2, y2, 0.5)?; // Lower resistance for efficient cycle
        }
        
        // Add NADH/FADH2 generation points
        grid.add_voltage_source(1, 0, 3.0)?; // Isocitrate dehydrogenase
        grid.add_voltage_source(2, 1, 3.0)?; // α-Ketoglutarate dehydrogenase
        grid.add_voltage_source(1, 2, 2.0)?; // Succinate dehydrogenase (FADH2)
        grid.add_voltage_source(0, 1, 3.0)?; // Malate dehydrogenase
        
        Ok(grid)
    }

    fn create_electron_transport_chain(&self) -> Result<AdaptiveGrid> {
        let mut grid = AdaptiveGrid::new(12, 4);
        
        // Complex I
        grid.add_voltage_source(0, 0, 10.0)?; // NADH input
        grid.add_resistor(0, 0, 3, 0, 2.0)?; // Complex I
        
        // Complex II
        grid.add_voltage_source(0, 1, 6.0)?; // FADH2 input
        grid.add_resistor(0, 1, 2, 1, 1.5)?; // Complex II
        
        // Coenzyme Q pool
        grid.add_resistor(3, 0, 6, 0, 0.5)?; // Q pool
        grid.add_resistor(2, 1, 6, 0, 0.5)?; // Q pool connection
        
        // Complex III
        grid.add_resistor(6, 0, 9, 0, 2.0)?; // Complex III
        
        // Complex IV
        grid.add_resistor(9, 0, 11, 0, 2.5)?; // Complex IV
        grid.add_voltage_source(11, 0, -20.0)?; // O2 reduction (negative for electron acceptance)
        
        // ATP synthase
        grid.add_voltage_source(11, 3, 15.0)?; // Proton gradient
        grid.add_resistor(11, 3, 11, 0, 3.0)?; // ATP synthase
        
        Ok(grid)
    }

    /// Create a comprehensive demonstration of the quantum biological computer
    pub fn create_quantum_biology_demonstration() -> Result<Self> {
        let mut engine = Self::new_with_quantum_computation(10.0); // 10 mM initial ATP
        
        // Add multiple neuron circuits
        for i in 0..5 {
            engine.add_neuron_circuit(format!("neuron_{}", i))?;
        }
        
        // Add metabolic pathways
        engine.add_metabolic_pathway("glycolysis".to_string(), MetabolicPathwayType::Glycolysis)?;
        engine.add_metabolic_pathway("tca_cycle".to_string(), MetabolicPathwayType::CitricAcidCycle)?;
        engine.add_metabolic_pathway("electron_transport".to_string(), MetabolicPathwayType::ElectronTransport)?;
        
        // Configure for quantum computation
        engine.simulation_parameters.enable_quantum_computation = true;
        engine.simulation_parameters.quantum_coherence_threshold = 0.5;
        engine.simulation_parameters.max_radical_damage = 0.3;
        engine.simulation_parameters.max_time = 5.0; // 5 seconds
        
        Ok(engine)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MetabolicPathwayType {
    Glycolysis,
    CitricAcidCycle,
    ElectronTransport,
}

pub struct SimulationResults {
    pub timepoints: Vec<f64>,
    pub hierarchical_states: Vec<circuits::hierarchical_framework::HierarchicalState>,
    pub atp_levels: Vec<f64>,
    pub energy_charges: Vec<f64>,
    pub total_simulation_time: f64,
    pub total_steps: usize,
    pub final_atp_level: f64,
}

impl SimulationResults {
    fn new() -> Self {
        Self {
            timepoints: Vec::new(),
            hierarchical_states: Vec::new(),
            atp_levels: Vec::new(),
            energy_charges: Vec::new(),
            total_simulation_time: 0.0,
            total_steps: 0,
            final_atp_level: 0.0,
        }
    }

    fn add_timepoint(&mut self, time: f64, state: circuits::hierarchical_framework::HierarchicalState, atp_pool: AtpPool) {
        self.timepoints.push(time);
        self.hierarchical_states.push(state);
        self.atp_levels.push(atp_pool.atp_concentration);
        self.energy_charges.push(atp_pool.energy_charge);
    }

    fn finalize(&mut self, final_time: f64, total_steps: usize) {
        self.total_simulation_time = final_time;
        self.total_steps = total_steps;
        self.final_atp_level = self.atp_levels.last().copied().unwrap_or(0.0);
    }

    pub fn get_summary(&self) -> SimulationSummary {
        let average_atp = self.atp_levels.iter().sum::<f64>() / self.atp_levels.len() as f64;
        let min_atp = self.atp_levels.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_atp = self.atp_levels.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let average_energy_charge = self.energy_charges.iter().sum::<f64>() / self.energy_charges.len() as f64;
        
        let atp_depletion_rate = if self.total_simulation_time > 0.0 {
            (self.atp_levels.first().unwrap_or(&0.0) - self.final_atp_level) / self.total_simulation_time
        } else {
            0.0
        };

        SimulationSummary {
            total_time: self.total_simulation_time,
            total_steps: self.total_steps,
            average_atp,
            min_atp,
            max_atp,
            final_atp: self.final_atp_level,
            average_energy_charge,
            atp_depletion_rate,
        }
    }
}

#[derive(Debug)]
pub struct SimulationSummary {
    pub total_time: f64,
    pub total_steps: usize,
    pub average_atp: f64,
    pub min_atp: f64,
    pub max_atp: f64,
    pub final_atp: f64,
    pub average_energy_charge: f64,
    pub atp_depletion_rate: f64,
}

/// Enhanced simulation results with quantum computation data
#[derive(Debug)]
pub struct QuantumEnhancedSimulationResults {
    pub timepoints: Vec<f64>,
    pub hierarchical_states: Vec<circuits::hierarchical_framework::HierarchicalState>,
    pub atp_levels: Vec<f64>,
    pub energy_charges: Vec<f64>,
    pub quantum_states: Vec<Option<BiologicalQuantumState>>,
    pub quantum_trajectories: Vec<BiologicalQuantumTrajectory>,
    pub coherence_analysis: Option<QuantumCoherenceAnalysis>,
    pub tissue_analysis: Option<TissueLevelAnalysis>,
    pub damage_analysis: Option<RadicalDamageAnalysis>,
    pub total_simulation_time: f64,
    pub total_steps: usize,
    pub final_atp_level: f64,
}

impl QuantumEnhancedSimulationResults {
    fn new() -> Self {
        Self {
            timepoints: Vec::new(),
            hierarchical_states: Vec::new(),
            atp_levels: Vec::new(),
            energy_charges: Vec::new(),
            quantum_states: Vec::new(),
            quantum_trajectories: Vec::new(),
            coherence_analysis: None,
            tissue_analysis: None,
            damage_analysis: None,
            total_simulation_time: 0.0,
            total_steps: 0,
            final_atp_level: 0.0,
        }
    }

    fn add_timepoint(
        &mut self,
        time: f64,
        state: circuits::hierarchical_framework::HierarchicalState,
        atp_pool: AtpPool,
        quantum_state: Option<BiologicalQuantumState>,
    ) {
        self.timepoints.push(time);
        self.hierarchical_states.push(state);
        self.atp_levels.push(atp_pool.atp_concentration);
        self.energy_charges.push(atp_pool.energy_charge);
        self.quantum_states.push(quantum_state);
    }

    fn finalize(&mut self, final_time: f64, total_steps: usize) {
        self.total_simulation_time = final_time;
        self.total_steps = total_steps;
        self.final_atp_level = self.atp_levels.last().copied().unwrap_or(0.0);
    }

    pub fn get_quantum_summary(&self) -> QuantumSimulationSummary {
        let average_atp = self.atp_levels.iter().sum::<f64>() / self.atp_levels.len() as f64;
        let min_atp = self.atp_levels.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_atp = self.atp_levels.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        let quantum_advantage = self.coherence_analysis.as_ref()
            .map(|ca| ca.quantum_advantage_factor)
            .unwrap_or(1.0);
        
        let radical_damage = self.damage_analysis.as_ref()
            .map(|da| da.cumulative_damage)
            .unwrap_or(0.0);

        QuantumSimulationSummary {
            total_time: self.total_simulation_time,
            total_steps: self.total_steps,
            average_atp,
            min_atp,
            max_atp,
            final_atp: self.final_atp_level,
            quantum_advantage_factor: quantum_advantage,
            coherence_lifetime: self.coherence_analysis.as_ref()
                .map(|ca| ca.efficiency_metrics.coherence_lifetime)
                .unwrap_or(0.0),
            radical_damage_accumulated: radical_damage,
            quantum_computation_success: quantum_advantage > 1.0,
        }
    }
}

#[derive(Debug)]
pub struct QuantumSimulationSummary {
    pub total_time: f64,
    pub total_steps: usize,
    pub average_atp: f64,
    pub min_atp: f64,
    pub max_atp: f64,
    pub final_atp: f64,
    pub quantum_advantage_factor: f64,
    pub coherence_lifetime: f64,
    pub radical_damage_accumulated: f64,
    pub quantum_computation_success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = NebuchadnezzarEngine::new(10.0);
        assert_eq!(engine.global_atp_pool.atp_concentration, 10.0);
        assert_eq!(engine.current_time, 0.0);
    }

    #[test]
    fn test_add_neuron_circuit() {
        let mut engine = NebuchadnezzarEngine::new(10.0);
        let result = engine.add_neuron_circuit("test_neuron".to_string());
        assert!(result.is_ok());
        assert_eq!(engine.hierarchical_system.molecular_level.len(), 1);
    }

    #[test]
    fn test_add_metabolic_pathway() {
        let mut engine = NebuchadnezzarEngine::new(15.0);
        let result = engine.add_metabolic_pathway("glycolysis".to_string(), MetabolicPathwayType::Glycolysis);
        assert!(result.is_ok());
        assert_eq!(engine.hierarchical_system.organelle_level.len(), 1);
    }

    #[test]
    fn test_simulation_parameters() {
        let params = SimulationParameters::default();
        assert_eq!(params.time_step, 0.001);
        assert_eq!(params.max_time, 10.0);
        assert_eq!(params.atp_threshold, 0.01);
    }
} 
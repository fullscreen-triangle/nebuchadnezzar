---
layout: page
title: "API Reference"
permalink: /api-reference/
---

# API Reference

Complete API documentation for the Nebuchadnezzar framework, including all modules, structs, traits, and functions.

## Core Modules

### `nebuchadnezzar::prelude`
Essential imports for most Nebuchadnezzar applications.

```rust
use nebuchadnezzar::prelude::*;
```

Includes:
- `AtpOscillatoryMembraneSimulator`
- `QuantumMembrane`
- `BiologicalMaxwellsDemon`
- `CircuitGrid`
- `TurbulanceEngine`
- Common result types and configurations

## ATP Oscillatory Systems

### `AtpOscillatoryMembraneSimulator`

Main simulation engine for ATP-based biological systems.

```rust
pub struct AtpOscillatoryMembraneSimulator {
    config: OscillatoryConfig,
    membranes: Vec<QuantumMembrane>,
    maxwell_demons: Vec<BiologicalMaxwellsDemon>,
    circuits: CircuitGrid,
}
```

#### Methods

##### `new(config: OscillatoryConfig) -> Self`
Creates a new ATP oscillatory system.

**Parameters:**
- `config`: System configuration parameters

**Example:**
```rust
let system = AtpOscillatoryMembraneSimulator::new(
    OscillatoryConfig {
        atp_concentration: 5.0e-3,
        adp_concentration: 0.5e-3,
        temperature: 310.0,
        ph: 7.4,
        ionic_strength: 0.15,
    }
);
```

##### `simulate_atp_cycles(cycles: u64) -> Result<SimulationResults>`
Runs simulation for specified ATP cycles.

**Parameters:**
- `cycles`: Number of ATP hydrolysis cycles to simulate

**Returns:**
- `SimulationResults`: Comprehensive simulation data

**Example:**
```rust
let results = system.simulate_atp_cycles(1000)?;
println!("Coherence: {:.3}", results.quantum_coherence);
```

##### `add_membrane(&mut self, membrane: QuantumMembrane)`
Adds quantum membrane to the system.

##### `add_maxwell_demon(&mut self, demon: BiologicalMaxwellsDemon)`
Adds Maxwell's demon for information catalysis.

### `OscillatoryConfig`

Configuration for ATP oscillatory systems.

```rust
pub struct OscillatoryConfig {
    pub atp_concentration: f64,      // Molar concentration
    pub adp_concentration: f64,      // Molar concentration
    pub temperature: f64,            // Kelvin
    pub ph: f64,                     // pH units
    pub ionic_strength: f64,         // Molar ionic strength
}
```

### `SimulationResults`

Results from ATP oscillatory simulation.

```rust
pub struct SimulationResults {
    pub cycles_completed: u64,
    pub avg_frequency: f64,                    // Hz
    pub quantum_coherence: f64,                // 0.0-1.0
    pub information_catalysis_efficiency: f64,  // 0.0-1.0
    pub energy_conservation_ratio: f64,         // Energy conserved
    pub oscillation_data: Vec<OscillationPoint>,
    pub membrane_transport_data: Vec<TransportEvent>,
}
```

## Quantum Membrane Systems

### `QuantumMembrane`

Quantum-coherent biological membrane with ENAQT capabilities.

```rust
pub struct QuantumMembrane {
    coherence_time: f64,
    environmental_coupling: f64,
    tunneling_probability: f64,
    fire_light_optimization: f64,
    ion_collective_field: CollectiveQuantumField,
}
```

#### Methods

##### `new(config: MembraneConfig) -> Self`
Creates new quantum membrane.

##### `transport_ion(ion: Ion, config: TransportConfig) -> Result<TransportResult>`
Simulates ion transport through quantum membrane.

**Parameters:**
- `ion`: Ion type (Na+, K+, Ca2+, H+, etc.)
- `config`: Transport configuration

**Returns:**
- `TransportResult`: Transport efficiency and quantum contribution data

##### `evolve_quantum_state(dt: f64) -> QuantumState`
Evolves quantum state using Schrödinger equation.

### `MembraneConfig`

Configuration for quantum membranes.

```rust
pub struct MembraneConfig {
    pub coherence_time: f64,           // Seconds
    pub tunneling_probability: f64,     // 0.0-1.0
    pub fire_light_optimization: f64,   // 0.0-1.0
    pub environmental_coupling: f64,    // 0.0-1.0
}
```

### `Ion`

Ion types for membrane transport.

```rust
pub enum Ion {
    Sodium,      // Na+
    Potassium,   // K+
    Calcium,     // Ca2+
    Hydrogen,    // H+
    Magnesium,   // Mg2+
    Chloride,    // Cl-
    Custom { charge: i32, mass: f64 },
}
```

## Biological Maxwell's Demons

### `BiologicalMaxwellsDemon`

Information catalyst for biological processes.

```rust
pub struct BiologicalMaxwellsDemon {
    input_filter: InformationFilter,
    output_channel: ResponseChannel,
    catalytic_cycles: u64,
    agency_recognition: bool,
    associative_memory: AssociativeMemoryNetwork,
}
```

#### Methods

##### `new(config: MaxwellDemonConfig) -> Self`
Creates new Maxwell's demon.

##### `process_information(input: &InformationSet) -> ProcessingResult`
Processes information through catalytic amplification.

**Returns:**
- `ProcessingResult`: Information reduction and response data

##### `detect_agency(input: &InformationSet) -> AgencyDetectionResult`
Detects individual agency in information patterns.

### `MaxwellDemonConfig`

Configuration for Maxwell's demons.

```rust
pub struct MaxwellDemonConfig {
    pub information_filter_threshold: f64,
    pub catalytic_cycles: u64,
    pub agency_recognition: bool,
    pub associative_memory_size: usize,
}
```

## Circuit Systems

### `CircuitGrid`

Hierarchical electrical circuit representation of biological systems.

```rust
pub struct CircuitGrid {
    dimensions: (usize, usize),
    enzyme_circuits: Vec<EnzymeCircuit>,
    ion_channels: Vec<IonChannelCircuit>,
    fractal_circuits: Vec<FractalCircuit>,
    quantum_circuits: Vec<QuantumCircuit>,
}
```

#### Methods

##### `new(width: usize, height: usize) -> Self`
Creates new circuit grid.

##### `add_enzyme_circuit(&mut self, circuit: EnzymeCircuit, position: Position)`
Adds enzyme circuit at specified position.

##### `add_ion_channel(&mut self, channel: IonChannelCircuit, position: Position)`
Adds ion channel circuit.

##### `simulate_interactions(atp_cycles: u64) -> Result<CircuitResults>`
Simulates circuit interactions over ATP cycles.

### `EnzymeCircuit`

Electrical circuit model of enzyme function.

```rust
pub struct EnzymeCircuit {
    enzyme_type: EnzymeType,
    kinetic_parameters: KineticParameters,
    electrical_properties: ElectricalProperties,
}
```

### `EnzymeType`

Types of enzymes that can be modeled.

```rust
pub enum EnzymeType {
    Hexokinase,
    ATPSynthase,
    Kinase,
    Phosphatase,
    Transporter,
    Custom(CustomEnzyme),
}
```

## Turbulance Integration

### `TurbulanceEngine`

Main interface for Turbulance language execution.

```rust
pub struct TurbulanceEngine {
    compiler: TurbulanceCompiler,
    runtime: TurbulanceRuntime,
    nebu_integration: Option<NebuIntegration>,
}
```

#### Methods

##### `new() -> Self`
Creates new Turbulance engine.

##### `execute_script(script: &str) -> TurbulanceResult<ExecutionResult>`
Executes Turbulance script.

**Parameters:**
- `script`: Turbulance source code

**Returns:**
- `ExecutionResult`: Script execution results

##### `set_nebu_integration(&mut self, integration: NebuIntegration)`
Sets Nebuchadnezzar integration for biological function access.

### `NebuIntegration`

Bridge between Turbulance and Nebuchadnezzar systems.

```rust
pub struct NebuIntegration {
    atp_simulator: AtpOscillatoryMembraneSimulator,
    pattern_recognition: PatternRecognitionSystem,
    evidence_collector: EvidenceCollector,
}
```

#### Biological Functions

##### `analyze_atp_dynamics(params: AtpAnalysisParams) -> BiologicalDataValue`
Analyzes ATP oscillatory dynamics.

##### `simulate_oscillation(config: OscillationConfig) -> BiologicalDataValue`
Simulates biological oscillations.

##### `quantum_membrane_transport(params: TransportParams) -> BiologicalDataValue`
Simulates quantum membrane transport.

##### `run_maxwell_demon(config: DemonConfig) -> BiologicalDataValue`
Executes Maxwell demon information processing.

## Data Types

### `TurbulanceValue`

Values in Turbulance runtime.

```rust
pub enum TurbulanceValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<TurbulanceValue>),
    Object(HashMap<String, TurbulanceValue>),
    Biological(BiologicalDataValue),
    Function(TurbulanceFunction),
}
```

### `BiologicalDataValue`

Biological data integration with Turbulance.

```rust
pub enum BiologicalDataValue {
    AtpPool {
        concentration: f64,
        compartment: String,
        turnover_rate: f64,
    },
    QuantumState {
        coherence_time: f64,
        entanglement_degree: f64,
        decoherence_rate: f64,
    },
    MembranePotential {
        voltage: f64,
        capacitance: f64,
        resistance: f64,
    },
    OscillationData {
        frequency: f64,
        amplitude: f64,
        phase: f64,
        stability: f64,
    },
}
```

## Error Handling

### `NebuError`

Main error type for Nebuchadnezzar operations.

```rust
pub enum NebuError {
    SimulationError(String),
    QuantumError(String),
    CircuitError(String),
    ConfigurationError(String),
    TurbulanceError(TurbulanceError),
    IoError(std::io::Error),
}
```

### `TurbulanceError`

Errors specific to Turbulance language operations.

```rust
pub enum TurbulanceError {
    ParseError(String),
    CompileError(String),
    RuntimeError(String),
    TypeMismatch(String),
    FunctionNotFound(String),
}
```

## Configuration Types

### `SimulationConfig`

Master configuration for Nebuchadnezzar simulations.

```rust
pub struct SimulationConfig {
    pub atp_config: AtpConfig,
    pub quantum_config: QuantumConfig,
    pub demon_config: MaxwellDemonConfig,
    pub circuit_config: CircuitConfig,
    pub parallel_config: Option<ParallelConfig>,
}
```

### `QuantumConfig`

Quantum system configuration.

```rust
pub struct QuantumConfig {
    pub coherence_time: f64,
    pub decoherence_model: DecoherenceModel,
    pub fire_light_wavelength: f64,
    pub environmental_coupling_strength: f64,
}
```

### `ParallelConfig`

Parallel processing configuration.

```rust
pub struct ParallelConfig {
    pub num_threads: usize,
    pub chunk_size: usize,
    pub load_balancing: LoadBalancing,
}
```

## Utility Functions

### Time Conversion

```rust
pub fn atp_cycles_to_seconds(cycles: u64, atp_turnover_rate: f64) -> f64
pub fn seconds_to_atp_cycles(seconds: f64, atp_turnover_rate: f64) -> u64
```

### Energy Calculations

```rust
pub fn calculate_atp_energy(concentration: f64, temperature: f64) -> f64
pub fn gibbs_free_energy_hydrolysis(atp: f64, adp: f64, pi: f64, temperature: f64) -> f64
```

### Quantum Utilities

```rust
pub fn quantum_tunneling_probability(barrier_height: f64, barrier_width: f64, particle_energy: f64) -> f64
pub fn decoherence_time(temperature: f64, coupling_strength: f64) -> f64
```

## Example Usage Patterns

### Basic Simulation Setup

```rust
use nebuchadnezzar::prelude::*;

// Configure system
let config = SimulationConfig {
    atp_config: AtpConfig {
        initial_concentration: 5.0e-3,
        hydrolysis_rate: 1000.0,
        synthesis_rate: 800.0,
        temperature_dependence: true,
    },
    quantum_config: QuantumConfig {
        coherence_time: 1e-3,
        decoherence_model: DecoherenceModel::Environmental,
        fire_light_wavelength: 650e-9,
        environmental_coupling_strength: 0.7,
    },
    // ... other configs
};

// Create and run system
let mut system = NebuSystem::with_config(config)?;
let results = system.run_simulation(1000)?;
```

### Turbulance Integration

```rust
use nebuchadnezzar::turbulance::*;

let script = r#"
    proposition test_hypothesis {
        "Quantum effects enhance ATP efficiency"
    }
    
    evidence data = analyze_atp_dynamics(
        concentration: 5.0e-3,
        cycles: 1000
    );
    
    motion evaluate {
        given data.efficiency > 0.8 {
            support test_hypothesis with data.coherence_analysis;
        }
    }
    
    considering evaluate;
"#;

let mut engine = TurbulanceEngine::new();
engine.set_nebu_integration(NebuIntegration::new());
let result = engine.execute_script(script)?;
```

---

*For more examples and tutorials, see [Examples](examples) and [Getting Started](getting-started).* 
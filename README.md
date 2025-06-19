<div align="center">
     <h1 align="center">Nebuchadnezzar</h1>



<img src="assets/img/logo.png" alt="Nebuchadnezzar Logo" width="400"/>

## Hierarchical Probabilistic Electric Circuit System for Biological Simulation

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/fullscreen-triangle/nebuchadnezzar/ci.yml?branch=master&style=for-the-badge)](https://github.com/fullscreen-triangle/nebuchadnezzar/actions)
[![Crates.io](https://img.shields.io/crates/v/nebuchadnezzar?style=for-the-badge)](https://crates.io/crates/nebuchadnezzar)
[![Documentation](https://img.shields.io/docsrs/nebuchadnezzar?style=for-the-badge)](https://docs.rs/nebuchadnezzar)
[![GitHub Stars](https://img.shields.io/github/stars/fullscreen-triangle/nebuchadnezzar?style=for-the-badge)](https://github.com/fullscreen-triangle/nebuchadnezzar/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/fullscreen-triangle/nebuchadnezzar?style=for-the-badge)](https://github.com/fullscreen-triangle/nebuchadnezzar/issues)
[![GitHub Forks](https://img.shields.io/github/forks/fullscreen-triangle/nebuchadnezzar?style=for-the-badge)](https://github.com/fullscreen-triangle/nebuchadnezzar/network/members)

</div>

**Nebuchadnezzar** is a systems biology framework that models intracellular processes through hierarchical probabilistic electrical circuits, using ATP as the fundamental rate unit instead of time. The framework is built on three foundational theorems that enable quantum computation in biological membranes, universal oscillatory dynamics, and manipulation of entropy as a tangible quantity.

## Foundational Theoretical Framework

### 1. Membrane Quantum Computation Theorem

Biological membranes function as room-temperature quantum computers through Environment-Assisted Quantum Transport (ENAQT). Unlike artificial quantum computers that require isolation from environmental noise, biological systems optimize environmental coupling to enhance quantum coherence.

**Core Principle:**
```
Coherence_biological = f(environmental_coupling, thermal_noise, membrane_structure)
```

**Mathematical Foundation:**
- **Quantum State Evolution:** |ψ(t)⟩ = exp(-iHt/ℏ)|ψ(0)⟩ where H includes environmental interactions
- **Decoherence Suppression:** Environmental noise constructively interferes to maintain coherence
- **Transport Efficiency:** Quantum tunneling and superposition optimize energy transfer rates

**Implementation in Circuits:**
```rust
struct QuantumMembrane {
    coherence_time: f64,           // Quantum coherence duration
    environmental_coupling: f64,   // Coupling strength to environment
    tunneling_probability: f64,    // Quantum tunneling rate
    superposition_states: Vec<QuantumState>,
}
```

### 2. Universal Oscillatory Framework

Oscillatory behavior emerges mathematically in any bounded nonlinear system. All biological processes exhibit oscillatory dynamics governed by Causal Selection principles.

**Causal Selection Theorem:**
```
For system S with constraints C: lim(t→∞) behavior(S) → oscillatory_attractor
```

**Mathematical Formulation:**
- **Phase Space Constraint:** dx/dt = F(x) where x ∈ bounded_domain
- **Lyapunov Stability:** Oscillatory attractors are globally stable
- **Frequency Locking:** Multiple oscillators synchronize through nonlinear coupling

**Oscillatory Categories:**
1. **Metabolic Oscillations:** ATP/ADP cycles, glycolytic oscillations
2. **Membrane Oscillations:** Action potentials, calcium waves
3. **Genetic Oscillations:** Circadian rhythms, cell cycle progression
4. **Mechanical Oscillations:** Muscle contractions, flagellar motion

### 3. Entropy Reformulation

Entropy is reformulated as a manipulable quantity through probabilistic points and resolutions. This enables direct control and optimization of thermodynamic processes.

**Probabilistic Points Framework:**
```
Entropy = ∑(probability_points × resolution_strength)
```

**Key Concepts:**
- **Points:** Discrete probability masses representing system states
- **Resolutions:** Transformation operators that modify point distributions
- **Perturbation Validation:** Changes propagate through resolution networks
- **Streaming Semantics:** Real-time entropy manipulation through point flow

**Mathematical Implementation:**
```rust
struct EntropyPoint {
    probability_mass: f64,
    position: Vec<f64>,
    resolution_connections: Vec<ResolutionId>,
}

struct Resolution {
    transformation_matrix: Matrix<f64>,
    perturbation_sensitivity: f64,
    validation_criteria: Vec<Constraint>,
}
```

## Core Philosophy

### ATP-Based Rate Modeling

Traditional systems biology uses time-based differential equations:
```
dx/dt = f(x, parameters, t)
```

**Nebuchadnezzar uses ATP-based differential equations:**
```
dx/dATP = f(x, [ATP], energy_charge, metabolic_state)
```

This fundamental shift provides:
- **Biologically meaningful rates**: ATP availability directly controls cellular processes
- **Energy-coupled modeling**: Reactions are linked through cellular energy charge
- **Metabolic constraints**: Natural rate limiting based on ATP availability
- **Pathway optimization**: Clear energetic objectives for pathway efficiency

### Quantum-Enhanced Circuit Modeling

The framework integrates quantum mechanical effects into classical circuit models:

**Quantum Circuit Elements:**
- **Quantum Resistors:** Resistance varies with quantum state coherence
- **Coherence Capacitors:** Store quantum coherence as electrical charge
- **Tunneling Junctions:** Enable quantum transport across membrane barriers
- **Entanglement Networks:** Couple distant circuit elements through quantum correlations

**Mathematical Integration:**
```rust
// Quantum-classical hybrid circuit equation
dV/dt = (I_classical + I_quantum) / C_total
where:
I_quantum = ∑(tunneling_current + coherent_transport)
C_total = C_classical + C_coherence
```

### Oscillatory Circuit Dynamics

All circuit elements exhibit inherent oscillatory behavior:

**Oscillator Classification:**
```rust
enum BiologicalOscillator {
    Metabolic {
        atp_frequency: f64,
        coupling_strength: f64,
    },
    Membrane {
        action_potential_freq: f64,
        calcium_wave_period: f64,
    },
    Genetic {
        transcription_cycle: f64,
        protein_degradation_rate: f64,
    },
}
```

**Synchronization Networks:**
- **Phase Locking:** Oscillators maintain fixed phase relationships
- **Frequency Entrainment:** External signals drive oscillator frequencies
- **Amplitude Modulation:** Oscillator strength varies with metabolic state

### Entropy-Based Optimization

Direct manipulation of system entropy through probabilistic point control:

**Optimization Objectives:**
```rust
enum EntropyObjective {
    MinimizeSystemEntropy,              // Increase order
    MaximizeInformationEntropy,         // Increase information content
    OptimizeEntropyProduction,          // Balance order vs. information
    ConstrainEntropyFlow(f64),          // Limit entropy change rate
}
```

**Resolution Networks:**
- **Point Migration:** Probability masses flow between system states
- **Resolution Cascades:** Changes propagate through connected resolutions
- **Validation Loops:** System maintains thermodynamic consistency

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEBUCHADNEZZAR FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────┐  │
│  │   QUANTUM LAYER     │    │   OSCILLATORY LAYER │    │  ENTROPY    │  │
│  │   (Membranes)       │    │   (Dynamics)        │    │  LAYER      │  │
│  │                     │    │                     │    │             │  │
│  │ • ENAQT Computation │◄──►│ • Causal Selection  │◄──►│ • Points    │  │
│  │ • Coherence Control │    │ • Phase Locking     │    │ • Resolutions│  │
│  │ • Tunneling Events  │    │ • Frequency Sync    │    │ • Validation│  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────┘  │
│           │                            │                         │       │
│           └────────────────────────────┼─────────────────────────┘       │
│                                        │                                 │
│  ┌─────────────────────────────────────┼─────────────────────────────┐   │
│  │              ATP-BASED CIRCUIT MODELING                        │   │
│  │                                     │                           │   │
│  │  ┌────────────────┐  ┌─────────────┼──────────┐  ┌─────────────┐ │   │
│  │  │ Circuit Grids  │  │ Hierarchical Networks │  │ Solvers     │ │   │
│  │  │ (Electrical)   │◄─┤ (Probabilistic)      │◄─┤ (Adaptive)  │ │   │
│  │  └────────────────┘  └──────────────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Foundational Mathematical Framework

### 1. ATP-Based Differential Equations

**Core Innovation:** Replace time-based rates with ATP-based rates:

#### Traditional Time-Based Model:
```rust
// Standard biochemical kinetics
dS/dt = -k * [S] * [E]  // Substrate consumption over time
dP/dt = +k * [S] * [E]  // Product formation over time
```

#### Nebuchadnezzar ATP-Based Model:
```rust
// ATP-coupled biochemical kinetics  
dS/dATP = -(k * [S] * [E]) / (v_ATP * [ATP])  // Substrate per ATP consumed
dP/dATP = +(k * [S] * [E]) / (v_ATP * [ATP])  // Product per ATP consumed

where:
- v_ATP = ATP consumption rate for this process
- [ATP] = current ATP concentration
- k = ATP-dependent rate constant
```

#### Mathematical Foundation:
```
Rate Transformation: dx/dATP = (dx/dt) / (dATP/dt)

Where:
- dx/dt = traditional rate equation
- dATP/dt = ATP consumption/production rate
- dx/dATP = change in quantity per unit ATP change
```

### 2. Electrical Circuit Analogs

**Biological Process → Electrical Circuit Mapping:**

#### Ion Channels as Resistors/Capacitors:
```rust
// Ion channel conductance (Siemens)
G_channel = g_max * P_open * (V - E_rev)

// Membrane as capacitor
C_mem * dV/dt = -I_total

// ATP-coupled version
C_mem * dV/dATP = -I_total / (dATP/dt)
```

#### Biochemical Reactions as Circuit Elements:
```rust
// Enzyme as variable resistor
R_enzyme = 1 / (k_cat * [E] * f([ATP]))

// Pathway as circuit network
V_pathway = Σ(I_i * R_i)  // Ohm's law for pathway flux

// ATP coupling
P_ATP = V_pathway * I_total  // ATP power consumption
```

### 3. Hierarchical Probabilistic Nodes

**Probabilistic Node Representation:**
```rust
struct ProbabilisticNode {
    // Uncertain parameters with probability distributions
    rate_distribution: ProbabilityDistribution<f64>,
    atp_cost_distribution: ProbabilityDistribution<f64>,
    
    // Cross-reaction influences
    feedback_probability: f64,
    cross_reaction_strength: HashMap<NodeId, f64>,
    
    // Expansion criteria
    uncertainty_threshold: f64,
    computational_importance: f64,
}
```

**Expansion Decision Logic:**
```rust
fn should_expand_node(node: &ProbabilisticNode, criteria: &ExpansionCriteria) -> bool {
    let uncertainty = node.calculate_uncertainty();
    let optimization_impact = node.optimization_sensitivity();
    let computational_cost = estimate_expansion_cost(node);
    
    uncertainty > criteria.uncertainty_threshold &&
    optimization_impact > criteria.impact_threshold &&
    computational_cost < criteria.budget_limit
}
```

## Core Components

### 1. Quantum-Enhanced Physical Layer

**Quantum Membrane Circuits:**
```rust
struct QuantumMembraneCircuit {
    coherence_elements: Vec<CoherenceCapacitor>,
    tunneling_junctions: Vec<TunnelingJunction>,
    entanglement_network: EntanglementMatrix,
    environmental_coupling: f64,
}
```

**ENAQT Implementation:**
- Environment-assisted quantum transport modeling
- Coherence time calculations under thermal noise
- Quantum tunneling probability distributions
- Superposition state management

### 2. Oscillatory Dynamics Layer

**Universal Oscillator Framework:**
```rust
struct BiologicalOscillator {
    intrinsic_frequency: f64,
    amplitude: f64,
    phase: f64,
    coupling_matrix: Matrix<f64>,
    synchronization_state: SyncState,
}
```

**Causal Selection Engine:**
- Automatic oscillator detection in nonlinear systems
- Phase-locking mechanism implementation
- Frequency entrainment algorithms
- Multi-scale oscillatory coupling

### 3. Entropy Manipulation System

**Probabilistic Point Management:**
```rust
struct EntropyManipulator {
    points: Vec<EntropyPoint>,
    resolutions: Vec<Resolution>,
    validation_network: ValidationGraph,
    perturbation_engine: PerturbationEngine,
}
```

**Resolution Network Operations:**
- Point probability redistribution
- Resolution cascade propagation
- Thermodynamic consistency validation
- Real-time entropy optimization

## Advanced Mathematical Framework

### 1. Quantum-ATP Coupled Equations

Integration of quantum mechanical effects with ATP-based kinetics:

```rust
// Quantum-enhanced ATP differential equations
dS/dATP = -(k_classical + k_quantum) * [S] * [E] / (v_ATP * [ATP])

where:
k_quantum = k_tunneling + k_coherent_transport
k_tunneling = A * exp(-B * barrier_height / kT) * coherence_factor
k_coherent_transport = C * |⟨ψ_initial|ψ_final⟩|² * environmental_coupling
```

### 2. Oscillatory Circuit Networks

Circuit equations incorporating universal oscillatory dynamics:

```rust
// Oscillatory circuit element
dV/dt = (I_total + I_oscillatory) / C_effective
I_oscillatory = A * sin(ωt + φ) * coupling_matrix * V_neighbors

// Multi-frequency coupling
ω_effective = ω_intrinsic + Σ(coupling_i * ω_i * phase_diff_i)
```

### 3. Entropy Point Dynamics

Mathematical formulation of probabilistic point evolution:

```rust
// Point migration equation
dp_i/dt = Σ(R_ij * p_j) - Σ(R_ji * p_i) + source_i - sink_i

// Resolution transformation
R_ij = resolution_strength * exp(-energy_barrier_ij / kT) * validation_factor

// Entropy calculation
S = -Σ(p_i * ln(p_i)) + perturbation_correction
```

## Advanced Applications

### Quantum Biology Simulation
- Photosynthetic light harvesting complex modeling
- Quantum effects in enzyme catalysis
- Coherent energy transfer in biological systems
- Room-temperature quantum computation validation

### Oscillatory Pattern Analysis
- Circadian rhythm optimization
- Metabolic oscillation control
- Neural network synchronization
- Cell cycle progression modeling

### Entropy Engineering
- Thermodynamic efficiency optimization
- Information processing capacity enhancement
- Self-organization pattern control
- Dissipative structure design

### Multi-Scale Integration
- Molecular to cellular level coupling
- Tissue-level emergent behavior prediction
- Organ system coordination modeling
- Whole-organism metabolic optimization

## Getting Started

### Installation

```bash
git clone https://github.com/your-org/nebuchadnezzar.git
cd nebuchadnezzar
cargo build --release
```

### Basic Usage

```rust
use nebuchadnezzar::prelude::*;

// Create ATP pool with physiological conditions
let mut atp_pool = AtpPool::new_physiological();

// Define a simple glycolysis pathway
let mut pathway = BiochemicalPathway::new("glycolysis");
pathway.add_reaction(
    "hexokinase",
    BiochemicalReaction::new()
        .substrate("glucose", 1.0)
        .product("glucose_6_phosphate", 1.0)
        .atp_cost(1.0)  // Consumes 1 ATP
        .rate_constant(AtpRateConstant::michaelis(1.0, 0.1))
);

// Create hierarchical circuit builder
let mut builder = PathwayCircuitBuilder::new();
let circuit = builder
    .from_pathway(&pathway)
    .with_atp_pool(&atp_pool)
    .with_expansion_criteria(ExpansionCriteria::default())
    .build()?;

// Optimize pathway for maximum ATP yield
let optimizer = PathwayOptimizer::new()
    .objective(PathwayObjective::MaximizeEnergyEfficiency)
    .constraint(OptimizationConstraint::AtpAvailability(10.0))
    .build();

let optimized_circuit = optimizer.optimize(&circuit)?;
```

### Example: Glycolysis Optimization

```rust
// Define glycolysis with ATP-based kinetics
let glycolysis = BiochemicalPathway::builder("glycolysis")
    .reaction("hexokinase")
        .substrate("glucose", 1.0)
        .product("G6P", 1.0)
        .atp_cost(1.0)
        .rate_datp(AtpRateConstant::michaelis(2.0, 0.5))
    .reaction("phosphofructokinase") 
        .substrate("F6P", 1.0)
        .product("F1,6BP", 1.0)
        .atp_cost(1.0)
        .rate_datp(AtpRateConstant::hill(1.5, 0.2, 2.0))  // Cooperative
    .reaction("pyruvate_kinase")
        .substrate("PEP", 1.0) 
        .product("pyruvate", 1.0)
        .atp_yield(1.0)
        .rate_datp(AtpRateConstant::linear(3.0))
    .build();

// Optimize for maximum net ATP yield
let result = PathwayOptimizer::new()
    .objective(PathwayObjective::MaximizeEnergyEfficiency)
    .optimize(&glycolysis)?;

println!("Optimized ATP efficiency: {:.2}", result.energy_efficiency);
println!("Net ATP yield: {:.2}", result.net_atp_balance);
```

## Key Advantages

1. **Biologically Meaningful**: ATP-based rates reflect real cellular constraints
2. **Computationally Efficient**: Hierarchical abstraction reduces complexity  
3. **Optimization Ready**: Clear energetic objectives and constraints
4. **Experimentally Grounded**: Parameters directly relate to measurable quantities
5. **Pathway Agnostic**: Framework applies to any intracellular process
6. **Uncertainty Handling**: Probabilistic nodes capture biological variability

## Applications

- **Metabolic Engineering**: Optimize pathways for biofuel/pharmaceutical production
- **Drug Target Discovery**: Identify energetically critical pathway steps
- **Systems Pharmacology**: Model drug effects on cellular energetics
- **Synthetic Biology**: Design efficient artificial metabolic circuits
- **Bioprocess Optimization**: Maximize yield while minimizing energy costs

## Documentation

- [Mathematical Foundations](docs/nebuchadnezzar.md)
- [API Reference](docs/api/)
- [Examples](examples/)
- [Benchmarks](benchmarks/)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use Nebuchadnezzar in your research, please cite:

```bibtex
@software{nebuchadnezzar2024,
  title={Nebuchadnezzar: Hierarchical Probabilistic Electric Circuit System for Biological Simulation},
  author={Kundai Farai Sachikonye},
  year={2024},
  url={https://github.com/fullscreen-triangle/nebuchadnezzar}
}
```

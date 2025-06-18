# Nebuchadnezzar

## Hierarchical Probabilistic Electric Circuit System for Biological Simulation

**Nebuchadnezzar** is a revolutionary systems biology framework that models intracellular processes through hierarchical probabilistic electrical circuits, using ATP as the fundamental rate unit instead of time. Named after the ancient king who built magnificent structures, Nebuchadnezzar constructs foundational biological evidence through multi-scale electrical circuit simulation.

## Core Philosophy

### Revolutionary Approach: ATP-Based Rate Modeling

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

### Hierarchical Probabilistic Abstraction

Unlike traditional approaches requiring complete cellular knowledge, Nebuchadnezzar:

1. **Starts with probabilistic nodes** representing uncertain or abstracted processes
2. **Expands to detailed circuits** only when needed for optimization or analysis
3. **Represents cross-reactions and feedback** as probabilistic relationships
4. **Maintains computational efficiency** through intelligent abstraction levels

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEBUCHADNEZZAR FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────┐  │
│  │   PHYSICAL LAYER    │    │   BIOLOGICAL LAYER  │    │ OPTIMIZATION│  │
│  │   (Rust Core)       │    │   (ATP Kinetics)    │    │   LAYER     │  │
│  │                     │    │                     │    │             │  │
│  │ • Ion Channels      │◄──►│ • ATP Pools         │◄──►│ • Objective │  │
│  │ • Membranes         │    │ • Rate Constants    │    │   Functions │  │
│  │ • Circuit Networks  │    │ • Energetics       │    │ • Constraints│  │
│  │ • Differential Eqs  │    │ • Pathway Models    │    │ • Optimizers│  │
│  └─────────────────────┘    └─────────────────────┘    └─────────────┘  │
│           │                            │                         │       │
│           └────────────────────────────┼─────────────────────────┘       │
│                                        │                                 │
│  ┌─────────────────────────────────────┼─────────────────────────────┐   │
│  │              HIERARCHICAL PROBABILISTIC NODES                   │   │
│  │                                     │                           │   │
│  │  ┌────────────────┐  ┌─────────────┼──────────┐  ┌─────────────┐ │   │
│  │  │ Abstract Nodes │  │ Detailed Circuits     │  │ Expansion   │ │   │
│  │  │ (Probabilistic)│◄─┤ (When Needed)        │◄─┤ Criteria    │ │   │
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

### 1. Physical Layer (Electrical Circuits)

**Ion Channel Networks:**
- Hodgkin-Huxley dynamics with ATP dependence
- Voltage-gated and ligand-gated channels
- Gap junction coupling between cellular compartments

**Membrane Circuit Models:**
- Capacitive membrane properties
- Ion-specific conductances
- Electrochemical gradient calculations

**Circuit Network Simulation:**
- Multi-compartment electrical models
- Time-dependent and ATP-dependent dynamics
- Stochastic and deterministic solvers

### 2. Biological Layer (ATP Kinetics)

**ATP Pool Dynamics:**
```rust
struct AtpPool {
    atp_concentration: f64,    // [ATP] in mM
    adp_concentration: f64,    // [ADP] in mM  
    energy_charge: f64,        // (ATP + 0.5*ADP)/(ATP+ADP+AMP)
    synthesis_rate: f64,       // ATP production rate
    hydrolysis_rate: f64,      // ATP consumption rate
}
```

**ATP-Dependent Rate Constants:**
```rust
// Michaelis-Menten ATP dependence
v = V_max * [S] * [ATP] / ((K_m + [S]) * (K_ATP + [ATP]))

// Hill equation for cooperative ATP binding
v = V_max * [S] * [ATP]^n / ((K_m + [S]) * (K_ATP^n + [ATP]^n))

// Linear ATP dependence
v = k * [S] * [ATP]
```

**Energetic Profiles:**
- Total ATP cost/yield analysis
- Energy efficiency calculations
- Rate-limiting step identification
- Pathway energetic feasibility

### 3. Differential Equation Solvers

**ATP-Based ODE System:**
```rust
// System state vector including ATP pool
struct SystemState {
    concentrations: Vec<f64>,  // Metabolite concentrations
    atp_pool: AtpPool,         // Energy state
    electrical_state: Vec<f64>, // Membrane potentials, currents
}

// ATP-based derivative calculation
fn calculate_derivatives(
    state: &SystemState, 
    atp_change: f64
) -> SystemState {
    // Calculate dx/dATP for each variable
    // Update ATP pool based on consumption/production
    // Update electrical states based on ATP-dependent processes
}
```

**Solver Methods:**
- **Explicit Methods:** Euler, Runge-Kutta for ATP-based ODEs
- **Implicit Methods:** Backward Euler for stiff ATP-coupled systems
- **Adaptive Methods:** Variable step-size based on ATP consumption rate
- **Stochastic Methods:** Gillespie-like algorithms for ATP-discrete events

### 4. Optimization Framework

**Objective Functions:**
```rust
enum PathwayObjective {
    MaximizeProduct(String),           // Maximize specific product yield
    MinimizeAtpCost,                   // Minimize ATP consumption
    MaximizeEnergyEfficiency,          // Maximize ATP yield/cost ratio
    MaximizeFlux(String),              // Maximize pathway flux
    MinimizeTime(f64),                 // Minimize time to target concentration
    MaximizeRobustness(f64),           // Maximize stability to perturbations
}
```

**Constraint Types:**
```rust
enum OptimizationConstraint {
    AtpAvailability(f64),              // Maximum ATP consumption rate
    ConcentrationBounds(String, f64, f64), // Min/max metabolite levels
    EnergyCharge(f64, f64),            // Energy charge bounds
    SteadyState(Vec<String>),          // Steady-state requirements
    KineticFeasibility(String),        // Realistic rate constants
}
```

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

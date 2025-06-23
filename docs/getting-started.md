---
layout: page
title: "Getting Started"
permalink: /getting-started/
---

# Getting Started with Nebuchadnezzar

This guide will help you set up the Nebuchadnezzar framework and run your first biological simulations using ATP-based timing and quantum-coherent membrane modeling.

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Rust**: Version 1.70.0 or later
- **Memory**: Minimum 4GB RAM (8GB recommended for complex simulations)
- **Storage**: 2GB free space for dependencies and simulation data

### Scientific Background
While not required, familiarity with the following concepts will enhance your experience:
- **Systems Biology**: Understanding of cellular processes and energy metabolism
- **Quantum Mechanics**: Basic quantum coherence and tunneling concepts
- **Circuit Theory**: Electrical circuit analysis and probabilistic systems
- **Bioinformatics**: Biological data analysis and pattern recognition

## Installation

### Option 1: From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/username/nebuchadnezzar.git
cd nebuchadnezzar

# Build the framework
cargo build --release

# Run tests to verify installation
cargo test

# Run benchmarks to assess performance
cargo bench
```

### Option 2: Using Cargo

```bash
# Add to your Cargo.toml
[dependencies]
nebuchadnezzar = "0.1.0"

# Or install directly
cargo install nebuchadnezzar
```

### Development Dependencies

For development and advanced features:

```bash
# Install additional development tools
cargo install cargo-expand  # For macro expansion debugging
cargo install flamegraph     # For performance profiling
cargo install criterion     # For benchmarking
```

## Verification

Verify your installation by running the comprehensive simulation example:

```bash
cargo run --example comprehensive_simulation
```

Expected output should include:
- ATP oscillation analysis results
- Quantum membrane coherence measurements  
- Maxwell's demon information processing metrics
- Circuit network analysis summary

## First Simulation

### Basic ATP Oscillatory System

Create a new Rust project and add this minimal example:

```rust
use nebuchadnezzar::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the ATP oscillatory membrane system
    let mut system = AtpOscillatoryMembraneSimulator::new(
        OscillatoryConfig {
            atp_concentration: 5.0e-3,    // 5mM ATP concentration
            adp_concentration: 0.5e-3,   // 0.5mM ADP concentration  
            temperature: 310.0,          // 37°C in Kelvin
            ph: 7.4,                     // Physiological pH
            ionic_strength: 0.15,        // 150mM ionic strength
        }
    );

    // Add quantum membrane with fire-light optimization
    let membrane = QuantumMembrane::new(MembraneConfig {
        coherence_time: 1e-3,           // 1ms quantum coherence
        tunneling_probability: 0.1,     // 10% tunneling rate
        fire_light_optimization: 0.85,  // 85% fire-light enhancement
        environmental_coupling: 0.7,    // 70% environmental coupling
    });
    system.add_membrane(membrane);

    // Add biological Maxwell's demon for information catalysis
    let maxwell_demon = BiologicalMaxwellsDemon::new(MaxwellDemonConfig {
        information_filter_threshold: 0.5,
        catalytic_cycles: 1000,
        agency_recognition: true,
        associative_memory_size: 10000,
    });
    system.add_maxwell_demon(maxwell_demon);

    // Run simulation for 1000 ATP cycles
    println!("Starting ATP oscillatory simulation...");
    let results = system.simulate_atp_cycles(1000)?;

    // Display results
    println!("\n=== Simulation Results ===");
    println!("ATP cycles completed: {}", results.cycles_completed);
    println!("Average oscillation frequency: {:.2} Hz", results.avg_frequency);
    println!("Quantum coherence maintained: {:.1}%", results.quantum_coherence * 100.0);
    println!("Maxwell demon efficiency: {:.1}%", results.information_catalysis_efficiency * 100.0);
    println!("Energy conservation: {:.6}", results.energy_conservation_ratio);

    Ok(())
}
```

Run this example:

```bash
cargo run
```

### Turbulance Language Integration

Create a Turbulance script to analyze the simulation results:

```rust
use nebuchadnezzar::turbulance::*;

fn main() -> TurbulanceResult<()> {
    let turbulance_script = r#"
        // Define research proposition
        proposition atp_oscillation_analysis {
            "ATP oscillations exhibit quantum-enhanced frequency stability"
        }
        
        // Collect evidence from simulation
        evidence oscillation_data = collect_atp_dynamics();
        evidence coherence_data = analyze_quantum_coherence();
        evidence demon_data = measure_information_catalysis();
        
        // Analyze patterns
        item frequency_stability = pattern_stability(oscillation_data.frequency);
        item coherence_correlation = correlate(coherence_data, oscillation_data);
        
        // Motion to test proposition
        motion test_quantum_enhancement {
            given frequency_stability > 0.95 {
                support atp_oscillation_analysis with coherence_correlation;
            }
            else {
                contradict atp_oscillation_analysis with "insufficient stability";
            }
        }
        
        // Execute motion
        considering test_quantum_enhancement;
    "#;

    // Initialize Turbulance engine with Nebuchadnezzar integration
    let mut engine = TurbulanceEngine::new();
    engine.set_nebu_integration(NebuIntegration::new());
    
    // Execute the scientific reasoning script
    let result = engine.execute_script(turbulance_script)?;
    
    println!("Proposition analysis complete:");
    println!("Result: {:?}", result);
    
    Ok(())
}
```

## Core Concepts

### ATP-Based Timing

Unlike traditional time-based simulations, Nebuchadnezzar uses ATP hydrolysis events as the fundamental timing unit:

```rust
// Traditional time-based approach
let time_step = 0.001; // 1ms
for t in (0..1000).map(|i| i as f64 * time_step) {
    system.update(t);
}

// Nebuchadnezzar ATP-based approach  
let atp_cycles = 1000;
for cycle in 0..atp_cycles {
    system.advance_atp_cycle(); // Natural biological timing
}
```

This approach ensures that:
- All processes scale naturally with cellular energy availability
- Simulation speed adapts to biological reality
- Energy conservation is maintained throughout
- Quantum effects remain coherent with metabolic processes

### Hierarchical Circuit Architecture

Biological systems are modeled as hierarchical electrical circuits:

```rust
// Create circuit hierarchy
let mut circuit_grid = CircuitGrid::new(64, 64); // 64x64 circuit grid

// Add enzyme circuits (molecular level)
let glycolysis_circuit = EnzymeCircuit::new(EnzymeType::Hexokinase);
circuit_grid.add_enzyme_circuit(glycolysis_circuit, Position::new(10, 10));

// Add ion channel circuits (membrane level)  
let sodium_channel = IonChannelCircuit::new(IonType::Sodium);
circuit_grid.add_ion_channel(sodium_channel, Position::new(20, 20));

// Add fractal circuits (multi-scale organization)
let fractal_circuit = FractalCircuit::new(FractalType::Metabolic, 3); // 3 levels
circuit_grid.add_fractal_circuit(fractal_circuit, Position::new(30, 30));

// Simulate circuit interactions
let circuit_results = circuit_grid.simulate_interactions(1000)?;
```

### Quantum-Classical Integration

The framework seamlessly integrates quantum effects with classical biological dynamics:

```rust
// Quantum membrane transport
let transport_result = quantum_membrane.transport_ion(
    Ion::Sodium,
    TransportConfig {
        classical_diffusion: true,      // Include classical diffusion
        quantum_tunneling: true,        // Enable quantum tunneling
        coherence_enhancement: true,    // Fire-light coherence boost
        environmental_noise: NoiseLevel::Biological, // Realistic noise
    }
)?;

println!("Transport efficiency: {:.1}%", transport_result.efficiency * 100.0);
println!("Quantum contribution: {:.1}%", transport_result.quantum_fraction * 100.0);
```

## Configuration

### Simulation Parameters

Customize your simulations through configuration:

```rust
use nebuchadnezzar::config::*;

let config = SimulationConfig {
    // ATP system parameters
    atp_config: AtpConfig {
        initial_concentration: 5.0e-3,
        hydrolysis_rate: 1000.0,
        synthesis_rate: 800.0,
        temperature_dependence: true,
    },
    
    // Quantum system parameters  
    quantum_config: QuantumConfig {
        coherence_time: 1e-3,
        decoherence_model: DecoherenceModel::Environmental,
        fire_light_wavelength: 650e-9, // 650nm red light
        environmental_coupling_strength: 0.7,
    },
    
    // Maxwell demon configuration
    demon_config: MaxwellDemonConfig {
        information_processing_rate: 1e6, // 1 MHz
        catalytic_amplification: 100.0,
        agency_recognition_threshold: 0.8,
        memory_persistence: Duration::from_secs(3600), // 1 hour
    },
    
    // Circuit system parameters
    circuit_config: CircuitConfig {
        grid_size: (64, 64),
        voltage_range: (-100e-3, 100e-3), // ±100mV
        current_precision: 1e-12,          // 1pA precision
        noise_level: NoiseLevel::Thermal,
    },
};

let mut system = NebuSystem::with_config(config)?;
```

### Performance Tuning

For large-scale simulations:

```rust
// Enable parallel processing
let parallel_config = ParallelConfig {
    num_threads: num_cpus::get(),
    chunk_size: 1024,
    load_balancing: LoadBalancing::WorkStealing,
};

// Use high-performance solvers
let solver_config = SolverConfig {
    method: SolverMethod::QuantumBiological,
    tolerance: 1e-12,
    max_iterations: 10000,
    adaptive_step_size: true,
};

system.set_parallel_config(parallel_config);
system.set_solver_config(solver_config);
```

## Next Steps

1. **[Explore the Theoretical Framework](theoretical-framework)**: Understand the six foundational theorems
2. **[Learn Turbulance Language](turbulance-language)**: Master the scientific programming language  
3. **[Review Examples](examples)**: Study practical applications and use cases
4. **[Check API Reference](api-reference)**: Comprehensive API documentation
5. **[Run Benchmarks](benchmarks)**: Performance analysis and validation

## Troubleshooting

### Common Issues

**Compilation Errors**:
```bash
# Update Rust toolchain
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

**Memory Issues**:
```rust
// Reduce simulation scale
let config = SimulationConfig {
    circuit_config: CircuitConfig {
        grid_size: (32, 32), // Smaller grid
        // ...
    },
    // ...
};
```

**Performance Issues**:
```bash
# Enable optimizations
export RUSTFLAGS="-C target-cpu=native"
cargo build --release

# Profile performance
cargo install flamegraph
sudo cargo flamegraph --example comprehensive_simulation
```

### Getting Help

- **Documentation**: Full API reference and examples
- **GitHub Issues**: Bug reports and feature requests  
- **Discussions**: Community support and questions
- **Scientific Publications**: Theoretical background and validation studies

---

*Ready to explore quantum biology with Nebuchadnezzar? Continue to the [Theoretical Framework](theoretical-framework) to understand the science behind the simulations.* 
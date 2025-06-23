---
layout: page
title: "Examples"
permalink: /examples/
---

# Examples

Practical examples demonstrating Nebuchadnezzar's capabilities.

## Basic ATP Simulation

```rust
use nebuchadnezzar::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = AtpOscillatoryMembraneSimulator::new(
        OscillatoryConfig {
            atp_concentration: 5.0e-3,
            adp_concentration: 0.5e-3,
            temperature: 310.0,
            ph: 7.4,
            ionic_strength: 0.15,
        }
    );

    let results = system.simulate_atp_cycles(1000)?;
    
    println!("Cycles: {}", results.cycles_completed);
    println!("Frequency: {:.2} Hz", results.avg_frequency);
    println!("Quantum coherence: {:.1}%", results.quantum_coherence * 100.0);
    
    Ok(())
}
```

## Quantum Membrane Transport

```rust
use nebuchadnezzar::prelude::*;

fn ion_transport_example() -> Result<(), Box<dyn std::error::Error>> {
    let membrane = QuantumMembrane::new(MembraneConfig {
        coherence_time: 1e-3,
        tunneling_probability: 0.1,
        fire_light_optimization: 0.8,
        environmental_coupling: 0.7,
    });

    let transport_result = membrane.transport_ion(
        Ion::Sodium,
        TransportConfig {
            classical_diffusion: true,
            quantum_tunneling: true,
            coherence_enhancement: true,
            environmental_noise: NoiseLevel::Biological,
        }
    )?;
    
    println!("Transport efficiency: {:.1}%", transport_result.efficiency * 100.0);
    println!("Quantum contribution: {:.1}%", transport_result.quantum_fraction * 100.0);
    
    Ok(())
}
```

## Turbulance Scientific Programming

```turbulance
// Research hypothesis
proposition quantum_enhancement {
    "Quantum coherence enhances ATP stability"
}

// Evidence collection
evidence baseline = analyze_atp_dynamics(quantum_enhancement: false);
evidence quantum = analyze_atp_dynamics(quantum_enhancement: true);

// Testing
motion test_hypothesis {
    item ratio = quantum.stability / baseline.stability;
    
    given ratio > 1.2 {
        support quantum_enhancement with ratio;
    }
    else {
        contradict quantum_enhancement with "insufficient enhancement";
    }
}

considering test_hypothesis;
```

## Maxwell's Demon Information Processing

```rust
use nebuchadnezzar::prelude::*;

fn maxwell_demon_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut demon = BiologicalMaxwellsDemon::new(MaxwellDemonConfig {
        information_filter_threshold: 0.5,
        catalytic_cycles: 1000,
        agency_recognition: true,
        associative_memory_size: 10000,
    });

    let input = InformationSet::new(vec![
        BiologicalPattern::ATPOscillation { frequency: 10.0, amplitude: 0.8 },
        BiologicalPattern::MembraneTransport { ion_flux: 1e6, selectivity: 0.9 },
    ]);

    let result = demon.process_information(&input)?;
    
    println!("Information reduction: {:.2} bits", result.information_reduction);
    println!("Amplification: {:.1}x", result.amplification_factor);
    
    Ok(())
}
```

## Circuit Network Analysis

```rust
use nebuchadnezzar::circuits::*;

fn circuit_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut grid = CircuitGrid::new(32, 32);

    // Add components
    grid.add_enzyme_circuit(
        EnzymeCircuit::new(EnzymeType::Hexokinase), 
        Position::new(10, 10)
    );
    
    grid.add_ion_channel(
        IonChannelCircuit::new(IonType::Sodium),
        Position::new(15, 15)
    );

    let results = grid.simulate_interactions(1000)?;
    
    println!("Network efficiency: {:.2}%", results.efficiency * 100.0);
    println!("Information flow: {:.2e} bits/s", results.information_flow_rate);
    
    Ok(())
}
```

## Performance Benchmarking

```rust
use nebuchadnezzar::benchmarks::*;

fn benchmark_example() -> Result<(), Box<dyn std::error::Error>> {
    let grid_sizes = vec![(32, 32), (64, 64), (128, 128)];
    let cycles = vec![1000, 5000, 10000];
    
    for grid_size in grid_sizes {
        for cycle_count in &cycles {
            let start = std::time::Instant::now();
            
            let mut system = create_system(grid_size)?;
            let results = system.simulate_atp_cycles(*cycle_count)?;
            
            let elapsed = start.elapsed();
            
            println!("Grid: {:?}, Cycles: {}, Time: {:.2}s", 
                grid_size, cycle_count, elapsed.as_secs_f64());
        }
    }
    
    Ok(())
}
```

---

*For complete examples, see the `examples/` directory in the repository.* 
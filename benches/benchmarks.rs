use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nebuchadnezzar::{
    atp_oscillatory_membrane_simulator::*,
    atp_oscillatory_membrane_solver::*,
    circuits::hierarchical_framework::*,
    biological_integration::*,
    quantum_metabolism_analyzer::*,
};
use std::time::Duration;

fn benchmark_biological_quantum_state_creation(c: &mut Criterion) {
    c.bench_function("biological_quantum_state_creation", |b| {
        b.iter(|| {
            black_box(BiologicalQuantumState::new(
                black_box(1000.0),  // atp_pool
                black_box(8),        // num_quantum_states
                black_box(3),        // num_oscillators
                black_box(6),        // num_tunneling_processes
            ))
        })
    });
}

fn benchmark_hamiltonian_calculation(c: &mut Criterion) {
    let state = BiologicalQuantumState::new(1000.0, 8, 3, 6);
    
    c.bench_function("hamiltonian_calculation", |b| {
        b.iter(|| {
            let hamiltonian = BiologicalQuantumHamiltonian::new(black_box(&state));
            black_box(hamiltonian.total_energy())
        })
    });
}

fn benchmark_solver_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_step");
    
    for atp_pool in [100.0, 1000.0, 10000.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("atp_pool", atp_pool),
            atp_pool,
            |b, &atp_pool| {
                let initial_state = BiologicalQuantumState::new(atp_pool, 8, 3, 6);
                let mut solver = BiologicalQuantumSolver::new(initial_state);
                
                b.iter(|| {
                    black_box(solver.step(black_box(0.001)))
                })
            },
        );
    }
    group.finish();
}

fn benchmark_hierarchical_circuit_expansion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_circuit_expansion");
    
    for num_nodes in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("nodes", num_nodes),
            num_nodes,
            |b, &num_nodes| {
                let mut circuit = HierarchicalCircuit::new();
                
                // Add probabilistic nodes
                for i in 0..*num_nodes {
                    circuit.add_probabilistic_node(ProbabilisticNode {
                        id: i,
                        uncertainty: 0.5,
                        computational_cost: 1.0,
                        impact_score: 0.8,
                        process_type: format!("process_{}", i),
                        parameters: vec![1.0, 2.0, 3.0],
                    });
                }
                
                let criteria = ExpansionCriteria {
                    uncertainty_threshold: 0.3,
                    impact_threshold: 0.5,
                    cost_threshold: 10.0,
                };
                
                b.iter(|| {
                    black_box(circuit.should_expand_node(black_box(0), black_box(&criteria)))
                })
            },
        );
    }
    group.finish();
}

fn benchmark_biological_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("biological_integration");
    
    let quantum_state = BiologicalQuantumState::new(1000.0, 8, 3, 6);
    let integrator = BiologicalIntegrator::new();
    
    group.bench_function("metabolic_response", |b| {
        b.iter(|| {
            black_box(integrator.analyze_metabolic_response(black_box(&quantum_state)))
        })
    });
    
    group.bench_function("signaling_response", |b| {
        b.iter(|| {
            black_box(integrator.analyze_signaling_response(black_box(&quantum_state)))
        })
    });
    
    group.bench_function("regulatory_response", |b| {
        b.iter(|| {
            black_box(integrator.analyze_regulatory_response(black_box(&quantum_state)))
        })
    });
    
    group.finish();
}

fn benchmark_quantum_metabolism_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantum_metabolism_analysis");
    
    let analyzer = QuantumMetabolismAnalyzer::new();
    
    // Create sample pathway
    let pathway = vec![
        ("glucose", "glucose_6_phosphate", 1.0, 0.5),
        ("glucose_6_phosphate", "fructose_6_phosphate", 0.8, 0.3),
        ("fructose_6_phosphate", "fructose_1_6_bisphosphate", 1.2, 0.7),
        ("fructose_1_6_bisphosphate", "pyruvate", 2.0, 1.0),
    ];
    
    group.bench_function("pathway_flux_calculation", |b| {
        b.iter(|| {
            for (substrate, product, vmax, km) in &pathway {
                black_box(analyzer.calculate_pathway_flux(
                    black_box(substrate),
                    black_box(product),
                    black_box(*vmax),
                    black_box(*km),
                    black_box(5.0), // substrate_concentration
                ));
            }
        })
    });
    
    group.bench_function("quantum_tunneling_rate", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_quantum_tunneling_rate(
                black_box(0.5),   // barrier_height
                black_box(1e-12), // coherence_time
            ))
        })
    });
    
    group.bench_function("coherence_enhancement", |b| {
        b.iter(|| {
            black_box(analyzer.calculate_coherence_enhancement(
                black_box(310.15), // temperature
                black_box(1e-12),  // coherence_time
            ))
        })
    });
    
    group.finish();
}

fn benchmark_monte_carlo_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("monte_carlo_sampling");
    group.measurement_time(Duration::from_secs(10));
    
    for num_samples in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("samples", num_samples),
            num_samples,
            |b, &num_samples| {
                let circuit = HierarchicalCircuit::new();
                
                b.iter(|| {
                    black_box(circuit.monte_carlo_analysis(black_box(num_samples)))
                })
            },
        );
    }
    group.finish();
}

fn benchmark_atp_differential_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("atp_differential_solving");
    group.measurement_time(Duration::from_secs(15));
    
    for num_steps in [100, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("steps", num_steps),
            num_steps,
            |b, &num_steps| {
                let initial_state = BiologicalQuantumState::new(1000.0, 8, 3, 6);
                let mut solver = BiologicalQuantumSolver::new(initial_state);
                
                b.iter(|| {
                    for _ in 0..*num_steps {
                        black_box(solver.step(black_box(0.001)));
                    }
                })
            },
        );
    }
    group.finish();
}

fn benchmark_entropy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_calculation");
    
    for num_states in [8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("quantum_states", num_states),
            num_states,
            |b, &num_states| {
                let state = BiologicalQuantumState::new(1000.0, *num_states, 3, 6);
                
                b.iter(|| {
                    black_box(state.oscillatory_entropy.calculate_endpoint_entropy())
                })
            },
        );
    }
    group.finish();
}

fn benchmark_parallel_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_computation");
    group.measurement_time(Duration::from_secs(20));
    
    let num_simulations = 1000;
    
    group.bench_function("sequential", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for i in 0..num_simulations {
                let state = BiologicalQuantumState::new(1000.0 + i as f64, 8, 3, 6);
                let mut solver = BiologicalQuantumSolver::new(state);
                results.push(solver.step(0.001));
            }
            black_box(results)
        })
    });
    
    group.bench_function("parallel", |b| {
        use rayon::prelude::*;
        
        b.iter(|| {
            let results: Vec<_> = (0..num_simulations)
                .into_par_iter()
                .map(|i| {
                    let state = BiologicalQuantumState::new(1000.0 + i as f64, 8, 3, 6);
                    let mut solver = BiologicalQuantumSolver::new(state);
                    solver.step(0.001)
                })
                .collect();
            black_box(results)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_biological_quantum_state_creation,
    benchmark_hamiltonian_calculation,
    benchmark_solver_step,
    benchmark_hierarchical_circuit_expansion,
    benchmark_biological_integration,
    benchmark_quantum_metabolism_analysis,
    benchmark_monte_carlo_sampling,
    benchmark_atp_differential_solving,
    benchmark_entropy_calculation,
    benchmark_parallel_computation
);

criterion_main!(benches); 
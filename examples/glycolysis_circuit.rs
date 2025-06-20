//! # Glycolysis Circuit Example
//! 
//! Demonstrates modeling the glycolytic pathway using hierarchical probabilistic circuits
//! with ATP-based differential equations and quantum membrane computation.

use nebuchadnezzar::prelude::*;
use nebuchadnezzar::{
    BiochemicalPathway, BiochemicalReaction, PathwayCircuitBuilder, 
    PathwayOptimizer, PathwayObjective, OptimizationConstraint
};

fn main() -> Result<()> {
    println!("=== Nebuchadnezzar: Glycolysis Circuit Analysis ===\n");
    
    // Create the glycolytic pathway
    let pathway = create_glycolysis_pathway();
    
    // Set up ATP pool with physiological concentrations
    let atp_pool = AtpPool::new(
        5.0,  // ATP concentration (mM)
        1.0,  // ADP concentration (mM)
        10.0  // Pi concentration (mM)
    );
    
    // Build hierarchical circuit from pathway
    let mut circuit = PathwayCircuitBuilder::new()
        .from_pathway(&pathway)
        .with_atp_pool(&atp_pool)
        .build()?;
    
    println!("Built hierarchical circuit with {} nodes", circuit.nodes.len());
    println!("Total computational load: {:.2}\n", circuit.calculate_total_load()?);
    
    // Solve the circuit system with ATP budget
    let atp_budget = 100.0; // mM ATP available
    let result = circuit.solve_hierarchical_system(atp_budget)?;
    
    println!("=== Circuit Solution Results ===");
    println!("ATP consumed: {:.2} mM", result.total_atp_consumed);
    println!("Resolution level: {}", result.resolution_level);
    println!("Nodes solved: {}", result.node_states.len());
    
    // Analyze individual node results
    for (node_id, node_state) in &result.node_states {
        match node_state {
            NodeState::Probabilistic(prob_result) => {
                println!("Node {}: Probabilistic solution", node_id);
                println!("  Mean flux: {:.4}", prob_result.mean_flux);
                println!("  Uncertainty: {:.4}", prob_result.flux_uncertainty);
                println!("  ATP consumed: {:.4}", prob_result.atp_consumed);
            }
            NodeState::Detailed(detail_result) => {
                println!("Node {}: Detailed circuit solution", node_id);
                println!("  Current flow: {:.4}", detail_result.current_flow);
                println!("  Power consumption: {:.4}", detail_result.power_consumption);
                println!("  ATP consumed: {:.4}", detail_result.atp_consumed);
                println!("  Converged: {}", detail_result.converged);
            }
        }
    }
    
    // Perform pathway optimization
    println!("\n=== Pathway Optimization ===");
    let optimizer = PathwayOptimizer::new()
        .objective(PathwayObjective::MaximizeAtpYield)
        .constraint(OptimizationConstraint::AtpAvailability(atp_budget))
        .constraint(OptimizationConstraint::ThermodynamicFeasibility)
        .build();
    
    let optimization_result = optimizer.optimize(&circuit)?;
    println!("ATP efficiency: {:.4}", optimization_result.atp_efficiency);
    println!("Optimization converged: {}", optimization_result.converged);
    
    // Demonstrate quantum computation integration
    println!("\n=== Quantum Computation Integration ===");
    demonstrate_quantum_computation(atp_budget)?;
    
    Ok(())
}

fn create_glycolysis_pathway() -> BiochemicalPathway {
    let mut pathway = BiochemicalPathway::new("Glycolysis");
    
    // Glucose phosphorylation (Hexokinase)
    let hexokinase = BiochemicalReaction::new()
        .substrate("Glucose", 1.0)
        .substrate("ATP", 1.0)
        .product("Glucose-6-phosphate", 1.0)
        .product("ADP", 1.0)
        .atp_cost(1.0)
        .rate_constant(AtpRateConstant::Michaelis(100.0, 0.1));
    
    pathway.add_reaction("Hexokinase", hexokinase);
    
    // Glucose-6-phosphate isomerization (Phosphoglucose isomerase)
    let pgi = BiochemicalReaction::new()
        .substrate("Glucose-6-phosphate", 1.0)
        .product("Fructose-6-phosphate", 1.0)
        .atp_cost(0.0)
        .rate_constant(AtpRateConstant::Linear(50.0));
    
    pathway.add_reaction("Phosphoglucose_isomerase", pgi);
    
    // Fructose-6-phosphate phosphorylation (Phosphofructokinase)
    let pfk = BiochemicalReaction::new()
        .substrate("Fructose-6-phosphate", 1.0)
        .substrate("ATP", 1.0)
        .product("Fructose-1,6-bisphosphate", 1.0)
        .product("ADP", 1.0)
        .atp_cost(1.0)
        .rate_constant(AtpRateConstant::Cooperative(80.0, 0.2, 2.0));
    
    pathway.add_reaction("Phosphofructokinase", pfk);
    
    // Fructose-1,6-bisphosphate cleavage (Aldolase)
    let aldolase = BiochemicalReaction::new()
        .substrate("Fructose-1,6-bisphosphate", 1.0)
        .product("Dihydroxyacetone_phosphate", 1.0)
        .product("Glyceraldehyde-3-phosphate", 1.0)
        .atp_cost(0.0)
        .rate_constant(AtpRateConstant::Linear(60.0));
    
    pathway.add_reaction("Aldolase", aldolase);
    
    // Triose phosphate isomerase
    let tpi = BiochemicalReaction::new()
        .substrate("Dihydroxyacetone_phosphate", 1.0)
        .product("Glyceraldehyde-3-phosphate", 1.0)
        .atp_cost(0.0)
        .rate_constant(AtpRateConstant::Linear(200.0));
    
    pathway.add_reaction("Triose_phosphate_isomerase", tpi);
    
    // Glyceraldehyde-3-phosphate dehydrogenase
    let gapdh = BiochemicalReaction::new()
        .substrate("Glyceraldehyde-3-phosphate", 1.0)
        .substrate("NAD+", 1.0)
        .substrate("Pi", 1.0)
        .product("1,3-Bisphosphoglycerate", 1.0)
        .product("NADH", 1.0)
        .atp_cost(0.0)
        .rate_constant(AtpRateConstant::Michaelis(120.0, 0.05));
    
    pathway.add_reaction("GAPDH", gapdh);
    
    // Phosphoglycerate kinase (ATP production)
    let pgk = BiochemicalReaction::new()
        .substrate("1,3-Bisphosphoglycerate", 1.0)
        .substrate("ADP", 1.0)
        .product("3-Phosphoglycerate", 1.0)
        .product("ATP", 1.0)
        .atp_cost(-1.0) // ATP production
        .rate_constant(AtpRateConstant::Linear(150.0));
    
    pathway.add_reaction("Phosphoglycerate_kinase", pgk);
    
    // Phosphoglycerate mutase
    let pgm = BiochemicalReaction::new()
        .substrate("3-Phosphoglycerate", 1.0)
        .product("2-Phosphoglycerate", 1.0)
        .atp_cost(0.0)
        .rate_constant(AtpRateConstant::Linear(90.0));
    
    pathway.add_reaction("Phosphoglycerate_mutase", pgm);
    
    // Enolase
    let enolase = BiochemicalReaction::new()
        .substrate("2-Phosphoglycerate", 1.0)
        .product("Phosphoenolpyruvate", 1.0)
        .product("H2O", 1.0)
        .atp_cost(0.0)
        .rate_constant(AtpRateConstant::Linear(70.0));
    
    pathway.add_reaction("Enolase", enolase);
    
    // Pyruvate kinase (ATP production)
    let pk = BiochemicalReaction::new()
        .substrate("Phosphoenolpyruvate", 1.0)
        .substrate("ADP", 1.0)
        .product("Pyruvate", 1.0)
        .product("ATP", 1.0)
        .atp_cost(-1.0) // ATP production
        .rate_constant(AtpRateConstant::Cooperative(110.0, 0.3, 1.8));
    
    pathway.add_reaction("Pyruvate_kinase", pk);
    
    pathway
}

fn demonstrate_quantum_computation(atp_budget: f64) -> Result<()> {
    // Create biological quantum computer
    let mut quantum_solver = BiologicalQuantumComputerSolver::new();
    
    // Set up initial quantum state
    let initial_state = BiologicalQuantumState {
        atp_coords: AtpCoordinates {
            atp_concentration: 5.0,
            adp_concentration: 1.0,
            pi_concentration: 10.0,
            energy_charge: 0.8,
            atp_oscillation_amplitude: 0.1,
            atp_oscillation_phase: 0.0,
        },
        oscillatory_coords: OscillatoryCoordinates {
            oscillations: vec![
                OscillationState {
                    name: "Membrane_potential".to_string(),
                    amplitude: 0.07, // 70 mV
                    frequency: 10.0, // 10 Hz
                    phase: 0.0,
                    damping_coefficient: 0.1,
                    atp_coupling_strength: 0.5,
                },
                OscillationState {
                    name: "Metabolic_flux".to_string(),
                    amplitude: 1.0,
                    frequency: 0.1, // 0.1 Hz (slow metabolic oscillation)
                    phase: 0.0,
                    damping_coefficient: 0.05,
                    atp_coupling_strength: 1.0,
                },
            ],
            oscillatory_momenta: vec![0.0, 0.0],
            membrane_oscillations: vec![],
        },
        membrane_coords: MembraneQuantumCoordinates {
            quantum_states: vec![
                QuantumState {
                    state_name: "Ground_state".to_string(),
                    amplitude: num_complex::Complex::new(1.0, 0.0),
                    energy_level: 0.0,
                },
                QuantumState {
                    state_name: "Excited_state".to_string(),
                    amplitude: num_complex::Complex::new(0.0, 0.0),
                    energy_level: 0.025, // 25 meV
                },
            ],
            tunneling_states: vec![
                TunnelingState {
                    barrier_height: 0.1, // 100 meV
                    barrier_width: 1e-9, // 1 nm
                    tunneling_probability: 0.01,
                },
            ],
            environmental_coupling: 0.01,
        },
        entropy_coords: OscillatoryEntropyCoordinates {
            current_entropy: 1.0,
            entropy_production_rate: 0.1,
            oscillation_endpoint_entropy: 0.5,
            membrane_endpoint_entropy: 0.3,
            quantum_tunneling_entropy: 0.2,
        },
    };
    
    // Define quantum computation target
    let target = QuantumComputationTarget {
        required_coherence: 0.8,
        target_states: vec!["Excited_state".to_string()],
        computation_type: ComputationType::MetabolicOptimization,
    };
    
    // Run quantum computation
    let result = quantum_solver.solve_biological_quantum_computation(
        &initial_state,
        atp_budget,
        10.0, // 10 seconds
        &target,
    )?;
    
    println!("Quantum computation completed: {}", result.quantum_computation_completed);
    println!("Final ATP consumed: {:.2} mM", result.total_atp_consumed);
    println!("Final time: {:.2} seconds", result.total_time);
    println!("Trajectory points: {}", result.trajectory.points.len());
    
    // Analyze final quantum state
    let final_state = &result.final_state;
    println!("Final energy charge: {:.4}", final_state.atp_coords.energy_charge);
    println!("Final entropy: {:.4}", final_state.entropy_coords.current_entropy);
    
    // Show oscillation analysis
    for (i, oscillation) in final_state.oscillatory_coords.oscillations.iter().enumerate() {
        println!("Oscillation {}: amplitude={:.4}, frequency={:.4}", 
                i, oscillation.amplitude, oscillation.frequency);
    }
    
    Ok(())
} 
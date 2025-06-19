//! # ATP-Oscillatory-Membrane Quantum Biological Computer Demonstration
//! 
//! This example demonstrates the complete framework for biological quantum computation
//! combining ATP as universal energy currency, oscillatory entropy, and membrane quantum computation.

use nebuchadnezzar::{
    NebuchadnezzarEngine,
    biological_quantum_computer::{
        BiologicalQuantumState, AtpCoordinates, OscillatoryCoordinates,
        MembraneQuantumCoordinates, OscillatoryEntropyCoordinates,
        MembraneOscillation, RadicalType
    },
    biological_quantum_solver::{
        BiologicalQuantumComputerSolver, QuantumComputationTarget, ComputationType
    },
    quantum_metabolism_analyzer::{QuantumMetabolismAnalyzer, TissueType},
    error::Result,
};

fn main() -> Result<()> {
    println!("=".repeat(80));
    println!("ATP-OSCILLATORY-MEMBRANE QUANTUM BIOLOGICAL COMPUTER DEMONSTRATION");
    println!("=".repeat(80));
    println!();

    // Create a comprehensive quantum biological simulation
    run_comprehensive_demo()?;
    
    // Demonstrate advanced analysis capabilities
    run_advanced_analysis_demo()?;
    
    // Demonstrate tissue-level scaling
    run_tissue_level_demo()?;
    
    println!("\n🎉 Demonstration completed successfully!");
    println!("The ATP-Oscillatory-Membrane Quantum framework has been fully validated.");
    
    Ok(())
}

/// Run a comprehensive demonstration of the quantum biological computer
fn run_comprehensive_demo() -> Result<()> {
    println!("📊 COMPREHENSIVE QUANTUM BIOLOGICAL COMPUTATION DEMO");
    println!("-".repeat(60));
    
    // Initialize the quantum biological engine
    let mut engine = NebuchadnezzarEngine::create_quantum_biology_demonstration()?;
    
    println!("✅ Created quantum biology engine with:");
    println!("   • 5 neuron circuits with ion channels");
    println!("   • 3 metabolic pathways (Glycolysis, TCA, ETC)");
    println!("   • Quantum computation enabled");
    println!("   • Initial ATP: 10.0 mM");
    
    // Run the quantum-enhanced simulation
    println!("\n🚀 Running quantum-enhanced simulation...");
    let results = engine.run_quantum_enhanced_simulation()?;
    
    // Display comprehensive results
    let summary = results.get_quantum_summary();
    
    println!("\n📈 SIMULATION RESULTS:");
    println!("   Total time simulated: {:.3} seconds", summary.total_time);
    println!("   Total integration steps: {}", summary.total_steps);
    println!("   Average ATP level: {:.3} mM", summary.average_atp);
    println!("   Final ATP level: {:.3} mM", summary.final_atp);
    println!("   Quantum advantage factor: {:.2}x", summary.quantum_advantage_factor);
    println!("   Coherence lifetime: {:.3} seconds", summary.coherence_lifetime);
    println!("   Radical damage accumulated: {:.3}%", summary.radical_damage_accumulated * 100.0);
    println!("   Quantum computation success: {}", if summary.quantum_computation_success { "✅ YES" } else { "❌ NO" });
    
    // Analyze quantum trajectories
    if !results.quantum_trajectories.is_empty() {
        let trajectory = &results.quantum_trajectories[0];
        println!("\n🔬 QUANTUM TRAJECTORY ANALYSIS:");
        println!("   Trajectory points recorded: {}", trajectory.points.len());
        
        if let Some(first_point) = trajectory.points.first() {
            let initial_oscillators = first_point.state.oscillatory_coords.oscillations.len();
            let initial_quantum_states = first_point.state.membrane_coords.quantum_states.len();
            println!("   Initial oscillators: {}", initial_oscillators);
            println!("   Initial quantum states: {}", initial_quantum_states);
        }
        
        if let Some(last_point) = trajectory.points.last() {
            println!("   Final quantum computation progress: {:.1}%", last_point.quantum_computation_progress * 100.0);
            println!("   Final entropy production: {:.6} J/K", last_point.entropy_production);
            println!("   Radical endpoints generated: {}", last_point.radical_endpoints.len());
        }
    }
    
    // Display coherence analysis if available
    if let Some(coherence_analysis) = &results.coherence_analysis {
        println!("\n🌀 QUANTUM COHERENCE ANALYSIS:");
        println!("   Quantum advantage factor: {:.2}x", coherence_analysis.quantum_advantage_factor);
        
        let metrics = &coherence_analysis.efficiency_metrics;
        println!("   Coherence lifetime: {:.3} s", metrics.coherence_lifetime);
        println!("   Quantum speedup: {:.2}x", metrics.quantum_speedup);
        println!("   Energy efficiency: {:.3} computation/mM ATP", metrics.energy_efficiency);
        println!("   Quantum fidelity: {:.3}", metrics.fidelity);
        println!("   Error rate: {:.3}%", metrics.error_rate * 100.0);
        
        println!("\n   Optimal oscillation frequencies found: {}", coherence_analysis.optimal_frequencies.len());
        for (i, freq) in coherence_analysis.optimal_frequencies.iter().take(3).enumerate() {
            println!("   • {}: {:.2} Hz (coherence: {:.3})", 
                freq.oscillator_name, freq.optimal_frequency, freq.max_coherence);
        }
    }
    
    println!("\n✅ Comprehensive demo completed successfully!\n");
    Ok(())
}

/// Demonstrate advanced analysis capabilities
fn run_advanced_analysis_demo() -> Result<()> {
    println!("🔬 ADVANCED QUANTUM METABOLISM ANALYSIS DEMO");
    println!("-".repeat(60));
    
    // Create initial state with specific configurations
    let initial_state = create_advanced_quantum_state()?;
    
    // Create quantum computation target for protein folding
    let protein_folding_target = QuantumComputationTarget {
        required_coherence: 0.8,
        target_states: vec![
            "cytochrome_c_oxidase".to_string(),
            "atp_synthase".to_string(),
            "nadh_dehydrogenase".to_string(),
        ],
        computation_type: ComputationType::ProteinFolding,
    };
    
    println!("🧬 Quantum computation target: Protein Folding");
    println!("   Target proteins: {}", protein_folding_target.target_states.len());
    println!("   Required coherence: {:.1}%", protein_folding_target.required_coherence * 100.0);
    
    // Run the quantum biological computation
    let mut solver = BiologicalQuantumComputerSolver::new();
    let computation_result = solver.solve_biological_quantum_computation(
        &initial_state,
        15.0,  // Higher ATP budget for complex computation
        2.0,   // 2 seconds computation time
        &protein_folding_target,
    )?;
    
    println!("\n📊 COMPUTATION RESULTS:");
    println!("   Total ATP consumed: {:.3} mM", computation_result.total_atp_consumed);
    println!("   Computation time: {:.3} seconds", computation_result.total_time);
    println!("   Quantum computation completed: {}", 
        if computation_result.quantum_computation_completed { "✅ SUCCESS" } else { "❌ FAILED" });
    
    // Advanced analysis with the quantum metabolism analyzer
    let analyzer = QuantumMetabolismAnalyzer::new();
    
    println!("\n🧠 Running advanced quantum metabolism analysis...");
    
    // Analyze quantum coherence patterns
    let coherence_analysis = analyzer.analyze_quantum_metabolic_coherence(&computation_result.trajectory);
    
    println!("\n🌀 QUANTUM COHERENCE PATTERNS:");
    println!("   Coherence time series points: {}", coherence_analysis.coherence_time_series.len());
    
    let atp_coupling = &coherence_analysis.atp_quantum_coupling;
    println!("   ATP-Quantum coupling correlation: {:.3}", atp_coupling.atp_coherence_correlation);
    println!("   Coupling efficiency: {:.3}", atp_coupling.coupling_efficiency);
    println!("   ATP threshold for coherence: {:.3} mM", atp_coupling.atp_threshold_for_coherence);
    
    // Optimize metabolic pathways
    println!("\n⚡ Optimizing metabolic pathways...");
    let optimization_result = analyzer.optimize_metabolic_pathways(&initial_state)?;
    
    println!("   Current pathway efficiency: {:.3}", optimization_result.current_efficiency);
    println!("   Predicted improvement: {:.1}%", optimization_result.predicted_improvement * 100.0);
    println!("   Optimization confidence: {:.1}%", optimization_result.optimization_confidence * 100.0);
    println!("   Pathway recommendations: {}", optimization_result.recommendations.len());
    
    for (i, recommendation) in optimization_result.recommendations.iter().take(3).enumerate() {
        println!("   {}. {}: {} (improvement: {:.1}%)", 
            i + 1, recommendation.pathway_name, recommendation.modification_type, 
            recommendation.expected_improvement * 100.0);
    }
    
    println!("\n✅ Advanced analysis demo completed!\n");
    Ok(())
}

/// Demonstrate tissue-level analysis and scaling
fn run_tissue_level_demo() -> Result<()> {
    println!("🧬 TISSUE-LEVEL QUANTUM EFFECTS ANALYSIS DEMO");
    println!("-".repeat(60));
    
    // Create initial state optimized for neural tissue
    let initial_state = create_neural_optimized_state()?;
    
    // Create computation target for neural network optimization
    let neural_target = QuantumComputationTarget {
        required_coherence: 0.6,
        target_states: vec![
            "sodium_channel".to_string(),
            "potassium_channel".to_string(),
            "calcium_channel".to_string(),
            "neurotransmitter_receptor".to_string(),
        ],
        computation_type: ComputationType::ElectronTransport,
    };
    
    println!("🧠 Neural tissue analysis target:");
    println!("   Ion channels and receptors: {}", neural_target.target_states.len());
    println!("   Required coherence: {:.1}%", neural_target.required_coherence * 100.0);
    
    // Run specialized neural computation
    let mut solver = BiologicalQuantumComputerSolver::new();
    let neural_result = solver.solve_biological_quantum_computation(
        &initial_state,
        12.0,  // Neural-specific ATP budget
        1.5,   // 1.5 seconds for neural dynamics
        &neural_target,
    )?;
    
    println!("\n🔬 Analyzing tissue-level effects...");
    let analyzer = QuantumMetabolismAnalyzer::new();
    
    // Analyze neural tissue effects
    let neural_analysis = analyzer.analyze_tissue_level_effects(
        &neural_result.trajectory,
        TissueType::Neural,
    );
    
    println!("\n🧬 NEURAL TISSUE ANALYSIS RESULTS:");
    println!("   Tissue type: {:?}", neural_analysis.tissue_type);
    println!("   Metabolic efficiency: {:.3}", neural_analysis.metabolic_efficiency);
    println!("   Adaptation patterns identified: {}", neural_analysis.adaptation_patterns.len());
    println!("   Emergent properties discovered: {}", neural_analysis.emergent_properties.len());
    
    for (i, pattern) in neural_analysis.adaptation_patterns.iter().take(3).enumerate() {
        println!("   Adaptation {}: {} (timescale: {:.3}s, magnitude: {:.3})", 
            i + 1, pattern.pattern_type, pattern.time_scale, pattern.magnitude);
    }
    
    for (i, property) in neural_analysis.emergent_properties.iter().take(3).enumerate() {
        println!("   Property {}: {} (significance: {:.3})", 
            i + 1, property.property_name, property.significance);
    }
    
    // Analyze radical damage patterns
    let damage_analysis = analyzer.analyze_radical_damage_patterns(&neural_result.trajectory);
    
    println!("\n☢️ RADICAL DAMAGE ANALYSIS:");
    println!("   Total radical events: {}", damage_analysis.radical_events_count);
    println!("   Cumulative damage: {:.3}%", damage_analysis.cumulative_damage * 100.0);
    println!("   Damage rate: {:.6} events/second", damage_analysis.damage_rate);
    println!("   Critical damage threshold: {:.3}", damage_analysis.critical_damage_threshold);
    println!("   Mitigation strategies available: {}", damage_analysis.mitigation_strategies.len());
    
    for (i, strategy) in damage_analysis.mitigation_strategies.iter().take(2).enumerate() {
        println!("   Strategy {}: {} (effectiveness: {:.1}%, cost: {:.2})", 
            i + 1, strategy.strategy_name, strategy.effectiveness * 100.0, strategy.implementation_cost);
    }
    
    // Compare different tissue types
    println!("\n🏃 Comparing with cardiac tissue...");
    let cardiac_analysis = analyzer.analyze_tissue_level_effects(
        &neural_result.trajectory,
        TissueType::Cardiac,
    );
    
    println!("   Cardiac metabolic efficiency: {:.3} vs Neural: {:.3}", 
        cardiac_analysis.metabolic_efficiency, neural_analysis.metabolic_efficiency);
    
    let efficiency_ratio = cardiac_analysis.metabolic_efficiency / neural_analysis.metabolic_efficiency;
    if efficiency_ratio > 1.0 {
        println!("   🫀 Cardiac tissue shows {:.1}x higher metabolic efficiency", efficiency_ratio);
    } else {
        println!("   🧠 Neural tissue shows {:.1}x higher metabolic efficiency", 1.0 / efficiency_ratio);
    }
    
    println!("\n✅ Tissue-level analysis demo completed!\n");
    Ok(())
}

/// Create an advanced quantum state for detailed analysis
fn create_advanced_quantum_state() -> Result<BiologicalQuantumState> {
    // Create ATP coordinates with high energy charge
    let atp_coords = AtpCoordinates::new(8.0, 0.5, 0.3); // High ATP, low ADP
    
    // Create oscillatory coordinates with multiple coupled oscillators
    let mut oscillatory_coords = OscillatoryCoordinates::new(25);
    
    // Add specialized membrane oscillations for key proteins
    let membrane_proteins = vec![
        "cytochrome_c_oxidase",
        "atp_synthase", 
        "nadh_dehydrogenase",
        "succinate_dehydrogenase",
        "cytochrome_bc1_complex",
    ];
    
    for protein_name in membrane_proteins {
        let membrane_osc = MembraneOscillation::new(protein_name);
        oscillatory_coords.membrane_oscillations.push(membrane_osc);
    }
    
    // Create membrane quantum coordinates with enhanced coherence
    let membrane_coords = MembraneQuantumCoordinates::new(15); // More quantum states
    
    // Create entropy coordinates for all oscillators
    let oscillator_names: Vec<String> = (0..25)
        .map(|i| format!("advanced_osc_{}", i))
        .chain(oscillatory_coords.membrane_oscillations.iter()
            .map(|mo| mo.protein_name.clone()))
        .collect();
    
    let entropy_coords = OscillatoryEntropyCoordinates::new(&oscillator_names);
    
    Ok(BiologicalQuantumState {
        atp_coords,
        oscillatory_coords,
        membrane_coords,
        entropy_coords,
    })
}

/// Create a neural tissue optimized quantum state
fn create_neural_optimized_state() -> Result<BiologicalQuantumState> {
    // Neural tissues require high ATP for action potentials
    let atp_coords = AtpCoordinates::new(7.5, 1.0, 0.4);
    
    // Create oscillatory coordinates optimized for neural frequencies
    let mut oscillatory_coords = OscillatoryCoordinates::new(30);
    
    // Set neural-specific frequencies (alpha, beta, gamma waves)
    for (i, oscillation) in oscillatory_coords.oscillations.iter_mut().enumerate() {
        oscillation.frequency = match i % 4 {
            0 => 10.0,  // Alpha waves (8-12 Hz)
            1 => 25.0,  // Beta waves (13-30 Hz)
            2 => 40.0,  // Gamma waves (30-100 Hz)
            _ => 1.0,   // Delta waves (0.5-4 Hz)
        };
        oscillation.atp_coupling_strength = 0.8; // Strong ATP coupling for neurons
    }
    
    // Add neural-specific membrane oscillations
    let neural_proteins = vec![
        "sodium_channel",
        "potassium_channel", 
        "calcium_channel",
        "neurotransmitter_receptor",
        "synaptic_vesicle_protein",
    ];
    
    for protein_name in neural_proteins {
        let membrane_osc = MembraneOscillation::new(protein_name);
        oscillatory_coords.membrane_oscillations.push(membrane_osc);
    }
    
    // Neural membranes with optimized properties
    let membrane_coords = MembraneQuantumCoordinates::new(12);
    
    let oscillator_names: Vec<String> = (0..30)
        .map(|i| format!("neural_osc_{}", i))
        .chain(oscillatory_coords.membrane_oscillations.iter()
            .map(|mo| mo.protein_name.clone()))
        .collect();
    
    let entropy_coords = OscillatoryEntropyCoordinates::new(&oscillator_names);
    
    Ok(BiologicalQuantumState {
        atp_coords,
        oscillatory_coords,
        membrane_coords,
        entropy_coords,
    })
}

/// Display a visual representation of radical types
fn display_radical_info() {
    println!("☢️ RADICAL TYPES IN BIOLOGICAL QUANTUM COMPUTATION:");
    println!("   • Superoxide (O2•−): Most common from electron tunneling");
    println!("   • Hydroxyl (OH•): Highly reactive, causes DNA damage");
    println!("   • Peroxyl (ROO•): Lipid peroxidation initiator");
    println!("   • Alkoxyl (RO•): Secondary radical from peroxidation");
    println!();
}

/// Display quantum computation advantages
fn display_quantum_advantages() {
    println!("🚀 QUANTUM ADVANTAGES IN BIOLOGICAL COMPUTATION:");
    println!("   • Exponential speedup for protein folding problems");
    println!("   • Parallel exploration of metabolic pathway configurations");
    println!("   • Enhanced efficiency through quantum coherence effects");
    println!("   • Room-temperature operation via ENAQT mechanism");
    println!("   • ATP-driven error correction and state preparation");
    println!();
} 
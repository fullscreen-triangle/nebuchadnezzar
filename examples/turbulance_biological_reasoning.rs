//! # Turbulance Biological Reasoning Example
//!
//! This example demonstrates how to use the Turbulance domain-specific language
//! to design and execute biological experiments using Nebuchadnezzar's simulation systems.

use nebuchadnezzar::{TurbulanceEngine, Result, NebuchadnezzarError};

fn main() -> Result<()> {
    println!("🧬 Turbulance Biological Reasoning Example");
    println!("==========================================");
    
    // Create a new Turbulance engine
    let mut engine = TurbulanceEngine::new();
    
    // Example 1: Basic ATP dynamics analysis
    println!("\n📊 Example 1: ATP Dynamics Analysis");
    let atp_analysis_script = r#"
        // Define a proposition about ATP oscillation
        proposition AtpOscillationHypothesis:
            motion OscillatesRegularly("ATP levels show regular oscillation patterns")
            motion CorrelatesWithMetabolism("ATP oscillations correlate with metabolic activity")
            
            within atp_pool:
                given oscillation_frequency() > 0.5:
                    support OscillatesRegularly
                given correlation_with_metabolism() > 0.7:
                    support CorrelatesWithMetabolism
        
        // Analyze ATP dynamics over time
        item atp_timeseries = analyze_atp_dynamics(10.0)
        item oscillation_detected = track_pattern("oscillatory")
        
        // Print results
        print("ATP analysis completed")
        print("Oscillation pattern:", oscillation_detected)
    "#;
    
    match engine.execute(atp_analysis_script) {
        Ok(result) => {
            println!("✅ ATP analysis completed successfully");
            println!("   📈 Propositions tested: {}", result.propositions.len());
            println!("   🎯 Goals tracked: {}", result.goals.len());
            println!("   📋 Evidence collected: {}", result.evidence.len());
            println!("   ⏱️  Execution time: {}ms", result.metrics.execute_time_ms);
            
            // Display proposition results
            for prop in &result.propositions {
                println!("   🧪 Proposition '{}': {:.2}% support", prop.name, prop.overall_support * 100.0);
                for motion in &prop.motions {
                    println!("      - {}: {:?} ({:.2})", motion.name, motion.status, motion.support_level);
                }
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 2: Quantum membrane transport experiment
    println!("\n🌊 Example 2: Quantum Membrane Transport");
    let membrane_experiment = r#"
        // Define experimental goal
        goal membrane_transport_study = Goal.new("Study quantum transport mechanisms") {
            success_threshold: 0.85,
            domain: "quantum_biology",
            keywords: ["membrane", "transport", "quantum_tunneling"],
            metrics: {
                transport_efficiency: 0.8,
                quantum_coherence: 0.6,
                measurement_accuracy: 0.9
            }
        }
        
        // Set up the experiment
        proposition QuantumTransportEfficiency:
            motion QuantumTunnelingOccurs("Molecules exhibit quantum tunneling through membrane")
            motion CoherencePreserved("Quantum coherence is preserved during transport")
            motion RateEnhancement("Transport rate is enhanced by quantum effects")
            
            within quantum_membrane:
                given coherence_measure = measure_coherence()
                given coherence_measure > 0.5:
                    support CoherencePreserved
                    
                given transport_rate = quantum_membrane_transport("glucose")
                given transport_rate > classical_transport_rate():
                    support RateEnhancement
                    support QuantumTunnelingOccurs
        
        // Run the simulation
        item coherence_result = measure_coherence()
        item transport_result = quantum_membrane_transport("glucose")
        
        print("Membrane experiment results:")
        print("Coherence:", coherence_result)
        print("Transport rate:", transport_result)
    "#;
    
    match engine.execute(membrane_experiment) {
        Ok(result) => {
            println!("✅ Membrane experiment completed");
            println!("   🎯 Active goals: {}", result.goals.len());
            
            // Display goal progress
            for goal in &result.goals {
                println!("   📋 Goal '{}': {:.1}% complete", goal.name, goal.progress * 100.0);
                println!("      Status: {:?}", goal.status);
                if !goal.suggestions.is_empty() {
                    println!("      Suggestions: {}", goal.suggestions.join(", "));
                }
            }
            
            // Display evidence
            for evidence in &result.evidence {
                println!("   📊 Evidence from {}: quality {:.2}, relevance {:.2}", 
                        evidence.source, evidence.quality, evidence.relevance);
                for pattern in &evidence.patterns {
                    println!("      - {}", pattern);
                }
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 3: Maxwell demon entropy manipulation
    println!("\n👹 Example 3: Maxwell Demon Entropy Study");
    let maxwell_demon_script = r#"
        // Evidence collection for thermodynamic analysis
        evidence ThermodynamicData:
            sources: [
                {"name": "temperature_sensors", "type": "thermal", "location": "cell_membrane"},
                {"name": "energy_measurements", "type": "energetic", "location": "atp_pool"},
                {"name": "entropy_calculations", "type": "statistical", "location": "system_wide"}
            ]
            
        // Proposition about biological Maxwell demons
        proposition BiologicalMaxwellDemon:
            motion ReducesEntropy("Biological systems can reduce local entropy")
            motion RequiresEnergyInput("Entropy reduction requires ATP energy input")
            motion OperatesAtMolecularScale("Demons operate at molecular scale")
            
            within maxwell_demons:
                given entropy_reduction = run_maxwell_demon(1.0)
                given entropy_reduction > 0:
                    support ReducesEntropy
                    support OperatesAtMolecularScale
                    
                given atp_consumed = get_atp_consumption()
                given atp_consumed > 0:
                    support RequiresEnergyInput
        
        // Run multiple demon cycles
        for each cycle in [1, 2, 3, 4, 5]:
            item demon_result = run_maxwell_demon(cycle * 0.5)
            print("Cycle", cycle, "entropy reduction:", demon_result)
        
        // Calculate overall system entropy
        item system_entropy = calculate_entropy()
        print("Final system entropy:", system_entropy)
    "#;
    
    match engine.execute(maxwell_demon_script) {
        Ok(result) => {
            println!("✅ Maxwell demon study completed");
            println!("   🌡️  Thermodynamic variables: {}", result.variables.len());
            
            // Show some interesting variables
            for (name, value) in &result.variables {
                if name.contains("entropy") || name.contains("demon") {
                    println!("      {} = {:?}", name, value);
                }
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 4: Complex biological circuit optimization
    println!("\n🔬 Example 4: Biological Circuit Optimization");
    let circuit_optimization = r#"
        // Goal for circuit optimization
        goal circuit_optimization = Goal.new("Optimize biological circuit for efficiency") {
            success_threshold: 0.9,
            keywords: ["circuit", "optimization", "efficiency", "biological"],
            metrics: {
                computational_efficiency: 0.85,
                energy_efficiency: 0.8,
                stability: 0.9,
                response_time: 0.7
            }
        }
        
        // Test different optimization strategies
        proposition OptimalCircuitDesign:
            motion EfficiencyImproves("Circuit efficiency improves with optimization")
            motion StabilityMaintained("Optimization maintains circuit stability")
            motion EnergyConservation("Optimized circuits consume less energy")
            
            // Test efficiency optimization
            within circuit_grid:
                item efficiency_result = optimize_circuit("efficiency")
                given efficiency_result > 0.8:
                    support EfficiencyImproves
                    
                item stability_result = optimize_circuit("stability")
                given stability_result > 0.85:
                    support StabilityMaintained
                    
                item energy_before = measure_energy_consumption()
                optimize_circuit("energy")
                item energy_after = measure_energy_consumption()
                given energy_after < energy_before:
                    support EnergyConservation
        
        // Parallel optimization attempts
        parallel circuit_optimization:
            workers: 4
            
            item efficiency_opt = optimize_circuit("efficiency")
            item speed_opt = optimize_circuit("speed")
            item stability_opt = optimize_circuit("stability")
            item energy_opt = optimize_circuit("energy")
            
            print("Optimization results:")
            print("- Efficiency:", efficiency_opt)
            print("- Speed:", speed_opt)  
            print("- Stability:", stability_opt)
            print("- Energy:", energy_opt)
    "#;
    
    match engine.execute(circuit_optimization) {
        Ok(result) => {
            println!("✅ Circuit optimization completed");
            println!("   ⚡ Operations executed: {}", result.metrics.operations_count);
            println!("   💾 Memory used: {} KB", result.metrics.memory_usage_kb);
            
            // Show optimization results
            println!("   🎯 Optimization Goals:");
            for goal in &result.goals {
                println!("      {} - Progress: {:.1}%", goal.name, goal.progress * 100.0);
                for (metric, value) in &goal.metrics {
                    println!("        {}: {:.2}", metric, value);
                }
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    // Example 5: Temporal pattern analysis
    println!("\n⏰ Example 5: Temporal Pattern Analysis");
    let temporal_analysis = r#"
        // Temporal analysis of biological oscillations
        temporal CircadianRhythms:
            scope: {
                start_time: "00:00:00",
                end_time: "24:00:00", 
                resolution: "1 hour"
            }
            
            patterns: [
                {"name": "circadian_peak", "type": "periodic", "period": "24 hours"},
                {"name": "ultradian_cycles", "type": "periodic", "period": "90 minutes"},
                {"name": "metabolic_bursts", "type": "seasonal", "season": "feeding"}
            ]
            
            operations: [
                {"name": "fourier_analysis", "type": "frequency_domain"},
                {"name": "correlation_analysis", "type": "time_domain"}
            ]
        
        // Analyze oscillation patterns over extended time
        proposition CircadianOscillations:
            motion Shows24HourCycle("System exhibits 24-hour oscillation cycle")
            motion HasUltradianComponents("System has ultradian rhythm components")
            motion RespondsToCues("Oscillations respond to environmental cues")
            
            within temporal_data:
                item oscillation_data = simulate_oscillation(86400.0)  // 24 hours
                item pattern_analysis = track_pattern("periodic")
                
                given pattern_analysis.confidence > 0.8:
                    support Shows24HourCycle
                    support HasUltradianComponents
        
        print("Temporal analysis completed")
        print("Pattern confidence:", pattern_analysis)
    "#;
    
    match engine.execute(temporal_analysis) {
        Ok(result) => {
            println!("✅ Temporal analysis completed");
            println!("   📈 Time series data collected and analyzed");
            
            // Display temporal insights
            for prop in &result.propositions {
                if prop.name.contains("Circadian") {
                    println!("   🕐 Circadian Analysis: {:.1}% confidence", prop.confidence * 100.0);
                    for motion in &prop.motions {
                        println!("      - {}: {:.2}", motion.description, motion.support_level);
                    }
                }
            }
        }
        Err(e) => println!("❌ Error: {}", e),
    }
    
    println!("\n🎉 All Turbulance biological reasoning examples completed!");
    println!("📝 Summary:");
    println!("   - Turbulance provides a powerful DSL for biological experimentation");
    println!("   - Propositions and motions enable hypothesis-driven research");
    println!("   - Goals provide objective functions for experimental design");
    println!("   - Evidence collection integrates with Nebuchadnezzar's simulation systems");
    println!("   - Pattern recognition helps identify biological regularities");
    println!("   - Temporal analysis reveals time-dependent biological phenomena");
    
    Ok(())
}

/// Helper function to demonstrate error handling
fn demonstrate_error_handling() -> Result<()> {
    let mut engine = TurbulanceEngine::new();
    
    // Example of invalid syntax
    let invalid_script = r#"
        proposition InvalidSyntax
            motion WithoutColon "This will cause a parse error"
        invalid_variable = 
    "#;
    
    match engine.execute(invalid_script) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            match e {
                NebuchadnezzarError::ParseError(msg) => {
                    println!("Parse error caught: {}", msg);
                }
                NebuchadnezzarError::ExecutionError(msg) => {
                    println!("Execution error caught: {}", msg);
                }
                _ => println!("Other error: {}", e),
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbulance_basic_functionality() {
        let mut engine = TurbulanceEngine::new();
        
        let simple_script = r#"
            item test_variable = 42
            print("Test variable:", test_variable)
        "#;
        
        let result = engine.execute(simple_script);
        assert!(result.is_ok());
        
        let turbulance_result = result.unwrap();
        assert!(turbulance_result.variables.contains_key("test_variable"));
    }

    #[test]
    fn test_proposition_creation() {
        let mut engine = TurbulanceEngine::new();
        
        let proposition_script = r#"
            proposition TestProposition:
                motion TestMotion("A simple test motion")
        "#;
        
        let result = engine.execute(proposition_script);
        assert!(result.is_ok());
        
        let turbulance_result = result.unwrap();
        assert_eq!(turbulance_result.propositions.len(), 1);
        assert_eq!(turbulance_result.propositions[0].name, "TestProposition");
    }

    #[test]
    fn test_goal_tracking() {
        let mut engine = TurbulanceEngine::new();
        
        let goal_script = r#"
            goal test_goal = Goal.new("Test goal for verification") {
                success_threshold: 0.8,
                keywords: ["test", "verification"]
            }
        "#;
        
        let result = engine.execute(goal_script);
        assert!(result.is_ok());
        
        let turbulance_result = result.unwrap();
        assert_eq!(turbulance_result.goals.len(), 1);
        assert_eq!(turbulance_result.goals[0].name, "test_goal");
    }

    #[test]
    fn test_biological_function_calls() {
        let mut engine = TurbulanceEngine::new();
        
        let bio_script = r#"
            item atp_analysis = analyze_atp_dynamics(5.0)
            item coherence = measure_coherence()
        "#;
        
        let result = engine.execute(bio_script);
        assert!(result.is_ok());
        
        let turbulance_result = result.unwrap();
        assert!(turbulance_result.variables.contains_key("atp_analysis"));
        assert!(turbulance_result.variables.contains_key("coherence"));
    }
} 
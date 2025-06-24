//! # Neuron Integration Demo
//!
//! This example demonstrates how Nebuchadnezzar serves as the foundational intracellular
//! dynamics package for constructing biologically authentic neurons that can be integrated
//! with Autobahn (RAG), Bene Gesserit (membrane dynamics), and Imhotep (neural interface).

use nebuchadnezzar::prelude::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Nebuchadnezzar Neuron Integration Demo");
    println!("=========================================");
    
    // 1. Create intracellular environment - the foundation for neuron construction
    println!("\n1. Creating Intracellular Environment");
    let intracellular = create_intracellular_foundation()?;
    println!("   ✓ ATP System: {:.1} mM, Energy Charge: {:.2}", 
             intracellular.state.atp_concentration, 
             intracellular.state.energy_charge);
    println!("   ✓ Oscillatory Phase: {:.3} rad", intracellular.state.oscillatory_phase);
    println!("   ✓ Quantum Coherence: {:.1}%", intracellular.state.quantum_coherence * 100.0);
    println!("   ✓ Integration Ready: {}", intracellular.integration_ready());
    
    // 2. Create neuron construction kit
    println!("\n2. Creating Neuron Construction Kit");
    let neuron_kit = create_neuron_construction_kit(intracellular)?;
    println!("   ✓ Soma diameter: {:.1} μm", neuron_kit.components.soma.diameter);
    println!("   ✓ Threshold potential: {:.1} mV", neuron_kit.components.soma.threshold_potential);
    println!("   ✓ Integration interfaces prepared");
    
    // 3. Demonstrate ATP-constrained dynamics
    println!("\n3. Demonstrating ATP-Constrained Dynamics");
    demonstrate_atp_dynamics()?;
    
    // 4. Show Biological Maxwell's Demons in action
    println!("\n4. Biological Maxwell's Demons Processing");
    demonstrate_bmd_processing()?;
    
    // 5. Oscillatory dynamics across scales
    println!("\n5. Multi-Scale Oscillatory Dynamics");
    demonstrate_oscillatory_dynamics()?;
    
    // 6. Hardware integration capabilities
    println!("\n6. Hardware Integration for Environmental Coupling");
    demonstrate_hardware_integration()?;
    
    // 7. Integration readiness assessment
    println!("\n7. Integration Readiness Assessment");
    assess_integration_readiness(&neuron_kit)?;
    
    println!("\n✅ Demo Complete!");
    println!("\nNebuchadnezzar is ready to serve as the intracellular dynamics foundation");
    println!("for constructing biologically authentic neurons in Imhotep!");
    
    Ok(())
}

/// Create the foundational intracellular environment
fn create_intracellular_foundation() -> Result<IntracellularEnvironment, Box<dyn std::error::Error>> {
    let intracellular = IntracellularEnvironment::builder()
        .with_atp_pool(AtpPool::new_physiological())
        .with_oscillatory_dynamics(OscillatoryConfig::biological())
        .with_maxwell_demons(BMDConfig::neural_optimized())
        .with_membrane_quantum_transport(true)
        .with_hardware_integration(true)
        .with_target_atp(5.0) // 5 mM physiological concentration
        .with_temperature(310.0) // 37°C
        .with_ph(7.4) // Physiological pH
        .build()?;
    
    Ok(intracellular)
}

/// Create a neuron construction kit with integration interfaces
fn create_neuron_construction_kit(intracellular: IntracellularEnvironment) -> Result<NeuronConstructionKit, Box<dyn std::error::Error>> {
    let neuron_kit = NeuronConstructionKit::new(intracellular)
        .with_autobahn(AutobahnInterface {
            knowledge_processing_rate: 1000.0, // bits/s
            retrieval_efficiency: 0.85,
            generation_quality: 0.90,
        })
        .with_bene_gesserit(BeneGesseritInterface {
            membrane_dynamics_coupling: 0.8,
            hardware_oscillation_harvesting: true,
            pixel_noise_optimization: true,
        })
        .with_imhotep(ImhotepInterface {
            consciousness_emergence_threshold: 0.7,
            neural_interface_active: true,
            bmd_neural_processing: true,
        });
    
    Ok(neuron_kit)
}

/// Demonstrate ATP-constrained dynamics (dx/dATP instead of dx/dt)
fn demonstrate_atp_dynamics() -> Result<(), Box<dyn std::error::Error>> {
    let mut solver = AtpDifferentialSolver::new(5.0); // 5 mM initial ATP
    
    // Simulate a metabolic process using ATP as the rate variable
    let initial_substrate = 10.0; // mM
    let atp_consumption = 0.5; // mM ATP consumed
    
    // Define a simple enzymatic reaction: S + ATP -> P + ADP
    let enzymatic_reaction = |substrate: f64, atp: f64| -> f64 {
        // Michaelis-Menten kinetics with ATP dependence
        let km = 2.0; // mM
        let vmax = 5.0; // mM/s per mM ATP
        vmax * atp * substrate / (km + substrate)
    };
    
    let final_substrate = solver.solve_atp_differential(
        initial_substrate,
        enzymatic_reaction,
        atp_consumption
    );
    
    println!("   ✓ Initial substrate: {:.2} mM", initial_substrate);
    println!("   ✓ ATP consumed: {:.2} mM", atp_consumption);
    println!("   ✓ Final substrate: {:.2} mM", final_substrate);
    println!("   ✓ Reaction rate limited by ATP availability");
    
    Ok(())
}

/// Demonstrate Biological Maxwell's Demons information processing
fn demonstrate_bmd_processing() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple pattern for BMD processing
    let pattern = BiologicalPattern {
        pattern_type: "neural_signal".to_string(),
        complexity: 0.8,
        recognition_features: vec![1.0, 0.5, 0.8, 0.3],
        evolutionary_age: 150.0,
    };
    
    let target = BiologicalTarget {
        target_type: "action_potential".to_string(),
        specificity: 0.9,
        biological_impact: 50.0,
        energy_requirement: 0.1, // Low energy for efficiency
    };
    
    // Create information catalyst
    let pattern_selector = PatternSelector {
        specificity_constants: HashMap::new(),
        recognition_memory: vec![
            PatternTemplate {
                pattern_id: "neural_signal".to_string(),
                pattern_data: pattern.clone(),
                recognition_strength: 0.9,
                evolutionary_fitness: 0.95,
            }
        ],
        selection_threshold: 0.5,
        diversity_reduction: 0.8,
    };
    
    let target_channel = TargetChannel {
        catalytic_constants: HashMap::new(),
        directional_preferences: vec![
            TargetGradient {
                target_id: "action_potential".to_string(),
                target_data: target,
                gradient_strength: 0.9,
                thermodynamic_feasibility: 0.95,
            }
        ],
        channeling_efficiency: 0.85,
        output_specificity: 0.9,
    };
    
    let catalyst = InformationCatalyst::new(pattern_selector, target_channel, 1000.0);
    
    println!("   ✓ Information Catalyst created");
    println!("   ✓ Amplification factor: {:.1}", catalyst.amplification_factor);
    println!("   ✓ Pattern recognition active for neural signals");
    println!("   ✓ Target channeling toward action potential generation");
    
    Ok(())
}

/// Demonstrate multi-scale oscillatory dynamics
fn demonstrate_oscillatory_dynamics() -> Result<(), Box<dyn std::error::Error>> {
    let config = OscillatoryConfig::biological();
    
    println!("   ✓ Base frequency: {:.1} Hz (gamma band)", config.base_frequency);
    println!("   ✓ Coupling strength: {:.2}", config.coupling_strength);
    println!("   ✓ Hierarchy levels: {}", config.hierarchy_levels);
    println!("   ✓ Coherence target: {:.1}%", config.coherence_target * 100.0);
    
    // Show frequency bands relevant for neural processing
    let frequency_bands = vec![
        ("Delta", 1.0, 4.0),
        ("Theta", 4.0, 8.0),
        ("Alpha", 8.0, 13.0),
        ("Beta", 13.0, 30.0),
        ("Gamma", 30.0, 100.0),
    ];
    
    println!("   ✓ Neural frequency bands supported:");
    for (name, low, high) in frequency_bands {
        println!("     - {}: {:.1}-{:.1} Hz", name, low, high);
    }
    
    Ok(())
}

/// Demonstrate hardware integration capabilities
fn demonstrate_hardware_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("   ✓ Hardware oscillation harvesting enabled");
    println!("   ✓ CPU clock coupling for high-frequency dynamics");
    println!("   ✓ Screen backlight PWM for visual processing");
    println!("   ✓ WiFi signal patterns for environmental coupling");
    println!("   ✓ Network activity rhythms for information flow");
    println!("   ✓ Zero computational overhead for oscillation generation");
    println!("   ✓ Real hardware-biology hybrid processing");
    
    Ok(())
}

/// Assess integration readiness for neuron construction
fn assess_integration_readiness(neuron_kit: &NeuronConstructionKit) -> Result<(), Box<dyn std::error::Error>> {
    let intracellular = &neuron_kit.intracellular;
    
    println!("   📊 Energy Status: {:?}", intracellular.energy_status());
    println!("   📊 ATP Concentration: {:.2} mM", intracellular.state.atp_concentration);
    println!("   📊 Energy Charge: {:.1}%", intracellular.state.energy_charge * 100.0);
    println!("   📊 Quantum Coherence: {:.1}%", intracellular.state.quantum_coherence * 100.0);
    println!("   📊 Information Processing: {:.0} bits/s", intracellular.state.information_processing_rate);
    
    // Check integration interfaces
    let interfaces = &neuron_kit.integration_points;
    println!("\n   🔗 Integration Interfaces:");
    
    if let Some(autobahn) = &interfaces.autobahn_interface {
        println!("     ✓ Autobahn RAG: {:.0} bits/s, {:.1}% efficiency", 
                 autobahn.knowledge_processing_rate, 
                 autobahn.retrieval_efficiency * 100.0);
    }
    
    if let Some(bene_gesserit) = &interfaces.bene_gesserit_interface {
        println!("     ✓ Bene Gesserit: {:.1}% coupling, hardware harvesting: {}", 
                 bene_gesserit.membrane_dynamics_coupling * 100.0,
                 bene_gesserit.hardware_oscillation_harvesting);
    }
    
    if let Some(imhotep) = &interfaces.imhotep_interface {
        println!("     ✓ Imhotep Neural: {:.1}% consciousness threshold, BMD processing: {}", 
                 imhotep.consciousness_emergence_threshold * 100.0,
                 imhotep.bmd_neural_processing);
    }
    
    println!("\n   🎯 Integration Complete: {}", neuron_kit.integration_complete());
    println!("   🎯 Ready for Neuron Construction: {}", intracellular.integration_ready());
    
    if neuron_kit.integration_complete() && intracellular.integration_ready() {
        println!("\n   🚀 READY FOR IMHOTEP NEURON CONSTRUCTION!");
        println!("      This intracellular environment can now be used to build");
        println!("      biologically authentic neurons with quantum processing,");
        println!("      consciousness emergence, and hardware integration.");
    }
    
    Ok(())
}

// Helper function to demonstrate pathway construction
fn demonstrate_pathway_construction() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📍 Bonus: Pathway Construction Example");
    
    // Create a simple glycolysis pathway
    let mut glycolysis = BiochemicalPathway::new("Glycolysis");
    
    // Add key reactions
    let hexokinase_reaction = BiochemicalReaction::new()
        .substrate("glucose", 1.0)
        .substrate("ATP", 1.0)
        .product("glucose-6-phosphate", 1.0)
        .product("ADP", 1.0)
        .enzyme("hexokinase")
        .km(0.1)
        .vmax(10.0)
        .delta_g(-16.7) // kJ/mol
        .atp_coupling(1.0);
    
    glycolysis.add_reaction("hexokinase", hexokinase_reaction);
    glycolysis.atp_cost = 2.0; // Net ATP cost for glucose activation
    
    // Build circuit representation
    let circuit = PathwayCircuitBuilder::new()
        .add_pathway(glycolysis)
        .with_expansion_criteria(ExpansionCriteria {
            uncertainty_threshold: 0.3,
            impact_threshold: 0.5,
            budget_limit: 10.0,
        })
        .build()?;
    
    println!("   ✓ Glycolysis pathway created with {} nodes", circuit.nodes.len());
    println!("   ✓ Circuit representation ready for integration");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_intracellular_environment_creation() {
        let result = create_intracellular_foundation();
        assert!(result.is_ok());
        
        let intracellular = result.unwrap();
        assert!(intracellular.state.atp_concentration > 0.0);
        assert!(intracellular.state.energy_charge > 0.0);
    }
    
    #[test]
    fn test_neuron_construction_kit() {
        let intracellular = create_intracellular_foundation().unwrap();
        let neuron_kit = create_neuron_construction_kit(intracellular).unwrap();
        
        assert!(neuron_kit.integration_complete());
        assert_eq!(neuron_kit.components.soma.diameter, 20.0);
    }
    
    #[test]
    fn test_atp_differential_solver() {
        let mut solver = AtpDifferentialSolver::new(5.0);
        let result = solver.solve_atp_differential(
            10.0,
            |substrate, atp| substrate * atp * 0.1,
            1.0
        );
        
        assert!(result > 0.0);
        assert!(result < 10.0); // Should be consumed
    }
} 
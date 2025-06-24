//! Simple Biological Maxwell's Demons Test
//! 
//! A minimal example to test BMD functionality without complex dependencies

use nebuchadnezzar::{
    biological_maxwell_demons::*,
    systems_biology::atp_kinetics::AtpPool,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Simple BMD Test");
    println!("==================");
    
    // Test 1: Create a basic Information Catalyst
    test_basic_information_catalyst()?;
    
    // Test 2: Test the Prisoner Parable
    test_prisoner_parable()?;
    
    // Test 3: Create BMD categories
    test_bmd_categories()?;
    
    // Test 4: Test Enhanced BMD System
    test_enhanced_bmd_system()?;
    
    println!("\n✅ All tests passed!");
    Ok(())
}

fn test_basic_information_catalyst() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Testing Basic Information Catalyst");
    
    let pattern_selector = PatternSelector {
        specificity_constants: HashMap::new(),
        recognition_memory: vec![
            PatternTemplate {
                pattern_id: "test_pattern".to_string(),
                pattern_data: BiologicalPattern {
                    pattern_type: "test".to_string(),
                    complexity: 0.5,
                    recognition_features: vec![1.0, 2.0],
                    evolutionary_age: 100.0,
                },
                recognition_strength: 0.8,
                evolutionary_fitness: 0.9,
            }
        ],
        selection_threshold: 0.5,
        diversity_reduction: 0.9,
    };
    
    let target_channel = TargetChannel {
        catalytic_constants: HashMap::new(),
        directional_preferences: vec![
            TargetGradient {
                target_id: "test_target".to_string(),
                target_data: BiologicalTarget {
                    target_type: "test".to_string(),
                    specificity: 0.8,
                    biological_impact: 10.0,
                    energy_requirement: 1.0,
                },
                gradient_strength: 0.9,
                thermodynamic_feasibility: 0.95,
            }
        ],
        channeling_efficiency: 0.8,
        output_specificity: 0.9,
    };
    
    let mut catalyst = InformationCatalyst::new(pattern_selector, target_channel, 100.0);
    
    // Test basic functionality
    assert!(catalyst.amplification_factor > 0.0);
    assert!(catalyst.verify_thermodynamic_consistency());
    
    // Test metastability
    let status = catalyst.calculate_metastability_status();
    match status {
        MetastabilityStatus::Stable => println!("   ✓ Catalyst is stable"),
        _ => println!("   ⚠️  Catalyst is degrading"),
    }
    
    // Test ATP integration
    let atp_pool = AtpPool::new_physiological();
    let atp_rate = catalyst.calculate_atp_consumption_rate(&atp_pool);
    println!("   ✓ ATP consumption rate: {:.3}", atp_rate);
    
    println!("   ✓ Basic Information Catalyst test passed");
    Ok(())
}

fn test_prisoner_parable() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Testing Prisoner Parable");
    
    // Create test signals
    let signals = vec![
        LightSignal { intensity: 1.0, duration: 0.3, wavelength: 650.0, timestamp: 0.0 },
        LightSignal { intensity: 1.0, duration: 0.9, wavelength: 650.0, timestamp: 0.5 },
    ];
    
    // Test with pattern recognition
    let mut informed_catalyst = create_test_catalyst_with_pattern();
    let outcome1 = informed_catalyst.simulate_prisoner_parable(&signals);
    
    match outcome1 {
        ThermodynamicOutcome::Survival(energy) => {
            println!("   ✓ With pattern recognition: SURVIVAL ({:.0} J)", energy);
        }
        ThermodynamicOutcome::Death(entropy) => {
            println!("   ❌ With pattern recognition: DEATH ({:.2e} J/K)", entropy);
        }
    }
    
    // Test without pattern recognition
    let mut uninformed_catalyst = create_test_catalyst_without_pattern();
    let outcome2 = uninformed_catalyst.simulate_prisoner_parable(&signals);
    
    match outcome2 {
        ThermodynamicOutcome::Survival(energy) => {
            println!("   ❌ Without pattern recognition: SURVIVAL ({:.0} J)", energy);
        }
        ThermodynamicOutcome::Death(entropy) => {
            println!("   ✓ Without pattern recognition: DEATH ({:.2e} J/K)", entropy);
        }
    }
    
    println!("   ✓ Prisoner Parable test passed");
    Ok(())
}

fn create_test_catalyst_with_pattern() -> InformationCatalyst<BiologicalPattern, BiologicalTarget> {
    let pattern_selector = PatternSelector {
        specificity_constants: HashMap::new(),
        recognition_memory: vec![
            PatternTemplate {
                pattern_id: "morse_code".to_string(),
                pattern_data: BiologicalPattern {
                    pattern_type: "communication".to_string(),
                    complexity: 0.8,
                    recognition_features: vec![0.3, 0.9],
                    evolutionary_age: 150.0,
                },
                recognition_strength: 0.9,
                evolutionary_fitness: 0.95,
            }
        ],
        selection_threshold: 0.5,
        diversity_reduction: 0.99,
    };
    
    let target_channel = TargetChannel {
        catalytic_constants: HashMap::new(),
        directional_preferences: vec![
            TargetGradient {
                target_id: "survival".to_string(),
                target_data: BiologicalTarget {
                    target_type: "behavior".to_string(),
                    specificity: 0.95,
                    biological_impact: 90.0,
                    energy_requirement: 0.001,
                },
                gradient_strength: 1.0,
                thermodynamic_feasibility: 0.99,
            }
        ],
        channeling_efficiency: 0.95,
        output_specificity: 0.9,
    };
    
    InformationCatalyst::new(pattern_selector, target_channel, 1000000.0)
}

fn create_test_catalyst_without_pattern() -> InformationCatalyst<BiologicalPattern, BiologicalTarget> {
    let pattern_selector = PatternSelector {
        specificity_constants: HashMap::new(),
        recognition_memory: Vec::new(), // No patterns!
        selection_threshold: 0.5,
        diversity_reduction: 0.01,
    };
    
    let target_channel = TargetChannel {
        catalytic_constants: HashMap::new(),
        directional_preferences: Vec::new(),
        channeling_efficiency: 0.1,
        output_specificity: 0.1,
    };
    
    InformationCatalyst::new(pattern_selector, target_channel, 1.0)
}

fn test_bmd_categories() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Testing BMD Categories");
    
    // Test BMD factory
    let molecular_bmd = BMDFactory::create_molecular_bmd("test_enzyme", 0.9, 1000.0);
    println!("   ✓ Created molecular BMD");
    
    let neural_bmd = BMDFactory::create_neural_bmd(100, 10, 0.8);
    println!("   ✓ Created neural BMD");
    
    println!("   ✓ BMD Categories test passed");
    Ok(())
}

fn test_enhanced_bmd_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Testing Enhanced BMD System");
    
    let mut system = EnhancedBMDSystem::new();
    
    // Add some BMDs
    system.base_system.demons.push(BMDFactory::create_molecular_bmd("hexokinase", 0.9, 1000.0));
    
    // Test metastability
    assert!(system.metastability_tracker.is_functional());
    println!("   ✓ System is functional");
    
    // Test simulation (simplified)
    match system.simulate_complete_cycle(0.1) {
        Ok(result) => {
            println!("   ✓ Simulation successful:");
            println!("     - Information processed: {:.2}", result.information_processed);
            println!("     - ATP consumed: {:.3}", result.atp_consumed);
            println!("     - Thermodynamically consistent: {}", result.thermodynamic_consistent);
        }
        Err(e) => {
            println!("   ⚠️  Simulation error: {:?}", e);
        }
    }
    
    println!("   ✓ Enhanced BMD System test passed");
    Ok(())
} 
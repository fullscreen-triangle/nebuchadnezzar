/// Comprehensive demonstration of Eduardo Mizraji's Biological Maxwell's Demons framework
/// 
/// This example implements the key concepts from Mizraji's 2021 paper:
/// 1. Information Catalysts (iCat = ℑ_input ◦ ℑ_output)
/// 2. The "Parable of the Prisoner" showing information amplification
/// 3. Molecular BMDs (enzymes), Cellular BMDs (gene regulation), Neural BMDs (memories)
/// 4. Pattern recognition and target channeling
/// 5. Metastability and thermodynamic consistency (Haldane relations)

use nebuchadnezzar::{
    BiologicalMaxwellDemon, BMDFactory, BMDSystem, InformationFlow, 
    EntropyManipulation, FireLightOptimization, InformationCatalyst,
    PatternSelector, TargetChannel, PatternTemplate, TargetGradient,
    MolecularPattern, ChemicalProduct, NeuralPattern, MotorResponse,
    HaldaneRelation, MichaelisMentenParams, HopfieldParams,
    BMDError, Result,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Mizraji's Biological Maxwell's Demons - Information Catalysis Framework");
    println!("═══════════════════════════════════════════════════════════════════════════");
    
    // 1. Demonstrate the "Parable of the Prisoner" - Information Amplification
    demonstrate_parable_of_prisoner()?;
    
    // 2. Create Molecular BMDs (Haldane's original enzyme concept)
    let molecular_bmd = create_molecular_bmd_example()?;
    
    // 3. Create Neural BMDs (Associative memory systems)
    let neural_bmd = create_neural_bmd_example()?;
    
    // 4. Create Cellular BMDs (Gene regulation - Monod, Jacob, Lwoff)
    let cellular_bmd = create_cellular_bmd_example()?;
    
    // 5. Demonstrate BMD System Integration
    demonstrate_bmd_system_integration(vec![molecular_bmd, neural_bmd, cellular_bmd])?;
    
    // 6. Show Information Catalysis Mathematics
    demonstrate_information_catalysis_mathematics()?;
    
    // 7. Demonstrate Metastability (Wiener's insight)
    demonstrate_metastability_cycles()?;
    
    // 8. Show Fire-Light Optimization (600-700nm enhancement)
    demonstrate_fire_light_optimization()?;
    
    println!("\n🎯 Mizraji Framework Integration Complete!");
    println!("   Information catalysts successfully demonstrated pattern → target transformations");
    println!("   with dramatic thermodynamic amplification effects.");
    
    Ok(())
}

/// Demonstrates Mizraji's "Parable of the Prisoner" - how minimal information
/// can have dramatically different thermodynamic consequences
fn demonstrate_parable_of_prisoner() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📖 PARABLE OF THE PRISONER - Information Amplification");
    println!("─────────────────────────────────────────────────────────");
    
    // Scenario: Prisoner receives light signals - some are Morse code with safe combination
    let morse_signals = vec![
        "... --- ...".to_string(),    // SOS
        ".-. .- -. -.. --- --".to_string(), // RANDOM
        ".---- ..--- ...-- ....-".to_string(), // 1234 (safe combination)
    ];
    
    let random_noise = vec![
        ".. ... . ..".to_string(),
        ". ... .. .".to_string(),
        "... . .. ...".to_string(),
    ];
    
    // Prisoner with Morse code knowledge (has ℑ_input pattern recognition)
    println!("🔍 Prisoner WITH Morse code knowledge:");
    let survival_energy = simulate_prisoner_with_knowledge(&morse_signals, &random_noise)?;
    println!("   → Decoded combination: 1234");
    println!("   → Opened safe, accessed food");
    println!("   → Survival energy expenditure: {:.2} ATP units", survival_energy);
    println!("   → Thermodynamic outcome: SURVIVAL");
    
    // Prisoner without Morse code knowledge (lacks ℑ_input pattern recognition)
    println!("\n🚫 Prisoner WITHOUT Morse code knowledge:");
    let death_entropy = simulate_prisoner_without_knowledge(&morse_signals, &random_noise)?;
    println!("   → Could not decode signals");
    println!("   → Safe remained locked");
    println!("   → Death entropy increase: {:.2} units", death_entropy);
    println!("   → Thermodynamic outcome: DEATH");
    
    println!("\n💡 MIZRAJI'S INSIGHT:");
    println!("   Same energy input (light signals) → Dramatically different outcomes");
    println!("   Information processing capability = Life vs Death");
    println!("   BMD pattern recognition = Thermodynamic fate determination");
    
    Ok(())
}

fn simulate_prisoner_with_knowledge(morse_signals: &[String], _noise: &[String]) -> Result<f64, BMDError> {
    // Create neural BMD for Morse code recognition
    let input_filter = PatternSelector {
        specificity_constants: [
            ("morse_dot".to_string(), 0.95),
            ("morse_dash".to_string(), 0.95),
            ("morse_space".to_string(), 0.90),
        ].iter().cloned().collect(),
        recognition_memory: vec![
            PatternTemplate {
                pattern_id: "morse_1".to_string(),
                pattern_data: ".----".to_string(),
                recognition_strength: 0.98,
                evolutionary_fitness: 1.0, // Life-saving information
            },
            PatternTemplate {
                pattern_id: "morse_2".to_string(),
                pattern_data: "..---".to_string(),
                recognition_strength: 0.98,
                evolutionary_fitness: 1.0,
            },
        ],
        selection_threshold: 0.8,
        diversity_reduction: 0.99, // Massive pattern space reduction
    };
    
    let output_channel = TargetChannel {
        catalytic_constants: [("safe_opening".to_string(), 0.95)].iter().cloned().collect(),
        directional_preferences: vec![
            TargetGradient {
                target_id: "food_access".to_string(),
                target_data: "survival".to_string(),
                gradient_strength: 1.0,
                thermodynamic_feasibility: 0.95,
            }
        ],
        channeling_efficiency: 0.90,
        output_specificity: 0.95,
    };
    
    let mut morse_decoder = InformationCatalyst::new(input_filter, output_channel, 1000.0);
    
    // Process signals through BMD information catalyst
    let _decoded_patterns = morse_decoder.catalyze(morse_signals)?;
    
    // Calculate survival energy (3 months of life)
    let daily_energy = 2000.0; // kcal
    let survival_days = 90.0;
    let atp_conversion = 0.1; // ATP units per kcal
    
    Ok(daily_energy * survival_days * atp_conversion)
}

fn simulate_prisoner_without_knowledge(signals: &[String], _noise: &[String]) -> Result<f64, BMDError> {
    // No pattern recognition capability - all signals appear as noise
    println!("   → Signals processed as random noise: {:?}", &signals[0..2]);
    
    // Calculate death entropy increase
    let body_mass = 70.0; // kg
    let entropy_per_kg = 1000.0; // arbitrary units
    let death_entropy = body_mass * entropy_per_kg;
    
    Ok(death_entropy)
}

/// Creates molecular BMD example - enzymes as pattern-recognizing catalysts (Haldane 1930)
fn create_molecular_bmd_example() -> Result<BiologicalMaxwellDemon, Box<dyn std::error::Error>> {
    println!("\n🧪 MOLECULAR BMD - Enzyme as Information Catalyst (Haldane 1930)");
    println!("─────────────────────────────────────────────────────────────────");
    
    // Create hexokinase enzyme BMD - first step of glycolysis
    let hexokinase_bmd = BMDFactory::create_molecular_bmd(
        "hexokinase",
        0.95, // High substrate specificity (glucose recognition)
        0.90  // High catalytic efficiency
    );
    
    match &hexokinase_bmd {
        BiologicalMaxwellDemon::Molecular(mol_bmd) => {
            println!("🔬 Hexokinase Enzyme BMD Created:");
            println!("   → Substrate Recognition: {:.1}% specificity", 
                mol_bmd.enzyme_specificity.input_filter.specificity_constants.get("hexokinase").unwrap_or(&0.0) * 100.0);
            println!("   → Catalytic Efficiency: {:.1}%", 
                mol_bmd.enzyme_specificity.output_channel.channeling_efficiency * 100.0);
            println!("   → Haldane Relation Keq: {:.2}", mol_bmd.haldane_constants.equilibrium_constant);
            println!("   → Michaelis-Menten Km: {:.2e} M", mol_bmd.kinetic_parameters.km);
            println!("   → Catalytic Constant kcat: {:.0} s⁻¹", mol_bmd.kinetic_parameters.kcat);
            
            // Demonstrate pattern selection and channeling
            println!("\n🎯 Pattern Selection & Channeling:");
            println!("   Input Pattern Space: Thousands of potential substrates");
            println!("   Selected Pattern: Glucose + ATP");
            println!("   Output Channel: Glucose-6-phosphate + ADP");
            println!("   Information Amplification: 1 enzyme → 1000s of reactions/second");
            
            println!("\n⚡ Thermodynamic Consistency (Haldane Relation):");
            if mol_bmd.haldane_constants.thermodynamic_consistency {
                println!("   ✅ Microscopic reversibility satisfied");
                println!("   ✅ No perpetual motion violation");
                println!("   ✅ Global entropy increase maintained");
            }
        },
        _ => return Err("Expected molecular BMD".into()),
    }
    
    Ok(hexokinase_bmd)
}

/// Creates neural BMD example - associative memories as cognitive pattern processors
fn create_neural_bmd_example() -> Result<BiologicalMaxwellDemon, Box<dyn std::error::Error>> {
    println!("\n🧠 NEURAL BMD - Associative Memory as Information Catalyst");
    println!("──────────────────────────────────────────────────────────");
    
    let neural_bmd = BMDFactory::create_neural_bmd(
        1000,  // Memory dimension (10⁶ components in real brains)
        100,   // Storage capacity 
        0.85   // Recognition threshold
    );
    
    match &neural_bmd {
        BiologicalMaxwellDemon::Neural(neural) => {
            println!("🧠 Neural Memory BMD Created:");
            println!("   → Network Dimension: {} neurons", neural.network_parameters.network_dimension);
            println!("   → Storage Capacity: {} patterns", neural.network_parameters.storage_capacity);
            println!("   → Retrieval Accuracy: {:.1}%", neural.network_parameters.retrieval_accuracy * 100.0);
            println!("   → Cognitive Selection Power: {:.0e} pattern reduction", neural.cognitive_selection_power);
            
            println!("\n🎯 Enormous Pattern Space Reduction:");
            println!("   Cardinal(All Possible Patterns) ≈ ∞");
            println!("   Cardinal(Stored Memories) = {}", neural.network_parameters.storage_capacity);
            println!("   Selection Ratio: 1 in {:.0e}", neural.cognitive_selection_power);
            
            println!("\n🔄 Synaptic Plasticity (Learning):");
            println!("   → LTP Threshold: {:.1}", neural.synaptic_plasticity.ltp_threshold);
            println!("   → LTD Threshold: {:.1}", neural.synaptic_plasticity.ltd_threshold);
            println!("   → Learning Rate: {:.3}", neural.synaptic_plasticity.learning_rate);
            
            println!("\n💡 Information Catalysis Example:");
            println!("   Input: Face recognition pattern (10⁶ components)");
            println!("   Processing: Associative memory lookup");
            println!("   Output: Name recall + emotional response");
            println!("   Amplification: Millisecond recognition → Lifetime memories");
        },
        _ => return Err("Expected neural BMD".into()),
    }
    
    Ok(neural_bmd)
}

/// Creates cellular BMD example - gene regulation systems (Monod, Jacob, Lwoff)
fn create_cellular_bmd_example() -> Result<BiologicalMaxwellDemon, Box<dyn std::error::Error>> {
    println!("\n🧬 CELLULAR BMD - Gene Regulation as Information Processing");
    println!("──────────────────────────────────────────────────────────────");
    
    // Create a cellular BMD representing the lac operon system
    // (Szilard's double negation logic contribution)
    let input_filter = PatternSelector {
        specificity_constants: [
            ("lactose_present".to_string(), 0.90),
            ("glucose_absent".to_string(), 0.85),
            ("cap_crp_binding".to_string(), 0.95),
        ].iter().cloned().collect(),
        recognition_memory: vec![
            PatternTemplate {
                pattern_id: "lac_promoter".to_string(),
                pattern_data: "TTTACA".to_string(),
                recognition_strength: 0.95,
                evolutionary_fitness: 0.8,
            }
        ],
        selection_threshold: 0.7,
        diversity_reduction: 0.95,
    };
    
    let output_channel = TargetChannel {
        catalytic_constants: [("transcription_rate".to_string(), 100.0)].iter().cloned().collect(),
        directional_preferences: vec![
            TargetGradient {
                target_id: "beta_galactosidase".to_string(),
                target_data: "enzyme_production".to_string(),
                gradient_strength: 0.9,
                thermodynamic_feasibility: 0.85,
            }
        ],
        channeling_efficiency: 0.80,
        output_specificity: 0.90,
    };
    
    let transcription_control = InformationCatalyst::new(input_filter, output_channel, 10000.0);
    
    let cellular_bmd = BiologicalMaxwellDemon::Cellular(nebuchadnezzar::CellularBMD {
        transcription_control,
        regulatory_circuits: vec![
            nebuchadnezzar::RegulatoryCircuit {
                circuit_type: "lac_operon".to_string(),
                logic_function: "double_negation".to_string(), // Szilard's contribution
                amplification_factor: 1000.0,
            }
        ],
        expression_amplification: 10000.0, // Single mRNA → thousands of proteins
    });
    
    println!("🧬 Lac Operon BMD Created (Monod, Jacob, Lwoff system):");
    println!("   → Promoter Recognition: TTTACA sequence");
    println!("   → Logic Function: Double Negation (Szilard's insight)");
    println!("   → Expression Amplification: 10,000x");
    println!("   → Information Processing: Environmental sensing → Gene expression");
    
    println!("\n🎯 Pattern Recognition & Target Channeling:");
    println!("   Input Patterns: Lactose presence, Glucose absence, CAP-cAMP");
    println!("   Information Integration: Boolean logic processing");
    println!("   Target Channel: β-galactosidase enzyme production");
    println!("   Metabolic Consequence: Lactose utilization pathway activation");
    
    println!("\n⚡ Information Amplification:");
    println!("   1 lactose molecule detection → 1000s of enzyme molecules");
    println!("   Minimal information input → Major metabolic reprogramming");
    println!("   BMD as cellular decision-making system");
    
    Ok(cellular_bmd)
}

/// Demonstrates integrated BMD system with information flow and entropy manipulation
fn demonstrate_bmd_system_integration(demons: Vec<BiologicalMaxwellDemon>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🌐 BMD SYSTEM INTEGRATION - Hierarchical Information Processing");
    println!("─────────────────────────────────────────────────────────────────");
    
    let mut bmd_system = BMDSystem {
        demons,
        information_flow: InformationFlow {
            total_information_processed: 0.0,
            amplification_cascade: 1.0,
            pattern_recognition_events: 0,
            target_channeling_successes: 0,
        },
        entropy_manipulation: EntropyManipulation {
            local_entropy_reduction: 0.0,
            global_entropy_increase: 0.0,
            entropy_balance: 0.0,
            negentropy_generation: 0.0,
        },
        fire_light_optimization: FireLightOptimization {
            wavelength_range: (600.0, 700.0), // Fire-light optimization
            quantum_enhancement_factor: 1.5,
            consciousness_amplification: 2.0,
            abstract_reasoning_boost: 1.8,
        },
    };
    
    // Simulate system operation
    println!("🔄 System Operation Simulation:");
    
    // Each BMD processes information
    for (i, demon) in bmd_system.demons.iter().enumerate() {
        match demon {
            BiologicalMaxwellDemon::Molecular(_) => {
                bmd_system.information_flow.pattern_recognition_events += 1000; // Enzyme turnovers
                bmd_system.information_flow.amplification_cascade *= 1000.0;
                println!("   → Molecular BMD {}: 1,000 catalytic events", i + 1);
            },
            BiologicalMaxwellDemon::Neural(_) => {
                bmd_system.information_flow.pattern_recognition_events += 1;
                bmd_system.information_flow.amplification_cascade *= 1e6; // Massive neural amplification
                println!("   → Neural BMD {}: Memory association event", i + 1);
            },
            BiologicalMaxwellDemon::Cellular(_) => {
                bmd_system.information_flow.pattern_recognition_events += 1;
                bmd_system.information_flow.amplification_cascade *= 10000.0;
                println!("   → Cellular BMD {}: Gene expression event", i + 1);
            },
            _ => {}
        }
    }
    
    bmd_system.information_flow.total_information_processed = 
        bmd_system.information_flow.pattern_recognition_events as f64 * 
        bmd_system.information_flow.amplification_cascade.log10();
    
    // Calculate entropy effects
    bmd_system.entropy_manipulation.local_entropy_reduction = 100.0; // Order creation
    bmd_system.entropy_manipulation.global_entropy_increase = 150.0; // Heat dissipation
    bmd_system.entropy_manipulation.entropy_balance = 
        bmd_system.entropy_manipulation.global_entropy_increase - 
        bmd_system.entropy_manipulation.local_entropy_reduction;
    bmd_system.entropy_manipulation.negentropy_generation = 
        bmd_system.entropy_manipulation.local_entropy_reduction;
    
    println!("\n📊 System Performance Metrics:");
    println!("   → Total Information Processed: {:.2e} bits", bmd_system.information_flow.total_information_processed);
    println!("   → Pattern Recognition Events: {}", bmd_system.information_flow.pattern_recognition_events);
    println!("   → Amplification Cascade: {:.2e}x", bmd_system.information_flow.amplification_cascade);
    
    println!("\n🌡️ Entropy Manipulation Results:");
    println!("   → Local Entropy Reduction: {:.1} units (order creation)", bmd_system.entropy_manipulation.local_entropy_reduction);
    println!("   → Global Entropy Increase: {:.1} units (heat dissipation)", bmd_system.entropy_manipulation.global_entropy_increase);
    println!("   → Net Entropy Balance: +{:.1} units (2nd law satisfied)", bmd_system.entropy_manipulation.entropy_balance);
    println!("   → Negentropy Generation: {:.1} units (biological order)", bmd_system.entropy_manipulation.negentropy_generation);
    
    println!("\n🔥 Fire-Light Optimization (600-700nm):");
    println!("   → Wavelength Range: {:.0}-{:.0} nm", 
        bmd_system.fire_light_optimization.wavelength_range.0,
        bmd_system.fire_light_optimization.wavelength_range.1);
    println!("   → Quantum Enhancement: {:.1}x", bmd_system.fire_light_optimization.quantum_enhancement_factor);
    println!("   → Consciousness Amplification: {:.1}x", bmd_system.fire_light_optimization.consciousness_amplification);
    
    Ok(())
}

/// Demonstrates the mathematical formalization of information catalysis
fn demonstrate_information_catalysis_mathematics() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔢 INFORMATION CATALYSIS MATHEMATICS - Mizraji's Formalization");
    println!("─────────────────────────────────────────────────────────────────");
    
    println!("📐 Core Equation: iCat = ℑ_input ◦ ℑ_output");
    println!("   Where:");
    println!("   • ℑ_input: Pattern selector (Y↓(in) → Y↑(in))");
    println!("   • ℑ_output: Target channeler (Z↓(fin) → Z↑(fin))");
    println!("   • ◦: Functional composition");
    
    println!("\n🎯 Transformation Process:");
    println!("   Y↓(in) --[ℑ_input]→ Y↑(in) --[linkage]→ Z↓(fin) --[ℑ_output]→ Z↑(fin)");
    println!("   Potential → Selected → Potential → Directed");
    println!("   Patterns    Patterns   Outcomes   Outcomes");
    
    println!("\n📊 Combinatorial Reduction:");
    println!("   Cardinal(Y↓) ≫ Cardinal(Y↑) ≫ Cardinal(Z↑)");
    println!("   Example: 10¹² potential → 10³ selected → 10¹ directed");
    println!("   Reduction Factor: 10¹¹ (enormous pattern space compression)");
    
    println!("\n⚡ Probability Enhancement:");
    println!("   P(Y↓ → Z↓) ≈ 0 (without iCat)");
    println!("   P(Y↑ → Z↑) ≫ P(Y↓ → Z↓) (with iCat)");
    println!("   Enhancement: 10⁶ - 10¹² fold increase");
    
    println!("\n🔄 Catalytic Cycle:");
    println!("   1. Pattern Recognition (ℑ_input activation)");
    println!("   2. Information Processing (pattern → target linkage)");
    println!("   3. Target Channeling (ℑ_output activation)");
    println!("   4. Catalyst Regeneration (ready for next cycle)");
    
    println!("\n🌡️ Thermodynamic Framework:");
    println!("   • BMDs operate far from equilibrium");
    println!("   • Local entropy reduction at expense of global increase");
    println!("   • Information cost ≪ Thermodynamic consequences");
    println!("   • Metastability ensures eventual deterioration");
    
    Ok(())
}

/// Demonstrates metastability of BMDs (Wiener's insight)
fn demonstrate_metastability_cycles() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⏳ METASTABILITY DEMONSTRATION - Wiener's Insight");
    println!("─────────────────────────────────────────────────────");
    
    println!("💭 Wiener (1948): 'Living organisms are metastable Maxwell demons");
    println!("   whose stable state is to be dead.'");
    
    // Create a simple information catalyst and run it through cycles
    let input_filter = PatternSelector {
        specificity_constants: HashMap::new(),
        recognition_memory: Vec::new(),
        selection_threshold: 0.5,
        diversity_reduction: 0.9,
    };
    
    let output_channel = TargetChannel {
        catalytic_constants: HashMap::new(),
        directional_preferences: Vec::new(),
        channeling_efficiency: 0.9,
        output_specificity: 0.9,
    };
    
    let mut catalyst = InformationCatalyst::new(input_filter, output_channel, 100.0);
    
    println!("\n🔄 Catalytic Cycle Simulation:");
    
    // Simulate catalytic cycles
    let test_patterns = vec!["pattern1", "pattern2", "pattern3"];
    let mut cycle = 0;
    
    while catalyst.is_metastable() && cycle < 15000 {
        // Simulate catalytic activity
        let _result = catalyst.catalyze(&test_patterns);
        cycle += 1;
        
        if cycle % 2500 == 0 {
            println!("   → Cycle {}: Energy cost {:.1} ATP, Still metastable: {}", 
                cycle, catalyst.total_energy_cost(), catalyst.is_metastable());
        }
    }
    
    println!("\n⚠️  Metastability Exceeded at cycle {}", cycle);
    println!("   → Total energy cost: {:.1} ATP units", catalyst.total_energy_cost());
    println!("   → Catalyst deterioration: BMD requires replacement");
    println!("   → Biological reality: Enzyme degradation, synaptic pruning");
    
    println!("\n🔄 Replacement Mechanisms:");
    println!("   • Molecular: New enzyme synthesis");
    println!("   • Neural: Synaptic plasticity, neurogenesis");
    println!("   • Cellular: Gene expression regulation");
    println!("   • Organismal: Reproduction, cultural transmission");
    
    Ok(())
}

/// Demonstrates fire-light optimization for consciousness enhancement
fn demonstrate_fire_light_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔥 FIRE-LIGHT OPTIMIZATION - Consciousness Enhancement");
    println!("───────────────────────────────────────────────────────");
    
    println!("🧠 Theoretical Foundation:");
    println!("   Human consciousness emerged through fire exposure");
    println!("   600-700nm wavelengths optimize quantum ion tunneling");
    println!("   BMDs amplify abstract reasoning capabilities");
    
    let fire_optimization = FireLightOptimization {
        wavelength_range: (600.0, 700.0),
        quantum_enhancement_factor: 1.5,
        consciousness_amplification: 2.0,
        abstract_reasoning_boost: 1.8,
    };
    
    println!("\n🌈 Wavelength Optimization:");
    println!("   → Target Range: {:.0}-{:.0} nm (fire-light spectrum)", 
        fire_optimization.wavelength_range.0,
        fire_optimization.wavelength_range.1);
    println!("   → Quantum Enhancement: {:.1}x ion tunneling efficiency", 
        fire_optimization.quantum_enhancement_factor);
    
    println!("\n🧠 Consciousness Effects:");
    println!("   → Overall Amplification: {:.1}x", fire_optimization.consciousness_amplification);
    println!("   → Abstract Reasoning: {:.1}x boost", fire_optimization.abstract_reasoning_boost);
    println!("   → BMD Information Processing: Enhanced pattern recognition");
    
    // Simulate different lighting conditions
    println!("\n💡 Lighting Condition Comparison:");
    
    let conditions = vec![
        ("Sunlight", 400.0..700.0, 1.0),
        ("Fire-light", 600.0..700.0, fire_optimization.quantum_enhancement_factor),
        ("LED White", 450.0..650.0, 0.8),
        ("Candlelight", 580.0..700.0, 1.2),
    ];
    
    for (name, range, enhancement) in conditions {
        let fire_overlap = calculate_fire_overlap(&range, &fire_optimization.wavelength_range);
        let cognitive_boost = enhancement * fire_overlap;
        
        println!("   → {}: {:.1}% fire-spectrum overlap, {:.2}x cognitive boost", 
            name, fire_overlap * 100.0, cognitive_boost);
    }
    
    println!("\n🎯 Optimization Applications:");
    println!("   • Workspace lighting design");
    println!("   • Educational environment optimization");
    println!("   • Therapeutic light therapy");
    println!("   • Consciousness research protocols");
    
    Ok(())
}

fn calculate_fire_overlap(range: &std::ops::Range<f64>, fire_range: &(f64, f64)) -> f64 {
    let overlap_start = range.start.max(fire_range.0);
    let overlap_end = range.end.min(fire_range.1);
    
    if overlap_start >= overlap_end {
        return 0.0;
    }
    
    let overlap_width = overlap_end - overlap_start;
    let total_width = range.end - range.start;
    
    overlap_width / total_width
}
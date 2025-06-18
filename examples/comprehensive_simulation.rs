//! # Comprehensive Cellular Simulation Example
//! 
//! This example demonstrates the full capabilities of the Nebuchadnezzar circuit library
//! by simulating a complete cellular system with hierarchical circuits, ATP-driven
//! differential equations, and dynamic circuit resolution.

use nebuchadnezzar::{
    NebuchadnezzarEngine, 
    MetabolicPathwayType,
    SimulationParameters,
    circuits::{
        CircuitFactory,
        AdaptiveGrid,
        HierarchicalSystem,
        hierarchical_framework::*,
        enzyme_circuits::*,
        circuit_grid::*,
    },
    systems_biology::AtpPool,
    solvers::{SystemState, IntegratorFactory},
    Result,
};

fn main() -> Result<()> {
    println!("🧬 Nebuchadnezzar: Comprehensive Cellular Simulation");
    println!("================================================");

    // Create the main simulation engine
    let mut engine = create_cellular_system()?;
    
    // Run the simulation
    println!("\n🚀 Starting simulation...");
    let results = engine.run_simulation()?;
    
    // Analyze and display results
    analyze_results(&results);
    
    // Demonstrate circuit resolution
    demonstrate_circuit_resolution()?;
    
    // Show hierarchical voltage dynamics
    demonstrate_voltage_hierarchy()?;
    
    // Demonstrate enzyme logic gates
    demonstrate_enzyme_logic_gates()?;

    println!("\n✅ Simulation completed successfully!");
    Ok(())
}

/// Create a complete cellular system with multiple circuits and pathways
fn create_cellular_system() -> Result<NebuchadnezzarEngine> {
    println!("\n🏗️  Building cellular system...");
    
    let mut engine = NebuchadnezzarEngine::new(20.0); // Start with 20 mM ATP
    
    // Configure simulation parameters
    engine.simulation_parameters = SimulationParameters {
        time_step: 0.001,
        max_time: 5.0,
        atp_threshold: 0.1,
        voltage_tolerance: 0.001,
        adaptive_stepping: true,
        resolution_threshold: 0.8,
        output_frequency: 50,
    };

    // Add neuronal circuits
    println!("  Adding neuronal circuits...");
    engine.add_neuron_circuit("pyramidal_neuron".to_string())?;
    engine.add_neuron_circuit("interneuron".to_string())?;
    
    // Add metabolic pathways
    println!("  Adding metabolic pathways...");
    engine.add_metabolic_pathway("glycolysis".to_string(), MetabolicPathwayType::Glycolysis)?;
    engine.add_metabolic_pathway("tca_cycle".to_string(), MetabolicPathwayType::CitricAcidCycle)?;
    engine.add_metabolic_pathway("electron_transport".to_string(), MetabolicPathwayType::ElectronTransport)?;
    
    // Add custom circuits
    add_custom_circuits(&mut engine)?;
    
    println!("  System construction complete!");
    println!("    - {} molecular circuits", engine.hierarchical_system.molecular_level.len());
    println!("    - {} organelle networks", engine.hierarchical_system.organelle_level.len());
    println!("    - Initial ATP: {:.2} mM", engine.global_atp_pool.atp_concentration);
    
    Ok(engine)
}

/// Add custom circuits to demonstrate specific functionality
fn add_custom_circuits(engine: &mut NebuchadnezzarEngine) -> Result<()> {
    println!("  Adding custom circuits...");
    
    // Create a calcium signaling circuit
    let calcium_circuit = create_calcium_signaling_circuit()?;
    engine.hierarchical_system.molecular_level.push(calcium_circuit);
    
    // Create a stress response circuit
    let stress_circuit = create_stress_response_circuit()?;
    engine.hierarchical_system.molecular_level.push(stress_circuit);
    
    Ok(())
}

/// Create a calcium signaling molecular circuit
fn create_calcium_signaling_circuit() -> Result<MolecularCircuit> {
    let mut circuit = MolecularCircuit {
        circuit_id: "calcium_signaling".to_string(),
        protein_complexes: vec![
            ProteinComplex {
                complex_id: "ip3_receptor".to_string(),
                subunits: vec!["ip3r1".to_string(), "ip3r2".to_string()],
                binding_energy: -12.5,
                conformational_states: vec![
                    ProteinState {
                        conformation_energy: 0.0,
                        binding_affinity: 0.1,
                        activity_level: 0.0,
                        last_transition_time: 0.0,
                    },
                    ProteinState {
                        conformation_energy: -8.0,
                        binding_affinity: 0.9,
                        activity_level: 1.0,
                        last_transition_time: 0.0,
                    },
                ],
                current_state_index: 0,
            }
        ],
        enzyme_circuits: vec![
            Box::new(ProbabilisticOR::new(
                "ip3".to_string(),
                "calcium_release".to_string(),
                0.9,
                0.8,
            )),
        ],
        ion_channels: vec![
            ProbabilisticIonChannel::new(
                "calcium_channel".to_string(),
                IonType::Calcium,
                2.0,
                0.1,
                ChannelKinetics::ligand_gated(0.001, 0.1),
                0.02,
            ),
        ],
        molecular_voltage: -60.0,
        atp_local: 2.0,
        temporal_state: TemporalEvidence::new(),
    };
    
    // Add temporal events for calcium signaling
    circuit.temporal_state.add_event(TemporalEvent {
        event_id: "calcium_spike".to_string(),
        event_type: EventType::IonChannelGating,
        strength: 1.0,
        decay_rate: 2.0,
        timestamp: 0.0,
        half_life: 0.5,
    });
    
    Ok(circuit)
}

/// Create a stress response molecular circuit
fn create_stress_response_circuit() -> Result<MolecularCircuit> {
    Ok(MolecularCircuit {
        circuit_id: "stress_response".to_string(),
        protein_complexes: vec![],
        enzyme_circuits: vec![
            Box::new(ProbabilisticXOR::new(
                "stress_signal".to_string(),
                "hsp70".to_string(),
                "NAD+".to_string(),
                "NADH".to_string(),
                0.85,
                0.7,
            )),
            Box::new(ProbabilisticAND::new(
                "misfolded_protein".to_string(),
                "ubiquitin".to_string(),
                "proteasome_target".to_string(),
                0.8,
                0.9,
            )),
        ],
        ion_channels: vec![],
        molecular_voltage: -70.0,
        atp_local: 1.5,
        temporal_state: TemporalEvidence::new(),
    })
}

/// Analyze and display simulation results
fn analyze_results(results: &nebuchadnezzar::SimulationResults) {
    println!("\n📊 Simulation Results Analysis");
    println!("=============================");
    
    let summary = results.get_summary();
    
    println!("Simulation Summary:");
    println!("  Total time: {:.3} seconds", summary.total_time);
    println!("  Total steps: {}", summary.total_steps);
    println!("  Average ATP: {:.3} mM", summary.average_atp);
    println!("  Min ATP: {:.3} mM", summary.min_atp);
    println!("  Max ATP: {:.3} mM", summary.max_atp);
    println!("  Final ATP: {:.3} mM", summary.final_atp);
    println!("  Average Energy Charge: {:.3}", summary.average_energy_charge);
    println!("  ATP Depletion Rate: {:.3} mM/s", summary.atp_depletion_rate);
    
    // Analyze voltage dynamics
    if !results.hierarchical_states.is_empty() {
        let final_state = &results.hierarchical_states.last().unwrap();
        println!("\nFinal System State:");
        println!("  Cellular voltage: {:.1} mV", final_state.voltage_hierarchy.cellular_voltage);
        
        for (organelle, voltage) in &final_state.voltage_hierarchy.organelle_voltages {
            println!("  {} voltage: {:.1} mV", organelle, voltage);
        }
        
        println!("  Active molecular circuits: {}", final_state.molecular_states.len());
        println!("  Active organelle networks: {}", final_state.organelle_states.len());
        println!("  Active cellular systems: {}", final_state.cellular_states.len());
    }
}

/// Demonstrate dynamic circuit resolution
fn demonstrate_circuit_resolution() -> Result<()> {
    println!("\n🔄 Demonstrating Dynamic Circuit Resolution");
    println!("==========================================");
    
    let mut adaptive_grid = AdaptiveGrid::new("resolution_demo".to_string(), 15.0);
    
    // Add some probabilistic nodes
    let nodes = vec![
        ("high_activity_node", 0.9, 0.95),
        ("medium_activity_node", 0.6, 0.7),
        ("low_activity_node", 0.3, 0.4),
    ];
    
    for (name, probability, importance) in nodes {
        let node = ProbabilisticNode {
            node_id: name.to_string(),
            node_type: NodeType::EnzymaticReaction {
                enzyme_class: "kinase".to_string(),
                substrate_binding_prob: 0.8,
                product_formation_prob: 0.9,
            },
            probability,
            atp_cost: 1.0,
            inputs: vec!["substrate".to_string()],
            outputs: vec!["product".to_string()],
            resolution_importance: importance,
            last_activity: probability,
        };
        
        adaptive_grid.base_grid.add_probabilistic_node(node);
    }
    
    println!("Initial state:");
    println!("  Probabilistic nodes: {}", adaptive_grid.base_grid.probabilistic_nodes.len());
    println!("  Resolved circuits: {}", adaptive_grid.base_grid.resolved_circuits.len());
    
    // Perform optimization steps
    for step in 0..3 {
        let optimization_result = adaptive_grid.optimize_step()?;
        println!("\nOptimization step {}:", step + 1);
        println!("  Nodes resolved: {:?}", optimization_result.nodes_resolved);
        println!("  Performance improvement: {:.3}", optimization_result.performance_improvement);
        println!("  Computational cost: {:.1}", optimization_result.computational_cost);
        println!("  Remaining probabilistic nodes: {}", adaptive_grid.base_grid.probabilistic_nodes.len());
        println!("  Total resolved circuits: {}", adaptive_grid.base_grid.resolved_circuits.len());
    }
    
    Ok(())
}

/// Demonstrate voltage hierarchy across scales
fn demonstrate_voltage_hierarchy() -> Result<()> {
    println!("\n⚡ Demonstrating Multi-Level Voltage Hierarchy");
    println!("============================================");
    
    let mut hierarchical_system = HierarchicalSystem::new();
    
    // Add molecular circuits with different voltages
    let molecular_voltages = vec![
        ("sodium_channel_cluster", -55.0),
        ("potassium_channel_cluster", -80.0),
        ("calcium_channel_cluster", -20.0),
    ];
    
    for (name, voltage) in molecular_voltages {
        let circuit = MolecularCircuit {
            circuit_id: name.to_string(),
            protein_complexes: vec![],
            enzyme_circuits: vec![],
            ion_channels: vec![],
            molecular_voltage: voltage,
            atp_local: 1.0,
            temporal_state: TemporalEvidence::new(),
        };
        hierarchical_system.molecular_level.push(circuit);
    }
    
    // Add organelle networks
    let organelle_types = vec![
        ("mitochondrion", -180.0),
        ("endoplasmic_reticulum", -60.0),
        ("nucleus", -10.0),
    ];
    
    for (name, voltage) in organelle_types {
        let organelle = OrganelleNetwork {
            organelle_id: name.to_string(),
            organelle_type: match name {
                "mitochondrion" => OrganelleType::Mitochondrion { cristae_density: 1.5 },
                "endoplasmic_reticulum" => OrganelleType::EndoplasmicReticulum { membrane_area: 100.0 },
                "nucleus" => OrganelleType::Nucleus { chromatin_state: "euchromatin".to_string() },
                _ => OrganelleType::Mitochondrion { cristae_density: 1.0 },
            },
            molecular_circuits: vec![],
            inter_circuit_connections: vec![],
            organelle_voltage: voltage,
            local_atp_pool: 3.0,
            metabolic_state: MetabolicState {
                energy_charge: 0.8,
                nadh_nad_ratio: 0.1,
                calcium_level: 0.0001,
                ph: 7.4,
                oxygen_level: 0.2,
            },
        };
        hierarchical_system.organelle_level.push(organelle);
    }
    
    println!("Voltage Hierarchy:");
    println!("  Cellular level: {:.1} mV", hierarchical_system.voltage_hierarchy.cellular_voltage);
    
    for (organelle, voltage) in &hierarchical_system.voltage_hierarchy.organelle_voltages {
        println!("  {} level: {:.1} mV", organelle, voltage);
    }
    
    for circuit in &hierarchical_system.molecular_level {
        println!("  Molecular ({}): {:.1} mV", circuit.circuit_id, circuit.molecular_voltage);
    }
    
    // Simulate voltage propagation
    let result = hierarchical_system.solve_hierarchical_system(2.0)?;
    
    println!("\nAfter ATP consumption (2.0 mM):");
    println!("  Updated cellular voltage: {:.1} mV", result.voltage_hierarchy.cellular_voltage);
    
    Ok(())
}

/// Demonstrate enzyme probabilistic logic gates
fn demonstrate_enzyme_logic_gates() -> Result<()> {
    println!("\n🧪 Demonstrating Enzyme Probabilistic Logic Gates");
    println!("===============================================");
    
    let atp_concentration = 5.0;
    
    // Demonstrate each type of enzyme logic gate
    println!("Testing enzyme circuits with ATP = {:.1} mM:\n", atp_concentration);
    
    // NOT gate (Isomerase)
    let isomerase = ProbabilisticNOT::glucose_phosphate_isomerase();
    println!("🔄 ProbabilisticNOT (Isomerase):");
    println!("  Enzyme: Glucose-6-phosphate isomerase");
    println!("  Can fire: {}", isomerase.can_fire(atp_concentration));
    println!("  Success probability: {:.3}", isomerase.success_probability(atp_concentration));
    println!("  ATP cost: {:.3} mM", isomerase.atp_cost());
    
    let flux = isomerase.compute_flux(atp_concentration)?;
    for (metabolite, rate) in &flux {
        println!("  Flux {}: {:.3}", metabolite, rate);
    }
    
    // SPLIT gate (Dismutase)
    let dismutase = ProbabilisticSPLIT::superoxide_dismutase();
    println!("\n🌪️  ProbabilisticSPLIT (Dismutase):");
    println!("  Enzyme: Superoxide dismutase");
    println!("  Can fire: {}", dismutase.can_fire(atp_concentration));
    println!("  Success probability: {:.3}", dismutase.success_probability(atp_concentration));
    println!("  ATP cost: {:.3} mM", dismutase.atp_cost());
    
    // XOR gate (Dehydrogenase)
    let dehydrogenase = ProbabilisticXOR::lactate_dehydrogenase();
    println!("\n🔀 ProbabilisticXOR (Dehydrogenase):");
    println!("  Enzyme: Lactate dehydrogenase");
    println!("  Can fire: {}", dehydrogenase.can_fire(atp_concentration));
    println!("  Success probability: {:.3}", dehydrogenase.success_probability(atp_concentration));
    println!("  ATP cost: {:.3} mM", dehydrogenase.atp_cost());
    
    // OR gate (Kinase)
    let kinase = ProbabilisticOR::hexokinase();
    println!("\n⚡ ProbabilisticOR (Kinase):");
    println!("  Enzyme: Hexokinase");
    println!("  Can fire: {}", kinase.can_fire(atp_concentration));
    println!("  Success probability: {:.3}", kinase.success_probability(atp_concentration));
    println!("  ATP cost: {:.3} mM", kinase.atp_cost());
    
    // AND gate (Ligase)
    let ligase = ProbabilisticAND::dna_ligase();
    println!("\n🔗 ProbabilisticAND (Ligase):");
    println!("  Enzyme: DNA ligase");
    println!("  Can fire: {}", ligase.can_fire(atp_concentration));
    println!("  Success probability: {:.3}", ligase.success_probability(atp_concentration));
    println!("  ATP cost: {:.3} mM", ligase.atp_cost());
    
    // Test circuit grid with multiple enzymes
    println!("\n🏭 Testing Glycolysis Enzyme Circuit:");
    let glycolysis_enzymes = EnzymeCircuitFactory::create_glycolysis_enzymes();
    println!("  Total enzymes in pathway: {}", glycolysis_enzymes.len());
    
    let mut total_atp_cost = 0.0;
    for (i, enzyme) in glycolysis_enzymes.iter().enumerate() {
        let can_fire = enzyme.can_fire(atp_concentration);
        let atp_cost = enzyme.atp_cost();
        total_atp_cost += atp_cost;
        
        println!("  Step {}: Can fire = {}, ATP cost = {:.3} mM", i + 1, can_fire, atp_cost);
    }
    
    println!("  Total pathway ATP cost: {:.3} mM", total_atp_cost);
    
    Ok(())
} 
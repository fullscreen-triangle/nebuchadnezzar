//! # Fractal-Holographic Cellular Circuits
//! 
//! This module implements self-similar fractal circuits and holographic information
//! encoding in cellular systems, extending the hierarchical paradigm to infinite scales.

use crate::error::{NebuchadnezzarError, Result};
use crate::circuits::Circuit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fractal dimension for biological networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalDimension {
    pub hausdorff_dimension: f64,
    pub box_counting_dimension: f64,
    pub correlation_dimension: f64,
    pub information_dimension: f64,
}

/// Self-similar circuit that repeats at all scales
#[derive(Debug, Clone)]
pub struct FractalCircuit {
    /// Base circuit pattern
    pub base_pattern: CircuitPattern,
    
    /// Scaling factor between levels
    pub scaling_factor: f64,
    
    /// Maximum recursion depth
    pub max_depth: usize,
    
    /// Current fractal dimension
    pub fractal_dimension: FractalDimension,
    
    /// Self-similarity ratio
    pub self_similarity: f64,
    
    /// Fractal generators at each scale
    pub generators: Vec<FractalGenerator>,
}

#[derive(Debug, Clone)]
pub struct CircuitPattern {
    pub nodes: Vec<FractalNode>,
    pub connections: Vec<FractalConnection>,
    pub symmetry_group: SymmetryGroup,
}

#[derive(Debug, Clone)]
pub struct FractalNode {
    pub position: (f64, f64, f64), // 3D position
    pub scale: f64,
    pub circuit_type: String,
    pub atp_density: f64,
    pub information_content: f64, // Bits per node
}

#[derive(Debug, Clone)]
pub struct FractalConnection {
    pub from_node: usize,
    pub to_node: usize,
    pub conductance: f64,
    pub fractal_weight: f64,
    pub information_flow: f64, // Bits per second
}

#[derive(Debug, Clone)]
pub struct SymmetryGroup {
    pub rotations: Vec<f64>,      // Rotation angles
    pub reflections: Vec<bool>,   // Reflection axes
    pub translations: Vec<(f64, f64, f64)>, // Translation vectors
}

#[derive(Debug, Clone)]
pub struct FractalGenerator {
    pub scale: f64,
    pub pattern_rules: Vec<GeneratorRule>,
    pub emergence_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct GeneratorRule {
    pub input_pattern: String,
    pub output_pattern: String,
    pub transformation_matrix: Vec<Vec<f64>>,
    pub probability: f64,
}

impl FractalCircuit {
    pub fn new(base_pattern: CircuitPattern, scaling_factor: f64) -> Self {
        Self {
            base_pattern,
            scaling_factor,
            max_depth: 10,
            fractal_dimension: FractalDimension {
                hausdorff_dimension: 2.5,
                box_counting_dimension: 2.3,
                correlation_dimension: 2.1,
                information_dimension: 1.9,
            },
            self_similarity: 0.618, // Golden ratio
            generators: Vec::new(),
        }
    }

    /// Generate fractal circuit at specified depth
    pub fn generate_fractal(&mut self, depth: usize) -> Result<CircuitPattern> {
        if depth == 0 {
            return Ok(self.base_pattern.clone());
        }
        
        if depth > self.max_depth {
            return Err(NebuchadnezzarError::ComputationError(
                "Maximum fractal depth exceeded".to_string()
            ));
        }

        let parent_pattern = self.generate_fractal(depth - 1)?;
        let mut new_pattern = CircuitPattern {
            nodes: Vec::new(),
            connections: Vec::new(),
            symmetry_group: parent_pattern.symmetry_group.clone(),
        };

        // Apply fractal transformation to each node
        for (i, parent_node) in parent_pattern.nodes.iter().enumerate() {
            // Create child nodes around each parent node
            let child_nodes = self.create_child_nodes(parent_node, depth)?;
            new_pattern.nodes.extend(child_nodes);

            // Create fractal connections
            let child_connections = self.create_child_connections(i, &parent_pattern, depth)?;
            new_pattern.connections.extend(child_connections);
        }

        // Update fractal dimension based on scaling
        self.update_fractal_dimension(&new_pattern)?;

        Ok(new_pattern)
    }

    fn create_child_nodes(&self, parent: &FractalNode, depth: usize) -> Result<Vec<FractalNode>> {
        let mut children = Vec::new();
        let scale_factor = self.scaling_factor.powi(depth as i32);
        
        // Create nodes in fractal pattern (e.g., Sierpinski triangle)
        let child_positions = vec![
            (parent.position.0 + 0.5 * scale_factor, parent.position.1, parent.position.2),
            (parent.position.0 - 0.25 * scale_factor, parent.position.1 + 0.433 * scale_factor, parent.position.2),
            (parent.position.0 - 0.25 * scale_factor, parent.position.1 - 0.433 * scale_factor, parent.position.2),
        ];

        for pos in child_positions {
            children.push(FractalNode {
                position: pos,
                scale: parent.scale * self.scaling_factor,
                circuit_type: parent.circuit_type.clone(),
                atp_density: parent.atp_density * self.self_similarity,
                information_content: parent.information_content * 1.5, // Information increases with detail
            });
        }

        Ok(children)
    }

    fn create_child_connections(&self, parent_index: usize, pattern: &CircuitPattern, depth: usize) -> Result<Vec<FractalConnection>> {
        let mut connections = Vec::new();
        let scale_factor = self.scaling_factor.powi(depth as i32);

        // Connect child nodes with fractal connectivity
        let base_conductance = 1.0 / scale_factor; // Conductance scales with size
        
        for i in 0..3 {
            for j in (i+1)..3 {
                connections.push(FractalConnection {
                    from_node: parent_index * 3 + i,
                    to_node: parent_index * 3 + j,
                    conductance: base_conductance,
                    fractal_weight: self.self_similarity.powi(depth as i32),
                    information_flow: 1000.0 * scale_factor, // Information flow scales
                });
            }
        }

        Ok(connections)
    }

    fn update_fractal_dimension(&mut self, pattern: &CircuitPattern) -> Result<()> {
        let n_nodes = pattern.nodes.len() as f64;
        let n_connections = pattern.connections.len() as f64;
        
        // Box-counting dimension estimation
        let log_n = n_nodes.ln();
        let log_scale = self.scaling_factor.ln();
        self.fractal_dimension.box_counting_dimension = log_n / log_scale;
        
        // Information dimension based on information content
        let total_info: f64 = pattern.nodes.iter()
            .map(|node| node.information_content)
            .sum();
        
        if total_info > 0.0 {
            self.fractal_dimension.information_dimension = total_info.ln() / log_scale;
        }

        Ok(())
    }

    /// Compute fractal ATP flow across scales
    pub fn fractal_atp_flow(&self, pattern: &CircuitPattern) -> Result<f64> {
        let mut total_flow = 0.0;

        for connection in &pattern.connections {
            if connection.from_node < pattern.nodes.len() && connection.to_node < pattern.nodes.len() {
                let from_node = &pattern.nodes[connection.from_node];
                let to_node = &pattern.nodes[connection.to_node];
                
                let atp_gradient = from_node.atp_density - to_node.atp_density;
                let flow = connection.conductance * atp_gradient * connection.fractal_weight;
                total_flow += flow;
            }
        }

        Ok(total_flow)
    }
}

/// Holographic information encoding in cellular circuits
#[derive(Debug, Clone)]
pub struct HolographicCircuit {
    /// Surface area encoding all volume information
    pub boundary_surface: HolographicSurface,
    
    /// Bulk circuit information
    pub bulk_circuit: BulkCircuitInfo,
    
    /// Holographic duality mapping
    pub ads_cft_correspondence: AdsCftMapping,
    
    /// Information entropy of the circuit
    pub von_neumann_entropy: f64,
    
    /// Entanglement entropy across regions
    pub entanglement_entropy: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct HolographicSurface {
    /// Boundary degrees of freedom
    pub boundary_dofs: Vec<BoundaryDegreeOfFreedom>,
    
    /// Surface area in Planck units
    pub surface_area: f64,
    
    /// Information density (bits per Planck area)
    pub information_density: f64,
}

#[derive(Debug, Clone)]
pub struct BoundaryDegreeOfFreedom {
    pub position: (f64, f64), // 2D boundary position
    pub field_value: f64,
    pub conjugate_momentum: f64,
    pub information_content: f64,
}

#[derive(Debug, Clone)]
pub struct BulkCircuitInfo {
    /// 3D circuit in the bulk
    pub bulk_nodes: Vec<(f64, f64, f64)>,
    pub bulk_connections: Vec<(usize, usize)>,
    pub bulk_curvature: f64, // Spacetime curvature
}

#[derive(Debug, Clone)]
pub struct AdsCftMapping {
    /// Conformal transformation parameters
    pub conformal_factor: f64,
    
    /// Radial coordinate mapping
    pub radial_cutoff: f64,
    
    /// Boundary-to-bulk propagator
    pub propagator: Vec<Vec<f64>>,
}

impl HolographicCircuit {
    pub fn new(surface_area: f64) -> Self {
        Self {
            boundary_surface: HolographicSurface {
                boundary_dofs: Vec::new(),
                surface_area,
                information_density: 1.0 / (4.0 * 1.616e-35_f64.powi(2)), // 1/4 bit per Planck area
            },
            bulk_circuit: BulkCircuitInfo {
                bulk_nodes: Vec::new(),
                bulk_connections: Vec::new(),
                bulk_curvature: 0.0,
            },
            ads_cft_correspondence: AdsCftMapping {
                conformal_factor: 1.0,
                radial_cutoff: 1e-6, // UV cutoff
                propagator: Vec::new(),
            },
            von_neumann_entropy: 0.0,
            entanglement_entropy: HashMap::new(),
        }
    }

    /// Encode bulk circuit information on boundary
    pub fn holographic_encoding(&mut self, bulk_circuit: BulkCircuitInfo) -> Result<()> {
        self.bulk_circuit = bulk_circuit;
        
        // Create boundary degrees of freedom
        let n_boundary_dofs = (self.boundary_surface.surface_area * self.boundary_surface.information_density) as usize;
        
        for i in 0..n_boundary_dofs {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_boundary_dofs as f64;
            let phi = std::f64::consts::PI * (i as f64 + 0.5) / n_boundary_dofs as f64;
            
            // Map bulk information to boundary
            let bulk_info = self.extract_bulk_information(theta, phi)?;
            
            self.boundary_surface.boundary_dofs.push(BoundaryDegreeOfFreedom {
                position: (theta, phi),
                field_value: bulk_info.0,
                conjugate_momentum: bulk_info.1,
                information_content: bulk_info.2,
            });
        }

        // Compute holographic entropy
        self.compute_holographic_entropy()?;

        Ok(())
    }

    fn extract_bulk_information(&self, theta: f64, phi: f64) -> Result<(f64, f64, f64)> {
        // Project bulk circuit onto boundary point
        let boundary_point = (theta.sin() * phi.cos(), theta.sin() * phi.sin());
        
        let mut field_value = 0.0;
        let mut momentum = 0.0;
        let mut information = 0.0;

        for (i, bulk_node) in self.bulk_circuit.bulk_nodes.iter().enumerate() {
            let distance = ((bulk_node.0 - boundary_point.0).powi(2) + 
                           (bulk_node.1 - boundary_point.1).powi(2)).sqrt();
            
            let weight = (-distance / self.ads_cft_correspondence.radial_cutoff).exp();
            
            field_value += weight * bulk_node.2; // Use z-coordinate as field value
            momentum += weight * (i as f64); // Node index as momentum
            information += weight; // Weighted information content
        }

        Ok((field_value, momentum, information))
    }

    fn compute_holographic_entropy(&mut self) -> Result<()> {
        // Von Neumann entropy: S = -Tr(ρ log ρ)
        let mut entropy = 0.0;
        
        for dof in &self.boundary_surface.boundary_dofs {
            let prob = dof.information_content / self.boundary_surface.boundary_dofs.len() as f64;
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }
        
        self.von_neumann_entropy = entropy;

        // Entanglement entropy for different regions
        self.compute_entanglement_entropy()?;

        Ok(())
    }

    fn compute_entanglement_entropy(&mut self) -> Result<()> {
        // Divide boundary into regions and compute entanglement
        let n_dofs = self.boundary_surface.boundary_dofs.len();
        
        // Left half vs right half
        let left_entropy = self.region_entropy(0, n_dofs / 2)?;
        let right_entropy = self.region_entropy(n_dofs / 2, n_dofs)?;
        
        self.entanglement_entropy.insert("left_right".to_string(), 
                                       (left_entropy + right_entropy - self.von_neumann_entropy).abs());

        // Top half vs bottom half  
        let top_entropy = self.region_entropy_by_coordinate(|dof| dof.position.1 > std::f64::consts::PI / 2.0)?;
        let bottom_entropy = self.region_entropy_by_coordinate(|dof| dof.position.1 <= std::f64::consts::PI / 2.0)?;
        
        self.entanglement_entropy.insert("top_bottom".to_string(),
                                       (top_entropy + bottom_entropy - self.von_neumann_entropy).abs());

        Ok(())
    }

    fn region_entropy(&self, start: usize, end: usize) -> Result<f64> {
        let mut entropy = 0.0;
        let region_size = (end - start) as f64;
        
        for i in start..end {
            if i < self.boundary_surface.boundary_dofs.len() {
                let dof = &self.boundary_surface.boundary_dofs[i];
                let prob = dof.information_content / region_size;
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
        }
        
        Ok(entropy)
    }

    fn region_entropy_by_coordinate<F>(&self, predicate: F) -> Result<f64>
    where
        F: Fn(&BoundaryDegreeOfFreedom) -> bool,
    {
        let mut entropy = 0.0;
        let mut count = 0;
        
        for dof in &self.boundary_surface.boundary_dofs {
            if predicate(dof) {
                count += 1;
            }
        }
        
        let region_size = count as f64;
        
        for dof in &self.boundary_surface.boundary_dofs {
            if predicate(dof) {
                let prob = dof.information_content / region_size;
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
        }
        
        Ok(entropy)
    }

    /// Reconstruct bulk circuit from boundary information
    pub fn holographic_reconstruction(&self) -> Result<BulkCircuitInfo> {
        let mut reconstructed = BulkCircuitInfo {
            bulk_nodes: Vec::new(),
            bulk_connections: Vec::new(),
            bulk_curvature: 0.0,
        };

        // Use boundary information to reconstruct bulk
        for dof in &self.boundary_surface.boundary_dofs {
            // Map boundary point to bulk using AdS/CFT
            let radial_coord = self.ads_cft_correspondence.radial_cutoff * 
                              (1.0 + dof.information_content);
            
            let bulk_x = radial_coord * dof.position.0.sin() * dof.position.1.cos();
            let bulk_y = radial_coord * dof.position.0.sin() * dof.position.1.sin();
            let bulk_z = radial_coord * dof.position.0.cos();
            
            reconstructed.bulk_nodes.push((bulk_x, bulk_y, bulk_z));
        }

        // Reconstruct connections based on boundary correlations
        for i in 0..reconstructed.bulk_nodes.len() {
            for j in (i+1)..reconstructed.bulk_nodes.len() {
                let dof_i = &self.boundary_surface.boundary_dofs[i];
                let dof_j = &self.boundary_surface.boundary_dofs[j];
                
                // Correlation between boundary points
                let correlation = dof_i.field_value * dof_j.field_value + 
                                dof_i.conjugate_momentum * dof_j.conjugate_momentum;
                
                if correlation > 0.5 { // Threshold for connection
                    reconstructed.bulk_connections.push((i, j));
                }
            }
        }

        Ok(reconstructed)
    }
}

/// Emergent complexity from simple rules
#[derive(Debug, Clone)]
pub struct EmergentComplexityCircuit {
    /// Simple local rules
    pub local_rules: Vec<LocalRule>,
    
    /// Emergent global behavior
    pub global_patterns: Vec<GlobalPattern>,
    
    /// Complexity measures
    pub complexity_metrics: ComplexityMetrics,
    
    /// Phase transition points
    pub critical_points: Vec<CriticalPoint>,
}

#[derive(Debug, Clone)]
pub struct LocalRule {
    pub rule_id: String,
    pub input_conditions: Vec<String>,
    pub output_actions: Vec<String>,
    pub probability: f64,
    pub energy_cost: f64,
}

#[derive(Debug, Clone)]
pub struct GlobalPattern {
    pub pattern_type: String,
    pub characteristic_scale: f64,
    pub persistence_time: f64,
    pub information_content: f64,
}

#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub logical_depth: f64,
    pub thermodynamic_depth: f64,
    pub effective_complexity: f64,
    pub algorithmic_information: f64,
}

#[derive(Debug, Clone)]
pub struct CriticalPoint {
    pub parameter_value: f64,
    pub order_parameter: f64,
    pub correlation_length: f64,
    pub critical_exponents: HashMap<String, f64>,
}

impl EmergentComplexityCircuit {
    pub fn new() -> Self {
        Self {
            local_rules: vec![
                LocalRule {
                    rule_id: "atp_hydrolysis".to_string(),
                    input_conditions: vec!["ATP_present".to_string(), "enzyme_active".to_string()],
                    output_actions: vec!["produce_ADP".to_string(), "release_energy".to_string()],
                    probability: 0.9,
                    energy_cost: -30.5, // kJ/mol released
                },
                LocalRule {
                    rule_id: "cooperative_binding".to_string(),
                    input_conditions: vec!["substrate_bound".to_string()],
                    output_actions: vec!["increase_affinity".to_string()],
                    probability: 0.7,
                    energy_cost: 2.0,
                },
            ],
            global_patterns: Vec::new(),
            complexity_metrics: ComplexityMetrics {
                logical_depth: 0.0,
                thermodynamic_depth: 0.0,
                effective_complexity: 0.0,
                algorithmic_information: 0.0,
            },
            critical_points: Vec::new(),
        }
    }

    /// Simulate emergence of complexity
    pub fn simulate_emergence(&mut self, steps: usize) -> Result<()> {
        for step in 0..steps {
            // Apply local rules
            self.apply_local_rules()?;
            
            // Detect emerging patterns
            self.detect_global_patterns(step)?;
            
            // Update complexity metrics
            self.update_complexity_metrics(step)?;
            
            // Check for critical transitions
            self.detect_critical_points(step)?;
        }

        Ok(())
    }

    fn apply_local_rules(&mut self) -> Result<()> {
        // Simplified rule application
        for rule in &self.local_rules {
            if fastrand::f64() < rule.probability {
                // Rule fires - would update system state
                println!("Rule {} fired", rule.rule_id);
            }
        }
        Ok(())
    }

    fn detect_global_patterns(&mut self, step: usize) -> Result<()> {
        // Detect emergent patterns from local interactions
        if step % 100 == 0 { // Check every 100 steps
            let pattern = GlobalPattern {
                pattern_type: "metabolic_wave".to_string(),
                characteristic_scale: (step as f64).sqrt(),
                persistence_time: 50.0,
                information_content: (step as f64).ln(),
            };
            self.global_patterns.push(pattern);
        }
        Ok(())
    }

    fn update_complexity_metrics(&mut self, step: usize) -> Result<()> {
        // Update complexity measures
        self.complexity_metrics.logical_depth = (step as f64).ln();
        self.complexity_metrics.thermodynamic_depth = step as f64 * 0.1;
        self.complexity_metrics.effective_complexity = 
            self.global_patterns.len() as f64 * (step as f64).sqrt();
        self.complexity_metrics.algorithmic_information = 
            self.local_rules.len() as f64 + self.global_patterns.len() as f64;
        
        Ok(())
    }

    fn detect_critical_points(&mut self, step: usize) -> Result<()> {
        // Detect phase transitions
        let order_parameter = self.complexity_metrics.effective_complexity / (step as f64 + 1.0);
        
        if order_parameter > 0.5 && self.critical_points.is_empty() {
            let critical_point = CriticalPoint {
                parameter_value: step as f64,
                order_parameter,
                correlation_length: 100.0,
                critical_exponents: HashMap::from([
                    ("beta".to_string(), 0.5),
                    ("gamma".to_string(), 1.0),
                    ("nu".to_string(), 1.0),
                ]),
            };
            self.critical_points.push(critical_point);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_circuit() {
        let base_pattern = CircuitPattern {
            nodes: vec![FractalNode {
                position: (0.0, 0.0, 0.0),
                scale: 1.0,
                circuit_type: "ATP_node".to_string(),
                atp_density: 5.0,
                information_content: 1.0,
            }],
            connections: Vec::new(),
            symmetry_group: SymmetryGroup {
                rotations: vec![0.0, 2.0 * std::f64::consts::PI / 3.0, 4.0 * std::f64::consts::PI / 3.0],
                reflections: vec![true, true, true],
                translations: Vec::new(),
            },
        };

        let mut fractal = FractalCircuit::new(base_pattern, 0.5);
        let result = fractal.generate_fractal(2);
        assert!(result.is_ok());
        
        let pattern = result.unwrap();
        assert!(pattern.nodes.len() > 1); // Should have generated more nodes
    }

    #[test]
    fn test_holographic_circuit() {
        let mut holo = HolographicCircuit::new(1e-12); // 1 μm² surface
        
        let bulk = BulkCircuitInfo {
            bulk_nodes: vec![(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            bulk_connections: vec![(0, 1), (1, 2)],
            bulk_curvature: 0.1,
        };

        let result = holo.holographic_encoding(bulk);
        assert!(result.is_ok());
        assert!(holo.von_neumann_entropy > 0.0);
    }

    #[test]
    fn test_emergent_complexity() {
        let mut circuit = EmergentComplexityCircuit::new();
        let result = circuit.simulate_emergence(500);
        assert!(result.is_ok());
        assert!(circuit.complexity_metrics.effective_complexity > 0.0);
    }
} 
//! # Enzyme Probabilistic Circuits: Biochemical Logic Gates
//! 
//! This module implements enzymes as probabilistic logic circuits that can
//! perform biochemical transformations with ATP-dependent success rates.

use crate::error::{NebuchadnezzarError, Result};
use crate::systems_biology::AtpPool;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base trait for all enzyme probabilistic circuits
pub trait EnzymeProbCircuit {
    fn can_fire(&self, atp_concentration: f64) -> bool;
    fn compute_flux(&self, atp_concentration: f64) -> Result<HashMap<String, f64>>;
    fn atp_cost(&self) -> f64;
    fn success_probability(&self, atp_concentration: f64) -> f64;
    fn get_enzyme_class(&self) -> EnzymeClass;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnzymeClass {
    Isomerase,
    Dismutase,
    Dehydrogenase,
    Kinase,
    Ligase,
    Hydrolase,
    Transferase,
}

/// Probabilistic NOT gate - Isomerase (A ⇌ B)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticNOT {
    pub p_success: f64,
    pub atp_requirement: f64,
    pub substrate: String,
    pub product: String,
    pub rate_constant: f64,
}

impl ProbabilisticNOT {
    pub fn new(substrate: String, product: String, p_success: f64) -> Self {
        Self {
            p_success,
            atp_requirement: 0.01,
            substrate,
            product,
            rate_constant: 1.0,
        }
    }

    pub fn glucose_phosphate_isomerase() -> Self {
        Self::new(
            "glucose-6-phosphate".to_string(),
            "fructose-6-phosphate".to_string(),
            0.95,
        )
    }
}

impl EnzymeProbCircuit for ProbabilisticNOT {
    fn can_fire(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.atp_requirement
    }

    fn compute_flux(&self, atp_concentration: f64) -> Result<HashMap<String, f64>> {
        let mut flux = HashMap::new();
        
        if self.can_fire(atp_concentration) {
            let success_prob = self.success_probability(atp_concentration);
            let flux_rate = self.rate_constant * success_prob;
            
            flux.insert(self.substrate.clone(), -flux_rate);
            flux.insert(self.product.clone(), flux_rate);
        }
        
        Ok(flux)
    }

    fn atp_cost(&self) -> f64 {
        self.atp_requirement
    }

    fn success_probability(&self, atp_concentration: f64) -> f64 {
        if atp_concentration >= self.atp_requirement {
            self.p_success
        } else {
            0.0
        }
    }

    fn get_enzyme_class(&self) -> EnzymeClass {
        EnzymeClass::Isomerase
    }
}

/// Probabilistic SPLIT gate - Dismutase (A → B + C)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticSPLIT {
    pub p_success: f64,
    pub p_both_products: f64,
    pub atp_requirement: f64,
    pub substrate: String,
    pub product_1: String,
    pub product_2: String,
    pub rate_constant: f64,
}

impl ProbabilisticSPLIT {
    pub fn new(substrate: String, product_1: String, product_2: String, p_success: f64, p_both_products: f64) -> Self {
        Self {
            p_success,
            p_both_products,
            atp_requirement: 0.02,
            substrate,
            product_1,
            product_2,
            rate_constant: 0.8,
        }
    }

    pub fn superoxide_dismutase() -> Self {
        Self::new(
            "superoxide".to_string(),
            "oxygen".to_string(),
            "hydrogen_peroxide".to_string(),
            0.85,
            0.8,
        )
    }
}

impl EnzymeProbCircuit for ProbabilisticSPLIT {
    fn can_fire(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.atp_requirement
    }

    fn compute_flux(&self, atp_concentration: f64) -> Result<HashMap<String, f64>> {
        let mut flux = HashMap::new();
        
        if self.can_fire(atp_concentration) {
            let success_prob = self.success_probability(atp_concentration);
            let flux_rate = self.rate_constant * success_prob;
            
            flux.insert(self.substrate.clone(), -flux_rate);
            
            // Probabilistic product formation
            if fastrand::f64() < self.p_both_products {
                flux.insert(self.product_1.clone(), flux_rate);
                flux.insert(self.product_2.clone(), flux_rate);
            } else {
                // Only one product forms
                if fastrand::f64() < 0.5 {
                    flux.insert(self.product_1.clone(), flux_rate);
                } else {
                    flux.insert(self.product_2.clone(), flux_rate);
                }
            }
        }
        
        Ok(flux)
    }

    fn atp_cost(&self) -> f64 {
        self.atp_requirement
    }

    fn success_probability(&self, atp_concentration: f64) -> f64 {
        if atp_concentration >= self.atp_requirement {
            self.p_success
        } else {
            0.0
        }
    }

    fn get_enzyme_class(&self) -> EnzymeClass {
        EnzymeClass::Dismutase
    }
}

/// Probabilistic XOR gate - Dehydrogenase (A + NAD+ ⇌ B + NADH)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticXOR {
    pub p_success: f64,
    pub p_cofactor_binding: f64,
    pub atp_requirement: f64,
    pub substrate: String,
    pub product: String,
    pub cofactor_oxidized: String,
    pub cofactor_reduced: String,
    pub rate_constant: f64,
}

impl ProbabilisticXOR {
    pub fn new(
        substrate: String,
        product: String,
        cofactor_oxidized: String,
        cofactor_reduced: String,
        p_success: f64,
        p_cofactor_binding: f64,
    ) -> Self {
        Self {
            p_success,
            p_cofactor_binding,
            atp_requirement: 0.05,
            substrate,
            product,
            cofactor_oxidized,
            cofactor_reduced,
            rate_constant: 1.2,
        }
    }

    pub fn lactate_dehydrogenase() -> Self {
        Self::new(
            "lactate".to_string(),
            "pyruvate".to_string(),
            "NAD+".to_string(),
            "NADH".to_string(),
            0.90,
            0.7,
        )
    }
}

impl EnzymeProbCircuit for ProbabilisticXOR {
    fn can_fire(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.atp_requirement
    }

    fn compute_flux(&self, atp_concentration: f64) -> Result<HashMap<String, f64>> {
        let mut flux = HashMap::new();
        
        if self.can_fire(atp_concentration) {
            let success_prob = self.success_probability(atp_concentration);
            
            // XOR logic: reaction proceeds only if cofactor binding is successful
            if fastrand::f64() < self.p_cofactor_binding {
                let flux_rate = self.rate_constant * success_prob;
                
                flux.insert(self.substrate.clone(), -flux_rate);
                flux.insert(self.cofactor_oxidized.clone(), -flux_rate);
                flux.insert(self.product.clone(), flux_rate);
                flux.insert(self.cofactor_reduced.clone(), flux_rate);
            }
        }
        
        Ok(flux)
    }

    fn atp_cost(&self) -> f64 {
        self.atp_requirement
    }

    fn success_probability(&self, atp_concentration: f64) -> f64 {
        if atp_concentration >= self.atp_requirement {
            self.p_success * self.p_cofactor_binding
        } else {
            0.0
        }
    }

    fn get_enzyme_class(&self) -> EnzymeClass {
        EnzymeClass::Dehydrogenase
    }
}

/// Probabilistic OR gate - Kinase (A + ATP → A-P + ADP)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticOR {
    pub p_success: f64,
    pub p_atp_binding: f64,
    pub atp_requirement: f64,
    pub substrate: String,
    pub phosphorylated_product: String,
    pub rate_constant: f64,
    pub km_atp: f64,
}

impl ProbabilisticOR {
    pub fn new(substrate: String, phosphorylated_product: String, p_success: f64, p_atp_binding: f64) -> Self {
        Self {
            p_success,
            p_atp_binding,
            atp_requirement: 1.0,
            substrate,
            phosphorylated_product,
            rate_constant: 0.9,
            km_atp: 0.1,
        }
    }

    pub fn hexokinase() -> Self {
        Self::new(
            "glucose".to_string(),
            "glucose-6-phosphate".to_string(),
            0.88,
            0.85,
        )
    }

    pub fn protein_kinase_a() -> Self {
        Self::new(
            "target_protein".to_string(),
            "phospho_target_protein".to_string(),
            0.92,
            0.90,
        )
    }
}

impl EnzymeProbCircuit for ProbabilisticOR {
    fn can_fire(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.atp_requirement
    }

    fn compute_flux(&self, atp_concentration: f64) -> Result<HashMap<String, f64>> {
        let mut flux = HashMap::new();
        
        if self.can_fire(atp_concentration) {
            let success_prob = self.success_probability(atp_concentration);
            
            // OR logic: reaction can proceed via multiple pathways
            let atp_binding_success = fastrand::f64() < self.p_atp_binding;
            let alternative_pathway = fastrand::f64() < 0.1; // 10% alternative pathway
            
            if atp_binding_success || alternative_pathway {
                let flux_rate = self.rate_constant * success_prob;
                
                flux.insert(self.substrate.clone(), -flux_rate);
                flux.insert("ATP".to_string(), -flux_rate);
                flux.insert(self.phosphorylated_product.clone(), flux_rate);
                flux.insert("ADP".to_string(), flux_rate);
            }
        }
        
        Ok(flux)
    }

    fn atp_cost(&self) -> f64 {
        self.atp_requirement
    }

    fn success_probability(&self, atp_concentration: f64) -> f64 {
        if atp_concentration >= self.atp_requirement {
            let atp_factor = atp_concentration / (self.km_atp + atp_concentration);
            self.p_success * atp_factor
        } else {
            0.0
        }
    }

    fn get_enzyme_class(&self) -> EnzymeClass {
        EnzymeClass::Kinase
    }
}

/// Probabilistic AND gate - Ligase (A + B + ATP → A-B + AMP + PPi)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticAND {
    pub p_success: f64,
    pub p_energy_coupling: f64,
    pub atp_requirement: f64,
    pub substrate_1: String,
    pub substrate_2: String,
    pub product: String,
    pub rate_constant: f64,
}

impl ProbabilisticAND {
    pub fn new(
        substrate_1: String,
        substrate_2: String,
        product: String,
        p_success: f64,
        p_energy_coupling: f64,
    ) -> Self {
        Self {
            p_success,
            p_energy_coupling,
            atp_requirement: 2.0,
            substrate_1,
            substrate_2,
            product,
            rate_constant: 0.6,
        }
    }

    pub fn dna_ligase() -> Self {
        Self::new(
            "dna_fragment_1".to_string(),
            "dna_fragment_2".to_string(),
            "ligated_dna".to_string(),
            0.75,
            0.9,
        )
    }

    pub fn acetyl_coa_carboxylase() -> Self {
        Self::new(
            "acetyl_coa".to_string(),
            "co2".to_string(),
            "malonyl_coa".to_string(),
            0.80,
            0.85,
        )
    }
}

impl EnzymeProbCircuit for ProbabilisticAND {
    fn can_fire(&self, atp_concentration: f64) -> bool {
        atp_concentration >= self.atp_requirement
    }

    fn compute_flux(&self, atp_concentration: f64) -> Result<HashMap<String, f64>> {
        let mut flux = HashMap::new();
        
        if self.can_fire(atp_concentration) {
            let success_prob = self.success_probability(atp_concentration);
            
            // AND logic: both substrates must be available AND energy coupling must work
            let substrate_1_available = fastrand::f64() < 0.8;
            let substrate_2_available = fastrand::f64() < 0.8;
            let energy_coupling_success = fastrand::f64() < self.p_energy_coupling;
            
            if substrate_1_available && substrate_2_available && energy_coupling_success {
                let flux_rate = self.rate_constant * success_prob;
                
                flux.insert(self.substrate_1.clone(), -flux_rate);
                flux.insert(self.substrate_2.clone(), -flux_rate);
                flux.insert("ATP".to_string(), -flux_rate);
                flux.insert(self.product.clone(), flux_rate);
                flux.insert("AMP".to_string(), flux_rate);
                flux.insert("PPi".to_string(), flux_rate);
            }
        }
        
        Ok(flux)
    }

    fn atp_cost(&self) -> f64 {
        self.atp_requirement
    }

    fn success_probability(&self, atp_concentration: f64) -> f64 {
        if atp_concentration >= self.atp_requirement {
            self.p_success * self.p_energy_coupling
        } else {
            0.0
        }
    }

    fn get_enzyme_class(&self) -> EnzymeClass {
        EnzymeClass::Ligase
    }
}

/// Factory for creating enzyme circuits
pub struct EnzymeCircuitFactory;

impl EnzymeCircuitFactory {
    pub fn create_isomerase(substrate: String, product: String) -> Box<dyn EnzymeProbCircuit> {
        Box::new(ProbabilisticNOT::new(substrate, product, 0.95))
    }

    pub fn create_dismutase(substrate: String, product_1: String, product_2: String) -> Box<dyn EnzymeProbCircuit> {
        Box::new(ProbabilisticSPLIT::new(substrate, product_1, product_2, 0.85, 0.8))
    }

    pub fn create_dehydrogenase(
        substrate: String,
        product: String,
        cofactor_ox: String,
        cofactor_red: String,
    ) -> Box<dyn EnzymeProbCircuit> {
        Box::new(ProbabilisticXOR::new(substrate, product, cofactor_ox, cofactor_red, 0.90, 0.7))
    }

    pub fn create_kinase(substrate: String, phospho_product: String) -> Box<dyn EnzymeProbCircuit> {
        Box::new(ProbabilisticOR::new(substrate, phospho_product, 0.88, 0.85))
    }

    pub fn create_ligase(substrate_1: String, substrate_2: String, product: String) -> Box<dyn EnzymeProbCircuit> {
        Box::new(ProbabilisticAND::new(substrate_1, substrate_2, product, 0.75, 0.9))
    }

    /// Create glycolysis enzyme circuits
    pub fn create_glycolysis_enzymes() -> Vec<Box<dyn EnzymeProbCircuit>> {
        vec![
            Box::new(ProbabilisticOR::hexokinase()),
            Box::new(ProbabilisticNOT::glucose_phosphate_isomerase()),
            Box::new(ProbabilisticOR::new(
                "fructose-6-phosphate".to_string(),
                "fructose-1,6-bisphosphate".to_string(),
                0.85,
                0.80,
            )), // Phosphofructokinase
        ]
    }
}

/// Enzyme circuit that can be resolved to detailed kinetic model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvableEnzymeCircuit {
    pub base_circuit: EnzymeClass,
    pub detailed_kinetics: Option<DetailedKineticModel>,
    pub resolution_threshold: f64,
    pub current_importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedKineticModel {
    pub enzyme_states: Vec<String>,
    pub transition_rates: HashMap<(usize, usize), f64>,
    pub michaelis_constants: HashMap<String, f64>,
    pub turnover_numbers: HashMap<String, f64>,
}

impl ResolvableEnzymeCircuit {
    pub fn should_resolve(&self) -> bool {
        self.current_importance > self.resolution_threshold
    }

    pub fn resolve_to_detailed_kinetics(&mut self) -> Result<()> {
        if self.should_resolve() {
            self.detailed_kinetics = Some(self.build_detailed_model()?);
        }
        Ok(())
    }

    fn build_detailed_model(&self) -> Result<DetailedKineticModel> {
        match self.base_circuit {
            EnzymeClass::Kinase => Ok(DetailedKineticModel {
                enzyme_states: vec![
                    "free".to_string(),
                    "substrate_bound".to_string(),
                    "atp_bound".to_string(),
                    "ternary_complex".to_string(),
                    "phosphorylated".to_string(),
                ],
                transition_rates: HashMap::new(),
                michaelis_constants: HashMap::from([
                    ("substrate".to_string(), 0.1),
                    ("ATP".to_string(), 0.05),
                ]),
                turnover_numbers: HashMap::from([
                    ("phosphorylation".to_string(), 100.0),
                ]),
            }),
            _ => Ok(DetailedKineticModel {
                enzyme_states: vec!["free".to_string(), "bound".to_string()],
                transition_rates: HashMap::new(),
                michaelis_constants: HashMap::new(),
                turnover_numbers: HashMap::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isomerase_circuit() {
        let isomerase = ProbabilisticNOT::glucose_phosphate_isomerase();
        assert!(isomerase.can_fire(1.0));
        assert_eq!(isomerase.get_enzyme_class(), EnzymeClass::Isomerase);
    }

    #[test]
    fn test_kinase_circuit() {
        let kinase = ProbabilisticOR::hexokinase();
        assert!(kinase.can_fire(2.0));
        assert!(!kinase.can_fire(0.5));
        assert_eq!(kinase.get_enzyme_class(), EnzymeClass::Kinase);
    }

    #[test]
    fn test_enzyme_factory() {
        let enzymes = EnzymeCircuitFactory::create_glycolysis_enzymes();
        assert!(!enzymes.is_empty());
    }
} 
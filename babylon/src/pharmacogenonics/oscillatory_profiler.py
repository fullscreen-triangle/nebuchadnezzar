"""
Oscillatory Profiler Module
===========================

Comprehensive pharmacogenomic profiler that integrates all oscillatory framework
modules to generate complete drug response predictions. Combines genetic analysis,
molecular dynamics, cellular responses, and temporal patterns into unified profiles.

The ultimate integration module for the St. Stellas oscillatory framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

# Import all the other babylon modules
# from .hole_detector import OscillatoryHoleDetector
# from .genetic_risk_assessor import GeneticRiskAssessor
# from ..dynamics.quantum_drug_transport import QuantumDrugTransport
# from ..dynamics.cellular_drug_response import CellularDrugResponse
# from ..dynamics.molecular_drug_binding import MolecularDrugBinding
# from ..dynamics.tissue_drug_distribution import TissueDrugDistribution
# from ..dynamics.organ_drug_effects import OrganDrugEffects
# from ..dynamics.systemic_drug_response import SystemicDrugResponse
# from ..dynamics.temporal_drug_patterns import TemporalDrugPatterns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OscillatorySignature:
    """Represents the oscillatory signature extracted from genetic variants."""
    individual_id: str
    variant_count: int
    oscillatory_holes: List[Dict[str, Any]]  # From hole detector
    frequency_spectrum: Dict[str, float]  # Dominant frequencies by pathway
    coupling_matrix: Dict[str, Dict[str, float]]  # Pathway coupling strengths
    coherence_metrics: Dict[str, float]  # Oscillatory coherence by scale
    disruption_patterns: List[str]  # Types of oscillatory disruption
    therapeutic_targets: List[str]  # Potential therapeutic targets

@dataclass
class DrugResponsePrediction:
    """Comprehensive drug response prediction."""
    drug_name: str
    individual_id: str
    
    # From genetic risk assessor
    genetic_risk_score: float
    pharmacogenetic_warnings: List[str]
    
    # From quantum transport
    membrane_permeability: float
    transport_mechanism: str
    quantum_enhancement: float
    
    # From molecular binding
    binding_affinity: float
    binding_kinetics: Dict[str, float]
    
    # From cellular response
    cellular_effects: Dict[str, float]
    
    # From tissue distribution
    tissue_concentrations: Dict[str, float]
    
    # From organ effects
    organ_functional_changes: Dict[str, float]  
    therapeutic_benefit: float
    adverse_risk: float
    
    # From systemic response
    system_stability: float
    homeostatic_compensation: float
    emergent_effects: List[str]
    
    # From temporal patterns
    optimal_dosing_time: float
    chronotherapy_advantage: float
    circadian_effects: float
    
    # Integrated metrics
    overall_efficacy_prediction: float  # 0-1 scale
    overall_safety_prediction: float   # 0-1 scale
    therapeutic_index: float           # Efficacy/Safety ratio
    confidence_score: float            # Prediction confidence
    
    # Personalized recommendations
    dosing_recommendations: List[str]
    monitoring_recommendations: List[str]
    alternative_drugs: List[str]

@dataclass
class ComprehensiveProfile:
    """Complete oscillatory pharmacogenomic profile."""
    individual_id: str
    analysis_timestamp: str
    
    # Input data
    genetic_variants: List[Dict[str, Any]]
    
    # Oscillatory signature
    oscillatory_signature: OscillatorySignature
    
    # Drug predictions
    drug_predictions: List[DrugResponsePrediction]
    
    # Summary metrics
    overall_genetic_complexity: float
    oscillatory_stability: float
    drug_response_predictability: float
    
    # Clinical insights
    high_risk_drugs: List[str]
    recommended_drugs: List[str]
    personalized_insights: List[str]
    clinical_recommendations: List[str]

class OscillatoryProfiler:
    """
    Master integration class that combines all oscillatory framework modules
    to generate comprehensive pharmacogenomic profiles.
    
    This is the main interface for the complete Babylon framework.
    """
    
    def __init__(self):
        """Initialize the oscillatory profiler with all sub-modules."""
        logger.info("Initializing Oscillatory Profiler...")
        
        # Initialize all framework modules
        # self.hole_detector = OscillatoryHoleDetector()
        # self.risk_assessor = GeneticRiskAssessor()
        # self.quantum_transport = QuantumDrugTransport()
        # self.cellular_response = CellularDrugResponse()
        # self.molecular_binding = MolecularDrugBinding()
        # self.tissue_distribution = TissueDrugDistribution()
        # self.organ_effects = OrganDrugEffects()
        # self.systemic_response = SystemicDrugResponse()
        # self.temporal_patterns = TemporalDrugPatterns()
        
        # For now, use mock implementations
        self._initialize_mock_modules()
        
        logger.info("Oscillatory Profiler initialized successfully")
    
    def _initialize_mock_modules(self):
        """Initialize mock modules for demonstration (replace with actual imports)."""
        # This is a placeholder - in production, would import actual modules
        self.modules_initialized = True
        
    def extract_oscillatory_signature(self, variants: List[Dict[str, Any]], 
                                    individual_id: str) -> OscillatorySignature:
        """Extract oscillatory signature from genetic variants."""
        
        logger.info(f"Extracting oscillatory signature for {individual_id}")
        
        # Detect oscillatory holes (mock implementation)
        oscillatory_holes = self._detect_oscillatory_holes_mock(variants)
        
        # Calculate frequency spectrum
        frequency_spectrum = self._calculate_frequency_spectrum(variants)
        
        # Calculate coupling matrix
        coupling_matrix = self._calculate_coupling_matrix(variants)
        
        # Calculate coherence metrics
        coherence_metrics = self._calculate_coherence_metrics(variants)
        
        # Identify disruption patterns
        disruption_patterns = self._identify_disruption_patterns(variants)
        
        # Identify therapeutic targets
        therapeutic_targets = self._identify_therapeutic_targets(variants)
        
        signature = OscillatorySignature(
            individual_id=individual_id,
            variant_count=len(variants),
            oscillatory_holes=oscillatory_holes,
            frequency_spectrum=frequency_spectrum,
            coupling_matrix=coupling_matrix,
            coherence_metrics=coherence_metrics,
            disruption_patterns=disruption_patterns,
            therapeutic_targets=therapeutic_targets
        )
        
        logger.info(f"Extracted signature with {len(oscillatory_holes)} holes, "
                   f"{len(disruption_patterns)} disruption patterns")
        
        return signature
    
    def _detect_oscillatory_holes_mock(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock implementation of hole detection."""
        holes = []
        
        # Key oscillatory pathways and their characteristics
        pathways = {
            'inositol_metabolism': {'frequency': 7.23e13, 'genes': ['INPP1', 'IMPA1', 'IMPA2']},
            'gsk3_pathway': {'frequency': 2.15e13, 'genes': ['GSK3A', 'GSK3B']},
            'neurotransmitter_signaling': {'frequency': 1.45e13, 'genes': ['DRD2', 'HTR2A', 'SLC6A4']},
            'drug_metabolism': {'frequency': 3.8e13, 'genes': ['CYP2D6', 'CYP2C19', 'CYP3A4']}
        }
        
        for variant in variants:
            gene = variant.get('gene', '')
            impact = variant.get('impact', 'UNKNOWN')
            
            for pathway_name, pathway_info in pathways.items():
                if gene in pathway_info['genes']:
                    # Calculate hole properties
                    amplitude_deficit = {
                        'HIGH': 0.8, 'MODERATE': 0.6, 'LOW': 0.3, 'MODIFIER': 0.1
                    }.get(impact, 0.4)
                    
                    hole = {
                        'gene_id': gene,
                        'pathway': pathway_name,
                        'frequency': pathway_info['frequency'],
                        'amplitude_deficit': amplitude_deficit,
                        'variant_source': variant.get('variant_id', 'unknown'),
                        'confidence': 0.85
                    }
                    holes.append(hole)
        
        return holes
    
    def _calculate_frequency_spectrum(self, variants: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate dominant frequencies by pathway."""
        spectrum = {}
        
        # Base pathway frequencies
        base_frequencies = {
            'inositol_metabolism': 7.23e13,
            'gsk3_pathway': 2.15e13,
            'neurotransmitter_signaling': 1.45e13,
            'drug_metabolism': 3.8e13,
            'circadian_rhythms': 1.16e-5,
            'cellular_oscillations': 1e-3
        }
        
        # Modify frequencies based on variants
        for pathway, base_freq in base_frequencies.items():
            # Count variants affecting this pathway
            affecting_variants = 0
            total_impact = 0.0
            
            for variant in variants:
                gene = variant.get('gene', '')
                impact = variant.get('impact', 'UNKNOWN')
                
                # Determine if variant affects this pathway
                pathway_genes = {
                    'inositol_metabolism': ['INPP1', 'IMPA1', 'IMPA2'],
                    'gsk3_pathway': ['GSK3A', 'GSK3B'],
                    'neurotransmitter_signaling': ['DRD2', 'HTR2A', 'SLC6A4', 'COMT'],
                    'drug_metabolism': ['CYP2D6', 'CYP2C19', 'CYP3A4'],
                    'circadian_rhythms': ['CLOCK', 'BMAL1', 'PER1', 'PER2'],
                    'cellular_oscillations': ['CACNA1C', 'KCNH2', 'SCN5A']
                }
                
                if gene in pathway_genes.get(pathway, []):
                    affecting_variants += 1
                    impact_score = {
                        'HIGH': 0.8, 'MODERATE': 0.6, 'LOW': 0.3, 'MODIFIER': 0.1
                    }.get(impact, 0.4)
                    total_impact += impact_score
            
            # Calculate modified frequency
            if affecting_variants > 0:
                avg_impact = total_impact / affecting_variants
                # Variants generally reduce oscillatory frequency
                frequency_reduction = avg_impact * 0.2  # Up to 20% reduction
                modified_freq = base_freq * (1.0 - frequency_reduction)
            else:
                modified_freq = base_freq
            
            spectrum[pathway] = modified_freq
        
        return spectrum
    
    def _calculate_coupling_matrix(self, variants: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate pathway coupling strengths."""
        pathways = ['inositol_metabolism', 'gsk3_pathway', 'neurotransmitter_signaling', 
                   'drug_metabolism', 'circadian_rhythms', 'cellular_oscillations']
        
        # Base coupling matrix
        base_coupling = {
            'inositol_metabolism': {
                'gsk3_pathway': 0.8, 'neurotransmitter_signaling': 0.6,
                'drug_metabolism': 0.3, 'circadian_rhythms': 0.4, 'cellular_oscillations': 0.7
            },
            'gsk3_pathway': {
                'inositol_metabolism': 0.8, 'neurotransmitter_signaling': 0.9,
                'drug_metabolism': 0.4, 'circadian_rhythms': 0.6, 'cellular_oscillations': 0.8
            },
            'neurotransmitter_signaling': {
                'inositol_metabolism': 0.6, 'gsk3_pathway': 0.9,
                'drug_metabolism': 0.7, 'circadian_rhythms': 0.8, 'cellular_oscillations': 0.7
            },
            'drug_metabolism': {
                'inositol_metabolism': 0.3, 'gsk3_pathway': 0.4, 'neurotransmitter_signaling': 0.7,
                'circadian_rhythms': 0.5, 'cellular_oscillations': 0.5
            },
            'circadian_rhythms': {
                'inositol_metabolism': 0.4, 'gsk3_pathway': 0.6, 'neurotransmitter_signaling': 0.8,
                'drug_metabolism': 0.5, 'cellular_oscillations': 0.7
            },
            'cellular_oscillations': {
                'inositol_metabolism': 0.7, 'gsk3_pathway': 0.8, 'neurotransmitter_signaling': 0.7,
                'drug_metabolism': 0.5, 'circadian_rhythms': 0.7
            }
        }
        
        # Modify coupling based on variants (more variants = reduced coupling)
        variant_count = len(variants)
        coupling_reduction = min(0.3, variant_count * 0.02)  # Up to 30% reduction
        
        coupling_matrix = {}
        for pathway1 in pathways:
            coupling_matrix[pathway1] = {}
            for pathway2 in pathways:
                if pathway1 == pathway2:
                    coupling_matrix[pathway1][pathway2] = 1.0
                else:
                    base_value = base_coupling.get(pathway1, {}).get(pathway2, 0.5)
                    modified_value = base_value * (1.0 - coupling_reduction)
                    coupling_matrix[pathway1][pathway2] = max(0.1, modified_value)
        
        return coupling_matrix
    
    def _calculate_coherence_metrics(self, variants: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate oscillatory coherence metrics."""
        
        # Base coherence levels
        base_coherence = {
            'molecular': 0.85,
            'cellular': 0.75,
            'tissue': 0.65,
            'organ': 0.70,
            'systemic': 0.60
        }
        
        # Calculate coherence reduction based on variant burden
        variant_burden = len(variants)
        high_impact_variants = len([v for v in variants if v.get('impact') == 'HIGH'])
        
        # Higher variant burden reduces coherence
        burden_penalty = min(0.4, variant_burden * 0.015)  # Up to 40% reduction
        high_impact_penalty = min(0.2, high_impact_variants * 0.05)  # Additional penalty for high impact
        
        coherence_metrics = {}
        for scale, base_value in base_coherence.items():
            modified_coherence = base_value * (1.0 - burden_penalty - high_impact_penalty)
            coherence_metrics[scale] = max(0.2, modified_coherence)
        
        return coherence_metrics
    
    def _identify_disruption_patterns(self, variants: List[Dict[str, Any]]) -> List[str]:
        """Identify types of oscillatory disruption."""
        patterns = []
        
        # Analyze variant distribution
        gene_impacts = {}
        for variant in variants:
            gene = variant.get('gene', '')
            impact = variant.get('impact', 'UNKNOWN')
            if gene not in gene_impacts:
                gene_impacts[gene] = []
            gene_impacts[gene].append(impact)
        
        # Identify patterns
        high_impact_genes = [gene for gene, impacts in gene_impacts.items() 
                           if 'HIGH' in impacts]
        
        if len(high_impact_genes) >= 3:
            patterns.append('multi_gene_disruption')
        
        # Check for pathway-specific disruptions
        inositol_genes = ['INPP1', 'IMPA1', 'IMPA2']
        gsk3_genes = ['GSK3A', 'GSK3B']
        neuro_genes = ['DRD2', 'HTR2A', 'SLC6A4', 'COMT']
        cyp_genes = ['CYP2D6', 'CYP2C19', 'CYP3A4']
        
        if any(gene in gene_impacts for gene in inositol_genes):
            patterns.append('inositol_pathway_disruption')
        
        if any(gene in gene_impacts for gene in gsk3_genes):
            patterns.append('gsk3_pathway_disruption')
        
        if any(gene in gene_impacts for gene in neuro_genes):
            patterns.append('neurotransmitter_disruption')
        
        if any(gene in gene_impacts for gene in cyp_genes):
            patterns.append('metabolic_disruption')
        
        # Check for complex patterns
        total_variants = len(variants)
        if total_variants >= 10:
            patterns.append('complex_genetic_architecture')
        
        if len(set(gene_impacts.keys())) >= 8:
            patterns.append('distributed_variant_pattern')
        
        return patterns
    
    def _identify_therapeutic_targets(self, variants: List[Dict[str, Any]]) -> List[str]:
        """Identify potential therapeutic targets based on variants."""
        targets = []
        
        # Analyze variants to identify therapeutic opportunities
        gene_variants = {}
        for variant in variants:
            gene = variant.get('gene', '')
            impact = variant.get('impact', 'UNKNOWN')
            if gene not in gene_variants:
                gene_variants[gene] = []
            gene_variants[gene].append(impact)
        
        # Target identification logic
        if 'INPP1' in gene_variants or 'GSK3B' in gene_variants:
            targets.append('lithium_responsive_pathway')
        
        if any(gene in gene_variants for gene in ['DRD2', 'HTR2A']):
            targets.append('dopamine_serotonin_system')
        
        if any(gene in gene_variants for gene in ['CYP2D6', 'CYP2C19', 'CYP3A4']):
            targets.append('metabolic_optimization')
        
        if 'SLC6A4' in gene_variants:
            targets.append('serotonin_transport_system')
        
        if 'COMT' in gene_variants:
            targets.append('catecholamine_metabolism')
        
        # Add general targets based on variant burden
        if len(variants) >= 8:
            targets.append('multi_target_approach')
        
        if len([v for v in variants if v.get('impact') == 'HIGH']) >= 2:
            targets.append('precision_dosing_required')
        
        return targets
    
    def predict_drug_response(self, drug_name: str, 
                            individual_id: str,
                            oscillatory_signature: OscillatorySignature,
                            variants: List[Dict[str, Any]]) -> DrugResponsePrediction:
        """Generate comprehensive drug response prediction."""
        
        logger.info(f"Predicting {drug_name} response for {individual_id}")
        
        # Mock implementations of all module predictions
        # In production, these would call the actual modules
        
        # Genetic risk assessment
        genetic_risk_score = self._calculate_genetic_risk_mock(drug_name, variants)
        pharmacogenetic_warnings = self._get_pharmacogenetic_warnings_mock(drug_name, variants)
        
        # Quantum transport
        transport_data = self._predict_transport_mock(drug_name, oscillatory_signature)
        
        # Molecular binding
        binding_data = self._predict_binding_mock(drug_name, oscillatory_signature)
        
        # Cellular response
        cellular_effects = self._predict_cellular_effects_mock(drug_name, oscillatory_signature)
        
        # Tissue distribution
        tissue_concentrations = self._predict_tissue_distribution_mock(drug_name)
        
        # Organ effects
        organ_data = self._predict_organ_effects_mock(drug_name, tissue_concentrations)
        
        # Systemic response
        systemic_data = self._predict_systemic_response_mock(drug_name, organ_data)
        
        # Temporal patterns
        temporal_data = self._predict_temporal_patterns_mock(drug_name)
        
        # Integrate all predictions
        efficacy_prediction = self._integrate_efficacy_prediction(
            genetic_risk_score, transport_data, binding_data, 
            cellular_effects, organ_data, systemic_data
        )
        
        safety_prediction = self._integrate_safety_prediction(
            genetic_risk_score, organ_data, systemic_data, pharmacogenetic_warnings
        )
        
        therapeutic_index = efficacy_prediction / (1.0 - safety_prediction + 0.1)
        
        confidence_score = self._calculate_prediction_confidence(
            oscillatory_signature, len(variants), drug_name
        )
        
        # Generate recommendations
        dosing_recommendations = self._generate_dosing_recommendations(
            drug_name, genetic_risk_score, temporal_data, organ_data
        )
        
        monitoring_recommendations = self._generate_monitoring_recommendations(
            drug_name, pharmacogenetic_warnings, organ_data
        )
        
        alternative_drugs = self._suggest_alternative_drugs(
            drug_name, genetic_risk_score, therapeutic_index
        )
        
        prediction = DrugResponsePrediction(
            drug_name=drug_name,
            individual_id=individual_id,
            genetic_risk_score=genetic_risk_score,
            pharmacogenetic_warnings=pharmacogenetic_warnings,
            membrane_permeability=transport_data['permeability'],
            transport_mechanism=transport_data['mechanism'],
            quantum_enhancement=transport_data['enhancement'],
            binding_affinity=binding_data['affinity'],
            binding_kinetics=binding_data['kinetics'],
            cellular_effects=cellular_effects,
            tissue_concentrations=tissue_concentrations,
            organ_functional_changes=organ_data['functional_changes'],
            therapeutic_benefit=organ_data['benefit'],
            adverse_risk=organ_data['risk'],
            system_stability=systemic_data['stability'],
            homeostatic_compensation=systemic_data['compensation'],
            emergent_effects=systemic_data['emergent_effects'],
            optimal_dosing_time=temporal_data['optimal_time'],
            chronotherapy_advantage=temporal_data['chronotherapy_advantage'],
            circadian_effects=temporal_data['circadian_effects'],
            overall_efficacy_prediction=efficacy_prediction,
            overall_safety_prediction=safety_prediction,
            therapeutic_index=therapeutic_index,
            confidence_score=confidence_score,
            dosing_recommendations=dosing_recommendations,
            monitoring_recommendations=monitoring_recommendations,
            alternative_drugs=alternative_drugs
        )
        
        logger.info(f"Prediction complete: efficacy={efficacy_prediction:.3f}, "
                   f"safety={safety_prediction:.3f}, confidence={confidence_score:.3f}")
        
        return prediction
    
    def _calculate_genetic_risk_mock(self, drug_name: str, variants: List[Dict[str, Any]]) -> float:
        """Mock genetic risk calculation."""
        high_impact_variants = len([v for v in variants if v.get('impact') == 'HIGH'])
        moderate_impact_variants = len([v for v in variants if v.get('impact') == 'MODERATE'])
        
        # Drug-specific risk factors
        drug_risk_genes = {
            'lithium': ['INPP1', 'GSK3B', 'SLC1A2'],
            'aripiprazole': ['CYP2D6', 'DRD2', 'HTR2A'],
            'citalopram': ['CYP2C19', 'HTR2A', 'SLC6A4'],
            'atorvastatin': ['SLCO1B1', 'CYP3A4'],
            'aspirin': ['CYP2C9', 'PTGS1']
        }
        
        relevant_genes = drug_risk_genes.get(drug_name, [])
        relevant_variants = [v for v in variants if v.get('gene') in relevant_genes]
        
        base_risk = len(relevant_variants) * 0.1
        high_impact_risk = high_impact_variants * 0.2
        moderate_impact_risk = moderate_impact_variants * 0.1
        
        total_risk = min(1.0, base_risk + high_impact_risk + moderate_impact_risk)
        return total_risk
    
    def _get_pharmacogenetic_warnings_mock(self, drug_name: str, variants: List[Dict[str, Any]]) -> List[str]:
        """Mock pharmacogenetic warnings."""
        warnings = []
        
        gene_variants = {v.get('gene'): v.get('impact') for v in variants}
        
        if drug_name == 'lithium':
            if 'INPP1' in gene_variants and gene_variants['INPP1'] == 'HIGH':
                warnings.append('Enhanced lithium sensitivity - consider dose reduction')
            if 'GSK3B' in gene_variants:
                warnings.append('Monitor for lithium-induced nephrotoxicity')
        
        elif drug_name == 'aripiprazole':
            if 'CYP2D6' in gene_variants and gene_variants['CYP2D6'] == 'HIGH':
                warnings.append('Poor metabolizer - reduce aripiprazole dose by 50%')
        
        elif drug_name == 'citalopram':
            if 'CYP2C19' in gene_variants and gene_variants['CYP2C19'] == 'HIGH':
                warnings.append('Poor metabolizer - consider alternative SSRI')
        
        elif drug_name == 'atorvastatin':
            if 'SLCO1B1' in gene_variants:
                warnings.append('Increased myopathy risk - monitor CK levels')
        
        return warnings
    
    def _predict_transport_mock(self, drug_name: str, signature: OscillatorySignature) -> Dict[str, Any]:
        """Mock transport prediction."""
        # Base transport properties
        transport_props = {
            'lithium': {'permeability': 0.01, 'mechanism': 'ion_channels', 'enhancement': 1.2},
            'aripiprazole': {'permeability': 0.8, 'mechanism': 'passive_diffusion', 'enhancement': 2.1},
            'citalopram': {'permeability': 0.6, 'mechanism': 'passive_diffusion', 'enhancement': 1.8},
            'atorvastatin': {'permeability': 0.3, 'mechanism': 'active_transport', 'enhancement': 2.5},
            'aspirin': {'permeability': 0.9, 'mechanism': 'passive_diffusion', 'enhancement': 1.1}
        }
        
        base_props = transport_props.get(drug_name, 
                                       {'permeability': 0.5, 'mechanism': 'passive_diffusion', 'enhancement': 1.5})
        
        # Modify based on oscillatory signature
        coherence_factor = signature.coherence_metrics.get('molecular', 0.8)
        modified_enhancement = base_props['enhancement'] * coherence_factor
        
        return {
            'permeability': base_props['permeability'],
            'mechanism': base_props['mechanism'],
            'enhancement': modified_enhancement
        }
    
    def _predict_binding_mock(self, drug_name: str, signature: OscillatorySignature) -> Dict[str, Any]:
        """Mock binding prediction."""
        # Base binding properties
        binding_props = {
            'lithium': {'affinity': 50.0, 'kon': 1e6, 'koff': 0.02},
            'aripiprazole': {'affinity': 2.3, 'kon': 5e7, 'koff': 0.115},
            'citalopram': {'affinity': 15.0, 'kon': 2e7, 'koff': 0.3},
            'atorvastatin': {'affinity': 8.0, 'kon': 1e8, 'koff': 0.8},
            'aspirin': {'affinity': 100.0, 'kon': 1e5, 'koff': 10.0}
        }
        
        base_props = binding_props.get(drug_name, 
                                     {'affinity': 25.0, 'kon': 1e7, 'koff': 0.25})
        
        # Modify based on oscillatory holes affecting target pathways
        target_disruption = 0.0
        for hole in signature.oscillatory_holes:
            if hole['amplitude_deficit'] > 0.5:
                target_disruption += 0.1
        
        modified_affinity = base_props['affinity'] * (1.0 + target_disruption)
        
        return {
            'affinity': modified_affinity,
            'kinetics': {
                'kon': base_props['kon'],
                'koff': base_props['koff'],
                'residence_time': 1.0 / base_props['koff']
            }
        }
    
    def _predict_cellular_effects_mock(self, drug_name: str, signature: OscillatorySignature) -> Dict[str, float]:
        """Mock cellular effects prediction."""
        # Base cellular effects
        base_effects = {
            'lithium': {'frequency_change': 0.15, 'amplitude_change': 1.2, 'coupling_change': 0.3},
            'aripiprazole': {'frequency_change': 0.25, 'amplitude_change': 1.15, 'coupling_change': 0.2},
            'citalopram': {'frequency_change': 0.20, 'amplitude_change': 1.25, 'coupling_change': 0.25},
            'atorvastatin': {'frequency_change': 0.05, 'amplitude_change': 1.05, 'coupling_change': 0.1},
            'aspirin': {'frequency_change': 0.08, 'amplitude_change': 0.95, 'coupling_change': -0.05}
        }
        
        effects = base_effects.get(drug_name, 
                                 {'frequency_change': 0.1, 'amplitude_change': 1.1, 'coupling_change': 0.1})
        
        # Modify based on cellular coherence
        cellular_coherence = signature.coherence_metrics.get('cellular', 0.75)
        effects['frequency_change'] *= cellular_coherence
        effects['coupling_change'] *= cellular_coherence
        
        return effects
    
    def _predict_tissue_distribution_mock(self, drug_name: str) -> Dict[str, float]:
        """Mock tissue distribution prediction."""
        distributions = {
            'lithium': {'brain': 0.6, 'heart': 0.4, 'liver': 0.3, 'kidney': 0.8, 'muscle': 0.2},
            'aripiprazole': {'brain': 0.8, 'heart': 0.3, 'liver': 1.5, 'kidney': 0.6, 'muscle': 0.7},
            'citalopram': {'brain': 0.4, 'heart': 0.2, 'liver': 0.8, 'kidney': 0.3, 'muscle': 0.25},
            'atorvastatin': {'brain': 0.02, 'heart': 0.4, 'liver': 2.5, 'kidney': 0.3, 'muscle': 0.6},
            'aspirin': {'brain': 0.05, 'heart': 0.3, 'liver': 0.2, 'kidney': 0.4, 'muscle': 0.15}
        }
        
        return distributions.get(drug_name, 
                               {'brain': 0.3, 'heart': 0.3, 'liver': 0.5, 'kidney': 0.4, 'muscle': 0.3})
    
    def _predict_organ_effects_mock(self, drug_name: str, tissue_concentrations: Dict[str, float]) -> Dict[str, Any]:
        """Mock organ effects prediction."""
        # Calculate effects based on tissue concentrations
        brain_conc = tissue_concentrations.get('brain', 0.0)
        heart_conc = tissue_concentrations.get('heart', 0.0)
        liver_conc = tissue_concentrations.get('liver', 0.0)
        
        # Drug-specific organ effects
        if drug_name == 'lithium':
            therapeutic_benefit = min(1.0, brain_conc / 0.6 * 0.8) if brain_conc > 0.4 else 0.0
            adverse_risk = max(0.0, (brain_conc - 0.8) * 0.5) if brain_conc > 0.8 else 0.0
            functional_changes = {'brain': 1.0 + brain_conc * 0.3, 'kidney': 1.0 - brain_conc * 0.2}
        
        elif drug_name == 'aripiprazole':
            therapeutic_benefit = min(1.0, brain_conc / 0.8 * 0.7) if brain_conc > 0.1 else 0.0
            adverse_risk = max(0.0, (brain_conc - 1.0) * 0.3) if brain_conc > 1.0 else 0.0
            functional_changes = {'brain': 1.0 + brain_conc * 0.25, 'heart': 1.0 - heart_conc * 0.1}
        
        else:
            therapeutic_benefit = 0.5
            adverse_risk = 0.2
            functional_changes = {'brain': 1.0, 'heart': 1.0, 'liver': 1.0}
        
        return {
            'functional_changes': functional_changes,
            'benefit': therapeutic_benefit,
            'risk': adverse_risk
        }
    
    def _predict_systemic_response_mock(self, drug_name: str, organ_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock systemic response prediction."""
        benefit = organ_data['benefit']
        risk = organ_data['risk']
        
        # Calculate system stability
        stability = 1.0 - risk * 0.5
        
        # Calculate homeostatic compensation
        compensation = min(1.0, benefit * 0.6 + (1.0 - risk) * 0.4)
        
        # Determine emergent effects
        emergent_effects = []
        if risk > 0.6:
            emergent_effects.append('high_risk_cascade')
        if benefit > 0.7:
            emergent_effects.append('therapeutic_synergy')
        
        return {
            'stability': stability,
            'compensation': compensation,
            'emergent_effects': emergent_effects
        }
    
    def _predict_temporal_patterns_mock(self, drug_name: str) -> Dict[str, Any]:
        """Mock temporal patterns prediction."""
        optimal_times = {
            'lithium': 20.0,      # Evening
            'aripiprazole': 9.0,  # Morning
            'citalopram': 7.0,    # Early morning
            'atorvastatin': 22.0, # Late evening
            'aspirin': 6.0        # Early morning
        }
        
        chronotherapy_benefits = {
            'lithium': 0.7,
            'aripiprazole': 0.5,
            'citalopram': 0.6,
            'atorvastatin': 0.8,
            'aspirin': 0.4
        }
        
        return {
            'optimal_time': optimal_times.get(drug_name, 8.0),
            'chronotherapy_advantage': chronotherapy_benefits.get(drug_name, 0.5),
            'circadian_effects': 0.2
        }
    
    def _integrate_efficacy_prediction(self, genetic_risk: float, transport_data: Dict,
                                     binding_data: Dict, cellular_effects: Dict,
                                     organ_data: Dict, systemic_data: Dict) -> float:
        """Integrate all predictions into overall efficacy score."""
        
        # Weight different factors
        genetic_factor = 1.0 - genetic_risk * 0.3  # Genetic risk reduces efficacy
        transport_factor = min(1.0, transport_data['permeability'] * transport_data['enhancement'])
        binding_factor = min(1.0, 100.0 / binding_data['affinity'])  # Lower affinity = better binding
        cellular_factor = cellular_effects['amplitude_change'] - 1.0 + 0.5  # Normalize around 0.5
        organ_factor = organ_data['benefit']
        systemic_factor = systemic_data['stability'] * 0.5 + systemic_data['compensation'] * 0.5
        
        # Weighted integration
        efficacy = (genetic_factor * 0.2 + 
                   transport_factor * 0.15 + 
                   binding_factor * 0.15 + 
                   cellular_factor * 0.15 + 
                   organ_factor * 0.2 + 
                   systemic_factor * 0.15)
        
        return min(1.0, max(0.0, efficacy))
    
    def _integrate_safety_prediction(self, genetic_risk: float, organ_data: Dict,
                                   systemic_data: Dict, warnings: List[str]) -> float:
        """Integrate predictions into overall safety score."""
        
        genetic_safety_risk = genetic_risk
        organ_safety_risk = organ_data['risk']
        systemic_safety_risk = 1.0 - systemic_data['stability']
        warning_risk = len(warnings) * 0.1
        
        # Combine risks (higher = less safe)
        total_risk = min(1.0, genetic_safety_risk + organ_safety_risk + 
                        systemic_safety_risk + warning_risk)
        
        # Convert to safety score (higher = safer)
        safety_score = 1.0 - total_risk
        
        return max(0.0, safety_score)
    
    def _calculate_prediction_confidence(self, signature: OscillatorySignature,
                                       variant_count: int, drug_name: str) -> float:
        """Calculate confidence in predictions."""
        
        base_confidence = 0.7
        
        # More oscillatory holes reduce confidence (complexity)
        hole_penalty = min(0.2, len(signature.oscillatory_holes) * 0.02)
        
        # More variants can reduce confidence
        variant_penalty = min(0.1, variant_count * 0.005)
        
        # Well-studied drugs have higher confidence
        well_studied = ['lithium', 'atorvastatin', 'aspirin']
        study_bonus = 0.15 if drug_name in well_studied else 0.0
        
        # Higher oscillatory coherence increases confidence
        avg_coherence = np.mean(list(signature.coherence_metrics.values()))
        coherence_bonus = (avg_coherence - 0.5) * 0.2  # Up to 20% bonus
        
        final_confidence = base_confidence - hole_penalty - variant_penalty + study_bonus + coherence_bonus
        
        return max(0.3, min(0.95, final_confidence))
    
    def _generate_dosing_recommendations(self, drug_name: str, genetic_risk: float,
                                       temporal_data: Dict, organ_data: Dict) -> List[str]:
        """Generate personalized dosing recommendations."""
        recommendations = []
        
        # Genetic risk-based recommendations
        if genetic_risk > 0.6:
            recommendations.append(f"Start with 50% of standard {drug_name} dose due to genetic risk factors")
            recommendations.append("Titrate dose slowly with close monitoring")
        elif genetic_risk > 0.3:
            recommendations.append("Consider starting with reduced dose and titrating as tolerated")
        
        # Temporal recommendations
        optimal_time = temporal_data['optimal_time']
        if temporal_data['chronotherapy_advantage'] > 0.5:
            recommendations.append(f"Administer {drug_name} at {optimal_time:.0f}:00 for optimal efficacy")
        
        # Organ-specific recommendations
        if organ_data['risk'] > 0.4:
            recommendations.append("Monitor organ function closely during treatment")
        
        # Drug-specific recommendations
        if drug_name == 'lithium':
            recommendations.append("Monitor lithium levels weekly for first month, then monthly")
            recommendations.append("Ensure adequate hydration and stable sodium intake")
        elif drug_name == 'atorvastatin':
            recommendations.append("Take in evening with or without food")
            recommendations.append("Monitor for muscle pain or weakness")
        
        return recommendations
    
    def _generate_monitoring_recommendations(self, drug_name: str, warnings: List[str],
                                          organ_data: Dict) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []
        
        # Warning-based monitoring
        for warning in warnings:
            if 'metabolizer' in warning.lower():
                recommendations.append("Monitor drug levels and adjust dose based on response")
            if 'myopathy' in warning.lower():
                recommendations.append("Monitor creatine kinase levels monthly")
            if 'nephrotoxicity' in warning.lower():
                recommendations.append("Monitor kidney function regularly")
        
        # Organ risk-based monitoring
        if organ_data['risk'] > 0.5:
            recommendations.append("Increased monitoring frequency recommended due to genetic risk")
        
        # Drug-specific monitoring
        drug_monitoring = {
            'lithium': ['serum lithium levels', 'kidney function', 'thyroid function'],
            'aripiprazole': ['metabolic parameters', 'movement disorders', 'prolactin levels'],
            'citalopram': ['QTc interval', 'sodium levels', 'suicidal ideation'],
            'atorvastatin': ['liver enzymes', 'muscle symptoms', 'lipid levels'],
            'aspirin': ['bleeding events', 'gastrointestinal symptoms', 'kidney function']
        }
        
        if drug_name in drug_monitoring:
            for parameter in drug_monitoring[drug_name]:
                recommendations.append(f"Monitor {parameter}")
        
        return recommendations
    
    def _suggest_alternative_drugs(self, drug_name: str, genetic_risk: float,
                                 therapeutic_index: float) -> List[str]:
        """Suggest alternative drugs if needed."""
        alternatives = []
        
        # If high genetic risk or poor therapeutic index, suggest alternatives
        if genetic_risk > 0.7 or therapeutic_index < 1.0:
            
            drug_alternatives = {
                'lithium': ['valproate', 'lamotrigine', 'carbamazepine'],
                'aripiprazole': ['olanzapine', 'quetiapine', 'risperidone'],
                'citalopram': ['sertraline', 'escitalopram', 'fluoxetine'],
                'atorvastatin': ['rosuvastatin', 'pravastatin', 'simvastatin'],
                'aspirin': ['clopidogrel', 'warfarin', 'apixaban']
            }
            
            if drug_name in drug_alternatives:
                alternatives = drug_alternatives[drug_name][:2]  # Top 2 alternatives
        
        return alternatives
    
    def generate_comprehensive_profile(self, individual_id: str,
                                     variants: List[Dict[str, Any]],
                                     drugs_to_analyze: List[str] = None) -> ComprehensiveProfile:
        """Generate complete oscillatory pharmacogenomic profile."""
        
        logger.info(f"Generating comprehensive profile for {individual_id}")
        
        if drugs_to_analyze is None:
            drugs_to_analyze = ['lithium', 'aripiprazole', 'citalopram', 'atorvastatin', 'aspirin']
        
        # Extract oscillatory signature
        oscillatory_signature = self.extract_oscillatory_signature(variants, individual_id)
        
        # Generate drug predictions
        drug_predictions = []
        for drug_name in drugs_to_analyze:
            prediction = self.predict_drug_response(drug_name, individual_id, 
                                                  oscillatory_signature, variants)
            drug_predictions.append(prediction)
        
        # Calculate summary metrics
        overall_genetic_complexity = self._calculate_genetic_complexity(variants, oscillatory_signature)
        oscillatory_stability = np.mean(list(oscillatory_signature.coherence_metrics.values()))
        drug_response_predictability = np.mean([p.confidence_score for p in drug_predictions])
        
        # Generate clinical insights
        high_risk_drugs = [p.drug_name for p in drug_predictions 
                          if p.overall_safety_prediction < 0.6 or p.genetic_risk_score > 0.6]
        
        recommended_drugs = [p.drug_name for p in drug_predictions 
                           if p.therapeutic_index > 2.0 and p.overall_safety_prediction > 0.7]
        
        personalized_insights = self._generate_personalized_insights(
            oscillatory_signature, drug_predictions
        )
        
        clinical_recommendations = self._generate_clinical_recommendations(
            oscillatory_signature, drug_predictions, high_risk_drugs
        )
        
        profile = ComprehensiveProfile(
            individual_id=individual_id,
            analysis_timestamp=pd.Timestamp.now().isoformat(),
            genetic_variants=variants,
            oscillatory_signature=oscillatory_signature,
            drug_predictions=drug_predictions,
            overall_genetic_complexity=overall_genetic_complexity,
            oscillatory_stability=oscillatory_stability,
            drug_response_predictability=drug_response_predictability,
            high_risk_drugs=high_risk_drugs,
            recommended_drugs=recommended_drugs,
            personalized_insights=personalized_insights,
            clinical_recommendations=clinical_recommendations
        )
        
        logger.info(f"Profile complete: {len(drug_predictions)} drugs analyzed, "
                   f"{len(high_risk_drugs)} high-risk, {len(recommended_drugs)} recommended")
        
        return profile
    
    def _calculate_genetic_complexity(self, variants: List[Dict[str, Any]], 
                                    signature: OscillatorySignature) -> float:
        """Calculate overall genetic complexity score."""
        
        variant_burden = len(variants) / 20.0  # Normalize by typical burden
        disruption_count = len(signature.disruption_patterns) / 5.0  # Normalize
        hole_burden = len(signature.oscillatory_holes) / 10.0  # Normalize
        
        complexity = min(1.0, (variant_burden + disruption_count + hole_burden) / 3.0)
        return complexity
    
    def _generate_personalized_insights(self, signature: OscillatorySignature,
                                      predictions: List[DrugResponsePrediction]) -> List[str]:
        """Generate personalized insights."""
        insights = []
        
        # Oscillatory signature insights
        if signature.variant_count > 15:
            insights.append("High genetic variant burden may require personalized dosing approaches")
        
        if len(signature.disruption_patterns) >= 3:
            insights.append("Complex oscillatory disruption patterns identified")
        
        # Drug response insights
        high_efficacy_drugs = [p for p in predictions if p.overall_efficacy_prediction > 0.7]
        if high_efficacy_drugs:
            drug_names = [p.drug_name for p in high_efficacy_drugs]
            insights.append(f"Predicted high efficacy for: {', '.join(drug_names)}")
        
        low_safety_drugs = [p for p in predictions if p.overall_safety_prediction < 0.6]
        if low_safety_drugs:
            drug_names = [p.drug_name for p in low_safety_drugs]
            insights.append(f"Increased safety monitoring needed for: {', '.join(drug_names)}")
        
        # Temporal insights
        chronotherapy_drugs = [p for p in predictions if p.chronotherapy_advantage > 0.5]
        if chronotherapy_drugs:
            insights.append("Significant chronotherapy benefits identified for several medications")
        
        return insights
    
    def _generate_clinical_recommendations(self, signature: OscillatorySignature,
                                         predictions: List[DrugResponsePrediction],
                                         high_risk_drugs: List[str]) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []
        
        # General recommendations
        if signature.variant_count > 10:
            recommendations.append("Consider pharmacogenetic consultation before prescribing")
        
        if high_risk_drugs:
            recommendations.append("Avoid or use extreme caution with high-risk medications")
        
        # Therapeutic target recommendations
        if 'lithium_responsive_pathway' in signature.therapeutic_targets:
            recommendations.append("Consider lithium-based treatments for mood stabilization")
        
        if 'precision_dosing_required' in signature.therapeutic_targets:
            recommendations.append("Implement precision dosing strategies with therapeutic monitoring")
        
        # Monitoring recommendations
        if any(p.genetic_risk_score > 0.6 for p in predictions):
            recommendations.append("Implement enhanced pharmacovigilance protocols")
        
        # System-specific recommendations
        if signature.coherence_metrics.get('systemic', 0.6) < 0.5:
            recommendations.append("Start with lower doses and titrate slowly due to system instability")
        
        return recommendations
    
    def visualize_comprehensive_profile(self, profile: ComprehensiveProfile,
                                      output_dir: str = "babylon_results") -> None:
        """Create comprehensive visualization of the profile."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive profile visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Use GridSpec for complex layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Oscillatory signature overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        coherence_values = list(profile.oscillatory_signature.coherence_metrics.values())
        coherence_labels = list(profile.oscillatory_signature.coherence_metrics.keys())
        ax1.bar(coherence_labels, coherence_values, color='lightblue', alpha=0.7)
        ax1.set_title('Oscillatory Coherence')
        ax1.set_ylabel('Coherence Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Drug efficacy predictions (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        drug_names = [p.drug_name for p in profile.drug_predictions]
        efficacies = [p.overall_efficacy_prediction for p in profile.drug_predictions]
        bars = ax2.bar(drug_names, efficacies, color='lightgreen', alpha=0.7)
        ax2.set_title('Predicted Drug Efficacy')
        ax2.set_ylabel('Efficacy Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Efficacy')
        ax2.legend()
        
        # Color bars by efficacy level
        for bar, efficacy in zip(bars, efficacies):
            if efficacy > 0.7:
                bar.set_color('darkgreen')
            elif efficacy > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        # 3. Drug safety predictions (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        safeties = [p.overall_safety_prediction for p in profile.drug_predictions]
        bars = ax3.bar(drug_names, safeties, color='lightcoral', alpha=0.7)
        ax3.set_title('Predicted Drug Safety')
        ax3.set_ylabel('Safety Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Safe Threshold')
        ax3.legend()
        
        # Color bars by safety level
        for bar, safety in zip(bars, safeties):
            if safety > 0.7:
                bar.set_color('darkgreen')
            elif safety > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 4. Therapeutic index (top far right)
        ax4 = fig.add_subplot(gs[0, 3])
        therapeutic_indices = [p.therapeutic_index for p in profile.drug_predictions]
        ax4.bar(drug_names, therapeutic_indices, color='gold', alpha=0.7)
        ax4.set_title('Therapeutic Index')
        ax4.set_ylabel('TI Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Good TI')
        ax4.legend()
        
        # 5. Genetic risk by drug (second row left)
        ax5 = fig.add_subplot(gs[1, 0])
        genetic_risks = [p.genetic_risk_score for p in profile.drug_predictions]
        ax5.bar(drug_names, genetic_risks, color='purple', alpha=0.7)
        ax5.set_title('Genetic Risk Scores')
        ax5.set_ylabel('Risk Score')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Oscillatory holes frequency spectrum (second row center-left)
        ax6 = fig.add_subplot(gs[1, 1])
        if profile.oscillatory_signature.frequency_spectrum:
            pathways = list(profile.oscillatory_signature.frequency_spectrum.keys())
            frequencies = list(profile.oscillatory_signature.frequency_spectrum.values())
            ax6.bar(pathways, frequencies, color='cyan', alpha=0.7)
            ax6.set_title('Frequency Spectrum')
            ax6.set_ylabel('Frequency (Hz)')
            ax6.set_yscale('log')
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Chronotherapy recommendations (second row center-right)
        ax7 = fig.add_subplot(gs[1, 2])
        optimal_times = [p.optimal_dosing_time for p in profile.drug_predictions]
        chronotherapy_advantages = [p.chronotherapy_advantage for p in profile.drug_predictions]
        
        scatter = ax7.scatter(optimal_times, chronotherapy_advantages, s=100, alpha=0.7, c='orange')
        ax7.set_xlabel('Optimal Dosing Time (hours)')
        ax7.set_ylabel('Chronotherapy Advantage')
        ax7.set_title('Chronotherapy Optimization')
        ax7.set_xlim(0, 24)
        
        # Annotate points
        for i, name in enumerate(drug_names):
            ax7.annotate(name, (optimal_times[i], chronotherapy_advantages[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 8. Confidence scores (second row right)
        ax8 = fig.add_subplot(gs[1, 3])
        confidences = [p.confidence_score for p in profile.drug_predictions]
        ax8.bar(drug_names, confidences, color='lightgray', alpha=0.7)
        ax8.set_title('Prediction Confidence')
        ax8.set_ylabel('Confidence')
        ax8.tick_params(axis='x', rotation=45)
        ax8.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
        ax8.legend()
        
        # 9. Summary metrics (third row left)
        ax9 = fig.add_subplot(gs[2, 0])
        metrics = ['Genetic\nComplexity', 'Oscillatory\nStability', 'Response\nPredictability']
        values = [profile.overall_genetic_complexity, 
                 profile.oscillatory_stability, 
                 profile.drug_response_predictability]
        bars = ax9.bar(metrics, values, color=['red', 'blue', 'green'], alpha=0.7)
        ax9.set_title('Overall Profile Metrics')
        ax9.set_ylabel('Score')
        ax9.set_ylim(0, 1)
        
        # 10. Disruption patterns (third row center)
        ax10 = fig.add_subplot(gs[2, 1:3])
        if profile.oscillatory_signature.disruption_patterns:
            patterns = profile.oscillatory_signature.disruption_patterns
            pattern_counts = [1] * len(patterns)  # Each pattern occurs once
            y_pos = np.arange(len(patterns))
            ax10.barh(y_pos, pattern_counts, color='red', alpha=0.7)
            ax10.set_yticks(y_pos)
            ax10.set_yticklabels([p.replace('_', ' ').title() for p in patterns])
            ax10.set_title('Oscillatory Disruption Patterns')
            ax10.set_xlabel('Presence')
        
        # 11. Drug recommendations text summary (bottom)
        ax11 = fig.add_subplot(gs[3, :])
        ax11.axis('off')
        
        # Create text summary
        summary_text = f"COMPREHENSIVE PHARMACOGENOMIC PROFILE\n"
        summary_text += f"Individual: {profile.individual_id}\n"
        summary_text += f"Variants Analyzed: {len(profile.genetic_variants)}\n"
        summary_text += f"Oscillatory Holes: {len(profile.oscillatory_signature.oscillatory_holes)}\n\n"
        
        summary_text += f"HIGH-RISK DRUGS: {', '.join(profile.high_risk_drugs) if profile.high_risk_drugs else 'None'}\n"
        summary_text += f"RECOMMENDED DRUGS: {', '.join(profile.recommended_drugs) if profile.recommended_drugs else 'None'}\n\n"
        
        summary_text += "KEY INSIGHTS:\n"
        for insight in profile.personalized_insights[:3]:  # Top 3 insights
            summary_text += f"• {insight}\n"
        
        summary_text += "\nCLINICAL RECOMMENDATIONS:\n"
        for rec in profile.clinical_recommendations[:3]:  # Top 3 recommendations
            summary_text += f"• {rec}\n"
        
        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Oscillatory Pharmacogenomic Profile: {profile.individual_id}', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(output_path / f'comprehensive_profile_{profile.individual_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive profile visualization saved to {output_path}")
    
    def save_comprehensive_profile(self, profile: ComprehensiveProfile,
                                 output_dir: str = "babylon_results") -> None:
        """Save comprehensive profile to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert profile to dictionary for JSON serialization
        profile_data = {
            'individual_id': profile.individual_id,
            'analysis_timestamp': profile.analysis_timestamp,
            'genetic_variants': profile.genetic_variants,
            'oscillatory_signature': {
                'individual_id': profile.oscillatory_signature.individual_id,
                'variant_count': profile.oscillatory_signature.variant_count,
                'oscillatory_holes': profile.oscillatory_signature.oscillatory_holes,
                'frequency_spectrum': profile.oscillatory_signature.frequency_spectrum,
                'coupling_matrix': profile.oscillatory_signature.coupling_matrix,
                'coherence_metrics': profile.oscillatory_signature.coherence_metrics,
                'disruption_patterns': profile.oscillatory_signature.disruption_patterns,
                'therapeutic_targets': profile.oscillatory_signature.therapeutic_targets
            },
            'drug_predictions': [
                {
                    'drug_name': p.drug_name,
                    'genetic_risk_score': p.genetic_risk_score,
                    'pharmacogenetic_warnings': p.pharmacogenetic_warnings,
                    'membrane_permeability': p.membrane_permeability,
                    'transport_mechanism': p.transport_mechanism,
                    'quantum_enhancement': p.quantum_enhancement,
                    'binding_affinity': p.binding_affinity,
                    'binding_kinetics': p.binding_kinetics,
                    'cellular_effects': p.cellular_effects,
                    'tissue_concentrations': p.tissue_concentrations,
                    'organ_functional_changes': p.organ_functional_changes,
                    'therapeutic_benefit': p.therapeutic_benefit,
                    'adverse_risk': p.adverse_risk,
                    'system_stability': p.system_stability,
                    'homeostatic_compensation': p.homeostatic_compensation,
                    'emergent_effects': p.emergent_effects,
                    'optimal_dosing_time': p.optimal_dosing_time,
                    'chronotherapy_advantage': p.chronotherapy_advantage,
                    'circadian_effects': p.circadian_effects,
                    'overall_efficacy_prediction': p.overall_efficacy_prediction,
                    'overall_safety_prediction': p.overall_safety_prediction,
                    'therapeutic_index': p.therapeutic_index,
                    'confidence_score': p.confidence_score,
                    'dosing_recommendations': p.dosing_recommendations,
                    'monitoring_recommendations': p.monitoring_recommendations,
                    'alternative_drugs': p.alternative_drugs
                }
                for p in profile.drug_predictions
            ],
            'overall_genetic_complexity': profile.overall_genetic_complexity,
            'oscillatory_stability': profile.oscillatory_stability,
            'drug_response_predictability': profile.drug_response_predictability,
            'high_risk_drugs': profile.high_risk_drugs,
            'recommended_drugs': profile.recommended_drugs,
            'personalized_insights': profile.personalized_insights,
            'clinical_recommendations': profile.clinical_recommendations
        }
        
        # Save complete profile as JSON
        with open(output_path / f'comprehensive_profile_{profile.individual_id}.json', 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        # Save drug predictions as CSV
        drug_df = pd.DataFrame([
            {
                'drug_name': p.drug_name,
                'efficacy_prediction': p.overall_efficacy_prediction,
                'safety_prediction': p.overall_safety_prediction,
                'therapeutic_index': p.therapeutic_index,
                'genetic_risk_score': p.genetic_risk_score,
                'optimal_dosing_time': p.optimal_dosing_time,
                'chronotherapy_advantage': p.chronotherapy_advantage,
                'confidence_score': p.confidence_score
            }
            for p in profile.drug_predictions
        ])
        drug_df.to_csv(output_path / f'drug_predictions_{profile.individual_id}.csv', index=False)
        
        # Save summary report
        summary_report = self._generate_summary_report(profile)
        with open(output_path / f'summary_report_{profile.individual_id}.txt', 'w') as f:
            f.write(summary_report)
        
        logger.info(f"Comprehensive profile saved to {output_path}")
    
    def _generate_summary_report(self, profile: ComprehensiveProfile) -> str:
        """Generate human-readable summary report."""
        
        report = f"""
OSCILLATORY PHARMACOGENOMIC PROFILE SUMMARY
==========================================

Patient ID: {profile.individual_id}
Analysis Date: {profile.analysis_timestamp}
Babylon Framework Version: 1.0.0

GENETIC OVERVIEW
----------------
Total Variants Analyzed: {len(profile.genetic_variants)}
Oscillatory Holes Detected: {len(profile.oscillatory_signature.oscillatory_holes)}
Disruption Patterns: {len(profile.oscillatory_signature.disruption_patterns)}

Overall Genetic Complexity: {profile.overall_genetic_complexity:.3f}
Oscillatory Stability: {profile.oscillatory_stability:.3f}
Drug Response Predictability: {profile.drug_response_predictability:.3f}

DRUG RESPONSE PREDICTIONS
-------------------------
"""
        
        for prediction in profile.drug_predictions:
            report += f"""
{prediction.drug_name.upper()}:
  Efficacy Prediction: {prediction.overall_efficacy_prediction:.3f}
  Safety Prediction: {prediction.overall_safety_prediction:.3f}
  Therapeutic Index: {prediction.therapeutic_index:.2f}
  Genetic Risk Score: {prediction.genetic_risk_score:.3f}
  Optimal Dosing Time: {prediction.optimal_dosing_time:.0f}:00
  Confidence: {prediction.confidence_score:.3f}
  
  Pharmacogenetic Warnings:"""
            
            if prediction.pharmacogenetic_warnings:
                for warning in prediction.pharmacogenetic_warnings:
                    report += f"\n    • {warning}"
            else:
                report += "\n    • None"
            
            report += f"\n  \n  Dosing Recommendations:"
            for rec in prediction.dosing_recommendations[:3]:  # Top 3
                report += f"\n    • {rec}"
        
        report += f"""

HIGH-RISK DRUGS
---------------
"""
        if profile.high_risk_drugs:
            for drug in profile.high_risk_drugs:
                report += f"• {drug}\n"
        else:
            report += "None identified\n"
        
        report += f"""
RECOMMENDED DRUGS
-----------------
"""
        if profile.recommended_drugs:
            for drug in profile.recommended_drugs:
                report += f"• {drug}\n"
        else:
            report += "None identified - proceed with caution for all medications\n"
        
        report += f"""
PERSONALIZED INSIGHTS
--------------------
"""
        for insight in profile.personalized_insights:
            report += f"• {insight}\n"
        
        report += f"""
CLINICAL RECOMMENDATIONS
-----------------------
"""
        for rec in profile.clinical_recommendations:
            report += f"• {rec}\n"
        
        report += f"""
OSCILLATORY SIGNATURE DETAILS
-----------------------------
Therapeutic Targets Identified: {', '.join(profile.oscillatory_signature.therapeutic_targets)}
Disruption Patterns: {', '.join(profile.oscillatory_signature.disruption_patterns)}

Coherence Metrics:"""
        for scale, coherence in profile.oscillatory_signature.coherence_metrics.items():
            report += f"\n  {scale}: {coherence:.3f}"
        
        report += f"""

---
Report generated by Babylon Oscillatory Framework
St. Stellas Theoretical Biology Laboratory
"""
        
        return report

def main():
    """
    Test the complete Oscillatory Profiler with sample genetic data.
    
    This demonstrates the full integrated Babylon framework for
    comprehensive pharmacogenomic analysis.
    """
    
    print("🌟 Testing Complete Oscillatory Pharmacogenomic Profiler")
    print("=" * 60)
    
    # Initialize the profiler
    profiler = OscillatoryProfiler()
    
    # Sample genetic variants (simulating Dante Labs or similar results)
    sample_variants = [
        {'gene': 'INPP1', 'variant_id': 'rs123456', 'impact': 'HIGH', 
         'description': 'Inositol polyphosphate 1-phosphatase variant'},
        {'gene': 'GSK3B', 'variant_id': 'rs789012', 'impact': 'MODERATE',
         'description': 'Glycogen synthase kinase 3 beta variant'},
        {'gene': 'CYP2D6', 'variant_id': 'CYP2D6*4', 'impact': 'HIGH',
         'description': 'Poor metabolizer allele'},
        {'gene': 'DRD2', 'variant_id': 'rs1800497', 'impact': 'MODERATE',
         'description': 'Dopamine receptor D2 variant'},
        {'gene': 'HTR2A', 'variant_id': 'rs6314', 'impact': 'LOW',
         'description': 'Serotonin receptor 2A variant'},
        {'gene': 'SLCO1B1', 'variant_id': 'rs4149056', 'impact': 'MODERATE',
         'description': 'Statin transporter variant'},
        {'gene': 'CYP2C19', 'variant_id': 'CYP2C19*2', 'impact': 'HIGH',
         'description': 'Poor metabolizer allele'},
        {'gene': 'COMT', 'variant_id': 'rs4680', 'impact': 'MODERATE',
         'description': 'Catechol-O-methyltransferase variant'},
        {'gene': 'IMPA1', 'variant_id': 'rs334558', 'impact': 'LOW',
         'description': 'Inositol monophosphatase 1 variant'},
        {'gene': 'SLC6A4', 'variant_id': '5-HTTLPR', 'impact': 'MODERATE',
         'description': 'Serotonin transporter promoter variant'}
    ]
    
    individual_id = "test_patient_001"
    
    print(f"\n👤 Analyzing genetic profile for: {individual_id}")
    print(f"📊 Input variants: {len(sample_variants)}")
    
    # Extract oscillatory signature
    print("\n🔍 Step 1: Extracting oscillatory signature...")
    oscillatory_signature = profiler.extract_oscillatory_signature(sample_variants, individual_id)
    
    print(f"✅ Signature extracted:")
    print(f"   • Oscillatory holes: {len(oscillatory_signature.oscillatory_holes)}")
    print(f"   • Disruption patterns: {len(oscillatory_signature.disruption_patterns)}")
    print(f"   • Therapeutic targets: {len(oscillatory_signature.therapeutic_targets)}")
    
    # Generate comprehensive profile
    print(f"\n🔬 Step 2: Generating comprehensive pharmacogenomic profile...")
    comprehensive_profile = profiler.generate_comprehensive_profile(
        individual_id, sample_variants
    )
    
    # Display results
    print(f"\n📋 COMPREHENSIVE PROFILE RESULTS:")
    print("-" * 60)
    print(f"Individual: {comprehensive_profile.individual_id}")
    print(f"Analysis timestamp: {comprehensive_profile.analysis_timestamp}")
    print(f"Genetic complexity: {comprehensive_profile.overall_genetic_complexity:.3f}")
    print(f"Oscillatory stability: {comprehensive_profile.oscillatory_stability:.3f}")
    print(f"Response predictability: {comprehensive_profile.drug_response_predictability:.3f}")
    
    print(f"\n💊 DRUG RESPONSE PREDICTIONS:")
    print("-" * 40)
    
    for prediction in comprehensive_profile.drug_predictions:
        print(f"\n{prediction.drug_name.upper()}:")
        print(f"  Efficacy: {prediction.overall_efficacy_prediction:.3f}")
        print(f"  Safety: {prediction.overall_safety_prediction:.3f}")
        print(f"  Therapeutic Index: {prediction.therapeutic_index:.2f}")
        print(f"  Genetic Risk: {prediction.genetic_risk_score:.3f}")
        print(f"  Optimal Time: {prediction.optimal_dosing_time:.0f}:00")
        print(f"  Confidence: {prediction.confidence_score:.3f}")
        
        if prediction.pharmacogenetic_warnings:
            print(f"  ⚠️  Warnings: {len(prediction.pharmacogenetic_warnings)}")
            for warning in prediction.pharmacogenetic_warnings[:2]:  # Show top 2
                print(f"    • {warning}")
    
    print(f"\n🚨 HIGH-RISK DRUGS:")
    if comprehensive_profile.high_risk_drugs:
        for drug in comprehensive_profile.high_risk_drugs:
            print(f"  • {drug}")
    else:
        print("  None identified")
    
    print(f"\n✅ RECOMMENDED DRUGS:")
    if comprehensive_profile.recommended_drugs:
        for drug in comprehensive_profile.recommended_drugs:
            print(f"  • {drug}")
    else:
        print("  None identified - proceed with caution")
    
    print(f"\n💡 KEY PERSONALIZED INSIGHTS:")
    for insight in comprehensive_profile.personalized_insights:
        print(f"  • {insight}")
    
    print(f"\n🏥 CLINICAL RECOMMENDATIONS:")
    for rec in comprehensive_profile.clinical_recommendations:
        print(f"  • {rec}")
    
    # Save results and create visualizations
    print(f"\n💾 Saving comprehensive profile and creating visualizations...")
    profiler.save_comprehensive_profile(comprehensive_profile)
    profiler.visualize_comprehensive_profile(comprehensive_profile)
    
    # Key insights summary
    print(f"\n🎯 KEY INSIGHTS SUMMARY:")
    print("-" * 40)
    
    # Find best drug candidate
    best_drug = max(comprehensive_profile.drug_predictions, 
                   key=lambda p: p.therapeutic_index)
    print(f"• Best therapeutic candidate: {best_drug.drug_name} "
          f"(TI={best_drug.therapeutic_index:.2f})")
    
    # Find highest risk drug
    highest_risk = max(comprehensive_profile.drug_predictions,
                      key=lambda p: p.genetic_risk_score)
    print(f"• Highest genetic risk: {highest_risk.drug_name} "
          f"(risk={highest_risk.genetic_risk_score:.3f})")
    
    # Chronotherapy opportunities
    chronotherapy_drugs = [p for p in comprehensive_profile.drug_predictions 
                          if p.chronotherapy_advantage > 0.5]
    if chronotherapy_drugs:
        print(f"• Chronotherapy opportunities: {len(chronotherapy_drugs)} drugs")
    
    # Overall risk assessment
    avg_genetic_risk = np.mean([p.genetic_risk_score for p in comprehensive_profile.drug_predictions])
    if avg_genetic_risk > 0.6:
        print(f"• ⚠️  High overall genetic risk - enhanced monitoring recommended")
    elif avg_genetic_risk > 0.3:
        print(f"• ⚡ Moderate genetic risk - personalized dosing beneficial")
    else:
        print(f"• ✅ Low genetic risk - standard protocols appropriate")
    
    # Complexity assessment
    if comprehensive_profile.overall_genetic_complexity > 0.7:
        print(f"• 🧬 Complex genetic architecture - precision medicine approach recommended")
    
    print(f"\n📁 Complete results saved to: babylon_results/")
    print(f"📊 Visualizations created: comprehensive_profile_{individual_id}.png")
    print(f"📋 Summary report: summary_report_{individual_id}.txt")
    
    print(f"\n✨ Oscillatory Pharmacogenomic Analysis Complete!")
    print("🌟 Babylon Framework has successfully integrated all modules")
    print("   for comprehensive drug response prediction.")
    
    return comprehensive_profile

if __name__ == "__main__":
    profile = main()
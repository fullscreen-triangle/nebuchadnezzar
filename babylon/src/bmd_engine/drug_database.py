"""
Drug Database Module
===================

Central database of pharmaceutical compounds with their molecular properties,
oscillatory frequencies, and BMD (Biological Maxwell Demon) characteristics.
Manages drug information for the oscillatory framework analysis.

Part of the St. Stellas BMD Engine for therapeutic prediction.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DrugProperties:
    """Comprehensive drug properties for oscillatory analysis."""
    name: str
    chemical_formula: str
    molecular_weight: float  # Da
    
    # Physical properties
    logp: float              # Lipophilicity
    pka: float              # Acid dissociation constant
    volume: float           # Molecular volume (Ų)
    surface_area: float     # Surface area (Ų)
    
    # Oscillatory properties
    primary_frequency: float      # Hz - main oscillatory frequency
    secondary_frequencies: List[float]  # Hz - harmonics
    frequency_bandwidth: float    # Hz - frequency spread
    oscillatory_amplitude: float  # Relative amplitude
    
    # BMD characteristics
    information_capacity: float   # bits - information processing capacity
    entropy_reduction: float      # J/K - thermodynamic ordering capability
    atp_coupling_efficiency: float  # 0-1 - efficiency of ATP-driven processes
    
    # Pharmacological properties
    bioavailability: float       # 0-1 - oral bioavailability
    half_life: float            # hours - elimination half-life
    protein_binding: float      # 0-1 - plasma protein binding
    
    # Target affinities (Kd values in nM)
    target_affinities: Dict[str, float] = field(default_factory=dict)
    
    # Mechanism and pathway information
    mechanism_of_action: str
    primary_pathways: List[str] = field(default_factory=list)
    side_effect_profiles: Dict[str, float] = field(default_factory=dict)
    
    # Oscillatory hole matching properties
    hole_matching_profile: Dict[str, float] = field(default_factory=dict)

class DrugDatabase:
    """
    Comprehensive database of pharmaceutical compounds with oscillatory properties.
    
    Manages drug information for BMD-based therapeutic prediction including
    molecular oscillatory frequencies, target affinities, and hole-matching profiles.
    """
    
    def __init__(self, data_file: Optional[str] = None):
        """Initialize drug database."""
        self.drugs = {}
        self.frequency_index = {}  # Fast lookup by frequency ranges
        self.pathway_index = {}    # Fast lookup by pathway
        
        if data_file and Path(data_file).exists():
            self.load_from_file(data_file)
        else:
            self._initialize_default_database()
            
        self._build_indices()
        logger.info(f"Drug database initialized with {len(self.drugs)} compounds")
    
    def _initialize_default_database(self):
        """Initialize database with default pharmaceutical compounds."""
        
        # Mood stabilizers
        self.add_drug(DrugProperties(
            name="lithium",
            chemical_formula="Li2CO3",
            molecular_weight=73.89,
            logp=-2.3,
            pka=13.8,
            volume=45.0,
            surface_area=85.0,
            primary_frequency=2.8e13,     # Matches inositol pathway
            secondary_frequencies=[7.23e13, 2.15e13],  # INPP1, GSK3 resonance
            frequency_bandwidth=5e12,
            oscillatory_amplitude=0.8,
            information_capacity=12.5,
            entropy_reduction=2.1e-20,
            atp_coupling_efficiency=0.85,
            bioavailability=1.0,
            half_life=24.0,
            protein_binding=0.0,
            target_affinities={
                'inpp1_enzyme': 50.0,      # nM - strong binding
                'gsk3b_kinase': 2000.0,    # μM range - indirect modulation
                'impa1_enzyme': 150.0      # nM - moderate binding
            },
            mechanism_of_action="Inositol depletion, GSK3 inhibition",
            primary_pathways=["inositol_metabolism", "gsk3_pathway", "neurotransmitter_signaling"],
            side_effect_profiles={
                'nephrotoxicity': 0.3,
                'tremor': 0.4,
                'weight_gain': 0.2
            },
            hole_matching_profile={
                'inositol_pathway_holes': 0.95,
                'gsk3_pathway_holes': 0.75,
                'neurotransmitter_holes': 0.60
            }
        ))
        
        # Antipsychotics
        self.add_drug(DrugProperties(
            name="aripiprazole",
            chemical_formula="C23H27Cl2N3O2",
            molecular_weight=448.39,
            logp=4.3,
            pka=7.6,
            volume=380.0,
            surface_area=420.0,
            primary_frequency=1.85e13,    # Dopamine receptor frequency
            secondary_frequencies=[1.23e13, 2.1e13],  # Serotonin, secondary sites
            frequency_bandwidth=3e12,
            oscillatory_amplitude=0.65,
            information_capacity=18.7,
            entropy_reduction=3.2e-20,
            atp_coupling_efficiency=0.72,
            bioavailability=0.87,
            half_life=75.0,
            protein_binding=0.99,
            target_affinities={
                'drd2_gpcr': 2.3,         # nM - partial agonist
                'htr2a_gpcr': 8.5,        # nM - antagonist  
                'drd3_gpcr': 4.1,         # nM
                'htr1a_gpcr': 15.0        # nM
            },
            mechanism_of_action="Dopamine partial agonist, 5-HT2A antagonist",
            primary_pathways=["dopamine_signaling", "serotonin_signaling", "gsk3_pathway"],
            side_effect_profiles={
                'akathisia': 0.25,
                'metabolic': 0.15,
                'sedation': 0.10
            },
            hole_matching_profile={
                'dopamine_pathway_holes': 0.90,
                'serotonin_pathway_holes': 0.80,
                'neurotransmitter_holes': 0.85
            }
        ))
        
        # Antidepressants
        self.add_drug(DrugProperties(
            name="citalopram",
            chemical_formula="C20H21FN2O",
            molecular_weight=324.39,
            logp=3.5,
            pka=9.78,
            volume=290.0,
            surface_area=340.0,
            primary_frequency=2.15e13,    # Serotonin transporter
            secondary_frequencies=[1.45e13, 3.2e13],  # Receptor interactions
            frequency_bandwidth=2.5e12,
            oscillatory_amplitude=0.70,
            information_capacity=15.2,
            entropy_reduction=2.8e-20,
            atp_coupling_efficiency=0.68,
            bioavailability=0.80,
            half_life=35.0,
            protein_binding=0.80,
            target_affinities={
                'slc6a4_transporter': 1.8,   # nM - SERT inhibition
                'htr2c_gpcr': 850.0,         # nM - weak affinity
                'htr2a_gpcr': 2600.0         # nM - very weak
            },
            mechanism_of_action="Selective serotonin reuptake inhibition",
            primary_pathways=["serotonin_signaling", "neurotransmitter_signaling"],
            side_effect_profiles={
                'sexual_dysfunction': 0.35,
                'nausea': 0.25,
                'insomnia': 0.20
            },
            hole_matching_profile={
                'serotonin_pathway_holes': 0.92,
                'neurotransmitter_holes': 0.70,
                'mood_regulation_holes': 0.85
            }
        ))
        
        # Statins
        self.add_drug(DrugProperties(
            name="atorvastatin",
            chemical_formula="C33H35FN2O5",
            molecular_weight=558.64,
            logp=4.1,
            pka=4.33,
            volume=450.0,
            surface_area=520.0,
            primary_frequency=1.67e13,    # HMG-CoA reductase
            secondary_frequencies=[2.8e13, 1.2e13],  # Metabolic pathways
            frequency_bandwidth=1.8e12,
            oscillatory_amplitude=0.55,
            information_capacity=22.3,
            entropy_reduction=4.1e-20,
            atp_coupling_efficiency=0.58,
            bioavailability=0.14,
            half_life=14.0,
            protein_binding=0.98,
            target_affinities={
                'hmgcr_enzyme': 8.2,      # nM - HMG-CoA reductase
                'cyp3a4_enzyme': 15000.0,  # nM - metabolism
                'slco1b1_transporter': 25.0  # nM - uptake transporter
            },
            mechanism_of_action="HMG-CoA reductase inhibition",
            primary_pathways=["cholesterol_metabolism", "mevalonate_pathway"],
            side_effect_profiles={
                'myopathy': 0.05,
                'hepatotoxicity': 0.02,
                'diabetes_risk': 0.08
            },
            hole_matching_profile={
                'metabolic_pathway_holes': 0.88,
                'cholesterol_synthesis_holes': 0.95,
                'lipid_metabolism_holes': 0.82
            }
        ))
        
        # Antiplatelet agents
        self.add_drug(DrugProperties(
            name="aspirin",
            chemical_formula="C9H8O4",
            molecular_weight=180.16,
            logp=1.2,
            pka=3.49,
            volume=140.0,
            surface_area=180.0,
            primary_frequency=3.42e13,    # COX enzyme oscillation
            secondary_frequencies=[4.1e13, 2.9e13],  # Protein interactions
            frequency_bandwidth=8e11,
            oscillatory_amplitude=0.45,
            information_capacity=8.9,
            entropy_reduction=1.6e-20,
            atp_coupling_efficiency=0.35,
            bioavailability=0.68,
            half_life=0.25,
            protein_binding=0.80,
            target_affinities={
                'ptgs1_enzyme': 100.0,    # nM - COX-1 (irreversible)
                'ptgs2_enzyme': 150.0,    # nM - COX-2
                'cyp2c9_enzyme': 5000.0   # nM - metabolism
            },
            mechanism_of_action="Irreversible COX inhibition",
            primary_pathways=["prostaglandin_synthesis", "platelet_aggregation"],
            side_effect_profiles={
                'gi_bleeding': 0.15,
                'peptic_ulcer': 0.08,
                'renal_dysfunction': 0.05
            },
            hole_matching_profile={
                'inflammation_pathway_holes': 0.78,
                'platelet_function_holes': 0.85,
                'vascular_holes': 0.65
            }
        ))
        
        # Anticonvulsants
        self.add_drug(DrugProperties(
            name="valproate",
            chemical_formula="C8H16O2",
            molecular_weight=144.21,
            logp=2.75,
            pka=4.95,
            volume=160.0,
            surface_area=200.0,
            primary_frequency=1.95e13,    # GABA system modulation
            secondary_frequencies=[2.8e13, 1.4e13],  # Multiple targets
            frequency_bandwidth=4e12,
            oscillatory_amplitude=0.60,
            information_capacity=11.8,
            entropy_reduction=2.3e-20,
            atp_coupling_efficiency=0.62,
            bioavailability=1.0,
            half_life=12.0,
            protein_binding=0.90,
            target_affinities={
                'gaba_system': 2500.0,    # μM range - indirect
                'gsk3b_kinase': 1200.0,   # nM - direct inhibition
                'hdac_enzymes': 800.0     # μM - histone deacetylase
            },
            mechanism_of_action="GABA enhancement, GSK3 inhibition, HDAC inhibition",
            primary_pathways=["gaba_signaling", "gsk3_pathway", "epigenetic_regulation"],
            side_effect_profiles={
                'weight_gain': 0.40,
                'tremor': 0.25,
                'hair_loss': 0.15
            },
            hole_matching_profile={
                'gaba_pathway_holes': 0.80,
                'gsk3_pathway_holes': 0.70,
                'neurotransmitter_holes': 0.75
            }
        ))
        
        # Benzodiazepines
        self.add_drug(DrugProperties(
            name="lorazepam",
            chemical_formula="C15H10Cl2N2O2",
            molecular_weight=321.16,
            logp=2.39,
            pka=1.3,
            volume=260.0,
            surface_area=320.0,
            primary_frequency=1.2e13,     # GABA-A receptor
            secondary_frequencies=[0.8e13, 1.5e13],  # Allosteric sites
            frequency_bandwidth=2e12,
            oscillatory_amplitude=0.85,
            information_capacity=14.1,
            entropy_reduction=2.9e-20,
            atp_coupling_efficiency=0.45,
            bioavailability=0.90,
            half_life=14.0,
            protein_binding=0.85,
            target_affinities={
                'gabra1_receptor': 5.2,   # nM - α1 subunit
                'gabra2_receptor': 8.1,   # nM - α2 subunit
                'gabra5_receptor': 12.0   # nM - α5 subunit
            },
            mechanism_of_action="GABA-A receptor positive allosteric modulation",
            primary_pathways=["gaba_signaling", "anxiolytic_response"],
            side_effect_profiles={
                'sedation': 0.60,
                'cognitive_impairment': 0.30,
                'dependence': 0.25
            },
            hole_matching_profile={
                'gaba_pathway_holes': 0.95,
                'anxiety_circuit_holes': 0.88,
                'inhibitory_holes': 0.90
            }
        ))
    
    def add_drug(self, drug_properties: DrugProperties) -> None:
        """Add drug to database."""
        self.drugs[drug_properties.name] = drug_properties
        logger.debug(f"Added drug: {drug_properties.name}")
    
    def get_drug(self, name: str) -> Optional[DrugProperties]:
        """Get drug properties by name."""
        return self.drugs.get(name.lower())
    
    def get_drugs_by_frequency_range(self, min_freq: float, max_freq: float) -> List[DrugProperties]:
        """Get drugs within specified frequency range."""
        matching_drugs = []
        
        for drug in self.drugs.values():
            if min_freq <= drug.primary_frequency <= max_freq:
                matching_drugs.append(drug)
            
            # Check secondary frequencies too
            for sec_freq in drug.secondary_frequencies:
                if min_freq <= sec_freq <= max_freq and drug not in matching_drugs:
                    matching_drugs.append(drug)
        
        return matching_drugs
    
    def get_drugs_by_pathway(self, pathway: str) -> List[DrugProperties]:
        """Get drugs that target specific pathway."""
        return [drug for drug in self.drugs.values() if pathway in drug.primary_pathways]
    
    def get_drugs_by_target(self, target: str) -> List[DrugProperties]:
        """Get drugs that bind to specific target."""
        matching_drugs = []
        
        for drug in self.drugs.values():
            if target in drug.target_affinities:
                matching_drugs.append(drug)
        
        return matching_drugs
    
    def calculate_frequency_match_score(self, drug_name: str, 
                                      target_frequency: float,
                                      tolerance: float = 0.1) -> float:
        """Calculate how well a drug's frequencies match a target frequency."""
        
        drug = self.get_drug(drug_name)
        if not drug:
            return 0.0
        
        # Check primary frequency
        primary_match = self._frequency_similarity(drug.primary_frequency, target_frequency, tolerance)
        
        # Check secondary frequencies
        secondary_matches = [
            self._frequency_similarity(freq, target_frequency, tolerance)
            for freq in drug.secondary_frequencies
        ]
        
        # Best match wins, with primary frequency weighted higher
        best_secondary = max(secondary_matches) if secondary_matches else 0.0
        
        return max(primary_match * 1.2, best_secondary)  # 20% bonus for primary frequency
    
    def _frequency_similarity(self, freq1: float, freq2: float, tolerance: float) -> float:
        """Calculate similarity between two frequencies."""
        
        if freq2 == 0:
            return 0.0
        
        relative_diff = abs(freq1 - freq2) / freq2
        
        if relative_diff <= tolerance:
            # Perfect match within tolerance
            return 1.0 - (relative_diff / tolerance) * 0.2  # Small penalty for deviation
        else:
            # Exponential decay for frequencies outside tolerance
            return np.exp(-5 * (relative_diff - tolerance))
    
    def rank_drugs_for_hole(self, hole_frequency: float, 
                           hole_pathway: str,
                           amplitude_deficit: float) -> List[tuple]:
        """
        Rank drugs by their suitability for filling an oscillatory hole.
        
        Returns list of (drug_name, score) tuples sorted by score (highest first).
        """
        
        drug_scores = []
        
        for drug_name, drug in self.drugs.items():
            # Frequency matching score (40% weight)
            freq_score = self.calculate_frequency_match_score(drug_name, hole_frequency)
            
            # Pathway relevance score (30% weight)
            pathway_score = 0.0
            if hole_pathway in drug.primary_pathways:
                pathway_score = 1.0
            elif any(pathway in hole_pathway or hole_pathway in pathway 
                    for pathway in drug.primary_pathways):
                pathway_score = 0.6
            
            # Hole matching profile score (20% weight)
            hole_match_score = drug.hole_matching_profile.get(f"{hole_pathway}_holes", 0.0)
            
            # Amplitude compatibility score (10% weight)
            # Drugs with higher oscillatory amplitude can better fill larger deficits
            amplitude_score = min(1.0, drug.oscillatory_amplitude / amplitude_deficit) if amplitude_deficit > 0 else 1.0
            
            # Overall score
            total_score = (freq_score * 0.4 + 
                          pathway_score * 0.3 + 
                          hole_match_score * 0.2 + 
                          amplitude_score * 0.1)
            
            drug_scores.append((drug_name, total_score))
        
        # Sort by score (descending)
        return sorted(drug_scores, key=lambda x: x[1], reverse=True)
    
    def get_drug_interactions(self, drug1_name: str, drug2_name: str) -> Dict[str, Any]:
        """Predict interactions between two drugs based on oscillatory properties."""
        
        drug1 = self.get_drug(drug1_name)
        drug2 = self.get_drug(drug2_name)
        
        if not drug1 or not drug2:
            return {'interaction_risk': 'unknown', 'reason': 'drug not found'}
        
        interactions = {
            'frequency_interference': self._calculate_frequency_interference(drug1, drug2),
            'pathway_overlap': self._calculate_pathway_overlap(drug1, drug2),
            'target_competition': self._calculate_target_competition(drug1, drug2),
            'atp_coupling_strain': self._calculate_atp_strain(drug1, drug2)
        }
        
        # Overall risk assessment
        risk_factors = [
            interactions['frequency_interference'],
            interactions['pathway_overlap'] * 0.8,  # Pathway overlap less critical
            interactions['target_competition'] * 1.2,  # Target competition more critical
            interactions['atp_coupling_strain'] * 0.6   # ATP strain moderate risk
        ]
        
        overall_risk = max(risk_factors)
        
        if overall_risk > 0.7:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'interaction_risk': risk_level,
            'risk_score': overall_risk,
            'detailed_interactions': interactions,
            'recommendations': self._generate_interaction_recommendations(interactions, risk_level)
        }
    
    def _calculate_frequency_interference(self, drug1: DrugProperties, drug2: DrugProperties) -> float:
        """Calculate potential frequency interference between drugs."""
        
        all_freq1 = [drug1.primary_frequency] + drug1.secondary_frequencies
        all_freq2 = [drug2.primary_frequency] + drug2.secondary_frequencies
        
        max_interference = 0.0
        
        for freq1 in all_freq1:
            for freq2 in all_freq2:
                # Check for resonance (same frequency) or beating (close frequencies)
                freq_diff = abs(freq1 - freq2)
                avg_freq = (freq1 + freq2) / 2
                
                relative_diff = freq_diff / avg_freq if avg_freq > 0 else 0
                
                if relative_diff < 0.05:  # Very close frequencies
                    interference = 1.0 - relative_diff / 0.05
                elif relative_diff < 0.1:  # Moderate interference
                    interference = 0.5 * (1.0 - (relative_diff - 0.05) / 0.05)
                else:
                    interference = 0.0
                
                max_interference = max(max_interference, interference)
        
        return max_interference
    
    def _calculate_pathway_overlap(self, drug1: DrugProperties, drug2: DrugProperties) -> float:
        """Calculate overlap in targeted pathways."""
        
        pathways1 = set(drug1.primary_pathways)
        pathways2 = set(drug2.primary_pathways)
        
        overlap = pathways1.intersection(pathways2)
        union = pathways1.union(pathways2)
        
        return len(overlap) / len(union) if union else 0.0
    
    def _calculate_target_competition(self, drug1: DrugProperties, drug2: DrugProperties) -> float:
        """Calculate competition for the same molecular targets."""
        
        targets1 = set(drug1.target_affinities.keys())
        targets2 = set(drug2.target_affinities.keys())
        
        common_targets = targets1.intersection(targets2)
        
        if not common_targets:
            return 0.0
        
        # Calculate competition intensity
        competition_scores = []
        
        for target in common_targets:
            kd1 = drug1.target_affinities[target]
            kd2 = drug2.target_affinities[target]
            
            # Both drugs bind strongly (low Kd) = high competition
            if kd1 < 100 and kd2 < 100:  # Both high affinity
                competition_scores.append(1.0)
            elif kd1 < 1000 and kd2 < 1000:  # Both moderate affinity
                competition_scores.append(0.6)
            else:
                competition_scores.append(0.2)
        
        return max(competition_scores) if competition_scores else 0.0
    
    def _calculate_atp_strain(self, drug1: DrugProperties, drug2: DrugProperties) -> float:
        """Calculate potential ATP depletion from combined drug action."""
        
        # Drugs with high ATP coupling efficiency can strain cellular energy
        total_atp_demand = drug1.atp_coupling_efficiency + drug2.atp_coupling_efficiency
        
        if total_atp_demand > 1.5:
            return min(1.0, (total_atp_demand - 1.0) / 1.0)  # Strain above baseline
        else:
            return 0.0
    
    def _generate_interaction_recommendations(self, interactions: Dict, risk_level: str) -> List[str]:
        """Generate clinical recommendations for drug interactions."""
        
        recommendations = []
        
        if risk_level == 'high':
            recommendations.append("Consider alternative medications or dose reduction")
            
        if interactions['frequency_interference'] > 0.6:
            recommendations.append("Monitor for oscillatory interference effects")
            
        if interactions['target_competition'] > 0.7:
            recommendations.append("Stagger dosing times to reduce target competition")
            
        if interactions['atp_coupling_strain'] > 0.5:
            recommendations.append("Monitor for fatigue and cellular energy depletion")
            
        if interactions['pathway_overlap'] > 0.8:
            recommendations.append("Enhanced monitoring for pathway-specific effects")
        
        if not recommendations:
            recommendations.append("Standard monitoring protocols appropriate")
        
        return recommendations
    
    def _build_indices(self) -> None:
        """Build search indices for fast lookups."""
        
        # Frequency index - group drugs by frequency ranges
        self.frequency_index = {
            'very_low': [],    # < 1e13 Hz
            'low': [],         # 1e13 - 2e13 Hz
            'medium': [],      # 2e13 - 3e13 Hz
            'high': [],        # 3e13 - 5e13 Hz
            'very_high': []    # > 5e13 Hz
        }
        
        # Pathway index
        self.pathway_index = {}
        
        for drug_name, drug in self.drugs.items():
            # Frequency indexing
            freq = drug.primary_frequency
            
            if freq < 1e13:
                self.frequency_index['very_low'].append(drug_name)
            elif freq < 2e13:
                self.frequency_index['low'].append(drug_name)
            elif freq < 3e13:
                self.frequency_index['medium'].append(drug_name)
            elif freq < 5e13:
                self.frequency_index['high'].append(drug_name)
            else:
                self.frequency_index['very_high'].append(drug_name)
            
            # Pathway indexing
            for pathway in drug.primary_pathways:
                if pathway not in self.pathway_index:
                    self.pathway_index[pathway] = []
                self.pathway_index[pathway].append(drug_name)
    
    def save_to_file(self, filename: str) -> None:
        """Save database to JSON file."""
        
        output_data = {}
        
        for drug_name, drug in self.drugs.items():
            output_data[drug_name] = {
                'name': drug.name,
                'chemical_formula': drug.chemical_formula,
                'molecular_weight': drug.molecular_weight,
                'logp': drug.logp,
                'pka': drug.pka,
                'volume': drug.volume,
                'surface_area': drug.surface_area,
                'primary_frequency': drug.primary_frequency,
                'secondary_frequencies': drug.secondary_frequencies,
                'frequency_bandwidth': drug.frequency_bandwidth,
                'oscillatory_amplitude': drug.oscillatory_amplitude,
                'information_capacity': drug.information_capacity,
                'entropy_reduction': drug.entropy_reduction,
                'atp_coupling_efficiency': drug.atp_coupling_efficiency,
                'bioavailability': drug.bioavailability,
                'half_life': drug.half_life,
                'protein_binding': drug.protein_binding,
                'target_affinities': drug.target_affinities,
                'mechanism_of_action': drug.mechanism_of_action,
                'primary_pathways': drug.primary_pathways,
                'side_effect_profiles': drug.side_effect_profiles,
                'hole_matching_profile': drug.hole_matching_profile
            }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Database saved to {filename}")
    
    def load_from_file(self, filename: str) -> None:
        """Load database from JSON file."""
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        for drug_name, drug_data in data.items():
            drug_props = DrugProperties(**drug_data)
            self.add_drug(drug_props)
        
        logger.info(f"Database loaded from {filename}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the drug database."""
        
        if not self.drugs:
            return {'total_drugs': 0}
        
        frequencies = [drug.primary_frequency for drug in self.drugs.values()]
        amplitudes = [drug.oscillatory_amplitude for drug in self.drugs.values()]
        molecular_weights = [drug.molecular_weight for drug in self.drugs.values()]
        
        pathway_counts = {}
        for drug in self.drugs.values():
            for pathway in drug.primary_pathways:
                pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
        
        return {
            'total_drugs': len(self.drugs),
            'frequency_stats': {
                'mean': np.mean(frequencies),
                'median': np.median(frequencies),
                'range': [min(frequencies), max(frequencies)]
            },
            'amplitude_stats': {
                'mean': np.mean(amplitudes),
                'range': [min(amplitudes), max(amplitudes)]
            },
            'molecular_weight_stats': {
                'mean': np.mean(molecular_weights),
                'range': [min(molecular_weights), max(molecular_weights)]
            },
            'pathway_distribution': pathway_counts,
            'frequency_distribution': {
                band: len(drugs) for band, drugs in self.frequency_index.items()
            }
        }

def main():
    """
    Test the drug database functionality.
    
    Demonstrates loading, searching, and analysis capabilities
    of the oscillatory drug database.
    """
    
    print("💊 Testing Drug Database")
    print("=" * 40)
    
    # Initialize database
    db = DrugDatabase()
    
    # Display database stats
    stats = db.get_database_stats()
    print(f"\n📊 Database Statistics:")
    print(f"Total drugs: {stats['total_drugs']}")
    print(f"Frequency range: {stats['frequency_stats']['range'][0]:.2e} - {stats['frequency_stats']['range'][1]:.2e} Hz")
    print(f"Mean molecular weight: {stats['molecular_weight_stats']['mean']:.1f} Da")
    
    print(f"\n🎯 Pathway Distribution:")
    for pathway, count in stats['pathway_distribution'].items():
        print(f"  • {pathway}: {count} drugs")
    
    # Test frequency-based searches
    print(f"\n🔍 Frequency-based Drug Search:")
    
    # Search for drugs in different frequency bands
    target_freq = 2.0e13  # Hz
    tolerance = 0.2
    
    print(f"Target frequency: {target_freq:.2e} Hz")
    
    matching_drugs = db.get_drugs_by_frequency_range(
        target_freq * (1 - tolerance), 
        target_freq * (1 + tolerance)
    )
    
    print(f"Drugs in frequency range ({tolerance*100:.0f}% tolerance):")
    for drug in matching_drugs:
        match_score = db.calculate_frequency_match_score(drug.name, target_freq)
        print(f"  • {drug.name}: {drug.primary_frequency:.2e} Hz (match: {match_score:.3f})")
    
    # Test pathway searches
    print(f"\n🧠 Pathway-based Search:")
    pathway = "neurotransmitter_signaling"
    pathway_drugs = db.get_drugs_by_pathway(pathway)
    
    print(f"Drugs targeting {pathway}:")
    for drug in pathway_drugs:
        print(f"  • {drug.name}: {drug.mechanism_of_action}")
    
    # Test hole matching
    print(f"\n🕳️  Oscillatory Hole Matching:")
    
    # Simulate an oscillatory hole
    hole_freq = 2.8e13  # Hz
    hole_pathway = "inositol_metabolism"
    hole_amplitude = 0.6
    
    print(f"Hole: {hole_pathway} at {hole_freq:.2e} Hz, amplitude deficit {hole_amplitude}")
    
    ranked_drugs = db.rank_drugs_for_hole(hole_freq, hole_pathway, hole_amplitude)
    
    print(f"Top 3 drug candidates:")
    for drug_name, score in ranked_drugs[:3]:
        drug = db.get_drug(drug_name)
        print(f"  • {drug_name}: score {score:.3f}")
        print(f"    Primary freq: {drug.primary_frequency:.2e} Hz")
        print(f"    Pathways: {', '.join(drug.primary_pathways)}")
    
    # Test drug interactions
    print(f"\n⚠️  Drug Interaction Analysis:")
    
    drug1 = "lithium"
    drug2 = "aripiprazole"
    
    interaction = db.get_drug_interactions(drug1, drug2)
    
    print(f"Interaction between {drug1} and {drug2}:")
    print(f"  Risk level: {interaction['interaction_risk']}")
    print(f"  Risk score: {interaction['risk_score']:.3f}")
    print(f"  Recommendations:")
    for rec in interaction['recommendations']:
        print(f"    • {rec}")
    
    # Display detailed drug information
    print(f"\n🔬 Detailed Drug Information:")
    
    featured_drug = "lithium"
    drug = db.get_drug(featured_drug)
    
    if drug:
        print(f"\n{drug.name.upper()} Profile:")
        print(f"  Formula: {drug.chemical_formula}")
        print(f"  MW: {drug.molecular_weight} Da")
        print(f"  Primary frequency: {drug.primary_frequency:.2e} Hz")
        print(f"  Oscillatory amplitude: {drug.oscillatory_amplitude:.2f}")
        print(f"  Information capacity: {drug.information_capacity:.1f} bits")
        print(f"  ATP coupling efficiency: {drug.atp_coupling_efficiency:.2f}")
        print(f"  Mechanism: {drug.mechanism_of_action}")
        
        print(f"  Target affinities:")
        for target, kd in drug.target_affinities.items():
            print(f"    • {target}: {kd:.1f} nM")
        
        print(f"  Hole matching profile:")
        for hole_type, score in drug.hole_matching_profile.items():
            print(f"    • {hole_type}: {score:.2f}")
    
    # Save database for future use
    db.save_to_file("babylon_drug_database.json")
    
    print(f"\n✅ Drug database testing complete!")
    print(f"💾 Database saved as 'babylon_drug_database.json'")
    
    return db

if __name__ == "__main__":
    database = main()
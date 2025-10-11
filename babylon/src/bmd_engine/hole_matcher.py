"""
Oscillatory Hole Matcher Module
===============================

Matches pharmaceutical compounds to oscillatory holes using frequency resonance,
pathway analysis, and BMD (Biological Maxwell Demon) principles. Optimizes
therapeutic selection for filling specific oscillatory deficits.

Part of the St. Stellas BMD Engine for therapeutic prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

# Import from our drug database
from .drug_database import DrugDatabase, DrugProperties

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OscillatoryHole:
    """Represents an oscillatory deficit in biological systems."""
    hole_id: str
    pathway: str
    frequency: float             # Hz - characteristic frequency
    amplitude_deficit: float     # 0-1 - how much oscillation is missing
    bandwidth: float            # Hz - frequency spread of the hole
    location: str               # Cellular/tissue location
    impact_severity: float      # 0-1 - clinical impact
    
    # Causal information
    genetic_cause: Optional[str] = None   # Gene variant causing hole
    variant_impact: Optional[str] = None  # HIGH, MODERATE, LOW
    
    # Temporal characteristics
    persistence: float = 1.0     # 0-1 - how persistent the hole is
    variability: float = 0.1     # 0-1 - how much the hole varies over time
    
    # Contextual information
    associated_pathways: List[str] = field(default_factory=list)
    biomarkers: Dict[str, float] = field(default_factory=dict)
    phenotypic_effects: List[str] = field(default_factory=list)

@dataclass
class DrugHoleMatch:
    """Represents a match between a drug and an oscillatory hole."""
    drug_name: str
    hole_id: str
    
    # Matching scores (0-1 scale)
    frequency_match: float       # How well drug frequency matches hole
    pathway_match: float         # Pathway relevance score
    amplitude_compatibility: float  # Can drug fill the amplitude deficit
    temporal_alignment: float    # Temporal compatibility
    
    # Overall matching
    overall_score: float         # Combined matching score
    confidence: float           # Confidence in the match
    
    # Predicted outcomes
    predicted_hole_filling: float     # 0-1 - expected hole reduction
    predicted_efficacy: float         # 0-1 - expected therapeutic benefit
    predicted_side_effects: Dict[str, float] = field(default_factory=dict)
    
    # Dosing recommendations
    optimal_dose_ratio: float = 1.0    # Relative to standard dose
    dosing_frequency: str = "standard"  # Timing recommendations
    
    # Mechanistic insights
    mechanism_description: str = ""
    resonance_type: str = "direct"     # direct, harmonic, subharmonic
    
@dataclass
class TherapeuticRecommendation:
    """Complete therapeutic recommendation for filling oscillatory holes."""
    patient_id: str
    holes_analyzed: List[str]
    
    # Primary recommendations
    primary_drugs: List[DrugHoleMatch]     # Best matches
    alternative_drugs: List[DrugHoleMatch] # Backup options
    contraindicated_drugs: List[str]      # Drugs to avoid
    
    # Combination therapy
    drug_combinations: List[Dict[str, Any]] = field(default_factory=list)
    interaction_warnings: List[str] = field(default_factory=list)
    
    # Monitoring
    biomarkers_to_monitor: List[str] = field(default_factory=list)
    monitoring_frequency: str = "monthly"
    
    # Personalization factors
    genetic_considerations: List[str] = field(default_factory=list)
    lifestyle_factors: List[str] = field(default_factory=list)
    
    # Expected outcomes
    projected_improvement: Dict[str, float] = field(default_factory=dict)
    timeline_to_effect: str = "2-4 weeks"

class OscillatoryHoleMatcher:
    """
    Advanced matcher for pairing pharmaceutical compounds with oscillatory holes.
    
    Uses multi-dimensional analysis including frequency resonance, pathway
    relevance, temporal dynamics, and personalized factors.
    """
    
    def __init__(self, drug_database: Optional[DrugDatabase] = None):
        """Initialize the hole matcher."""
        self.drug_db = drug_database if drug_database else DrugDatabase()
        
        # Matching algorithms and parameters
        self.frequency_tolerance = 0.15      # ±15% frequency matching
        self.pathway_weight = 0.3           # Pathway importance in scoring
        self.frequency_weight = 0.4         # Frequency importance in scoring
        self.amplitude_weight = 0.2         # Amplitude compatibility importance
        self.temporal_weight = 0.1          # Temporal alignment importance
        
        logger.info("Oscillatory hole matcher initialized")
    
    def create_oscillatory_hole(self, hole_data: Dict[str, Any]) -> OscillatoryHole:
        """Create an oscillatory hole from data."""
        
        # Extract required fields
        hole_id = hole_data.get('hole_id', f"hole_{np.random.randint(1000, 9999)}")
        pathway = hole_data.get('pathway', 'unknown_pathway')
        frequency = float(hole_data.get('frequency', 1e13))
        amplitude_deficit = float(hole_data.get('amplitude_deficit', 0.5))
        
        # Extract optional fields with defaults
        bandwidth = hole_data.get('bandwidth', frequency * 0.1)  # 10% of frequency
        location = hole_data.get('location', 'cellular')
        impact_severity = hole_data.get('impact_severity', 0.5)
        
        return OscillatoryHole(
            hole_id=hole_id,
            pathway=pathway,
            frequency=frequency,
            amplitude_deficit=amplitude_deficit,
            bandwidth=bandwidth,
            location=location,
            impact_severity=impact_severity,
            genetic_cause=hole_data.get('genetic_cause'),
            variant_impact=hole_data.get('variant_impact'),
            persistence=hole_data.get('persistence', 1.0),
            variability=hole_data.get('variability', 0.1),
            associated_pathways=hole_data.get('associated_pathways', []),
            biomarkers=hole_data.get('biomarkers', {}),
            phenotypic_effects=hole_data.get('phenotypic_effects', [])
        )
    
    def calculate_frequency_match(self, drug: DrugProperties, 
                                 hole: OscillatoryHole) -> Tuple[float, str]:
        """Calculate frequency matching score between drug and hole."""
        
        drug_frequencies = [drug.primary_frequency] + drug.secondary_frequencies
        hole_freq = hole.frequency
        hole_bandwidth = hole.bandwidth
        
        best_match = 0.0
        resonance_type = "none"
        
        for drug_freq in drug_frequencies:
            # Direct resonance (same frequency)
            freq_diff = abs(drug_freq - hole_freq) / hole_freq
            
            if freq_diff <= self.frequency_tolerance:
                # Within tolerance - calculate match quality
                match_score = 1.0 - (freq_diff / self.frequency_tolerance) * 0.3
                if match_score > best_match:
                    best_match = match_score
                    resonance_type = "direct"
            
            # Harmonic resonance (drug frequency is multiple of hole frequency)
            for harmonic in [2, 3, 4, 5]:
                harmonic_freq = hole_freq * harmonic
                harmonic_diff = abs(drug_freq - harmonic_freq) / harmonic_freq
                
                if harmonic_diff <= self.frequency_tolerance:
                    match_score = (1.0 - harmonic_diff / self.frequency_tolerance) * (0.8 / harmonic)
                    if match_score > best_match:
                        best_match = match_score
                        resonance_type = f"harmonic_{harmonic}"
            
            # Subharmonic resonance (hole frequency is multiple of drug frequency)
            for subharmonic in [2, 3, 4, 5]:
                subharmonic_freq = drug_freq * subharmonic
                subharmonic_diff = abs(subharmonic_freq - hole_freq) / hole_freq
                
                if subharmonic_diff <= self.frequency_tolerance:
                    match_score = (1.0 - subharmonic_diff / self.frequency_tolerance) * (0.6 / subharmonic)
                    if match_score > best_match:
                        best_match = match_score
                        resonance_type = f"subharmonic_{subharmonic}"
        
        return best_match, resonance_type
    
    def calculate_pathway_match(self, drug: DrugProperties, 
                               hole: OscillatoryHole) -> float:
        """Calculate pathway relevance score."""
        
        drug_pathways = set(drug.primary_pathways)
        hole_pathways = set([hole.pathway] + hole.associated_pathways)
        
        # Direct pathway match
        direct_matches = drug_pathways.intersection(hole_pathways)
        if direct_matches:
            return 1.0
        
        # Partial pathway matches (substring matching)
        partial_score = 0.0
        
        for drug_pathway in drug_pathways:
            for hole_pathway in hole_pathways:
                # Check if pathways have common elements
                if drug_pathway in hole_pathway or hole_pathway in drug_pathway:
                    partial_score = max(partial_score, 0.6)
                
                # Check for related pathway families
                pathway_families = {
                    'neurotransmitter': ['dopamine', 'serotonin', 'gaba', 'glutamate'],
                    'metabolism': ['glucose', 'lipid', 'cholesterol', 'energy'],
                    'signaling': ['gsk3', 'mapk', 'pi3k', 'calcium']
                }
                
                for family, keywords in pathway_families.items():
                    drug_in_family = any(kw in drug_pathway.lower() for kw in keywords)
                    hole_in_family = any(kw in hole_pathway.lower() for kw in keywords)
                    
                    if drug_in_family and hole_in_family:
                        partial_score = max(partial_score, 0.4)
        
        # Use hole matching profile if available
        hole_match_key = f"{hole.pathway}_holes"
        if hole_match_key in drug.hole_matching_profile:
            profile_score = drug.hole_matching_profile[hole_match_key]
            partial_score = max(partial_score, profile_score)
        
        return partial_score
    
    def calculate_amplitude_compatibility(self, drug: DrugProperties,
                                        hole: OscillatoryHole) -> float:
        """Calculate how well drug can fill the amplitude deficit."""
        
        drug_amplitude = drug.oscillatory_amplitude
        hole_deficit = hole.amplitude_deficit
        
        # Perfect match when drug amplitude equals or slightly exceeds deficit
        amplitude_ratio = drug_amplitude / hole_deficit if hole_deficit > 0 else 1.0
        
        if 0.8 <= amplitude_ratio <= 1.2:
            # Ideal range - drug amplitude matches deficit well
            return 1.0 - abs(amplitude_ratio - 1.0) * 0.5
        elif amplitude_ratio > 1.2:
            # Drug too strong - may overfill hole
            return max(0.3, 1.0 - (amplitude_ratio - 1.2) * 0.8)
        else:
            # Drug too weak - underfill hole
            return amplitude_ratio * 0.8
    
    def calculate_temporal_alignment(self, drug: DrugProperties,
                                   hole: OscillatoryHole) -> float:
        """Calculate temporal compatibility between drug and hole."""
        
        # Base score on drug half-life vs hole persistence
        half_life_hours = drug.half_life
        hole_persistence = hole.persistence
        hole_variability = hole.variability
        
        # Ideal half-life depends on hole characteristics
        if hole_persistence > 0.8:  # Persistent hole
            # Need sustained drug action
            ideal_half_life_range = (12, 48)  # 12-48 hours
        elif hole_variability > 0.5:  # Highly variable hole
            # Need flexible dosing
            ideal_half_life_range = (4, 12)   # 4-12 hours
        else:
            # Standard dosing
            ideal_half_life_range = (8, 24)   # 8-24 hours
        
        min_ideal, max_ideal = ideal_half_life_range
        
        if min_ideal <= half_life_hours <= max_ideal:
            # Within ideal range
            return 1.0
        elif half_life_hours < min_ideal:
            # Too short - may need frequent dosing
            return 0.5 + 0.5 * (half_life_hours / min_ideal)
        else:
            # Too long - may accumulate
            return max(0.3, 1.0 - (half_life_hours - max_ideal) / max_ideal)
    
    def match_drug_to_hole(self, drug_name: str, 
                          hole: OscillatoryHole) -> DrugHoleMatch:
        """Create a comprehensive match between a drug and hole."""
        
        drug = self.drug_db.get_drug(drug_name)
        if not drug:
            raise ValueError(f"Drug {drug_name} not found in database")
        
        # Calculate individual matching scores
        frequency_match, resonance_type = self.calculate_frequency_match(drug, hole)
        pathway_match = self.calculate_pathway_match(drug, hole)
        amplitude_compatibility = self.calculate_amplitude_compatibility(drug, hole)
        temporal_alignment = self.calculate_temporal_alignment(drug, hole)
        
        # Calculate overall score
        overall_score = (frequency_match * self.frequency_weight +
                        pathway_match * self.pathway_weight +
                        amplitude_compatibility * self.amplitude_weight +
                        temporal_alignment * self.temporal_weight)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_match_confidence(
            drug, hole, frequency_match, pathway_match
        )
        
        # Predict outcomes
        predicted_hole_filling = self._predict_hole_filling(
            overall_score, amplitude_compatibility, hole
        )
        
        predicted_efficacy = self._predict_efficacy(
            predicted_hole_filling, hole.impact_severity, pathway_match
        )
        
        # Generate mechanism description
        mechanism_description = self._generate_mechanism_description(
            drug, hole, resonance_type, pathway_match
        )
        
        # Calculate dosing recommendations
        optimal_dose_ratio = self._calculate_optimal_dose_ratio(
            amplitude_compatibility, hole.amplitude_deficit
        )
        
        dosing_frequency = self._recommend_dosing_frequency(
            temporal_alignment, drug.half_life, hole.variability
        )
        
        return DrugHoleMatch(
            drug_name=drug_name,
            hole_id=hole.hole_id,
            frequency_match=frequency_match,
            pathway_match=pathway_match,
            amplitude_compatibility=amplitude_compatibility,
            temporal_alignment=temporal_alignment,
            overall_score=overall_score,
            confidence=confidence,
            predicted_hole_filling=predicted_hole_filling,
            predicted_efficacy=predicted_efficacy,
            predicted_side_effects=drug.side_effect_profiles.copy(),
            optimal_dose_ratio=optimal_dose_ratio,
            dosing_frequency=dosing_frequency,
            mechanism_description=mechanism_description,
            resonance_type=resonance_type
        )
    
    def find_best_matches_for_hole(self, hole: OscillatoryHole,
                                  max_matches: int = 5) -> List[DrugHoleMatch]:
        """Find the best drug matches for a given oscillatory hole."""
        
        all_matches = []
        
        # Test all drugs in database
        for drug_name in self.drug_db.drugs.keys():
            try:
                match = self.match_drug_to_hole(drug_name, hole)
                all_matches.append(match)
            except Exception as e:
                logger.warning(f"Error matching {drug_name} to hole {hole.hole_id}: {e}")
        
        # Sort by overall score and return top matches
        all_matches.sort(key=lambda m: m.overall_score, reverse=True)
        return all_matches[:max_matches]
    
    def find_matches_for_multiple_holes(self, holes: List[OscillatoryHole],
                                      max_matches_per_hole: int = 3) -> Dict[str, List[DrugHoleMatch]]:
        """Find matches for multiple holes simultaneously."""
        
        hole_matches = {}
        
        for hole in holes:
            logger.info(f"Finding matches for hole: {hole.hole_id} ({hole.pathway})")
            matches = self.find_best_matches_for_hole(hole, max_matches_per_hole)
            hole_matches[hole.hole_id] = matches
        
        return hole_matches
    
    def generate_therapeutic_recommendation(self, patient_id: str,
                                          holes: List[OscillatoryHole]) -> TherapeuticRecommendation:
        """Generate comprehensive therapeutic recommendations."""
        
        logger.info(f"Generating therapeutic recommendations for patient {patient_id}")
        
        # Find matches for all holes
        hole_matches = self.find_matches_for_multiple_holes(holes)
        
        # Aggregate all potential drugs
        all_drug_matches = []
        for hole_id, matches in hole_matches.items():
            all_drug_matches.extend(matches)
        
        # Group by drug name and calculate combined scores
        drug_scores = {}
        for match in all_drug_matches:
            drug_name = match.drug_name
            if drug_name not in drug_scores:
                drug_scores[drug_name] = {
                    'matches': [],
                    'total_score': 0.0,
                    'confidence': 0.0
                }
            
            drug_scores[drug_name]['matches'].append(match)
            drug_scores[drug_name]['total_score'] += match.overall_score
            drug_scores[drug_name]['confidence'] = max(
                drug_scores[drug_name]['confidence'], match.confidence
            )
        
        # Select primary drugs (highest combined scores)
        primary_candidates = sorted(
            drug_scores.items(), 
            key=lambda x: x[1]['total_score'], 
            reverse=True
        )[:3]
        
        primary_drugs = []
        for drug_name, scores in primary_candidates:
            # Use the best match for this drug
            best_match = max(scores['matches'], key=lambda m: m.overall_score)
            primary_drugs.append(best_match)
        
        # Select alternatives (good matches not in primary)
        alternative_drugs = []
        for drug_name, scores in sorted(drug_scores.items(), 
                                      key=lambda x: x[1]['total_score'], 
                                      reverse=True)[3:6]:
            best_match = max(scores['matches'], key=lambda m: m.overall_score)
            alternative_drugs.append(best_match)
        
        # Identify contraindicated drugs (very poor matches)
        contraindicated_drugs = [
            drug_name for drug_name, scores in drug_scores.items()
            if scores['total_score'] < 0.2 * len(scores['matches'])
        ]
        
        # Analyze drug combinations
        drug_combinations = self._analyze_drug_combinations(primary_drugs)
        
        # Generate interaction warnings
        interaction_warnings = self._generate_interaction_warnings(primary_drugs)
        
        # Identify biomarkers to monitor
        biomarkers_to_monitor = self._identify_monitoring_biomarkers(holes, primary_drugs)
        
        # Generate genetic considerations
        genetic_considerations = self._generate_genetic_considerations(holes)
        
        # Project improvements
        projected_improvement = self._project_improvements(holes, primary_drugs)
        
        return TherapeuticRecommendation(
            patient_id=patient_id,
            holes_analyzed=[h.hole_id for h in holes],
            primary_drugs=primary_drugs,
            alternative_drugs=alternative_drugs,
            contraindicated_drugs=contraindicated_drugs,
            drug_combinations=drug_combinations,
            interaction_warnings=interaction_warnings,
            biomarkers_to_monitor=biomarkers_to_monitor,
            monitoring_frequency="monthly",
            genetic_considerations=genetic_considerations,
            lifestyle_factors=[],  # Would be populated based on specific holes
            projected_improvement=projected_improvement,
            timeline_to_effect="2-4 weeks"
        )
    
    def _calculate_match_confidence(self, drug: DrugProperties, hole: OscillatoryHole,
                                  frequency_match: float, pathway_match: float) -> float:
        """Calculate confidence in the drug-hole match."""
        
        base_confidence = 0.6
        
        # High frequency and pathway match increases confidence
        strong_match_bonus = min(0.3, (frequency_match + pathway_match - 1.0) * 0.3)
        
        # Well-characterized pathways increase confidence
        pathway_confidence = {
            'neurotransmitter_signaling': 0.15,
            'gsk3_pathway': 0.20,
            'serotonin_signaling': 0.15,
            'dopamine_signaling': 0.15
        }
        
        pathway_bonus = pathway_confidence.get(hole.pathway, 0.05)
        
        # Genetic cause known increases confidence
        genetic_bonus = 0.1 if hole.genetic_cause else 0.0
        
        final_confidence = min(0.95, base_confidence + strong_match_bonus + 
                              pathway_bonus + genetic_bonus)
        
        return max(0.2, final_confidence)
    
    def _predict_hole_filling(self, overall_score: float, 
                            amplitude_compatibility: float,
                            hole: OscillatoryHole) -> float:
        """Predict how well the drug will fill the oscillatory hole."""
        
        # Base prediction on overall match quality
        base_filling = overall_score * 0.8
        
        # Amplitude compatibility is crucial for actual hole filling
        amplitude_factor = amplitude_compatibility * 0.6
        
        # Hole severity affects fillability
        severity_factor = (1.0 - hole.impact_severity * 0.3)
        
        predicted_filling = base_filling + amplitude_factor
        predicted_filling *= severity_factor
        
        return min(1.0, max(0.0, predicted_filling))
    
    def _predict_efficacy(self, hole_filling: float, 
                         impact_severity: float,
                         pathway_match: float) -> float:
        """Predict therapeutic efficacy."""
        
        # Efficacy depends on how much hole is filled and its clinical impact
        efficacy = hole_filling * impact_severity
        
        # Pathway relevance affects translation to clinical benefit
        pathway_factor = 0.7 + 0.3 * pathway_match
        
        return min(1.0, efficacy * pathway_factor)
    
    def _generate_mechanism_description(self, drug: DrugProperties,
                                      hole: OscillatoryHole,
                                      resonance_type: str,
                                      pathway_match: float) -> str:
        """Generate description of therapeutic mechanism."""
        
        base_desc = f"{drug.name} acts through {drug.mechanism_of_action}"
        
        if resonance_type.startswith("direct"):
            resonance_desc = f"provides direct oscillatory resonance at {hole.frequency:.2e} Hz"
        elif resonance_type.startswith("harmonic"):
            resonance_desc = f"provides harmonic resonance to restore {hole.pathway} oscillations"
        elif resonance_type.startswith("subharmonic"):
            resonance_desc = f"provides subharmonic drive to enhance {hole.pathway} dynamics"
        else:
            resonance_desc = "modulates pathway activity"
        
        pathway_desc = ""
        if pathway_match > 0.7:
            pathway_desc = f" with strong {hole.pathway} pathway engagement"
        elif pathway_match > 0.4:
            pathway_desc = f" with moderate {hole.pathway} pathway interaction"
        
        return f"{base_desc} and {resonance_desc}{pathway_desc}."
    
    def _calculate_optimal_dose_ratio(self, amplitude_compatibility: float,
                                    amplitude_deficit: float) -> float:
        """Calculate optimal dose relative to standard dosing."""
        
        # If amplitude matches well, use standard dose
        if amplitude_compatibility > 0.8:
            return 1.0
        
        # If drug is too weak for the deficit, may need higher dose
        if amplitude_deficit > 0.8:
            return min(1.5, 1.0 + amplitude_deficit * 0.5)
        
        # If drug is too strong, reduce dose
        if amplitude_compatibility < 0.5:
            return max(0.5, amplitude_compatibility + 0.3)
        
        return 1.0
    
    def _recommend_dosing_frequency(self, temporal_alignment: float,
                                   half_life: float,
                                   hole_variability: float) -> str:
        """Recommend dosing frequency."""
        
        if hole_variability > 0.7:  # Highly variable hole
            if half_life < 8:
                return "multiple_daily_doses"
            else:
                return "twice_daily"
        
        if temporal_alignment < 0.5:  # Poor temporal match
            if half_life < 6:
                return "frequent_dosing"
            elif half_life > 36:
                return "extended_release"
        
        return "standard"
    
    def _analyze_drug_combinations(self, primary_drugs: List[DrugHoleMatch]) -> List[Dict[str, Any]]:
        """Analyze potential drug combinations."""
        
        combinations = []
        
        if len(primary_drugs) < 2:
            return combinations
        
        # Analyze pairwise combinations
        for i in range(len(primary_drugs)):
            for j in range(i + 1, len(primary_drugs)):
                drug1 = primary_drugs[i]
                drug2 = primary_drugs[j]
                
                # Check for complementary mechanisms
                drug1_props = self.drug_db.get_drug(drug1.drug_name)
                drug2_props = self.drug_db.get_drug(drug2.drug_name)
                
                if drug1_props and drug2_props:
                    interaction = self.drug_db.get_drug_interactions(
                        drug1.drug_name, drug2.drug_name
                    )
                    
                    if interaction['interaction_risk'] in ['low', 'moderate']:
                        combinations.append({
                            'drugs': [drug1.drug_name, drug2.drug_name],
                            'rationale': 'Complementary mechanisms with acceptable interaction risk',
                            'interaction_risk': interaction['interaction_risk'],
                            'combined_score': (drug1.overall_score + drug2.overall_score) / 2
                        })
        
        return combinations
    
    def _generate_interaction_warnings(self, primary_drugs: List[DrugHoleMatch]) -> List[str]:
        """Generate interaction warnings for selected drugs."""
        
        warnings = []
        
        for i in range(len(primary_drugs)):
            for j in range(i + 1, len(primary_drugs)):
                drug1 = primary_drugs[i].drug_name
                drug2 = primary_drugs[j].drug_name
                
                interaction = self.drug_db.get_drug_interactions(drug1, drug2)
                
                if interaction['interaction_risk'] == 'high':
                    warnings.append(
                        f"High interaction risk between {drug1} and {drug2}: "
                        f"{interaction.get('reason', 'significant interaction potential')}"
                    )
                elif interaction['interaction_risk'] == 'moderate':
                    warnings.extend(interaction.get('recommendations', []))
        
        return warnings
    
    def _identify_monitoring_biomarkers(self, holes: List[OscillatoryHole],
                                      primary_drugs: List[DrugHoleMatch]) -> List[str]:
        """Identify biomarkers to monitor during treatment."""
        
        biomarkers = set()
        
        # Add biomarkers from holes
        for hole in holes:
            biomarkers.update(hole.biomarkers.keys())
        
        # Add drug-specific monitoring
        for drug_match in primary_drugs:
            drug = self.drug_db.get_drug(drug_match.drug_name)
            if drug:
                # Add pathway-specific biomarkers
                if 'lithium' in drug.name:
                    biomarkers.update(['lithium_level', 'creatinine', 'tsh'])
                elif 'serotonin' in drug.primary_pathways:
                    biomarkers.update(['serotonin_level', 'platelet_serotonin'])
                elif 'dopamine' in drug.primary_pathways:
                    biomarkers.update(['prolactin', 'metabolic_panel'])
        
        return list(biomarkers)
    
    def _generate_genetic_considerations(self, holes: List[OscillatoryHole]) -> List[str]:
        """Generate genetic considerations for therapy."""
        
        considerations = []
        
        for hole in holes:
            if hole.genetic_cause:
                considerations.append(
                    f"Genetic variant in {hole.genetic_cause} causing {hole.pathway} disruption"
                )
                
                if hole.variant_impact == 'HIGH':
                    considerations.append(
                        "High-impact variant may require enhanced therapeutic intervention"
                    )
        
        return considerations
    
    def _project_improvements(self, holes: List[OscillatoryHole],
                            primary_drugs: List[DrugHoleMatch]) -> Dict[str, float]:
        """Project expected improvements by pathway."""
        
        improvements = {}
        
        # Group holes by pathway
        pathway_holes = {}
        for hole in holes:
            if hole.pathway not in pathway_holes:
                pathway_holes[hole.pathway] = []
            pathway_holes[hole.pathway].append(hole)
        
        # Calculate improvement for each pathway
        for pathway, pathway_hole_list in pathway_holes.items():
            pathway_improvement = 0.0
            
            for hole in pathway_hole_list:
                # Find best drug for this hole
                best_drug = None
                best_filling = 0.0
                
                for drug_match in primary_drugs:
                    if drug_match.hole_id == hole.hole_id:
                        if drug_match.predicted_hole_filling > best_filling:
                            best_filling = drug_match.predicted_hole_filling
                            best_drug = drug_match
                
                if best_drug:
                    hole_improvement = best_filling * hole.impact_severity
                    pathway_improvement = max(pathway_improvement, hole_improvement)
            
            improvements[pathway] = pathway_improvement
        
        return improvements

def main():
    """
    Test the oscillatory hole matcher.
    
    Demonstrates matching drugs to oscillatory holes using
    frequency resonance and pathway analysis.
    """
    
    print("🎯 Testing Oscillatory Hole Matcher")
    print("=" * 50)
    
    # Initialize matcher with drug database
    matcher = OscillatoryHoleMatcher()
    
    print(f"Initialized with {len(matcher.drug_db.drugs)} drugs in database")
    
    # Create test oscillatory holes
    print("\n🕳️  Creating test oscillatory holes...")
    
    test_holes = [
        {
            'hole_id': 'hole_001',
            'pathway': 'inositol_metabolism',
            'frequency': 7.23e13,  # INPP1 frequency
            'amplitude_deficit': 0.7,
            'bandwidth': 5e12,
            'location': 'neuronal',
            'impact_severity': 0.8,
            'genetic_cause': 'INPP1',
            'variant_impact': 'HIGH',
            'associated_pathways': ['gsk3_pathway'],
            'phenotypic_effects': ['mood_instability', 'cognitive_impairment']
        },
        
        {
            'hole_id': 'hole_002', 
            'pathway': 'serotonin_signaling',
            'frequency': 1.45e13,
            'amplitude_deficit': 0.5,
            'bandwidth': 2e12,
            'location': 'synaptic',
            'impact_severity': 0.6,
            'genetic_cause': 'SLC6A4',
            'variant_impact': 'MODERATE',
            'phenotypic_effects': ['depression', 'anxiety']
        },
        
        {
            'hole_id': 'hole_003',
            'pathway': 'dopamine_signaling', 
            'frequency': 1.85e13,
            'amplitude_deficit': 0.8,
            'bandwidth': 3e12,
            'location': 'striatal',
            'impact_severity': 0.9,
            'genetic_cause': 'DRD2',
            'variant_impact': 'HIGH',
            'associated_pathways': ['gsk3_pathway'],
            'phenotypic_effects': ['psychosis', 'motor_dysfunction']
        }
    ]
    
    holes = [matcher.create_oscillatory_hole(hole_data) for hole_data in test_holes]
    
    for hole in holes:
        print(f"  • {hole.hole_id}: {hole.pathway} at {hole.frequency:.2e} Hz")
        print(f"    Deficit: {hole.amplitude_deficit:.1f}, Severity: {hole.impact_severity:.1f}")
        print(f"    Genetic cause: {hole.genetic_cause} ({hole.variant_impact})")
    
    # Test single hole matching
    print(f"\n🔍 Finding matches for individual holes...")
    
    for hole in holes[:2]:  # Test first two holes
        print(f"\nHole: {hole.hole_id} ({hole.pathway})")
        matches = matcher.find_best_matches_for_hole(hole, max_matches=3)
        
        print("Top matches:")
        for i, match in enumerate(matches):
            print(f"  {i+1}. {match.drug_name}:")
            print(f"     Overall score: {match.overall_score:.3f}")
            print(f"     Frequency match: {match.frequency_match:.3f} ({match.resonance_type})")
            print(f"     Pathway match: {match.pathway_match:.3f}")
            print(f"     Predicted hole filling: {match.predicted_hole_filling:.3f}")
            print(f"     Predicted efficacy: {match.predicted_efficacy:.3f}")
            print(f"     Confidence: {match.confidence:.3f}")
            print(f"     Dose ratio: {match.optimal_dose_ratio:.2f}x")
            print(f"     Mechanism: {match.mechanism_description}")
    
    # Test comprehensive therapeutic recommendation
    print(f"\n🏥 Generating comprehensive therapeutic recommendation...")
    
    patient_id = "test_patient_001"
    recommendation = matcher.generate_therapeutic_recommendation(patient_id, holes)
    
    print(f"\nTherapeutic Recommendation for Patient: {patient_id}")
    print(f"Holes analyzed: {len(recommendation.holes_analyzed)}")
    
    print(f"\nPrimary Drug Recommendations:")
    for i, drug_match in enumerate(recommendation.primary_drugs):
        print(f"  {i+1}. {drug_match.drug_name}")
        print(f"     Overall score: {drug_match.overall_score:.3f}")
        print(f"     Target hole: {drug_match.hole_id}")
        print(f"     Predicted efficacy: {drug_match.predicted_efficacy:.3f}")
        print(f"     Dosing: {drug_match.optimal_dose_ratio:.2f}x {drug_match.dosing_frequency}")
    
    print(f"\nAlternative Options:")
    for drug_match in recommendation.alternative_drugs:
        print(f"  • {drug_match.drug_name} (score: {drug_match.overall_score:.3f})")
    
    if recommendation.contraindicated_drugs:
        print(f"\nContraindicated Drugs:")
        for drug in recommendation.contraindicated_drugs:
            print(f"  • {drug}")
    
    if recommendation.drug_combinations:
        print(f"\nPotential Drug Combinations:")
        for combo in recommendation.drug_combinations:
            print(f"  • {' + '.join(combo['drugs'])}")
            print(f"    Rationale: {combo['rationale']}")
            print(f"    Interaction risk: {combo['interaction_risk']}")
    
    if recommendation.interaction_warnings:
        print(f"\n⚠️  Interaction Warnings:")
        for warning in recommendation.interaction_warnings:
            print(f"  • {warning}")
    
    print(f"\nMonitoring Recommendations:")
    print(f"  Biomarkers: {', '.join(recommendation.biomarkers_to_monitor)}")
    print(f"  Frequency: {recommendation.monitoring_frequency}")
    
    if recommendation.genetic_considerations:
        print(f"\nGenetic Considerations:")
        for consideration in recommendation.genetic_considerations:
            print(f"  • {consideration}")
    
    print(f"\nProjected Improvements:")
    for pathway, improvement in recommendation.projected_improvement.items():
        print(f"  • {pathway}: {improvement:.1%} improvement expected")
    
    print(f"  Timeline to effect: {recommendation.timeline_to_effect}")
    
    # Test frequency matching details
    print(f"\n🎵 Detailed Frequency Matching Analysis...")
    
    test_drug = "lithium"
    test_hole = holes[0]  # Inositol metabolism hole
    
    drug_props = matcher.drug_db.get_drug(test_drug)
    if drug_props:
        freq_match, resonance = matcher.calculate_frequency_match(drug_props, test_hole)
        pathway_match = matcher.calculate_pathway_match(drug_props, test_hole)
        amplitude_compat = matcher.calculate_amplitude_compatibility(drug_props, test_hole)
        temporal_align = matcher.calculate_temporal_alignment(drug_props, test_hole)
        
        print(f"\nDetailed analysis: {test_drug} vs {test_hole.hole_id}")
        print(f"  Drug frequency: {drug_props.primary_frequency:.2e} Hz")
        print(f"  Hole frequency: {test_hole.frequency:.2e} Hz")
        print(f"  Frequency match: {freq_match:.3f} ({resonance})")
        print(f"  Pathway match: {pathway_match:.3f}")
        print(f"  Amplitude compatibility: {amplitude_compat:.3f}")
        print(f"  Temporal alignment: {temporal_align:.3f}")
        
        print(f"\n  Drug pathways: {', '.join(drug_props.primary_pathways)}")
        print(f"  Hole pathway: {test_hole.pathway}")
        print(f"  Drug amplitude: {drug_props.oscillatory_amplitude:.2f}")
        print(f"  Hole deficit: {test_hole.amplitude_deficit:.2f}")
        print(f"  Drug half-life: {drug_props.half_life:.1f} hours")
    
    print(f"\n✅ Oscillatory hole matcher testing complete!")
    print("🎯 Successfully matched drugs to oscillatory holes using")
    print("   frequency resonance and pathway analysis.")
    
    return matcher, recommendation

if __name__ == "__main__":
    matcher, recommendation = main()
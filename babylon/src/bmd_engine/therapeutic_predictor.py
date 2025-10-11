"""
Therapeutic Predictor Module
===========================

Predicts therapeutic amplification and clinical outcomes using the complete BMD
(Biological Maxwell Demon) framework. Integrates drug database, hole matching,
and resonance calculations to provide comprehensive therapeutic predictions.

Part of the St. Stellas BMD Engine for therapeutic prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta

# Import from other BMD modules
from .drug_database import DrugDatabase, DrugProperties
from .hole_matcher import OscillatoryHoleMatcher, OscillatoryHole, DrugHoleMatch, TherapeuticRecommendation
from .resonance_calculator import ResonanceCalculator, OscillatorySpectrum, ResonanceAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TherapeuticOutcome:
    """Predicted therapeutic outcome for a specific intervention."""
    intervention_id: str
    patient_id: str
    drug_name: str
    target_holes: List[str]
    
    # Predicted efficacy metrics
    symptom_improvement: Dict[str, float] = field(default_factory=dict)  # symptom -> % improvement
    biomarker_changes: Dict[str, float] = field(default_factory=dict)   # biomarker -> expected change
    functional_improvements: Dict[str, float] = field(default_factory=dict)  # domain -> improvement
    
    # Predicted timeline
    onset_time: float = 7.0        # days to onset
    peak_effect_time: float = 28.0  # days to peak effect
    duration_of_effect: float = 365.0  # days of sustained effect
    
    # Confidence and reliability
    prediction_confidence: float = 0.7  # 0-1 confidence in prediction
    evidence_quality: str = "moderate"  # low, moderate, high
    
    # Risk assessment
    side_effect_probability: Dict[str, float] = field(default_factory=dict)
    contraindication_risk: float = 0.0  # 0-1 overall risk
    monitoring_requirements: List[str] = field(default_factory=list)
    
    # Personalization factors
    genetic_modifiers: List[str] = field(default_factory=list)
    comorbidity_interactions: List[str] = field(default_factory=list)

@dataclass
class AmplificationAnalysis:
    """Analysis of therapeutic amplification mechanisms."""
    amplification_type: str  # "resonant", "additive", "synergistic", "cascade"
    amplification_factor: float  # Multiplicative enhancement
    
    # Mechanism details
    primary_mechanism: str
    secondary_mechanisms: List[str] = field(default_factory=list)
    
    # Quantitative factors
    frequency_coupling: float = 0.0    # 0-1 - frequency-based amplification
    pathway_synergy: float = 0.0       # 0-1 - pathway interaction enhancement
    temporal_optimization: float = 0.0  # 0-1 - timing-based enhancement
    dose_optimization: float = 0.0     # 0-1 - dose-response optimization
    
    # Stability and sustainability
    amplification_stability: float = 0.8  # 0-1 - stability over time
    tolerance_risk: float = 0.1         # 0-1 - risk of tolerance development
    
    # Clinical relevance
    clinical_significance: float = 0.0   # 0-1 - clinical importance
    therapeutic_window: Tuple[float, float] = (1.0, 5.0)  # (min, max) amplification range

@dataclass
class CombinationTherapyPrediction:
    """Prediction for combination therapeutic approaches."""
    combination_id: str
    drugs: List[str]
    target_holes: List[str]
    
    # Interaction analysis
    drug_interactions: Dict[str, str] = field(default_factory=dict)  # pair -> interaction type
    synergy_score: float = 0.0      # -1 to 1 (antagonistic to synergistic)
    
    # Combined outcomes
    combined_efficacy: float = 0.0   # Overall efficacy prediction
    interaction_amplification: float = 1.0  # Multiplicative factor from interactions
    
    # Safety considerations
    interaction_risks: List[str] = field(default_factory=list)
    monitoring_complexity: float = 1.0  # Relative monitoring complexity
    
    # Optimization
    optimal_dosing_ratios: Dict[str, float] = field(default_factory=dict)
    optimal_timing: Dict[str, float] = field(default_factory=dict)  # drug -> time offset

class TherapeuticPredictor:
    """
    Comprehensive therapeutic outcome predictor using the BMD framework.
    
    Integrates all BMD components to predict therapeutic amplification,
    clinical outcomes, and optimal treatment strategies.
    """
    
    def __init__(self, drug_database: Optional[DrugDatabase] = None,
                 hole_matcher: Optional[OscillatoryHoleMatcher] = None,
                 resonance_calculator: Optional[ResonanceCalculator] = None):
        """Initialize therapeutic predictor with BMD components."""
        
        # Initialize or use provided components
        self.drug_db = drug_database if drug_database else DrugDatabase()
        self.hole_matcher = hole_matcher if hole_matcher else OscillatoryHoleMatcher(self.drug_db)
        self.resonance_calc = resonance_calculator if resonance_calculator else ResonanceCalculator()
        
        # Prediction models and parameters
        self.baseline_amplification = 1.0
        self.max_amplification = 10.0
        self.synergy_threshold = 0.6
        self.antagonism_threshold = -0.3
        
        # Clinical correlation factors
        self.clinical_correlation = {
            'frequency_match': 0.4,     # Frequency matching importance
            'pathway_match': 0.3,       # Pathway relevance
            'amplitude_match': 0.2,     # Amplitude compatibility
            'temporal_match': 0.1       # Temporal alignment
        }
        
        logger.info("Therapeutic predictor initialized with BMD framework")
    
    def predict_therapeutic_outcome(self, patient_id: str, drug_name: str,
                                  holes: List[OscillatoryHole],
                                  patient_factors: Optional[Dict[str, Any]] = None) -> TherapeuticOutcome:
        """Predict comprehensive therapeutic outcome for a patient-drug combination."""
        
        logger.info(f"Predicting therapeutic outcome: {drug_name} for patient {patient_id}")
        
        if patient_factors is None:
            patient_factors = {}
        
        # Get drug properties
        drug = self.drug_db.get_drug(drug_name)
        if not drug:
            raise ValueError(f"Drug {drug_name} not found in database")
        
        # Find best hole matches
        hole_matches = []
        for hole in holes:
            match = self.hole_matcher.match_drug_to_hole(drug_name, hole)
            hole_matches.append(match)
        
        # Analyze amplification mechanisms
        amplification_analysis = self.analyze_therapeutic_amplification(drug, holes, hole_matches)
        
        # Predict symptom improvements
        symptom_improvements = self._predict_symptom_improvements(
            drug, holes, hole_matches, amplification_analysis
        )
        
        # Predict biomarker changes
        biomarker_changes = self._predict_biomarker_changes(
            drug, holes, hole_matches, amplification_analysis
        )
        
        # Predict functional improvements
        functional_improvements = self._predict_functional_improvements(
            holes, hole_matches, amplification_analysis
        )
        
        # Predict timeline
        timeline = self._predict_therapeutic_timeline(
            drug, amplification_analysis, patient_factors
        )
        
        # Assess confidence and evidence quality
        confidence, evidence_quality = self._assess_prediction_confidence(
            drug, holes, hole_matches, amplification_analysis
        )
        
        # Assess risks
        side_effect_probs = self._predict_side_effect_probabilities(
            drug, amplification_analysis, patient_factors
        )
        
        contraindication_risk = self._assess_contraindication_risk(
            drug, patient_factors, amplification_analysis
        )
        
        # Generate monitoring requirements
        monitoring_requirements = self._generate_monitoring_requirements(
            drug, holes, amplification_analysis
        )
        
        # Identify genetic and comorbidity modifiers
        genetic_modifiers = self._identify_genetic_modifiers(patient_factors, drug)
        comorbidity_interactions = self._identify_comorbidity_interactions(patient_factors, drug)
        
        return TherapeuticOutcome(
            intervention_id=f"{patient_id}_{drug_name}_{datetime.now().strftime('%Y%m%d')}",
            patient_id=patient_id,
            drug_name=drug_name,
            target_holes=[hole.hole_id for hole in holes],
            symptom_improvement=symptom_improvements,
            biomarker_changes=biomarker_changes,
            functional_improvements=functional_improvements,
            onset_time=timeline['onset'],
            peak_effect_time=timeline['peak'],
            duration_of_effect=timeline['duration'],
            prediction_confidence=confidence,
            evidence_quality=evidence_quality,
            side_effect_probability=side_effect_probs,
            contraindication_risk=contraindication_risk,
            monitoring_requirements=monitoring_requirements,
            genetic_modifiers=genetic_modifiers,
            comorbidity_interactions=comorbidity_interactions
        )
    
    def analyze_therapeutic_amplification(self, drug: DrugProperties, 
                                        holes: List[OscillatoryHole],
                                        hole_matches: List[DrugHoleMatch]) -> AmplificationAnalysis:
        """Analyze therapeutic amplification mechanisms and factors."""
        
        # Create oscillatory spectra for analysis
        drug_data = {
            'name': drug.name,
            'primary_frequency': drug.primary_frequency,
            'secondary_frequencies': drug.secondary_frequencies,
            'oscillatory_amplitude': drug.oscillatory_amplitude,
            'frequency_bandwidth': drug.frequency_bandwidth
        }
        
        drug_spectrum = self.resonance_calc.create_spectrum_from_drug(drug_data)
        
        # Analyze resonance with each hole
        resonance_analyses = []
        for hole in holes:
            hole_data = {
                'hole_id': hole.hole_id,
                'frequency': hole.frequency,
                'amplitude_deficit': hole.amplitude_deficit,
                'bandwidth': hole.bandwidth
            }
            hole_spectrum = self.resonance_calc.create_spectrum_from_hole(hole_data)
            
            resonance = self.resonance_calc.perform_comprehensive_resonance_analysis(
                drug_spectrum, hole_spectrum
            )
            resonance_analyses.append(resonance)
        
        # Determine amplification type and factors
        amplification_type, amplification_factor = self._determine_amplification_type(
            resonance_analyses, hole_matches
        )
        
        # Analyze individual amplification components
        frequency_coupling = np.mean([r.frequency_match_score for r in resonance_analyses])
        pathway_synergy = np.mean([match.pathway_match for match in hole_matches])
        temporal_optimization = np.mean([match.temporal_alignment for match in hole_matches])
        dose_optimization = self._calculate_dose_optimization(drug, holes, hole_matches)
        
        # Primary mechanism determination
        primary_mechanism = self._determine_primary_mechanism(
            amplification_type, resonance_analyses, hole_matches
        )
        
        # Secondary mechanisms
        secondary_mechanisms = self._identify_secondary_mechanisms(
            resonance_analyses, hole_matches, drug
        )
        
        # Stability and tolerance assessment
        amplification_stability = self._assess_amplification_stability(
            resonance_analyses, drug, holes
        )
        
        tolerance_risk = self._assess_tolerance_risk(drug, amplification_factor)
        
        # Clinical significance
        clinical_significance = self._assess_clinical_significance(
            amplification_factor, holes, hole_matches
        )
        
        # Therapeutic window
        therapeutic_window = self._calculate_therapeutic_window(
            amplification_factor, drug, holes
        )
        
        return AmplificationAnalysis(
            amplification_type=amplification_type,
            amplification_factor=amplification_factor,
            primary_mechanism=primary_mechanism,
            secondary_mechanisms=secondary_mechanisms,
            frequency_coupling=frequency_coupling,
            pathway_synergy=pathway_synergy,
            temporal_optimization=temporal_optimization,
            dose_optimization=dose_optimization,
            amplification_stability=amplification_stability,
            tolerance_risk=tolerance_risk,
            clinical_significance=clinical_significance,
            therapeutic_window=therapeutic_window
        )
    
    def predict_combination_therapy(self, patient_id: str, 
                                  drug_combinations: List[List[str]],
                                  holes: List[OscillatoryHole],
                                  patient_factors: Optional[Dict[str, Any]] = None) -> List[CombinationTherapyPrediction]:
        """Predict outcomes for combination therapy approaches."""
        
        combination_predictions = []
        
        for i, drug_combination in enumerate(drug_combinations):
            combo_id = f"combo_{i+1}_{patient_id}"
            
            # Analyze drug interactions
            drug_interactions = self._analyze_drug_interactions(drug_combination)
            
            # Calculate synergy score
            synergy_score = self._calculate_synergy_score(drug_combination, holes)
            
            # Predict combined efficacy
            individual_outcomes = []
            for drug_name in drug_combination:
                outcome = self.predict_therapeutic_outcome(
                    patient_id, drug_name, holes, patient_factors
                )
                individual_outcomes.append(outcome)
            
            combined_efficacy, interaction_amplification = self._calculate_combined_efficacy(
                individual_outcomes, synergy_score, drug_interactions
            )
            
            # Assess interaction risks
            interaction_risks = self._assess_interaction_risks(drug_combination, drug_interactions)
            
            # Calculate monitoring complexity
            monitoring_complexity = len(drug_combination) * 1.2 + len(interaction_risks) * 0.3
            
            # Optimize dosing and timing
            optimal_dosing_ratios = self._optimize_combination_dosing(
                drug_combination, synergy_score, drug_interactions
            )
            
            optimal_timing = self._optimize_combination_timing(drug_combination)
            
            combination_predictions.append(CombinationTherapyPrediction(
                combination_id=combo_id,
                drugs=drug_combination,
                target_holes=[hole.hole_id for hole in holes],
                drug_interactions=drug_interactions,
                synergy_score=synergy_score,
                combined_efficacy=combined_efficacy,
                interaction_amplification=interaction_amplification,
                interaction_risks=interaction_risks,
                monitoring_complexity=monitoring_complexity,
                optimal_dosing_ratios=optimal_dosing_ratios,
                optimal_timing=optimal_timing
            ))
        
        return combination_predictions
    
    def optimize_therapeutic_strategy(self, patient_id: str,
                                    holes: List[OscillatoryHole],
                                    available_drugs: List[str],
                                    patient_factors: Optional[Dict[str, Any]] = None,
                                    max_drugs: int = 3) -> Dict[str, Any]:
        """Find optimal therapeutic strategy for a patient."""
        
        logger.info(f"Optimizing therapeutic strategy for patient {patient_id}")
        
        # Predict outcomes for all single drugs
        single_drug_outcomes = {}
        for drug_name in available_drugs:
            try:
                outcome = self.predict_therapeutic_outcome(
                    patient_id, drug_name, holes, patient_factors
                )
                single_drug_outcomes[drug_name] = outcome
            except Exception as e:
                logger.warning(f"Error predicting outcome for {drug_name}: {e}")
        
        # Generate and evaluate combination strategies
        combination_strategies = []
        if max_drugs > 1:
            # Generate 2-drug combinations
            for i, drug1 in enumerate(available_drugs):
                for drug2 in available_drugs[i+1:]:
                    combination_strategies.append([drug1, drug2])
            
            # Generate 3-drug combinations if allowed
            if max_drugs >= 3:
                for i, drug1 in enumerate(available_drugs):
                    for j, drug2 in enumerate(available_drugs[i+1:], i+1):
                        for drug3 in available_drugs[j+1:]:
                            combination_strategies.append([drug1, drug2, drug3])
        
        # Predict combination outcomes
        combination_outcomes = []
        if combination_strategies:
            combination_outcomes = self.predict_combination_therapy(
                patient_id, combination_strategies[:10], holes, patient_factors  # Limit to 10 combos
            )
        
        # Rank all strategies
        ranked_strategies = self._rank_therapeutic_strategies(
            single_drug_outcomes, combination_outcomes, holes
        )
        
        # Generate optimization insights
        optimization_insights = self._generate_optimization_insights(
            ranked_strategies, holes, patient_factors
        )
        
        return {
            'patient_id': patient_id,
            'single_drug_outcomes': single_drug_outcomes,
            'combination_outcomes': combination_outcomes,
            'ranked_strategies': ranked_strategies,
            'optimization_insights': optimization_insights,
            'recommendation': ranked_strategies[0] if ranked_strategies else None
        }
    
    def _determine_amplification_type(self, resonance_analyses: List[ResonanceAnalysis],
                                    hole_matches: List[DrugHoleMatch]) -> Tuple[str, float]:
        """Determine the type and magnitude of therapeutic amplification."""
        
        # Analyze resonance patterns
        strong_resonances = [r for r in resonance_analyses if r.resonance_strength > 0.7]
        moderate_resonances = [r for r in resonance_analyses if 0.4 <= r.resonance_strength <= 0.7]
        
        # Analyze hole filling efficiency
        high_filling = [m for m in hole_matches if m.predicted_hole_filling > 0.7]
        moderate_filling = [m for m in hole_matches if 0.4 <= m.predicted_hole_filling <= 0.7]
        
        # Determine amplification type
        if len(strong_resonances) > 0 and len(high_filling) > 0:
            amplification_type = "resonant"
            # Resonant amplification can be very strong
            base_factor = np.mean([r.power_amplification for r in resonance_analyses])
            amplification_factor = min(self.max_amplification, base_factor * 1.5)
        
        elif len(moderate_resonances) >= 2 or len(moderate_filling) >= 2:
            amplification_type = "synergistic"
            # Multiple moderate effects combine synergistically
            synergy_bonus = min(3.0, 1.0 + len(moderate_resonances) * 0.3)
            amplification_factor = self.baseline_amplification * synergy_bonus
        
        elif len(strong_resonances) > 0 or len(high_filling) > 0:
            amplification_type = "additive"
            # Single strong effect provides additive benefit
            max_resonance = max([r.resonance_strength for r in resonance_analyses])
            max_filling = max([m.predicted_hole_filling for m in hole_matches])
            amplification_factor = self.baseline_amplification + max(max_resonance, max_filling)
        
        else:
            amplification_type = "cascade"
            # Weak direct effects may still trigger cascades
            cascade_potential = np.mean([m.predicted_efficacy for m in hole_matches])
            amplification_factor = self.baseline_amplification + cascade_potential * 0.5
        
        return amplification_type, min(self.max_amplification, amplification_factor)
    
    def _predict_symptom_improvements(self, drug: DrugProperties,
                                    holes: List[OscillatoryHole],
                                    hole_matches: List[DrugHoleMatch],
                                    amplification: AmplificationAnalysis) -> Dict[str, float]:
        """Predict symptom improvements based on hole filling and amplification."""
        
        symptom_improvements = {}
        
        # Map holes to symptoms
        hole_symptom_map = {
            'inositol_metabolism': ['mood_swings', 'irritability', 'depression'],
            'serotonin_signaling': ['depression', 'anxiety', 'sleep_disturbance'],
            'dopamine_signaling': ['anhedonia', 'motivation', 'psychosis', 'movement'],
            'gsk3_pathway': ['cognitive_function', 'memory', 'mood_stability'],
            'gaba_signaling': ['anxiety', 'sleep', 'seizures'],
            'neurotransmitter_signaling': ['mood', 'cognition', 'behavior']
        }
        
        # Calculate improvements for each symptom
        for hole, match in zip(holes, hole_matches):
            pathway = hole.pathway
            hole_filling = match.predicted_hole_filling
            
            if pathway in hole_symptom_map:
                for symptom in hole_symptom_map[pathway]:
                    # Base improvement from hole filling
                    base_improvement = hole_filling * hole.impact_severity
                    
                    # Apply amplification
                    amplified_improvement = base_improvement * amplification.amplification_factor
                    
                    # Limit to realistic range (0-80% improvement)
                    final_improvement = min(0.8, amplified_improvement) * 100
                    
                    # Combine with existing predictions (take maximum)
                    if symptom in symptom_improvements:
                        symptom_improvements[symptom] = max(
                            symptom_improvements[symptom], final_improvement
                        )
                    else:
                        symptom_improvements[symptom] = final_improvement
        
        return symptom_improvements
    
    def _predict_biomarker_changes(self, drug: DrugProperties,
                                 holes: List[OscillatoryHole],
                                 hole_matches: List[DrugHoleMatch],
                                 amplification: AmplificationAnalysis) -> Dict[str, float]:
        """Predict biomarker changes from therapeutic intervention."""
        
        biomarker_changes = {}
        
        # Drug-specific biomarker effects
        drug_biomarker_map = {
            'lithium': {
                'inositol_1P': -30.0,      # % decrease in inositol-1-phosphate
                'gsk3_activity': -40.0,    # % decrease in GSK3 activity
                'lithium_level': 0.8,      # mEq/L target level
                'creatinine': 5.0          # % increase (side effect)
            },
            'aripiprazole': {
                'prolactin': 15.0,         # % increase
                'dopamine_turnover': -20.0, # % decrease in turnover
                'metabolic_markers': 8.0    # % worsening
            },
            'citalopram': {
                'serotonin_level': 25.0,   # % increase
                'platelet_serotonin': -30.0, # % decrease (uptake blocked)
                'qtc_interval': 10.0        # ms increase
            }
        }
        
        # Get drug-specific effects
        if drug.name in drug_biomarker_map:
            base_changes = drug_biomarker_map[drug.name]
            
            # Apply amplification to therapeutic markers
            for biomarker, change in base_changes.items():
                if change < 0 or 'level' in biomarker.lower():  # Therapeutic effects
                    amplified_change = change * amplification.amplification_factor
                else:  # Side effects - amplification may increase these too
                    amplified_change = change * (1.0 + amplification.amplification_factor * 0.2)
                
                biomarker_changes[biomarker] = amplified_change
        
        # Pathway-specific biomarker changes
        pathway_biomarkers = {
            'inositol_metabolism': {'inositol_ratio': -25.0, 'ip3_levels': 40.0},
            'serotonin_signaling': {'5hiaa': 20.0, 'tryptophan': -10.0},
            'dopamine_signaling': {'hva': 15.0, 'dopamine_metabolites': -20.0}
        }
        
        # Add pathway-specific changes
        for hole, match in zip(holes, hole_matches):
            pathway = hole.pathway
            if pathway in pathway_biomarkers:
                for biomarker, base_change in pathway_biomarkers[pathway].items():
                    # Scale by hole filling and amplification
                    scaled_change = base_change * match.predicted_hole_filling * amplification.amplification_factor * 0.5
                    
                    if biomarker in biomarker_changes:
                        biomarker_changes[biomarker] += scaled_change
                    else:
                        biomarker_changes[biomarker] = scaled_change
        
        return biomarker_changes
    
    def _predict_functional_improvements(self, holes: List[OscillatoryHole],
                                       hole_matches: List[DrugHoleMatch],
                                       amplification: AmplificationAnalysis) -> Dict[str, float]:
        """Predict functional domain improvements."""
        
        functional_improvements = {}
        
        # Map pathways to functional domains
        pathway_functions = {
            'inositol_metabolism': {'executive_function': 0.6, 'emotional_regulation': 0.8},
            'serotonin_signaling': {'mood_regulation': 0.9, 'sleep_quality': 0.7, 'appetite': 0.6},
            'dopamine_signaling': {'motivation': 0.8, 'motor_function': 0.7, 'reward_processing': 0.9},
            'gsk3_pathway': {'cognitive_flexibility': 0.7, 'memory': 0.6, 'neuroplasticity': 0.8},
            'gaba_signaling': {'anxiety_control': 0.9, 'sleep_initiation': 0.8, 'seizure_threshold': 0.7},
            'neurotransmitter_signaling': {'overall_brain_function': 0.5, 'connectivity': 0.6}
        }
        
        # Calculate functional improvements
        for hole, match in zip(holes, hole_matches):
            pathway = hole.pathway
            if pathway in pathway_functions:
                for function, max_improvement in pathway_functions[pathway].items():
                    # Base improvement from hole filling
                    base_improvement = match.predicted_hole_filling * hole.impact_severity * max_improvement
                    
                    # Apply amplification
                    amplified_improvement = base_improvement * amplification.amplification_factor
                    
                    # Convert to percentage and limit
                    improvement_percent = min(70.0, amplified_improvement * 100)
                    
                    # Combine with existing improvements
                    if function in functional_improvements:
                        functional_improvements[function] = max(
                            functional_improvements[function], improvement_percent
                        )
                    else:
                        functional_improvements[function] = improvement_percent
        
        return functional_improvements
    
    def _predict_therapeutic_timeline(self, drug: DrugProperties,
                                    amplification: AmplificationAnalysis,
                                    patient_factors: Dict[str, Any]) -> Dict[str, float]:
        """Predict therapeutic timeline based on drug properties and amplification."""
        
        # Base timeline from drug properties
        base_onset = 7.0  # days
        base_peak = 28.0  # days
        base_duration = 365.0  # days
        
        # Drug-specific modifiers
        drug_timeline_modifiers = {
            'lithium': {'onset_multiplier': 2.0, 'peak_multiplier': 1.5, 'duration_multiplier': 1.2},
            'aripiprazole': {'onset_multiplier': 0.5, 'peak_multiplier': 0.8, 'duration_multiplier': 1.0},
            'citalopram': {'onset_multiplier': 1.5, 'peak_multiplier': 1.2, 'duration_multiplier': 1.0},
            'atorvastatin': {'onset_multiplier': 0.3, 'peak_multiplier': 0.5, 'duration_multiplier': 1.5},
            'aspirin': {'onset_multiplier': 0.02, 'peak_multiplier': 0.1, 'duration_multiplier': 0.5}
        }
        
        modifiers = drug_timeline_modifiers.get(drug.name, 
            {'onset_multiplier': 1.0, 'peak_multiplier': 1.0, 'duration_multiplier': 1.0})
        
        # Apply drug-specific modifiers
        onset = base_onset * modifiers['onset_multiplier']
        peak = base_peak * modifiers['peak_multiplier']
        duration = base_duration * modifiers['duration_multiplier']
        
        # Amplification effects on timeline
        if amplification.amplification_type == "resonant":
            # Resonant amplification may speed onset
            onset *= 0.7
            peak *= 0.8
        elif amplification.amplification_type == "synergistic":
            # Synergistic effects may take longer to develop
            onset *= 1.2
            peak *= 1.3
        
        # High amplification may speed things up
        if amplification.amplification_factor > 3.0:
            onset *= 0.8
            peak *= 0.9
        
        # Patient factor modifiers
        age = patient_factors.get('age', 50)
        if age > 65:
            onset *= 1.3  # Slower onset in elderly
            peak *= 1.2
        elif age < 25:
            onset *= 0.8  # Faster response in young adults
        
        # Genetic factors
        if 'fast_metabolizer' in patient_factors.get('genetic_factors', []):
            onset *= 0.7
            duration *= 0.8
        elif 'slow_metabolizer' in patient_factors.get('genetic_factors', []):
            onset *= 1.5
            duration *= 1.3
        
        return {
            'onset': max(0.1, onset),  # At least 2.4 hours
            'peak': max(1.0, peak),    # At least 1 day
            'duration': max(7.0, duration)  # At least 1 week
        }
    
    def _assess_prediction_confidence(self, drug: DrugProperties,
                                    holes: List[OscillatoryHole],
                                    hole_matches: List[DrugHoleMatch],
                                    amplification: AmplificationAnalysis) -> Tuple[float, str]:
        """Assess confidence in therapeutic predictions."""
        
        base_confidence = 0.6
        
        # Drug database confidence
        drug_confidence_factors = {
            'lithium': 0.9,      # Well-studied
            'aspirin': 0.9,      # Very well-studied
            'atorvastatin': 0.8, # Well-studied
            'aripiprazole': 0.7, # Moderately studied
            'citalopram': 0.7    # Moderately studied
        }
        
        drug_confidence = drug_confidence_factors.get(drug.name, 0.5)
        
        # Hole characterization confidence
        hole_confidence = np.mean([
            0.9 if hole.genetic_cause else 0.6 for hole in holes
        ])
        
        # Match quality confidence
        match_confidence = np.mean([match.confidence for match in hole_matches])
        
        # Amplification mechanism confidence
        amplification_confidence_map = {
            'resonant': 0.8,     # Well-understood mechanism
            'synergistic': 0.7,  # Moderately understood
            'additive': 0.9,     # Simple mechanism
            'cascade': 0.5       # Complex, less predictable
        }
        
        amplification_confidence = amplification_confidence_map.get(
            amplification.amplification_type, 0.6
        )
        
        # Overall confidence calculation
        overall_confidence = (base_confidence * 0.2 + 
                            drug_confidence * 0.3 + 
                            hole_confidence * 0.2 + 
                            match_confidence * 0.2 + 
                            amplification_confidence * 0.1)
        
        # Evidence quality assessment
        if overall_confidence > 0.8:
            evidence_quality = "high"
        elif overall_confidence > 0.6:
            evidence_quality = "moderate"
        else:
            evidence_quality = "low"
        
        return min(0.95, overall_confidence), evidence_quality
    
    def _predict_side_effect_probabilities(self, drug: DrugProperties,
                                         amplification: AmplificationAnalysis,
                                         patient_factors: Dict[str, Any]) -> Dict[str, float]:
        """Predict side effect probabilities."""
        
        # Base side effect profiles from drug database
        base_side_effects = drug.side_effect_profiles.copy()
        
        # Amplification may increase side effect risk
        amplification_multiplier = 1.0 + (amplification.amplification_factor - 1.0) * 0.3
        
        modified_side_effects = {}
        for side_effect, base_prob in base_side_effects.items():
            # Apply amplification multiplier
            modified_prob = base_prob * amplification_multiplier
            
            # Patient-specific modifiers
            age = patient_factors.get('age', 50)
            if age > 65:
                modified_prob *= 1.3  # Increased risk in elderly
            
            # Genetic factors
            genetic_factors = patient_factors.get('genetic_factors', [])
            if 'poor_metabolizer' in genetic_factors:
                modified_prob *= 1.5
            
            modified_side_effects[side_effect] = min(0.9, modified_prob)
        
        return modified_side_effects
    
    def _assess_contraindication_risk(self, drug: DrugProperties,
                                    patient_factors: Dict[str, Any],
                                    amplification: AmplificationAnalysis) -> float:
        """Assess overall contraindication risk."""
        
        base_risk = 0.1  # 10% base risk
        
        # Drug-specific risks
        drug_specific_risks = {
            'lithium': 0.3,      # Higher risk due to narrow therapeutic window
            'aripiprazole': 0.2,  # Metabolic and movement disorder risks
            'citalopram': 0.15,   # QT prolongation risk
            'atorvastatin': 0.1,  # Muscle toxicity risk
            'aspirin': 0.2        # Bleeding risk
        }
        
        drug_risk = drug_specific_risks.get(drug.name, base_risk)
        
        # Patient factor modifications
        age = patient_factors.get('age', 50)
        if age > 75:
            drug_risk *= 1.5
        
        comorbidities = patient_factors.get('comorbidities', [])
        high_risk_combos = {
            'kidney_disease': ['lithium'],
            'heart_disease': ['citalopram', 'aripiprazole'],
            'liver_disease': ['atorvastatin'],
            'bleeding_disorder': ['aspirin']
        }
        
        for condition in comorbidities:
            if condition in high_risk_combos and drug.name in high_risk_combos[condition]:
                drug_risk *= 2.0
        
        # Amplification may increase risk
        if amplification.amplification_factor > 5.0:
            drug_risk *= 1.3
        
        return min(0.8, drug_risk)
    
    def _generate_monitoring_requirements(self, drug: DrugProperties,
                                        holes: List[OscillatoryHole],
                                        amplification: AmplificationAnalysis) -> List[str]:
        """Generate monitoring requirements for therapeutic intervention."""
        
        monitoring_requirements = []
        
        # Drug-specific monitoring
        drug_monitoring = {
            'lithium': ['Serum lithium levels', 'Kidney function', 'Thyroid function'],
            'aripiprazole': ['Metabolic parameters', 'Movement assessment', 'Prolactin levels'],
            'citalopram': ['ECG/QTc interval', 'Sodium levels', 'Mood assessment'],
            'atorvastatin': ['Liver enzymes', 'Muscle symptoms', 'Lipid levels'],
            'aspirin': ['Bleeding assessment', 'GI symptoms', 'Platelet function']
        }
        
        if drug.name in drug_monitoring:
            monitoring_requirements.extend(drug_monitoring[drug.name])
        
        # Amplification-specific monitoring
        if amplification.amplification_factor > 3.0:
            monitoring_requirements.append('Enhanced efficacy monitoring')
        
        if amplification.tolerance_risk > 0.3:
            monitoring_requirements.append('Tolerance development assessment')
        
        # Pathway-specific monitoring
        pathway_monitoring = {
            'inositol_metabolism': ['Inositol metabolite levels'],
            'serotonin_signaling': ['Serotonin function tests'],
            'dopamine_signaling': ['Dopamine function assessment'],
            'gsk3_pathway': ['Cognitive function tests']
        }
        
        for hole in holes:
            if hole.pathway in pathway_monitoring:
                monitoring_requirements.extend(pathway_monitoring[hole.pathway])
        
        # Remove duplicates
        return list(set(monitoring_requirements))
    
    def _identify_genetic_modifiers(self, patient_factors: Dict[str, Any],
                                  drug: DrugProperties) -> List[str]:
        """Identify genetic factors that may modify drug response."""
        
        genetic_modifiers = []
        
        genetic_variants = patient_factors.get('genetic_variants', [])
        
        # Drug-specific genetic modifiers
        drug_genetic_modifiers = {
            'lithium': ['INPP1', 'GSK3B', 'SLC1A2'],
            'aripiprazole': ['CYP2D6', 'DRD2', 'HTR2A'],
            'citalopram': ['CYP2C19', 'SLC6A4', 'HTR2A'],
            'atorvastatin': ['SLCO1B1', 'CYP3A4'],
            'aspirin': ['CYP2C9', 'PTGS1']
        }
        
        relevant_genes = drug_genetic_modifiers.get(drug.name, [])
        
        for variant in genetic_variants:
            gene = variant.get('gene', '')
            if gene in relevant_genes:
                impact = variant.get('impact', 'UNKNOWN')
                genetic_modifiers.append(f"{gene} variant ({impact} impact)")
        
        return genetic_modifiers
    
    def _identify_comorbidity_interactions(self, patient_factors: Dict[str, Any],
                                         drug: DrugProperties) -> List[str]:
        """Identify comorbidity-drug interactions."""
        
        interactions = []
        
        comorbidities = patient_factors.get('comorbidities', [])
        
        # Drug-comorbidity interaction map
        drug_comorbidity_interactions = {
            'lithium': {
                'kidney_disease': 'Increased nephrotoxicity risk',
                'heart_disease': 'Monitor for cardiac effects',
                'thyroid_disease': 'Enhanced thyroid dysfunction risk'
            },
            'aripiprazole': {
                'diabetes': 'May worsen glucose control',
                'cardiovascular_disease': 'Monitor for QT prolongation',
                'parkinsons_disease': 'May improve or worsen symptoms'
            },
            'citalopram': {
                'heart_disease': 'QT prolongation risk',
                'hyponatremia': 'Increased SIADH risk',
                'bleeding_disorders': 'Enhanced bleeding risk'
            }
        }
        
        drug_interactions = drug_comorbidity_interactions.get(drug.name, {})
        
        for comorbidity in comorbidities:
            if comorbidity in drug_interactions:
                interactions.append(f"{comorbidity}: {drug_interactions[comorbidity]}")
        
        return interactions
    
    def _determine_primary_mechanism(self, amplification_type: str,
                                   resonance_analyses: List[ResonanceAnalysis],
                                   hole_matches: List[DrugHoleMatch]) -> str:
        """Determine the primary therapeutic mechanism."""
        
        if amplification_type == "resonant":
            # Find the strongest resonance
            strongest_resonance = max(resonance_analyses, key=lambda r: r.resonance_strength)
            if strongest_resonance.resonance_type.startswith('direct'):
                return "Direct frequency resonance"
            elif strongest_resonance.resonance_type.startswith('harmonic'):
                return "Harmonic resonance coupling"
            else:
                return "Subharmonic frequency matching"
        
        elif amplification_type == "synergistic":
            return "Multi-pathway synergistic modulation"
        
        elif amplification_type == "additive":
            return "Direct pathway modulation"
        
        else:  # cascade
            return "Oscillatory cascade activation"
    
    def _identify_secondary_mechanisms(self, resonance_analyses: List[ResonanceAnalysis],
                                     hole_matches: List[DrugHoleMatch],
                                     drug: DrugProperties) -> List[str]:
        """Identify secondary therapeutic mechanisms."""
        
        mechanisms = []
        
        # Look for secondary resonances
        secondary_resonances = [r for r in resonance_analyses if r.resonance_strength > 0.3]
        if len(secondary_resonances) > 1:
            mechanisms.append("Multiple frequency coupling")
        
        # Check for pathway cross-talk
        pathways = [match.hole_id.split('_')[0] for match in hole_matches if match.pathway_match > 0.5]
        if len(set(pathways)) > 1:
            mechanisms.append("Cross-pathway modulation")
        
        # Drug-specific mechanisms
        if 'gsk3' in drug.primary_pathways:
            mechanisms.append("GSK3 pathway modulation")
        
        if 'neurotransmitter' in drug.primary_pathways:
            mechanisms.append("Neurotransmitter system regulation")
        
        return mechanisms
    
    def _assess_amplification_stability(self, resonance_analyses: List[ResonanceAnalysis],
                                      drug: DrugProperties, 
                                      holes: List[OscillatoryHole]) -> float:
        """Assess the stability of therapeutic amplification over time."""
        
        # Base stability from resonance coupling stability
        resonance_stabilities = [r.coupling_stability for r in resonance_analyses]
        base_stability = np.mean(resonance_stabilities) if resonance_stabilities else 0.8
        
        # Drug half-life affects stability
        half_life = drug.half_life
        if 8 <= half_life <= 48:  # Ideal range
            half_life_factor = 1.0
        elif half_life < 8:
            half_life_factor = 0.8  # May need frequent dosing
        else:
            half_life_factor = 0.9  # May accumulate
        
        # Hole persistence affects stability
        hole_persistences = [hole.persistence for hole in holes]
        hole_stability_factor = np.mean(hole_persistences) if hole_persistences else 1.0
        
        overall_stability = base_stability * half_life_factor * hole_stability_factor
        
        return min(1.0, max(0.1, overall_stability))
    
    def _assess_tolerance_risk(self, drug: DrugProperties, amplification_factor: float) -> float:
        """Assess risk of tolerance development."""
        
        # Base tolerance risk by drug class
        base_tolerance_risks = {
            'lithium': 0.1,      # Low tolerance risk
            'aripiprazole': 0.2,  # Moderate risk
            'citalopram': 0.3,    # Higher risk (receptor changes)
            'atorvastatin': 0.05, # Very low risk
            'aspirin': 0.1        # Low risk
        }
        
        base_risk = base_tolerance_risks.get(drug.name, 0.2)
        
        # Higher amplification may increase tolerance risk
        amplification_risk_factor = 1.0 + (amplification_factor - 1.0) * 0.2
        
        total_risk = base_risk * amplification_risk_factor
        
        return min(0.8, total_risk)
    
    def _assess_clinical_significance(self, amplification_factor: float,
                                    holes: List[OscillatoryHole],
                                    hole_matches: List[DrugHoleMatch]) -> float:
        """Assess clinical significance of the therapeutic effect."""
        
        # Base significance from hole impact severity
        hole_severities = [hole.impact_severity for hole in holes]
        base_significance = np.mean(hole_severities)
        
        # Amplification enhances significance
        amplification_enhancement = min(1.0, amplification_factor / 5.0)  # Cap at 5x
        
        # Predicted efficacy contributes to significance
        efficacies = [match.predicted_efficacy for match in hole_matches]
        efficacy_factor = np.mean(efficacies) if efficacies else 0.5
        
        clinical_significance = base_significance * (0.5 + 0.5 * amplification_enhancement) * efficacy_factor
        
        return min(1.0, clinical_significance)
    
    def _calculate_therapeutic_window(self, amplification_factor: float,
                                    drug: DrugProperties, 
                                    holes: List[OscillatoryHole]) -> Tuple[float, float]:
        """Calculate therapeutic window for amplified effects."""
        
        # Base therapeutic window
        base_min = 1.0
        base_max = amplification_factor * 2.0
        
        # Narrow window for drugs with safety concerns
        safety_drugs = ['lithium', 'citalopram']  # Narrow therapeutic windows
        if drug.name in safety_drugs:
            window_factor = 0.7
        else:
            window_factor = 1.0
        
        # High impact holes may require more careful dosing
        max_impact = max([hole.impact_severity for hole in holes]) if holes else 0.5
        if max_impact > 0.8:
            window_factor *= 0.8
        
        therapeutic_min = base_min
        therapeutic_max = base_max * window_factor
        
        return (therapeutic_min, min(10.0, therapeutic_max))
    
    def _calculate_dose_optimization(self, drug: DrugProperties,
                                   holes: List[OscillatoryHole],
                                   hole_matches: List[DrugHoleMatch]) -> float:
        """Calculate dose optimization factor."""
        
        # Base optimization from hole filling efficiency
        filling_efficiencies = [match.predicted_hole_filling for match in hole_matches]
        base_optimization = np.mean(filling_efficiencies) if filling_efficiencies else 0.5
        
        # Drug properties affect optimization potential
        # Higher amplitude drugs have more optimization potential
        amplitude_factor = drug.oscillatory_amplitude
        
        # ATP coupling efficiency affects dose response
        atp_factor = drug.atp_coupling_efficiency
        
        dose_optimization = base_optimization * amplitude_factor * atp_factor
        
        return min(1.0, dose_optimization)
    
    def _analyze_drug_interactions(self, drug_combination: List[str]) -> Dict[str, str]:
        """Analyze interactions between drugs in combination."""
        
        interactions = {}
        
        # Use drug database to check pairwise interactions
        for i, drug1 in enumerate(drug_combination):
            for drug2 in drug_combination[i+1:]:
                pair_key = f"{drug1}_{drug2}"
                
                # Get interaction analysis from drug database
                interaction_data = self.drug_db.get_drug_interactions(drug1, drug2)
                interactions[pair_key] = interaction_data.get('interaction_risk', 'unknown')
        
        return interactions
    
    def _calculate_synergy_score(self, drug_combination: List[str],
                               holes: List[OscillatoryHole]) -> float:
        """Calculate synergy score for drug combination."""
        
        # Predict individual effects
        individual_scores = []
        for drug_name in drug_combination:
            drug_score = 0.0
            for hole in holes:
                match = self.hole_matcher.match_drug_to_hole(drug_name, hole)
                drug_score += match.overall_score
            individual_scores.append(drug_score)
        
        # Expected additive effect
        expected_additive = sum(individual_scores)
        
        # Check for pathway overlap (synergy potential)
        drug_pathways = []
        for drug_name in drug_combination:
            drug = self.drug_db.get_drug(drug_name)
            if drug:
                drug_pathways.extend(drug.primary_pathways)
        
        # Calculate pathway overlap
        unique_pathways = set(drug_pathways)
        total_pathways = len(drug_pathways)
        overlap_ratio = (total_pathways - len(unique_pathways)) / total_pathways if total_pathways > 0 else 0
        
        # Synergy score calculation
        # Positive for synergy, negative for antagonism
        if overlap_ratio > 0.5:  # High overlap
            synergy_score = overlap_ratio - 0.5  # 0 to 0.5 range
        elif overlap_ratio < 0.2:  # Low overlap, potential antagonism
            synergy_score = overlap_ratio - 0.2  # -0.2 to 0 range
        else:
            synergy_score = 0.0  # Neutral
        
        return max(-1.0, min(1.0, synergy_score))
    
    def _calculate_combined_efficacy(self, individual_outcomes: List[TherapeuticOutcome],
                                   synergy_score: float,
                                   drug_interactions: Dict[str, str]) -> Tuple[float, float]:
        """Calculate combined efficacy for drug combination."""
        
        # Base combined efficacy (simple addition with diminishing returns)
        individual_efficacies = []
        for outcome in individual_outcomes:
            # Use average symptom improvement as efficacy measure
            if outcome.symptom_improvement:
                efficacy = np.mean(list(outcome.symptom_improvement.values())) / 100.0
            else:
                efficacy = 0.5  # Default
            individual_efficacies.append(efficacy)
        
        # Diminishing returns formula
        combined_base = 1.0 - np.prod([1.0 - eff for eff in individual_efficacies])
        
        # Synergy/antagonism adjustment
        synergy_factor = 1.0 + synergy_score * 0.5  # ±50% modification
        
        # Interaction penalty
        high_risk_interactions = len([risk for risk in drug_interactions.values() if risk == 'high'])
        interaction_penalty = 1.0 - high_risk_interactions * 0.2
        
        combined_efficacy = combined_base * synergy_factor * interaction_penalty
        interaction_amplification = synergy_factor
        
        return min(1.0, max(0.0, combined_efficacy)), interaction_amplification
    
    def _assess_interaction_risks(self, drug_combination: List[str],
                                drug_interactions: Dict[str, str]) -> List[str]:
        """Assess risks from drug interactions."""
        
        risks = []
        
        # High-risk interactions
        high_risk_pairs = [pair for pair, risk in drug_interactions.items() if risk == 'high']
        for pair in high_risk_pairs:
            risks.append(f"High interaction risk: {pair.replace('_', ' + ')}")
        
        # Moderate risk interactions
        moderate_risk_pairs = [pair for pair, risk in drug_interactions.items() if risk == 'moderate']
        for pair in moderate_risk_pairs:
            risks.append(f"Monitor for interactions: {pair.replace('_', ' + ')}")
        
        # Additive side effects
        if len(drug_combination) > 2:
            risks.append("Potential additive side effects from multiple drugs")
        
        return risks
    
    def _optimize_combination_dosing(self, drug_combination: List[str],
                                   synergy_score: float,
                                   drug_interactions: Dict[str, str]) -> Dict[str, float]:
        """Optimize dosing ratios for drug combination."""
        
        dosing_ratios = {}
        
        # Base ratios (1.0 = standard dose)
        for drug in drug_combination:
            dosing_ratios[drug] = 1.0
        
        # Adjust for synergy
        if synergy_score > 0.3:  # Significant synergy
            # Reduce doses when synergy is present
            reduction_factor = 1.0 - synergy_score * 0.3
            for drug in drug_combination:
                dosing_ratios[drug] *= reduction_factor
        
        # Adjust for interactions
        for pair, risk in drug_interactions.items():
            if risk == 'high':
                # Reduce doses for high-risk combinations
                drug1, drug2 = pair.split('_')
                if drug1 in dosing_ratios:
                    dosing_ratios[drug1] *= 0.7
                if drug2 in dosing_ratios:
                    dosing_ratios[drug2] *= 0.7
        
        return dosing_ratios
    
    def _optimize_combination_timing(self, drug_combination: List[str]) -> Dict[str, float]:
        """Optimize timing offsets for drug combination."""
        
        timing = {}
        
        # Get drug half-lives for timing optimization
        for i, drug_name in enumerate(drug_combination):
            drug = self.drug_db.get_drug(drug_name)
            if drug:
                # Stagger doses based on half-life
                if drug.half_life < 8:  # Short half-life
                    timing[drug_name] = i * 2.0  # 2-hour offsets
                elif drug.half_life > 24:  # Long half-life
                    timing[drug_name] = i * 12.0  # 12-hour offsets
                else:
                    timing[drug_name] = i * 6.0  # 6-hour offsets
            else:
                timing[drug_name] = i * 4.0  # Default 4-hour offset
        
        return timing
    
    def _rank_therapeutic_strategies(self, single_drug_outcomes: Dict[str, TherapeuticOutcome],
                                   combination_outcomes: List[CombinationTherapyPrediction],
                                   holes: List[OscillatoryHole]) -> List[Dict[str, Any]]:
        """Rank all therapeutic strategies by predicted effectiveness."""
        
        strategies = []
        
        # Add single drug strategies
        for drug_name, outcome in single_drug_outcomes.items():
            # Calculate overall score
            if outcome.symptom_improvement:
                symptom_score = np.mean(list(outcome.symptom_improvement.values())) / 100.0
            else:
                symptom_score = 0.0
            
            safety_score = 1.0 - outcome.contraindication_risk
            confidence_penalty = outcome.prediction_confidence
            
            overall_score = (symptom_score * 0.5 + safety_score * 0.3) * confidence_penalty
            
            strategies.append({
                'type': 'single_drug',
                'drugs': [drug_name],
                'overall_score': overall_score,
                'efficacy_score': symptom_score,
                'safety_score': safety_score,
                'confidence': outcome.prediction_confidence,
                'outcome': outcome
            })
        
        # Add combination strategies
        for combo in combination_outcomes:
            safety_score = 1.0 - (len(combo.interaction_risks) * 0.1)  # Penalty for risks
            confidence_score = 0.7  # Lower confidence for combinations
            
            overall_score = (combo.combined_efficacy * 0.5 + safety_score * 0.3) * confidence_score
            
            strategies.append({
                'type': 'combination',
                'drugs': combo.drugs,
                'overall_score': overall_score,
                'efficacy_score': combo.combined_efficacy,
                'safety_score': safety_score,
                'confidence': confidence_score,
                'outcome': combo
            })
        
        # Sort by overall score
        strategies.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return strategies
    
    def _generate_optimization_insights(self, ranked_strategies: List[Dict[str, Any]],
                                      holes: List[OscillatoryHole],
                                      patient_factors: Optional[Dict[str, Any]]) -> List[str]:
        """Generate insights about therapeutic optimization."""
        
        insights = []
        
        if not ranked_strategies:
            insights.append("No viable therapeutic strategies identified")
            return insights
        
        # Best strategy insights
        best_strategy = ranked_strategies[0]
        if best_strategy['type'] == 'single_drug':
            insights.append(f"Single drug therapy with {best_strategy['drugs'][0]} appears optimal")
        else:
            drug_list = ' + '.join(best_strategy['drugs'])
            insights.append(f"Combination therapy ({drug_list}) shows highest potential")
        
        # Efficacy insights
        if best_strategy['efficacy_score'] > 0.7:
            insights.append("High therapeutic efficacy predicted")
        elif best_strategy['efficacy_score'] < 0.4:
            insights.append("Limited therapeutic benefit expected - consider alternative approaches")
        
        # Safety insights
        if best_strategy['safety_score'] < 0.6:
            insights.append("Safety concerns identified - enhanced monitoring recommended")
        
        # Confidence insights
        if best_strategy['confidence'] < 0.6:
            insights.append("Low prediction confidence - consider additional genetic testing")
        
        # Hole-specific insights
        high_impact_holes = [hole for hole in holes if hole.impact_severity > 0.7]
        if high_impact_holes:
            insights.append(f"{len(high_impact_holes)} high-impact oscillatory holes identified")
        
        # Patient-specific insights
        if patient_factors:
            age = patient_factors.get('age', 50)
            if age > 65:
                insights.append("Elderly patient - consider dose adjustments and enhanced monitoring")
        
        return insights

def main():
    """
    Test the complete therapeutic predictor functionality.
    
    Demonstrates comprehensive therapeutic outcome prediction using
    the integrated BMD framework.
    """
    
    print("🔮 Testing Therapeutic Predictor")
    print("=" * 50)
    
    # Initialize predictor with all BMD components
    predictor = TherapeuticPredictor()
    
    print("Initialized therapeutic predictor with:")
    print(f"  • Drug database: {len(predictor.drug_db.drugs)} compounds")
    print(f"  • Hole matcher: Ready")
    print(f"  • Resonance calculator: Ready")
    
    # Create test patient scenario
    print("\n🧬 Creating test patient scenario...")
    
    patient_id = "patient_complex_001"
    patient_factors = {
        'age': 45,
        'comorbidities': ['mild_kidney_dysfunction'],
        'genetic_factors': ['normal_metabolizer'],
        'genetic_variants': [
            {'gene': 'INPP1', 'variant_id': 'rs123456', 'impact': 'HIGH'},
            {'gene': 'CYP2D6', 'variant_id': 'CYP2D6*1', 'impact': 'NORMAL'},
            {'gene': 'DRD2', 'variant_id': 'rs1800497', 'impact': 'MODERATE'}
        ]
    }
    
    # Create oscillatory holes
    test_holes_data = [
        {
            'hole_id': 'inositol_hole_severe',
            'pathway': 'inositol_metabolism',
            'frequency': 7.23e13,
            'amplitude_deficit': 0.8,
            'bandwidth': 5e12,
            'location': 'neuronal',
            'impact_severity': 0.9,
            'genetic_cause': 'INPP1',
            'variant_impact': 'HIGH',
            'persistence': 1.0,
            'variability': 0.1
        },
        {
            'hole_id': 'dopamine_hole_moderate',
            'pathway': 'dopamine_signaling',
            'frequency': 1.85e13,
            'amplitude_deficit': 0.6,
            'bandwidth': 3e12,
            'location': 'striatal',
            'impact_severity': 0.7,
            'genetic_cause': 'DRD2',
            'variant_impact': 'MODERATE',
            'persistence': 0.9,
            'variability': 0.2
        }
    ]
    
    holes = [predictor.hole_matcher.create_oscillatory_hole(hole_data) 
             for hole_data in test_holes_data]
    
    print(f"Patient profile:")
    print(f"  • Age: {patient_factors['age']}")
    print(f"  • Genetic variants: {len(patient_factors['genetic_variants'])}")
    print(f"  • Oscillatory holes: {len(holes)}")
    for hole in holes:
        print(f"    - {hole.hole_id}: {hole.pathway} (severity: {hole.impact_severity:.1f})")
    
    # Test single drug predictions
    print(f"\n💊 Testing single drug therapeutic predictions...")
    
    test_drugs = ['lithium', 'aripiprazole', 'citalopram']
    
    single_outcomes = {}
    for drug_name in test_drugs:
        print(f"\nPredicting outcome for {drug_name}...")
        
        outcome = predictor.predict_therapeutic_outcome(
            patient_id, drug_name, holes, patient_factors
        )
        single_outcomes[drug_name] = outcome
        
        print(f"  Prediction confidence: {outcome.prediction_confidence:.3f}")
        print(f"  Evidence quality: {outcome.evidence_quality}")
        print(f"  Contraindication risk: {outcome.contraindication_risk:.3f}")
        print(f"  Timeline - Onset: {outcome.onset_time:.1f} days, "
              f"Peak: {outcome.peak_effect_time:.1f} days")
        
        if outcome.symptom_improvement:
            print(f"  Top symptom improvements:")
            sorted_symptoms = sorted(outcome.symptom_improvement.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            for symptom, improvement in sorted_symptoms:
                print(f"    • {symptom}: {improvement:.1f}% improvement")
        
        if outcome.genetic_modifiers:
            print(f"  Genetic modifiers: {len(outcome.genetic_modifiers)}")
    
    # Test amplification analysis
    print(f"\n⚡ Testing therapeutic amplification analysis...")
    
    lithium_drug = predictor.drug_db.get_drug('lithium')
    lithium_matches = [predictor.hole_matcher.match_drug_to_hole('lithium', hole) for hole in holes]
    
    amplification = predictor.analyze_therapeutic_amplification(lithium_drug, holes, lithium_matches)
    
    print(f"Lithium amplification analysis:")
    print(f"  • Amplification type: {amplification.amplification_type}")
    print(f"  • Amplification factor: {amplification.amplification_factor:.2f}x")
    print(f"  • Primary mechanism: {amplification.primary_mechanism}")
    print(f"  • Frequency coupling: {amplification.frequency_coupling:.3f}")
    print(f"  • Pathway synergy: {amplification.pathway_synergy:.3f}")
    print(f"  • Clinical significance: {amplification.clinical_significance:.3f}")
    print(f"  • Therapeutic window: {amplification.therapeutic_window[0]:.1f} - {amplification.therapeutic_window[1]:.1f}")
    
    if amplification.secondary_mechanisms:
        print(f"  • Secondary mechanisms: {', '.join(amplification.secondary_mechanisms)}")
    
    # Test combination therapy predictions
    print(f"\n🔗 Testing combination therapy predictions...")
    
    combinations_to_test = [
        ['lithium', 'aripiprazole'],
        ['lithium', 'citalopram']
    ]
    
    combination_outcomes = predictor.predict_combination_therapy(
        patient_id, combinations_to_test, holes, patient_factors
    )
    
    for combo in combination_outcomes:
        drug_list = ' + '.join(combo.drugs)
        print(f"\n{drug_list} combination:")
        print(f"  • Synergy score: {combo.synergy_score:.3f}")
        print(f"  • Combined efficacy: {combo.combined_efficacy:.3f}")
        print(f"  • Interaction amplification: {combo.interaction_amplification:.2f}x")
        print(f"  • Monitoring complexity: {combo.monitoring_complexity:.1f}x")
        
        if combo.interaction_risks:
            print(f"  • Interaction risks: {len(combo.interaction_risks)}")
            for risk in combo.interaction_risks[:2]:
                print(f"    - {risk}")
        
        if combo.optimal_dosing_ratios:
            print(f"  • Optimal dosing ratios:")
            for drug, ratio in combo.optimal_dosing_ratios.items():
                print(f"    - {drug}: {ratio:.2f}x standard dose")
    
    # Test comprehensive strategy optimization
    print(f"\n🎯 Testing therapeutic strategy optimization...")
    
    available_drugs = ['lithium', 'aripiprazole', 'citalopram', 'valproate']
    
    optimization_result = predictor.optimize_therapeutic_strategy(
        patient_id, holes, available_drugs, patient_factors, max_drugs=2
    )
    
    print(f"\nOptimization results for patient {patient_id}:")
    
    ranked_strategies = optimization_result['ranked_strategies']
    print(f"  • Strategies evaluated: {len(ranked_strategies)}")
    
    print(f"\nTop 3 therapeutic strategies:")
    for i, strategy in enumerate(ranked_strategies[:3]):
        drug_list = ' + '.join(strategy['drugs'])
        print(f"  {i+1}. {drug_list}")
        print(f"     Overall score: {strategy['overall_score']:.3f}")
        print(f"     Efficacy: {strategy['efficacy_score']:.3f}")
        print(f"     Safety: {strategy['safety_score']:.3f}")
        print(f"     Confidence: {strategy['confidence']:.3f}")
    
    # Optimization insights
    insights = optimization_result['optimization_insights']
    if insights:
        print(f"\nOptimization insights:")
        for insight in insights:
            print(f"  • {insight}")
    
    # Best recommendation
    best_strategy = optimization_result['recommendation']
    if best_strategy:
        print(f"\n🏆 RECOMMENDED STRATEGY:")
        drug_list = ' + '.join(best_strategy['drugs'])
        print(f"  Strategy: {drug_list}")
        print(f"  Overall score: {best_strategy['overall_score']:.3f}")
        print(f"  Expected efficacy: {best_strategy['efficacy_score']:.1%}")
        print(f"  Safety profile: {best_strategy['safety_score']:.1%}")
        
        if best_strategy['type'] == 'single_drug':
            outcome = best_strategy['outcome']
            print(f"  Monitoring requirements: {len(outcome.monitoring_requirements)}")
        else:
            combo = best_strategy['outcome']
            print(f"  Drug interactions: {len(combo.drug_interactions)}")
    
    # Summary insights
    print(f"\n💡 Key Therapeutic Insights:")
    print("-" * 50)
    
    # Amplification insights
    if amplification.amplification_factor > 3.0:
        print(f"• Strong therapeutic amplification detected ({amplification.amplification_factor:.1f}x)")
    
    # Best drug insights
    best_single = max(single_outcomes.items(), key=lambda x: x[1].prediction_confidence)
    print(f"• Highest confidence single drug: {best_single[0]} ({best_single[1].prediction_confidence:.3f})")
    
    # Combination insights
    best_combo = max(combination_outcomes, key=lambda x: x.combined_efficacy)
    combo_drugs = ' + '.join(best_combo.drugs)
    print(f"• Best combination therapy: {combo_drugs} (efficacy: {best_combo.combined_efficacy:.3f})")
    
    # Risk insights
    high_risk_outcomes = [name for name, outcome in single_outcomes.items() 
                         if outcome.contraindication_risk > 0.3]
    if high_risk_outcomes:
        print(f"• High-risk drugs identified: {', '.join(high_risk_outcomes)}")
    
    # Genetic insights
    genetic_modifier_count = len(patient_factors['genetic_variants'])
    if genetic_modifier_count > 2:
        print(f"• Complex genetic profile detected ({genetic_modifier_count} variants)")
    
    print(f"\n✅ Therapeutic predictor testing complete!")
    print("🔮 Successfully demonstrated comprehensive therapeutic outcome")
    print("   prediction using the integrated BMD framework.")
    
    return predictor, optimization_result

if __name__ == "__main__":
    predictor, results = main()
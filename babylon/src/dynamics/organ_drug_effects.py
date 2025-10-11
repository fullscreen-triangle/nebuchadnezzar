"""
Organ Drug Effects Module
=========================

Models organ-level therapeutic and adverse effects resulting from tissue
drug distribution. Integrates tissue-level drug concentrations to predict
organ function changes and clinical outcomes.

Based on organ-specific oscillatory signatures and functional endpoints.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrganSystem:
    """Represents an organ system for drug effect modeling."""
    organ_id: str
    organ_name: str
    component_tissues: List[str]  # Tissues that make up this organ
    baseline_function: float  # Normal functional capacity (0-1 scale)
    functional_reserve: float  # Reserve capacity before dysfunction
    drug_sensitivity: float  # Sensitivity to drug effects (0-1 scale)
    oscillatory_coherence: float  # Coherence of oscillatory patterns
    biomarker_ranges: Dict[str, Tuple[float, float]]  # Normal biomarker ranges
    critical_thresholds: Dict[str, float]  # Thresholds for dysfunction

@dataclass
class DrugOrganProfile:
    """Drug's organ-specific effect profile."""
    drug_name: str
    therapeutic_targets: List[str]  # Target organs for therapeutic effect
    off_target_organs: List[str]  # Organs with potential adverse effects
    dose_response_curves: Dict[str, Dict[str, float]]  # organ -> {ec50, hill, emax}
    therapeutic_window: Tuple[float, float]  # (min_effective, max_safe) concentrations
    mechanism_type: str  # 'agonist', 'antagonist', 'allosteric', 'enzyme_inhibitor'

@dataclass
class OrganEffect:
    """Result of drug effect on organ function."""
    drug_name: str
    organ_id: str
    tissue_concentrations: Dict[str, float]  # tissue_id -> concentration
    functional_change: float  # Fold change in organ function
    biomarker_changes: Dict[str, float]  # biomarker -> fold change
    therapeutic_benefit: float  # 0-1 scale of therapeutic benefit
    adverse_effect_risk: float  # 0-1 scale of adverse effect risk
    oscillatory_disruption: float  # Disruption of organ oscillatory patterns
    clinical_outcome: str  # 'therapeutic', 'toxic', 'subtherapeutic', 'neutral'
    effect_time_course: List[Tuple[float, float]]  # (time, effect) pairs
    confidence: float = 0.0

class OrganDrugEffects:
    """
    Models drug effects at the organ level by integrating tissue drug
    concentrations and predicting functional outcomes.
    
    Combines tissue distribution data with organ-specific response models
    to predict therapeutic and adverse effects.
    """
    
    def __init__(self):
        """Initialize organ drug effects simulator."""
        self.organ_systems = self._load_organ_systems()
        
    def _load_organ_systems(self) -> Dict[str, OrganSystem]:
        """Load organ system definitions for effect modeling."""
        
        systems = {
            'brain': OrganSystem(
                organ_id='brain',
                organ_name='Central Nervous System',
                component_tissues=['brain'],
                baseline_function=1.0,
                functional_reserve=0.3,  # Limited CNS reserve
                drug_sensitivity=0.8,    # High sensitivity to drugs
                oscillatory_coherence=0.9,  # High coherence needed
                biomarker_ranges={
                    'neurotransmitter_activity': (0.8, 1.2),
                    'cognitive_function': (0.9, 1.1),
                    'motor_control': (0.85, 1.15)
                },
                critical_thresholds={
                    'seizure_threshold': 0.3,      # Below this = seizure risk
                    'cognitive_impairment': 0.7,   # Below this = impairment
                    'motor_dysfunction': 0.6       # Below this = motor issues
                }
            ),
            
            'cardiovascular': OrganSystem(
                organ_id='cardiovascular',
                organ_name='Cardiovascular System',
                component_tissues=['heart'],
                baseline_function=1.0,
                functional_reserve=0.5,  # Moderate cardiac reserve
                drug_sensitivity=0.6,    # Moderate sensitivity
                oscillatory_coherence=0.95,  # Critical rhythm coherence
                biomarker_ranges={
                    'cardiac_output': (4.0, 8.0),      # L/min
                    'ejection_fraction': (0.55, 0.70), # Fraction
                    'heart_rate': (60, 100)            # BPM
                },
                critical_thresholds={
                    'heart_failure': 0.4,     # EF < 40%
                    'arrhythmia': 0.2,        # Rhythm disruption
                    'cardiac_arrest': 0.1     # Critical dysfunction
                }
            ),
            
            'hepatic': OrganSystem(
                organ_id='hepatic',
                organ_name='Hepatic System',
                component_tissues=['liver'],
                baseline_function=1.0,
                functional_reserve=0.8,  # High hepatic reserve
                drug_sensitivity=0.4,    # Lower sensitivity (robust organ)
                oscillatory_coherence=0.7,  # Moderate coherence needs
                biomarker_ranges={
                    'alt_ast': (10, 40),      # U/L
                    'bilirubin': (0.2, 1.2), # mg/dL
                    'albumin': (3.5, 5.0)    # g/dL
                },
                critical_thresholds={
                    'hepatotoxicity': 0.6,   # Function < 60%
                    'liver_failure': 0.3,    # Function < 30%
                    'fulminant_failure': 0.1 # Critical failure
                }
            ),
            
            'renal': OrganSystem(
                organ_id='renal',
                organ_name='Renal System',
                component_tissues=['kidney'],
                baseline_function=1.0,
                functional_reserve=0.6,  # Good renal reserve
                drug_sensitivity=0.5,    # Moderate sensitivity
                oscillatory_coherence=0.8,  # High coherence for filtration
                biomarker_ranges={
                    'creatinine': (0.6, 1.2),     # mg/dL
                    'gfr': (90, 120),             # mL/min/1.73m²
                    'urea': (7, 20)               # mg/dL
                },
                critical_thresholds={
                    'kidney_injury': 0.7,    # GFR < 70%
                    'renal_failure': 0.3,    # GFR < 30%
                    'dialysis_needed': 0.15   # Critical dysfunction
                }
            ),
            
            'musculoskeletal': OrganSystem(
                organ_id='musculoskeletal',
                organ_name='Musculoskeletal System',
                component_tissues=['muscle'],
                baseline_function=1.0,
                functional_reserve=0.4,  # Moderate muscle reserve
                drug_sensitivity=0.3,    # Low sensitivity
                oscillatory_coherence=0.6,  # Lower coherence needs
                biomarker_ranges={
                    'muscle_strength': (0.8, 1.2),
                    'endurance': (0.7, 1.3),
                    'creatine_kinase': (30, 200)  # U/L
                },
                critical_thresholds={
                    'myopathy': 0.6,         # Strength < 60%
                    'rhabdomyolysis': 0.3,   # Severe muscle damage
                    'paralysis': 0.1         # Critical dysfunction
                }
            )
        }
        
        logger.info(f"Loaded {len(systems)} organ systems")
        return systems
    
    def create_drug_organ_profiles(self) -> Dict[str, DrugOrganProfile]:
        """Create organ-level effect profiles for drugs."""
        
        profiles = {
            'lithium': DrugOrganProfile(
                drug_name='lithium',
                therapeutic_targets=['brain'],
                off_target_organs=['renal', 'cardiovascular'],
                dose_response_curves={
                    'brain': {'ec50': 0.6, 'hill': 1.5, 'emax': 0.8},      # Therapeutic
                    'renal': {'ec50': 1.2, 'hill': 2.0, 'emax': -0.3},     # Nephrotoxic
                    'cardiovascular': {'ec50': 2.0, 'hill': 1.0, 'emax': -0.2}  # Cardiac effects
                },
                therapeutic_window=(0.4, 1.2),  # μM - narrow window
                mechanism_type='enzyme_inhibitor'  # GSK3β, INPP1
            ),
            
            'aripiprazole': DrugOrganProfile(
                drug_name='aripiprazole',
                therapeutic_targets=['brain'],
                off_target_organs=['cardiovascular', 'musculoskeletal'],
                dose_response_curves={
                    'brain': {'ec50': 0.15, 'hill': 1.2, 'emax': 0.7},         # Antipsychotic
                    'cardiovascular': {'ec50': 0.8, 'hill': 1.8, 'emax': -0.1}, # QT prolongation
                    'musculoskeletal': {'ec50': 1.5, 'hill': 2.5, 'emax': -0.2}  # Extrapyramidal
                },
                therapeutic_window=(0.05, 0.5),  # μM
                mechanism_type='allosteric'  # Partial D2 agonist
            ),
            
            'citalopram': DrugOrganProfile(
                drug_name='citalopram',
                therapeutic_targets=['brain'],
                off_target_organs=['cardiovascular'],
                dose_response_curves={
                    'brain': {'ec50': 0.08, 'hill': 1.0, 'emax': 0.6},     # Antidepressant
                    'cardiovascular': {'ec50': 0.3, 'hill': 2.0, 'emax': -0.15}  # QT effects
                },
                therapeutic_window=(0.02, 0.25),  # μM
                mechanism_type='antagonist'  # SERT inhibitor
            ),
            
            'atorvastatin': DrugOrganProfile(
                drug_name='atorvastatin',
                therapeutic_targets=['hepatic', 'cardiovascular'],
                off_target_organs=['musculoskeletal'],
                dose_response_curves={
                    'hepatic': {'ec50': 0.05, 'hill': 1.5, 'emax': 0.8},        # Cholesterol synthesis
                    'cardiovascular': {'ec50': 0.1, 'hill': 1.0, 'emax': 0.6},   # Cardioprotective
                    'musculoskeletal': {'ec50': 2.0, 'hill': 3.0, 'emax': -0.4}   # Myopathy
                },
                therapeutic_window=(0.01, 0.5),  # μM
                mechanism_type='enzyme_inhibitor'  # HMG-CoA reductase
            ),
            
            'aspirin': DrugOrganProfile(
                drug_name='aspirin',
                therapeutic_targets=['cardiovascular'],
                off_target_organs=['renal', 'hepatic'],
                dose_response_curves={
                    'cardiovascular': {'ec50': 0.02, 'hill': 1.0, 'emax': 0.5}, # Antiplatelet
                    'renal': {'ec50': 0.5, 'hill': 2.0, 'emax': -0.2},          # Nephrotoxic
                    'hepatic': {'ec50': 1.0, 'hill': 1.5, 'emax': -0.1}         # Hepatotoxic
                },
                therapeutic_window=(0.005, 0.2),  # μM
                mechanism_type='enzyme_inhibitor'  # COX inhibitor
            )
        }
        
        return profiles
    
    def calculate_organ_drug_concentration(self, organ: OrganSystem,
                                         tissue_concentrations: Dict[str, float]) -> float:
        """Calculate effective organ drug concentration from tissue concentrations."""
        
        # Weight tissue concentrations by their contribution to organ function
        total_concentration = 0.0
        total_weight = 0.0
        
        for tissue_id in organ.component_tissues:
            if tissue_id in tissue_concentrations:
                # Simple weighting - could be made more sophisticated
                tissue_weight = 1.0  # Equal weighting for now
                total_concentration += tissue_concentrations[tissue_id] * tissue_weight
                total_weight += tissue_weight
        
        if total_weight > 0:
            organ_concentration = total_concentration / total_weight
        else:
            organ_concentration = 0.0
        
        return organ_concentration
    
    def calculate_dose_response(self, concentration: float, 
                              dose_response_curve: Dict[str, float]) -> float:
        """Calculate drug response using Hill equation."""
        
        ec50 = dose_response_curve['ec50']
        hill = dose_response_curve['hill']
        emax = dose_response_curve['emax']
        
        # Hill equation: Effect = Emax * C^n / (EC50^n + C^n)
        if concentration <= 0:
            return 0.0
        
        numerator = emax * (concentration ** hill)
        denominator = (ec50 ** hill) + (concentration ** hill)
        
        effect = numerator / denominator
        
        return effect
    
    def calculate_functional_change(self, organ: OrganSystem,
                                  drug_effect: float,
                                  oscillatory_disruption: float) -> float:
        """Calculate change in organ function from drug effect."""
        
        # Base functional change from drug effect
        base_change = drug_effect
        
        # Oscillatory disruption reduces function
        oscillatory_penalty = oscillatory_disruption * organ.oscillatory_coherence
        
        # Apply drug sensitivity
        adjusted_change = base_change * organ.drug_sensitivity
        
        # Total functional change
        total_change = adjusted_change - oscillatory_penalty
        
        # Apply functional reserve (organs can compensate to some extent)
        if total_change < 0:  # Negative effects
            compensated_change = total_change * (1.0 - organ.functional_reserve)
        else:  # Positive effects
            compensated_change = total_change
        
        # New function level
        new_function = organ.baseline_function + compensated_change
        
        # Calculate fold change
        fold_change = new_function / organ.baseline_function
        
        return fold_change
    
    def calculate_biomarker_changes(self, organ: OrganSystem,
                                  functional_change: float) -> Dict[str, float]:
        """Calculate changes in organ-specific biomarkers."""
        
        biomarker_changes = {}
        
        for biomarker, (low_normal, high_normal) in organ.biomarker_ranges.items():
            # Biomarker changes correlate with functional changes
            if functional_change < 1.0:  # Decreased function
                # Dysfunction typically increases biomarkers (e.g., enzymes, creatinine)
                if 'activity' in biomarker or 'function' in biomarker:
                    # Activity markers decrease with dysfunction
                    biomarker_change = functional_change
                else:
                    # Damage markers increase with dysfunction
                    biomarker_change = 2.0 - functional_change
            else:  # Improved function
                if 'activity' in biomarker or 'function' in biomarker:
                    # Activity markers improve with function
                    biomarker_change = functional_change
                else:
                    # Damage markers decrease with improved function
                    biomarker_change = 2.0 - functional_change
            
            biomarker_changes[biomarker] = biomarker_change
        
        return biomarker_changes
    
    def calculate_oscillatory_disruption(self, organ: OrganSystem,
                                       drug_concentration: float,
                                       drug_profile: DrugOrganProfile) -> float:
        """Calculate disruption of organ oscillatory patterns."""
        
        # Base disruption from drug concentration
        base_disruption = min(0.5, drug_concentration / 10.0)  # Cap at 50% disruption
        
        # Mechanism-dependent disruption
        mechanism_factors = {
            'agonist': 0.3,          # Moderate disruption
            'antagonist': 0.5,       # Higher disruption
            'allosteric': 0.2,       # Lower disruption
            'enzyme_inhibitor': 0.4   # Moderate-high disruption
        }
        
        mechanism_factor = mechanism_factors.get(drug_profile.mechanism_type, 0.4)
        
        # Off-target effects increase disruption
        if organ.organ_id in drug_profile.off_target_organs:
            off_target_factor = 1.5
        else:
            off_target_factor = 1.0
        
        total_disruption = base_disruption * mechanism_factor * off_target_factor
        
        return min(0.8, total_disruption)  # Cap at 80% disruption
    
    def determine_clinical_outcome(self, organ: OrganSystem,
                                 functional_change: float,
                                 drug_concentration: float,
                                 drug_profile: DrugOrganProfile) -> str:
        """Determine clinical outcome category."""
        
        # Check if in therapeutic window
        min_therapeutic, max_safe = drug_profile.therapeutic_window
        
        if organ.organ_id in drug_profile.therapeutic_targets:
            # Target organ
            if min_therapeutic <= drug_concentration <= max_safe:
                if functional_change >= 1.1:  # Improved function
                    return 'therapeutic'
                else:
                    return 'subtherapeutic'
            elif drug_concentration > max_safe:
                return 'toxic'
            else:
                return 'subtherapeutic'
        else:
            # Off-target organ
            if functional_change < 0.8:  # Significant dysfunction
                return 'toxic'
            elif functional_change < 0.9:  # Mild dysfunction
                return 'adverse'
            else:
                return 'neutral'
    
    def simulate_organ_drug_effect(self, drug_profile: DrugOrganProfile,
                                 organ_id: str,
                                 tissue_concentrations: Dict[str, float],
                                 simulation_time: float = 48.0) -> OrganEffect:
        """Simulate drug effect on specific organ."""
        
        if organ_id not in self.organ_systems:
            raise ValueError(f"Unknown organ: {organ_id}")
        
        organ = self.organ_systems[organ_id]
        
        # Calculate organ drug concentration
        organ_concentration = self.calculate_organ_drug_concentration(
            organ, tissue_concentrations
        )
        
        # Calculate dose-response if organ has response curve
        if organ_id in drug_profile.dose_response_curves:
            drug_effect = self.calculate_dose_response(
                organ_concentration,
                drug_profile.dose_response_curves[organ_id]
            )
        else:
            drug_effect = 0.0  # No direct effect
        
        # Calculate oscillatory disruption
        oscillatory_disruption = self.calculate_oscillatory_disruption(
            organ, organ_concentration, drug_profile
        )
        
        # Calculate functional change
        functional_change = self.calculate_functional_change(
            organ, drug_effect, oscillatory_disruption
        )
        
        # Calculate biomarker changes
        biomarker_changes = self.calculate_biomarker_changes(organ, functional_change)
        
        # Calculate therapeutic benefit and adverse risk
        if organ_id in drug_profile.therapeutic_targets:
            therapeutic_benefit = max(0.0, functional_change - 1.0)  # Benefit if > baseline
            adverse_effect_risk = max(0.0, 1.0 - functional_change) * 0.5  # Risk from dysfunction
        else:
            therapeutic_benefit = 0.0
            adverse_effect_risk = max(0.0, 1.0 - functional_change)  # Risk from dysfunction
        
        # Determine clinical outcome
        clinical_outcome = self.determine_clinical_outcome(
            organ, functional_change, organ_concentration, drug_profile
        )
        
        # Calculate time course (simplified)
        time_points = np.linspace(0, simulation_time, 100)
        effect_time_course = []
        
        for t in time_points:
            # Simple exponential approach to steady state
            time_effect = functional_change * (1 - np.exp(-t / 12.0))  # 12h time constant
            effect_time_course.append((t, time_effect))
        
        # Calculate confidence
        confidence = self._calculate_organ_effect_confidence(
            drug_profile, organ, organ_concentration
        )
        
        return OrganEffect(
            drug_name=drug_profile.drug_name,
            organ_id=organ_id,
            tissue_concentrations=tissue_concentrations,
            functional_change=functional_change,
            biomarker_changes=biomarker_changes,
            therapeutic_benefit=therapeutic_benefit,
            adverse_effect_risk=adverse_effect_risk,
            oscillatory_disruption=oscillatory_disruption,
            clinical_outcome=clinical_outcome,
            effect_time_course=effect_time_course,
            confidence=confidence
        )
    
    def _calculate_organ_effect_confidence(self, drug_profile: DrugOrganProfile,
                                         organ: OrganSystem,
                                         concentration: float) -> float:
        """Calculate confidence in organ effect prediction."""
        
        base_confidence = 0.6
        
        # Higher confidence for organs with defined dose-response curves
        if organ.organ_id in drug_profile.dose_response_curves:
            curve_boost = 0.2
        else:
            curve_boost = -0.2
        
        # Higher confidence for target organs
        if organ.organ_id in drug_profile.therapeutic_targets:
            target_boost = 0.15
        else:
            target_boost = 0.0
        
        # Concentration within therapeutic window increases confidence
        min_therapeutic, max_safe = drug_profile.therapeutic_window
        if min_therapeutic <= concentration <= max_safe:
            concentration_boost = 0.1
        else:
            concentration_boost = -0.1
        
        # Well-studied drugs have higher confidence
        well_studied_drugs = ['lithium', 'atorvastatin', 'aspirin']
        if drug_profile.drug_name in well_studied_drugs:
            drug_boost = 0.1
        else:
            drug_boost = 0.0
        
        final_confidence = min(0.95, base_confidence + curve_boost + 
                              target_boost + concentration_boost + drug_boost)
        
        return max(0.2, final_confidence)
    
    def simulate_drug_library(self, tissue_concentrations_data: Dict[str, Dict[str, float]]) -> List[OrganEffect]:
        """Simulate organ effects for all drug-organ combinations."""
        
        drug_profiles = self.create_drug_organ_profiles()
        all_effects = []
        
        for drug_name, drug_profile in drug_profiles.items():
            logger.info(f"Simulating organ effects for {drug_name}")
            
            # Get tissue concentrations for this drug (use mock data if not provided)
            if drug_name in tissue_concentrations_data:
                tissue_concentrations = tissue_concentrations_data[drug_name]
            else:
                # Generate mock tissue concentrations
                tissue_concentrations = self._generate_mock_tissue_concentrations(drug_name)
            
            # Simulate effects on all organs
            for organ_id in self.organ_systems.keys():
                effect = self.simulate_organ_drug_effect(
                    drug_profile, organ_id, tissue_concentrations
                )
                all_effects.append(effect)
        
        logger.info(f"Completed organ effect simulation for {len(all_effects)} combinations")
        return all_effects
    
    def _generate_mock_tissue_concentrations(self, drug_name: str) -> Dict[str, float]:
        """Generate mock tissue concentrations for demonstration."""
        
        # Mock concentrations based on typical distribution patterns
        mock_concentrations = {
            'lithium': {
                'brain': 0.6, 'heart': 0.4, 'liver': 0.3,
                'kidney': 0.8, 'muscle': 0.2, 'fat': 0.05
            },
            'aripiprazole': {
                'brain': 0.8, 'heart': 0.3, 'liver': 1.5,
                'kidney': 0.6, 'muscle': 0.7, 'fat': 2.0
            },
            'citalopram': {
                'brain': 0.4, 'heart': 0.2, 'liver': 0.8,
                'kidney': 0.3, 'muscle': 0.25, 'fat': 0.6
            },
            'atorvastatin': {
                'brain': 0.02, 'heart': 0.4, 'liver': 2.5,
                'kidney': 0.3, 'muscle': 0.6, 'fat': 1.8
            },
            'aspirin': {
                'brain': 0.05, 'heart': 0.3, 'liver': 0.2,
                'kidney': 0.4, 'muscle': 0.15, 'fat': 0.03
            }
        }
        
        return mock_concentrations.get(drug_name, {
            'brain': 0.1, 'heart': 0.1, 'liver': 0.1,
            'kidney': 0.1, 'muscle': 0.1, 'fat': 0.1
        })
    
    def analyze_organ_effects(self, effects: List[OrganEffect]) -> Dict[str, Any]:
        """Analyze patterns in organ drug effects."""
        
        analysis = {}
        
        # Clinical outcome distribution
        outcomes = [e.clinical_outcome for e in effects]
        analysis['clinical_outcomes'] = {
            'therapeutic': outcomes.count('therapeutic'),
            'toxic': outcomes.count('toxic'),
            'subtherapeutic': outcomes.count('subtherapeutic'),
            'adverse': outcomes.count('adverse'),
            'neutral': outcomes.count('neutral')
        }
        
        # Therapeutic benefit analysis
        therapeutic_benefits = [e.therapeutic_benefit for e in effects]
        analysis['therapeutic_benefit'] = {
            'mean': np.mean(therapeutic_benefits),
            'max': np.max(therapeutic_benefits),
            'beneficial_effects': len([b for b in therapeutic_benefits if b > 0.1])
        }
        
        # Adverse effect analysis
        adverse_risks = [e.adverse_effect_risk for e in effects]
        analysis['adverse_effects'] = {
            'mean': np.mean(adverse_risks),
            'high_risk_count': len([r for r in adverse_risks if r > 0.3])
        }
        
        # Organ-specific analysis
        organ_analysis = {}
        for effect in effects:
            if effect.organ_id not in organ_analysis:
                organ_analysis[effect.organ_id] = []
            organ_analysis[effect.organ_id].append(effect)
        
        analysis['organ_specific'] = {}
        for organ, organ_effects in organ_analysis.items():
            functional_changes = [e.functional_change for e in organ_effects]
            therapeutic_outcomes = len([e for e in organ_effects if e.clinical_outcome == 'therapeutic'])
            
            analysis['organ_specific'][organ] = {
                'mean_functional_change': np.mean(functional_changes),
                'therapeutic_outcomes': therapeutic_outcomes,
                'total_effects': len(organ_effects)
            }
        
        # Drug-specific analysis
        drug_analysis = {}
        for effect in effects:
            if effect.drug_name not in drug_analysis:
                drug_analysis[effect.drug_name] = []
            drug_analysis[effect.drug_name].append(effect)
        
        analysis['drug_specific'] = {}
        for drug, drug_effects in drug_analysis.items():
            therapeutic_count = len([e for e in drug_effects if e.clinical_outcome == 'therapeutic'])
            toxic_count = len([e for e in drug_effects if e.clinical_outcome == 'toxic'])
            
            analysis['drug_specific'][drug] = {
                'therapeutic_effects': therapeutic_count,
                'toxic_effects': toxic_count,
                'safety_ratio': therapeutic_count / max(1, toxic_count)
            }
        
        return analysis
    
    def visualize_organ_effects(self, effects: List[OrganEffect],
                              output_dir: str = "babylon_results") -> None:
        """Create visualizations of organ drug effects."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive organ effects visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Clinical outcomes pie chart
        outcomes = [e.clinical_outcome for e in effects]
        outcome_counts = pd.Series(outcomes).value_counts()
        
        axes[0, 0].pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Clinical Outcome Distribution')
        
        # 2. Functional changes by organ
        organs = list(set(e.organ_id for e in effects))
        organ_func_changes = {}
        
        for organ in organs:
            organ_effects = [e for e in effects if e.organ_id == organ]
            func_changes = [e.functional_change for e in organ_effects]
            organ_func_changes[organ] = func_changes
        
        axes[0, 1].boxplot([organ_func_changes[organ] for organ in organs], 
                          labels=organs)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        axes[0, 1].set_ylabel('Functional Change (fold)')
        axes[0, 1].set_title('Functional Changes by Organ')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # 3. Therapeutic benefit vs adverse risk
        therapeutic_benefits = [e.therapeutic_benefit for e in effects]
        adverse_risks = [e.adverse_effect_risk for e in effects]
        
        scatter = axes[0, 2].scatter(therapeutic_benefits, adverse_risks, alpha=0.7, s=60)
        axes[0, 2].set_xlabel('Therapeutic Benefit')
        axes[0, 2].set_ylabel('Adverse Effect Risk')
        axes[0, 2].set_title('Risk-Benefit Analysis')
        axes[0, 2].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Risk = Benefit')
        axes[0, 2].legend()
        
        # 4. Drug safety profiles
        drugs = list(set(e.drug_name for e in effects))
        drug_safety = {}
        
        for drug in drugs:
            drug_effects = [e for e in effects if e.drug_name == drug]
            therapeutic_count = len([e for e in drug_effects if e.clinical_outcome == 'therapeutic'])
            toxic_count = len([e for e in drug_effects if e.clinical_outcome in ['toxic', 'adverse']])
            total_count = len(drug_effects)
            
            therapeutic_ratio = therapeutic_count / total_count
            toxic_ratio = toxic_count / total_count
            
            drug_safety[drug] = {'therapeutic': therapeutic_ratio, 'toxic': toxic_ratio}
        
        x = np.arange(len(drugs))
        therapeutic_ratios = [drug_safety[drug]['therapeutic'] for drug in drugs]
        toxic_ratios = [drug_safety[drug]['toxic'] for drug in drugs]
        
        width = 0.35
        axes[1, 0].bar(x - width/2, therapeutic_ratios, width, label='Therapeutic', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, toxic_ratios, width, label='Toxic', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Drugs')
        axes[1, 0].set_ylabel('Ratio of Effects')
        axes[1, 0].set_title('Drug Safety Profiles')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(drugs, rotation=45)
        axes[1, 0].legend()
        
        # 5. Oscillatory disruption by organ
        organ_disruptions = {}
        for organ in organs:
            organ_effects = [e for e in effects if e.organ_id == organ]
            disruptions = [e.oscillatory_disruption for e in organ_effects]
            organ_disruptions[organ] = np.mean(disruptions)
        
        bars = axes[1, 1].bar(organ_disruptions.keys(), organ_disruptions.values(), 
                             color='orange', alpha=0.7)
        axes[1, 1].set_ylabel('Oscillatory Disruption')
        axes[1, 1].set_title('Mean Oscillatory Disruption by Organ')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Color bars by disruption level
        for bar, disruption in zip(bars, organ_disruptions.values()):
            if disruption > 0.4:
                bar.set_color('red')
            elif disruption > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 6. Confidence distribution
        confidences = [e.confidence for e in effects]
        
        axes[1, 2].hist(confidences, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 2].set_xlabel('Prediction Confidence')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Effect Prediction Confidence')
        axes[1, 2].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'organ_drug_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Organ effects visualizations saved to {output_path}")
    
    def save_results(self, effects: List[OrganEffect], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save organ drug effects simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed effects as JSON
        effects_data = []
        for effect in effects:
            effects_data.append({
                'drug_name': effect.drug_name,
                'organ_id': effect.organ_id,
                'tissue_concentrations': effect.tissue_concentrations,
                'functional_change': effect.functional_change,
                'biomarker_changes': effect.biomarker_changes,
                'therapeutic_benefit': effect.therapeutic_benefit,
                'adverse_effect_risk': effect.adverse_effect_risk,
                'oscillatory_disruption': effect.oscillatory_disruption,
                'clinical_outcome': effect.clinical_outcome,
                'effect_time_course': effect.effect_time_course,
                'confidence': effect.confidence
            })
        
        with open(output_path / 'organ_drug_effects.json', 'w') as f:
            json.dump(effects_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_data = []
        for effect in effects:
            csv_data.append({
                'drug_name': effect.drug_name,
                'organ_id': effect.organ_id,
                'functional_change': effect.functional_change,
                'therapeutic_benefit': effect.therapeutic_benefit,
                'adverse_effect_risk': effect.adverse_effect_risk,
                'oscillatory_disruption': effect.oscillatory_disruption,
                'clinical_outcome': effect.clinical_outcome,
                'confidence': effect.confidence
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / 'organ_drug_effects.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'organ_effects_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Organ drug effects results saved to {output_path}")

def main():
    """
    Test organ drug effects simulation.
    
    This demonstrates how tissue drug concentrations translate to
    organ-level functional changes and clinical outcomes.
    """
    
    print("🫀 Testing Organ Drug Effects")
    print("=" * 50)
    
    # Initialize organ effects simulator
    simulator = OrganDrugEffects()
    
    # Display loaded organ systems
    print(f"\n🏥 Loaded Organ Systems:")
    for organ_id, organ in simulator.organ_systems.items():
        print(f"  • {organ_id}: {organ.organ_name}")
        print(f"    Tissues: {organ.component_tissues}")
        print(f"    Sensitivity: {organ.drug_sensitivity:.1f}")
        print(f"    Reserve: {organ.functional_reserve:.1f}")
        print(f"    Coherence: {organ.oscillatory_coherence:.1f}")
    
    # Create and display drug profiles
    drug_profiles = simulator.create_drug_organ_profiles()
    print(f"\n💊 Drug Organ Profiles:")
    for drug_name, profile in drug_profiles.items():
        print(f"  • {drug_name}:")
        print(f"    Target organs: {profile.therapeutic_targets}")
        print(f"    Off-target organs: {profile.off_target_organs}")
        print(f"    Therapeutic window: {profile.therapeutic_window}")
        print(f"    Mechanism: {profile.mechanism_type}")
    
    # Simulate organ effects for all combinations
    print(f"\n🔬 Simulating organ drug effects...")
    effects = simulator.simulate_drug_library({})  # Using mock data
    
    # Display results
    print(f"\n📊 ORGAN DRUG EFFECTS RESULTS:")
    print("-" * 80)
    
    # Group results by drug for better display
    drug_effects = {}
    for effect in effects:
        if effect.drug_name not in drug_effects:
            drug_effects[effect.drug_name] = []
        drug_effects[effect.drug_name].append(effect)
    
    for drug_name, drug_organ_effects in drug_effects.items():
        print(f"\n{drug_name.upper()}:")
        
        # Sort by therapeutic benefit (descending)
        drug_organ_effects.sort(key=lambda e: e.therapeutic_benefit, reverse=True)
        
        for effect in drug_organ_effects:
            print(f"  {effect.organ_id.capitalize():>15}: "
                  f"Function={effect.functional_change:.2f}× "
                  f"Benefit={effect.therapeutic_benefit:.2f} "
                  f"Risk={effect.adverse_effect_risk:.2f} "
                  f"Outcome={effect.clinical_outcome} "
                  f"Conf={effect.confidence:.3f}")
    
    # Analyze patterns
    print("\n🔍 Analyzing organ effect patterns...")
    analysis = simulator.analyze_organ_effects(effects)
    
    print(f"\n📈 ORGAN EFFECT ANALYSIS:")
    print("-" * 50)
    
    outcomes_analysis = analysis['clinical_outcomes']
    print("Clinical Outcomes:")
    for outcome, count in outcomes_analysis.items():
        print(f"  • {outcome.capitalize()}: {count}")
    
    benefit_analysis = analysis['therapeutic_benefit']
    print(f"\nTherapeutic Benefit:")
    print(f"  • Mean benefit: {benefit_analysis['mean']:.3f}")
    print(f"  • Maximum benefit: {benefit_analysis['max']:.3f}")
    print(f"  • Beneficial effects: {benefit_analysis['beneficial_effects']}")
    
    adverse_analysis = analysis['adverse_effects']
    print(f"\nAdverse Effects:")
    print(f"  • Mean risk: {adverse_analysis['mean']:.3f}")
    print(f"  • High-risk effects: {adverse_analysis['high_risk_count']}")
    
    print(f"\nOrgan-Specific Effects:")
    for organ, metrics in analysis['organ_specific'].items():
        print(f"  • {organ.capitalize()}: Function change = {metrics['mean_functional_change']:.2f}×, "
              f"{metrics['therapeutic_outcomes']} therapeutic outcomes")
    
    print(f"\nDrug Safety Profiles:")
    for drug, safety in analysis['drug_specific'].items():
        print(f"  • {drug}: {safety['therapeutic_effects']} therapeutic, "
              f"{safety['toxic_effects']} toxic, "
              f"safety ratio = {safety['safety_ratio']:.1f}")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    simulator.save_results(effects, analysis)
    simulator.visualize_organ_effects(effects)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find most therapeutic effect
    best_therapeutic = max(effects, key=lambda e: e.therapeutic_benefit)
    print(f"• Best therapeutic effect: {best_therapeutic.drug_name} → {best_therapeutic.organ_id} "
          f"(benefit = {best_therapeutic.therapeutic_benefit:.3f})")
    
    # Find highest risk effect
    highest_risk = max(effects, key=lambda e: e.adverse_effect_risk)
    print(f"• Highest adverse risk: {highest_risk.drug_name} → {highest_risk.organ_id} "
          f"(risk = {highest_risk.adverse_effect_risk:.3f})")
    
    # Find most disruptive effect
    most_disruptive = max(effects, key=lambda e: e.oscillatory_disruption)
    print(f"• Most oscillatory disruption: {most_disruptive.drug_name} → {most_disruptive.organ_id} "
          f"({most_disruptive.oscillatory_disruption:.3f})")
    
    # Count positive outcomes
    positive_outcomes = len([e for e in effects if e.clinical_outcome == 'therapeutic'])
    print(f"• Therapeutic outcomes: {positive_outcomes}/{len(effects)}")
    
    # High confidence predictions
    high_conf = len([e for e in effects if e.confidence > 0.7])
    print(f"• High-confidence predictions: {high_conf}/{len(effects)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Organ drug effects simulation complete!")
    
    return effects

if __name__ == "__main__":
    effects = main()

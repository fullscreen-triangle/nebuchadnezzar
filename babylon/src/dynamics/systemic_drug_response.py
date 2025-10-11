"""
Systemic Drug Response Module
============================

Models system-wide drug responses by integrating organ-level effects into
comprehensive physiological outcomes. Predicts emergent properties and
cross-organ interactions in drug response.

Based on systems oscillatory coupling and homeostatic feedback loops.
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
class SystemicRegulation:
    """Represents systemic regulatory mechanisms."""
    regulation_id: str
    regulation_name: str
    controlled_organs: List[str]  # Organs under this regulation
    feedback_strength: float  # Strength of feedback control (0-1)
    oscillatory_frequency: float  # Hz - regulatory oscillation frequency
    homeostatic_setpoint: float  # Target value for regulated parameter
    adaptation_time: float  # Hours - time to adapt to perturbations
    compensation_capacity: float  # Maximum compensation ability (0-1)

@dataclass
class SystemicDrugProfile:
    """System-wide drug response profile."""
    drug_name: str
    primary_system_effects: List[str]  # Primary systems affected
    secondary_system_effects: List[str]  # Secondary/compensatory effects
    systemic_half_life: float  # Hours - system-wide response half-life
    cross_organ_interactions: Dict[str, List[str]]  # organ -> list of interacting organs
    homeostatic_disruption: float  # 0-1 scale of homeostatic disruption
    adaptation_requirement: bool  # Whether system adaptation is required

@dataclass
class SystemicResponse:
    """Result of systemic drug response simulation."""
    drug_name: str
    overall_therapeutic_index: float  # Overall benefit/risk ratio
    system_stability: float  # 0-1 scale of system stability after drug
    homeostatic_compensation: float  # Degree of compensatory response
    emergent_effects: List[str]  # Unexpected system-level effects
    organ_interactions: Dict[str, Dict[str, float]]  # organ -> {interacting_organ: strength}
    regulatory_disruption: Dict[str, float]  # regulation_id -> disruption level
    adaptation_timeline: List[Tuple[float, str]]  # (time, adaptation_event)
    system_resilience: float  # Ability to maintain function despite perturbation
    clinical_phenotype: str  # 'therapeutic', 'toxic', 'adaptive', 'maladaptive'
    systemic_biomarkers: Dict[str, float]  # System-level biomarker changes
    confidence: float = 0.0

class SystemicDrugResponse:
    """
    Models system-wide drug responses by integrating organ effects and
    predicting emergent physiological outcomes.
    
    Focuses on cross-organ interactions, homeostatic regulation, and
    system-level adaptation to drug perturbations.
    """
    
    def __init__(self):
        """Initialize systemic drug response simulator."""
        self.systemic_regulations = self._load_systemic_regulations()
        
    def _load_systemic_regulations(self) -> Dict[str, SystemicRegulation]:
        """Load systemic regulatory mechanisms."""
        
        regulations = {
            'cardiovascular_regulation': SystemicRegulation(
                regulation_id='cardiovascular_regulation',
                regulation_name='Cardiovascular Homeostasis',
                controlled_organs=['cardiovascular', 'renal', 'brain'],
                feedback_strength=0.9,  # Strong cardiovascular feedback
                oscillatory_frequency=1.2e13,  # Cardiac rhythm coupling
                homeostatic_setpoint=1.0,  # Normal cardiac output
                adaptation_time=2.0,  # 2 hours to adapt
                compensation_capacity=0.7  # Good compensation ability
            ),
            
            'metabolic_regulation': SystemicRegulation(
                regulation_id='metabolic_regulation',
                regulation_name='Metabolic Homeostasis',
                controlled_organs=['hepatic', 'musculoskeletal', 'renal'],
                feedback_strength=0.8,  # Strong metabolic feedback
                oscillatory_frequency=0.8e13,  # Metabolic oscillations
                homeostatic_setpoint=1.0,  # Normal metabolic rate
                adaptation_time=6.0,  # 6 hours to adapt metabolism
                compensation_capacity=0.8  # High metabolic flexibility
            ),
            
            'neuroendocrine_regulation': SystemicRegulation(
                regulation_id='neuroendocrine_regulation',
                regulation_name='Neuroendocrine Control',
                controlled_organs=['brain', 'cardiovascular', 'hepatic'],
                feedback_strength=0.85,  # Strong neuroendocrine control
                oscillatory_frequency=2.5e13,  # Neural oscillations
                homeostatic_setpoint=1.0,  # Normal neuroendocrine function
                adaptation_time=4.0,  # 4 hours for neuroendocrine adaptation
                compensation_capacity=0.6  # Moderate compensation
            ),
            
            'fluid_electrolyte_regulation': SystemicRegulation(
                regulation_id='fluid_electrolyte_regulation',
                regulation_name='Fluid-Electrolyte Balance',
                controlled_organs=['renal', 'cardiovascular'],
                feedback_strength=0.95,  # Very strong fluid regulation
                oscillatory_frequency=1.0e13,  # Renal oscillations
                homeostatic_setpoint=1.0,  # Normal fluid balance
                adaptation_time=1.0,  # 1 hour - rapid fluid adaptation
                compensation_capacity=0.9  # Excellent fluid compensation
            ),
            
            'detox_clearance_regulation': SystemicRegulation(
                regulation_id='detox_clearance_regulation',
                regulation_name='Detoxification & Clearance',
                controlled_organs=['hepatic', 'renal'],
                feedback_strength=0.7,  # Moderate detox feedback
                oscillatory_frequency=1.5e13,  # Detox enzyme oscillations
                homeostatic_setpoint=1.0,  # Normal clearance capacity
                adaptation_time=12.0,  # 12 hours - slower enzyme adaptation
                compensation_capacity=0.5  # Limited compensation capacity
            )
        }
        
        logger.info(f"Loaded {len(regulations)} systemic regulations")
        return regulations
    
    def create_systemic_drug_profiles(self) -> Dict[str, SystemicDrugProfile]:
        """Create system-wide drug response profiles."""
        
        profiles = {
            'lithium': SystemicDrugProfile(
                drug_name='lithium',
                primary_system_effects=['neuroendocrine_regulation'],
                secondary_system_effects=['fluid_electrolyte_regulation', 'cardiovascular_regulation'],
                systemic_half_life=18.0,  # Hours
                cross_organ_interactions={
                    'brain': ['renal', 'cardiovascular'],
                    'renal': ['cardiovascular', 'brain'],
                    'cardiovascular': ['brain', 'renal']
                },
                homeostatic_disruption=0.4,  # Moderate disruption
                adaptation_requirement=True  # Requires system adaptation
            ),
            
            'aripiprazole': SystemicDrugProfile(
                drug_name='aripiprazole',
                primary_system_effects=['neuroendocrine_regulation'],
                secondary_system_effects=['metabolic_regulation', 'cardiovascular_regulation'],
                systemic_half_life=75.0,  # Very long systemic half-life
                cross_organ_interactions={
                    'brain': ['cardiovascular', 'musculoskeletal'],
                    'cardiovascular': ['brain'],
                    'musculoskeletal': ['brain', 'metabolic_regulation']
                },
                homeostatic_disruption=0.3,  # Lower disruption
                adaptation_requirement=True  # Dopamine system adaptation
            ),
            
            'citalopram': SystemicDrugProfile(
                drug_name='citalopram',
                primary_system_effects=['neuroendocrine_regulation'],
                secondary_system_effects=['cardiovascular_regulation'],
                systemic_half_life=35.0,  # Hours
                cross_organ_interactions={
                    'brain': ['cardiovascular'],
                    'cardiovascular': ['brain']
                },
                homeostatic_disruption=0.25,  # Mild disruption
                adaptation_requirement=True  # Serotonin system adaptation
            ),
            
            'atorvastatin': SystemicDrugProfile(
                drug_name='atorvastatin',
                primary_system_effects=['metabolic_regulation'],
                secondary_system_effects=['cardiovascular_regulation', 'detox_clearance_regulation'],
                systemic_half_life=14.0,  # Hours
                cross_organ_interactions={
                    'hepatic': ['cardiovascular', 'musculoskeletal'],
                    'cardiovascular': ['hepatic'],
                    'musculoskeletal': ['hepatic', 'metabolic_regulation']
                },
                homeostatic_disruption=0.35,  # Moderate metabolic disruption
                adaptation_requirement=True  # Metabolic pathway adaptation
            ),
            
            'aspirin': SystemicDrugProfile(
                drug_name='aspirin',
                primary_system_effects=['cardiovascular_regulation'],
                secondary_system_effects=['fluid_electrolyte_regulation', 'detox_clearance_regulation'],
                systemic_half_life=0.3,  # Very short - but systemic effects persist
                cross_organ_interactions={
                    'cardiovascular': ['renal', 'hepatic'],
                    'renal': ['cardiovascular'],
                    'hepatic': ['cardiovascular', 'renal']
                },
                homeostatic_disruption=0.2,  # Mild disruption
                adaptation_requirement=False  # No significant adaptation needed
            )
        }
        
        return profiles
    
    def calculate_organ_interactions(self, drug_profile: SystemicDrugProfile,
                                   organ_effects: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate cross-organ interaction effects."""
        
        interactions = {}
        
        for organ, interacting_organs in drug_profile.cross_organ_interactions.items():
            if organ not in organ_effects:
                continue
                
            organ_effect = organ_effects[organ]
            interactions[organ] = {}
            
            for interacting_organ in interacting_organs:
                # Calculate interaction strength based on:
                # 1. Primary organ effect magnitude
                # 2. Anatomical/physiological coupling
                # 3. Drug-specific interaction patterns
                
                base_interaction = abs(organ_effect - 1.0)  # Deviation from baseline
                
                # Coupling strengths between organs
                coupling_matrix = {
                    ('brain', 'cardiovascular'): 0.8,     # Strong neurocardiac coupling
                    ('brain', 'renal'): 0.6,              # Moderate neurorenal coupling
                    ('cardiovascular', 'renal'): 0.9,     # Strong cardiorenal coupling
                    ('hepatic', 'cardiovascular'): 0.7,   # Hepatocardiac coupling
                    ('hepatic', 'musculoskeletal'): 0.6,  # Hepatomuscular coupling
                    ('renal', 'cardiovascular'): 0.9,     # Strong renal-cardiac coupling
                }
                
                # Check both directions for coupling
                coupling_key = (organ, interacting_organ)
                reverse_coupling_key = (interacting_organ, organ)
                
                coupling_strength = coupling_matrix.get(coupling_key, 
                                   coupling_matrix.get(reverse_coupling_key, 0.4))
                
                # Calculate interaction effect
                interaction_effect = base_interaction * coupling_strength
                
                # Apply drug-specific modulation
                drug_modulation = {
                    'lithium': 1.2,      # Strong systemic interactions
                    'aripiprazole': 0.8,  # Moderate interactions
                    'citalopram': 0.7,    # Milder interactions
                    'atorvastatin': 1.0,  # Baseline interactions
                    'aspirin': 0.9        # Slightly reduced interactions
                }
                
                modulation = drug_modulation.get(drug_profile.drug_name, 1.0)
                final_interaction = interaction_effect * modulation
                
                interactions[organ][interacting_organ] = final_interaction
        
        return interactions
    
    def calculate_homeostatic_compensation(self, drug_profile: SystemicDrugProfile,
                                         organ_effects: Dict[str, float]) -> float:
        """Calculate overall homeostatic compensation response."""
        
        total_compensation = 0.0
        active_regulations = []
        
        # Check each regulatory system
        for regulation_id, regulation in self.systemic_regulations.items():
            if regulation_id in drug_profile.primary_system_effects or \
               regulation_id in drug_profile.secondary_system_effects:
                
                # Calculate perturbation to this regulatory system
                perturbation = 0.0
                regulated_organ_count = 0
                
                for organ in regulation.controlled_organs:
                    if organ in organ_effects:
                        organ_deviation = abs(organ_effects[organ] - regulation.homeostatic_setpoint)
                        perturbation += organ_deviation
                        regulated_organ_count += 1
                
                if regulated_organ_count > 0:
                    avg_perturbation = perturbation / regulated_organ_count
                    
                    # Calculate compensation based on regulation strength and capacity
                    compensation = (avg_perturbation * 
                                  regulation.feedback_strength * 
                                  regulation.compensation_capacity)
                    
                    total_compensation += compensation
                    active_regulations.append(regulation_id)
        
        # Normalize by number of active regulations
        if len(active_regulations) > 0:
            normalized_compensation = total_compensation / len(active_regulations)
        else:
            normalized_compensation = 0.0
        
        return min(1.0, normalized_compensation)
    
    def calculate_system_stability(self, drug_profile: SystemicDrugProfile,
                                 organ_effects: Dict[str, float],
                                 homeostatic_compensation: float) -> float:
        """Calculate overall system stability after drug perturbation."""
        
        # Base stability from organ effect variability
        organ_deviations = [abs(effect - 1.0) for effect in organ_effects.values()]
        effect_variability = np.std(organ_deviations) if organ_deviations else 0.0
        
        # Stability decreases with effect variability
        variability_penalty = effect_variability * 0.5
        
        # Homeostatic disruption reduces stability
        disruption_penalty = drug_profile.homeostatic_disruption * 0.3
        
        # Compensation improves stability
        compensation_benefit = homeostatic_compensation * 0.4
        
        # Base stability
        base_stability = 1.0
        
        # Calculate final stability
        stability = (base_stability - 
                    variability_penalty - 
                    disruption_penalty + 
                    compensation_benefit)
        
        return max(0.0, min(1.0, stability))
    
    def predict_emergent_effects(self, drug_profile: SystemicDrugProfile,
                                organ_effects: Dict[str, float],
                                organ_interactions: Dict[str, Dict[str, float]]) -> List[str]:
        """Predict emergent system-level effects."""
        
        emergent_effects = []
        
        # Check for cascade effects
        max_interaction = 0.0
        for organ_interactions_dict in organ_interactions.values():
            for interaction_strength in organ_interactions_dict.values():
                max_interaction = max(max_interaction, interaction_strength)
        
        if max_interaction > 0.5:
            emergent_effects.append('cross_organ_cascade')
        
        # Check for compensatory overshot
        compensatory_organs = []
        for organ, effect in organ_effects.items():
            if effect > 1.3:  # Significant upregulation
                compensatory_organs.append(organ)
        
        if len(compensatory_organs) >= 2:
            emergent_effects.append('compensatory_overactivation')
        
        # Check for system oscillation disruption
        disrupted_regulations = []
        for regulation_id in drug_profile.primary_system_effects:
            if regulation_id in self.systemic_regulations:
                regulation = self.systemic_regulations[regulation_id]
                affected_organs = [o for o in regulation.controlled_organs if o in organ_effects]
                if len(affected_organs) >= 2:
                    disrupted_regulations.append(regulation_id)
        
        if len(disrupted_regulations) >= 2:
            emergent_effects.append('multi_system_dysregulation')
        
        # Drug-specific emergent effects
        drug_specific_effects = {
            'lithium': ['electrolyte_cascade', 'thyroid_axis_disruption'],
            'aripiprazole': ['metabolic_syndrome_risk', 'tardive_dyskinesia_risk'],
            'citalopram': ['serotonin_syndrome_risk', 'qt_prolongation_risk'],
            'atorvastatin': ['pleiotropic_cardiovascular_effects', 'myopathy_cascade'],
            'aspirin': ['antiplatelet_cascade', 'reye_syndrome_risk']
        }
        
        specific_effects = drug_specific_effects.get(drug_profile.drug_name, [])
        
        # Add specific effects based on conditions
        for effect in specific_effects:
            if effect == 'electrolyte_cascade' and 'renal' in organ_effects and organ_effects['renal'] < 0.8:
                emergent_effects.append(effect)
            elif max_interaction > 0.3:  # Other effects need significant interactions
                emergent_effects.append(effect)
        
        return emergent_effects
    
    def calculate_therapeutic_index(self, organ_effects: Dict[str, float],
                                  organ_interactions: Dict[str, Dict[str, float]],
                                  drug_profile: SystemicDrugProfile) -> float:
        """Calculate overall therapeutic index (benefit/risk ratio)."""
        
        # Calculate total therapeutic benefit
        therapeutic_benefit = 0.0
        for regulation_id in drug_profile.primary_system_effects:
            regulation = self.systemic_regulations.get(regulation_id)
            if regulation:
                for organ in regulation.controlled_organs:
                    if organ in organ_effects:
                        # Benefit from therapeutic improvement
                        if organ_effects[organ] > 1.0:
                            therapeutic_benefit += (organ_effects[organ] - 1.0) * 0.5
        
        # Calculate total risk
        risk = 0.0
        
        # Risk from organ dysfunction
        for organ, effect in organ_effects.items():
            if effect < 0.8:  # Significant dysfunction
                risk += (1.0 - effect) * 0.7
        
        # Risk from interactions
        for organ_interactions_dict in organ_interactions.values():
            for interaction_strength in organ_interactions_dict.values():
                if interaction_strength > 0.4:
                    risk += interaction_strength * 0.3
        
        # Risk from homeostatic disruption
        risk += drug_profile.homeostatic_disruption * 0.2
        
        # Calculate therapeutic index
        if risk > 0:
            therapeutic_index = therapeutic_benefit / (risk + 0.1)  # Avoid division by zero
        else:
            therapeutic_index = therapeutic_benefit * 10  # High index for no risk
        
        return min(10.0, therapeutic_index)  # Cap at 10
    
    def determine_clinical_phenotype(self, therapeutic_index: float,
                                   system_stability: float,
                                   emergent_effects: List[str]) -> str:
        """Determine overall clinical phenotype."""
        
        # Base classification on therapeutic index and stability
        if therapeutic_index > 2.0 and system_stability > 0.7:
            base_phenotype = 'therapeutic'
        elif therapeutic_index < 0.5 or system_stability < 0.4:
            base_phenotype = 'toxic'
        elif len(emergent_effects) > 2:
            base_phenotype = 'maladaptive'
        else:
            base_phenotype = 'adaptive'
        
        # Modify based on emergent effects
        high_risk_effects = ['multi_system_dysregulation', 'compensatory_overactivation']
        if any(effect in emergent_effects for effect in high_risk_effects):
            if base_phenotype == 'therapeutic':
                base_phenotype = 'adaptive'  # Downgrade due to complexity
            elif base_phenotype == 'adaptive':
                base_phenotype = 'maladaptive'  # Further downgrade
        
        return base_phenotype
    
    def calculate_system_resilience(self, drug_profile: SystemicDrugProfile,
                                  homeostatic_compensation: float,
                                  system_stability: float) -> float:
        """Calculate system resilience to drug perturbation."""
        
        # Base resilience from compensation ability
        base_resilience = homeostatic_compensation * 0.6
        
        # Stability contributes to resilience
        stability_contribution = system_stability * 0.3
        
        # Adaptation requirement affects resilience
        if drug_profile.adaptation_requirement:
            adaptation_penalty = 0.1
        else:
            adaptation_penalty = 0.0
        
        # Drug-specific resilience factors
        drug_resilience_factors = {
            'lithium': 0.7,      # Requires careful monitoring
            'aripiprazole': 0.8,  # Generally well-tolerated systemically
            'citalopram': 0.85,   # Good systemic tolerance
            'atorvastatin': 0.75, # Moderate systemic impact
            'aspirin': 0.9        # High systemic resilience
        }
        
        drug_factor = drug_resilience_factors.get(drug_profile.drug_name, 0.8)
        
        resilience = (base_resilience + stability_contribution - adaptation_penalty) * drug_factor
        
        return max(0.0, min(1.0, resilience))
    
    def generate_adaptation_timeline(self, drug_profile: SystemicDrugProfile) -> List[Tuple[float, str]]:
        """Generate timeline of system adaptation events."""
        
        timeline = []
        
        # Immediate effects (0-2 hours)
        timeline.append((0.5, 'initial_drug_distribution'))
        timeline.append((1.0, 'primary_target_engagement'))
        
        # Early adaptation (2-8 hours)
        if drug_profile.adaptation_requirement:
            timeline.append((3.0, 'homeostatic_response_initiation'))
            timeline.append((6.0, 'compensatory_mechanism_activation'))
        
        # Medium-term adaptation (8-24 hours)
        for regulation_id in drug_profile.primary_system_effects:
            regulation = self.systemic_regulations.get(regulation_id)
            if regulation:
                adapt_time = regulation.adaptation_time
                if adapt_time <= 24:
                    timeline.append((adapt_time, f'{regulation_id}_adaptation'))
        
        # Long-term effects (1-7 days)
        if drug_profile.systemic_half_life > 24:
            timeline.append((48.0, 'steady_state_establishment'))
            if len(drug_profile.secondary_system_effects) > 1:
                timeline.append((96.0, 'secondary_system_adaptation'))
        
        # Sort timeline by time
        timeline.sort(key=lambda x: x[0])
        
        return timeline
    
    def calculate_systemic_biomarkers(self, drug_profile: SystemicDrugProfile,
                                    organ_effects: Dict[str, float]) -> Dict[str, float]:
        """Calculate system-level biomarker changes."""
        
        biomarkers = {}
        
        # Inflammatory markers
        inflammatory_stress = sum(abs(effect - 1.0) for effect in organ_effects.values()) / len(organ_effects)
        biomarkers['c_reactive_protein'] = 1.0 + inflammatory_stress * 0.3
        biomarkers['interleukin_6'] = 1.0 + inflammatory_stress * 0.5
        
        # Stress response markers
        stress_response = drug_profile.homeostatic_disruption
        biomarkers['cortisol'] = 1.0 + stress_response * 0.4
        biomarkers['catecholamines'] = 1.0 + stress_response * 0.6
        
        # Metabolic markers
        metabolic_organs = ['hepatic', 'musculoskeletal']
        metabolic_effects = [organ_effects.get(organ, 1.0) for organ in metabolic_organs]
        avg_metabolic_effect = np.mean(metabolic_effects)
        
        biomarkers['glucose'] = avg_metabolic_effect * 0.9 + 0.1  # Inverse relationship for some drugs
        biomarkers['lipid_profile'] = avg_metabolic_effect
        
        # Cardiovascular markers
        if 'cardiovascular' in organ_effects:
            cv_effect = organ_effects['cardiovascular']
            biomarkers['bnp'] = 2.0 - cv_effect  # Inverse relationship
            biomarkers['troponin'] = 1.0 + max(0, 1.0 - cv_effect) * 0.3
        
        # Renal markers
        if 'renal' in organ_effects:
            renal_effect = organ_effects['renal']
            biomarkers['creatinine'] = 2.0 - renal_effect  # Inverse relationship
            biomarkers['bun'] = 2.0 - renal_effect
        
        return biomarkers
    
    def simulate_systemic_drug_response(self, drug_profile: SystemicDrugProfile,
                                      organ_effects: Dict[str, float]) -> SystemicResponse:
        """Simulate complete systemic drug response."""
        
        # Calculate cross-organ interactions
        organ_interactions = self.calculate_organ_interactions(drug_profile, organ_effects)
        
        # Calculate homeostatic compensation
        homeostatic_compensation = self.calculate_homeostatic_compensation(drug_profile, organ_effects)
        
        # Calculate system stability
        system_stability = self.calculate_system_stability(
            drug_profile, organ_effects, homeostatic_compensation
        )
        
        # Predict emergent effects
        emergent_effects = self.predict_emergent_effects(
            drug_profile, organ_effects, organ_interactions
        )
        
        # Calculate therapeutic index
        therapeutic_index = self.calculate_therapeutic_index(
            organ_effects, organ_interactions, drug_profile
        )
        
        # Determine clinical phenotype
        clinical_phenotype = self.determine_clinical_phenotype(
            therapeutic_index, system_stability, emergent_effects
        )
        
        # Calculate system resilience
        system_resilience = self.calculate_system_resilience(
            drug_profile, homeostatic_compensation, system_stability
        )
        
        # Generate adaptation timeline
        adaptation_timeline = self.generate_adaptation_timeline(drug_profile)
        
        # Calculate regulatory disruption
        regulatory_disruption = {}
        for regulation_id in drug_profile.primary_system_effects + drug_profile.secondary_system_effects:
            if regulation_id in self.systemic_regulations:
                regulation = self.systemic_regulations[regulation_id]
                disruption = 0.0
                for organ in regulation.controlled_organs:
                    if organ in organ_effects:
                        disruption += abs(organ_effects[organ] - regulation.homeostatic_setpoint)
                
                avg_disruption = disruption / len(regulation.controlled_organs) if regulation.controlled_organs else 0.0
                regulatory_disruption[regulation_id] = min(1.0, avg_disruption)
        
        # Calculate systemic biomarkers
        systemic_biomarkers = self.calculate_systemic_biomarkers(drug_profile, organ_effects)
        
        # Calculate confidence
        confidence = self._calculate_systemic_confidence(drug_profile, system_stability, len(emergent_effects))
        
        return SystemicResponse(
            drug_name=drug_profile.drug_name,
            overall_therapeutic_index=therapeutic_index,
            system_stability=system_stability,
            homeostatic_compensation=homeostatic_compensation,
            emergent_effects=emergent_effects,
            organ_interactions=organ_interactions,
            regulatory_disruption=regulatory_disruption,
            adaptation_timeline=adaptation_timeline,
            system_resilience=system_resilience,
            clinical_phenotype=clinical_phenotype,
            systemic_biomarkers=systemic_biomarkers,
            confidence=confidence
        )
    
    def _calculate_systemic_confidence(self, drug_profile: SystemicDrugProfile,
                                     system_stability: float,
                                     num_emergent_effects: int) -> float:
        """Calculate confidence in systemic response prediction."""
        
        base_confidence = 0.6
        
        # Higher stability increases confidence
        stability_boost = system_stability * 0.2
        
        # Fewer emergent effects increase confidence
        complexity_penalty = min(0.3, num_emergent_effects * 0.1)
        
        # Well-studied drugs have higher confidence
        well_studied_drugs = ['lithium', 'aspirin', 'atorvastatin']
        if drug_profile.drug_name in well_studied_drugs:
            drug_boost = 0.15
        else:
            drug_boost = 0.0
        
        # Simple drug profiles have higher confidence
        if len(drug_profile.primary_system_effects) == 1:
            simplicity_boost = 0.1
        else:
            simplicity_boost = 0.0
        
        final_confidence = min(0.95, base_confidence + stability_boost - 
                              complexity_penalty + drug_boost + simplicity_boost)
        
        return max(0.2, final_confidence)
    
    def simulate_drug_library(self, organ_effects_data: Dict[str, Dict[str, float]]) -> List[SystemicResponse]:
        """Simulate systemic responses for all drugs."""
        
        drug_profiles = self.create_systemic_drug_profiles()
        all_responses = []
        
        for drug_name, drug_profile in drug_profiles.items():
            logger.info(f"Simulating systemic response for {drug_name}")
            
            # Get organ effects for this drug (use mock data if not provided)
            if drug_name in organ_effects_data:
                organ_effects = organ_effects_data[drug_name]
            else:
                # Generate mock organ effects
                organ_effects = self._generate_mock_organ_effects(drug_name)
            
            response = self.simulate_systemic_drug_response(drug_profile, organ_effects)
            all_responses.append(response)
        
        logger.info(f"Completed systemic response simulation for {len(all_responses)} drugs")
        return all_responses
    
    def _generate_mock_organ_effects(self, drug_name: str) -> Dict[str, float]:
        """Generate mock organ effects for demonstration."""
        
        # Mock organ functional changes based on typical drug effects
        mock_effects = {
            'lithium': {
                'brain': 1.2, 'renal': 0.8, 'cardiovascular': 0.9,
                'hepatic': 1.0, 'musculoskeletal': 1.0
            },
            'aripiprazole': {
                'brain': 1.3, 'cardiovascular': 0.95, 'musculoskeletal': 0.9,
                'hepatic': 1.1, 'renal': 1.0
            },
            'citalopram': {
                'brain': 1.25, 'cardiovascular': 0.92, 'hepatic': 1.0,
                'renal': 1.0, 'musculoskeletal': 1.0
            },
            'atorvastatin': {
                'hepatic': 1.4, 'cardiovascular': 1.2, 'musculoskeletal': 0.85,
                'brain': 1.0, 'renal': 1.0
            },
            'aspirin': {
                'cardiovascular': 1.15, 'renal': 0.93, 'hepatic': 0.95,
                'brain': 1.0, 'musculoskeletal': 1.0
            }
        }
        
        return mock_effects.get(drug_name, {
            'brain': 1.0, 'cardiovascular': 1.0, 'hepatic': 1.0,
            'renal': 1.0, 'musculoskeletal': 1.0
        })
    
    def analyze_systemic_responses(self, responses: List[SystemicResponse]) -> Dict[str, Any]:
        """Analyze patterns in systemic drug responses."""
        
        analysis = {}
        
        # Clinical phenotype distribution
        phenotypes = [r.clinical_phenotype for r in responses]
        analysis['clinical_phenotypes'] = {
            'therapeutic': phenotypes.count('therapeutic'),
            'toxic': phenotypes.count('toxic'),
            'adaptive': phenotypes.count('adaptive'),
            'maladaptive': phenotypes.count('maladaptive')
        }
        
        # Therapeutic index statistics
        therapeutic_indices = [r.overall_therapeutic_index for r in responses]
        analysis['therapeutic_index'] = {
            'mean': np.mean(therapeutic_indices),
            'max': np.max(therapeutic_indices),
            'drugs_with_good_index': len([ti for ti in therapeutic_indices if ti > 2.0])
        }
        
        # System stability analysis
        stabilities = [r.system_stability for r in responses]
        analysis['system_stability'] = {
            'mean': np.mean(stabilities),
            'stable_responses': len([s for s in stabilities if s > 0.7])
        }
        
        # Emergent effects analysis
        all_emergent_effects = []
        for response in responses:
            all_emergent_effects.extend(response.emergent_effects)
        
        emergent_effect_counts = {}
        for effect in all_emergent_effects:
            emergent_effect_counts[effect] = emergent_effect_counts.get(effect, 0) + 1
        
        analysis['emergent_effects'] = {
            'total_unique_effects': len(set(all_emergent_effects)),
            'most_common_effects': dict(sorted(emergent_effect_counts.items(), 
                                             key=lambda x: x[1], reverse=True)[:5])
        }
        
        # Resilience analysis
        resiliences = [r.system_resilience for r in responses]
        analysis['system_resilience'] = {
            'mean': np.mean(resiliences),
            'resilient_systems': len([r for r in resiliences if r > 0.8])
        }
        
        # Drug-specific analysis
        drug_analysis = {}
        for response in responses:
            drug_analysis[response.drug_name] = {
                'therapeutic_index': response.overall_therapeutic_index,
                'stability': response.system_stability,
                'resilience': response.system_resilience,
                'phenotype': response.clinical_phenotype,
                'emergent_effects_count': len(response.emergent_effects)
            }
        
        analysis['drug_specific'] = drug_analysis
        
        return analysis
    
    def visualize_systemic_responses(self, responses: List[SystemicResponse],
                                   output_dir: str = "babylon_results") -> None:
        """Create visualizations of systemic drug responses."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive systemic response visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Clinical phenotypes
        phenotypes = [r.clinical_phenotype for r in responses]
        phenotype_counts = pd.Series(phenotypes).value_counts()
        
        axes[0, 0].pie(phenotype_counts.values, labels=phenotype_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Clinical Phenotype Distribution')
        
        # 2. Therapeutic index vs system stability
        therapeutic_indices = [r.overall_therapeutic_index for r in responses]
        stabilities = [r.system_stability for r in responses]
        drug_names = [r.drug_name for r in responses]
        
        scatter = axes[0, 1].scatter(therapeutic_indices, stabilities, alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Therapeutic Index')
        axes[0, 1].set_ylabel('System Stability')
        axes[0, 1].set_title('Therapeutic Index vs System Stability')
        
        # Annotate points
        for i, name in enumerate(drug_names):
            axes[0, 1].annotate(name, (therapeutic_indices[i], stabilities[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Add quadrant lines
        axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Stability Threshold')
        axes[0, 1].axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='Therapeutic Threshold')
        axes[0, 1].legend()
        
        # 3. System resilience by drug
        resiliences = [r.system_resilience for r in responses]
        
        bars = axes[0, 2].bar(drug_names, resiliences, color='lightcoral', alpha=0.7)
        axes[0, 2].set_ylabel('System Resilience')
        axes[0, 2].set_title('System Resilience by Drug')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='High Resilience')
        axes[0, 2].legend()
        
        # Color bars by resilience level
        for bar, resilience in zip(bars, resiliences):
            if resilience > 0.8:
                bar.set_color('green')
            elif resilience > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 4. Homeostatic compensation
        compensations = [r.homeostatic_compensation for r in responses]
        
        axes[1, 0].bar(drug_names, compensations, color='lightblue', alpha=0.7)
        axes[1, 0].set_ylabel('Homeostatic Compensation')
        axes[1, 0].set_title('Homeostatic Compensation by Drug')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Number of emergent effects
        emergent_counts = [len(r.emergent_effects) for r in responses]
        
        axes[1, 1].bar(drug_names, emergent_counts, color='lightyellow', alpha=0.7)
        axes[1, 1].set_ylabel('Number of Emergent Effects')
        axes[1, 1].set_title('Emergent Effects by Drug')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Confidence distribution
        confidences = [r.confidence for r in responses]
        
        axes[1, 2].hist(confidences, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 2].set_xlabel('Prediction Confidence')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Systemic Prediction Confidence')
        axes[1, 2].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'systemic_drug_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Systemic response visualizations saved to {output_path}")
    
    def save_results(self, responses: List[SystemicResponse], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save systemic drug response simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed responses as JSON
        responses_data = []
        for response in responses:
            responses_data.append({
                'drug_name': response.drug_name,
                'overall_therapeutic_index': response.overall_therapeutic_index,
                'system_stability': response.system_stability,
                'homeostatic_compensation': response.homeostatic_compensation,
                'emergent_effects': response.emergent_effects,
                'organ_interactions': response.organ_interactions,
                'regulatory_disruption': response.regulatory_disruption,
                'adaptation_timeline': response.adaptation_timeline,
                'system_resilience': response.system_resilience,
                'clinical_phenotype': response.clinical_phenotype,
                'systemic_biomarkers': response.systemic_biomarkers,
                'confidence': response.confidence
            })
        
        with open(output_path / 'systemic_drug_responses.json', 'w') as f:
            json.dump(responses_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_data = []
        for response in responses:
            csv_data.append({
                'drug_name': response.drug_name,
                'therapeutic_index': response.overall_therapeutic_index,
                'system_stability': response.system_stability,
                'homeostatic_compensation': response.homeostatic_compensation,
                'system_resilience': response.system_resilience,
                'clinical_phenotype': response.clinical_phenotype,
                'emergent_effects_count': len(response.emergent_effects),
                'confidence': response.confidence
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / 'systemic_drug_responses.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'systemic_response_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Systemic drug response results saved to {output_path}")

def main():
    """
    Test systemic drug response simulation.
    
    This demonstrates how organ-level effects integrate into
    system-wide physiological responses and clinical outcomes.
    """
    
    print("🌐 Testing Systemic Drug Response")
    print("=" * 50)
    
    # Initialize systemic response simulator
    simulator = SystemicDrugResponse()
    
    # Display loaded regulatory systems
    print(f"\n🔄 Loaded Systemic Regulations:")
    for regulation_id, regulation in simulator.systemic_regulations.items():
        print(f"  • {regulation_id}: {regulation.regulation_name}")
        print(f"    Controlled organs: {regulation.controlled_organs}")
        print(f"    Feedback strength: {regulation.feedback_strength:.1f}")
        print(f"    Adaptation time: {regulation.adaptation_time:.1f}h")
        print(f"    Compensation capacity: {regulation.compensation_capacity:.1f}")
    
    # Create and display drug profiles
    drug_profiles = simulator.create_systemic_drug_profiles()
    print(f"\n💊 Systemic Drug Profiles:")
    for drug_name, profile in drug_profiles.items():
        print(f"  • {drug_name}:")
        print(f"    Primary effects: {profile.primary_system_effects}")
        print(f"    Secondary effects: {profile.secondary_system_effects}")
        print(f"    Systemic half-life: {profile.systemic_half_life:.1f}h")
        print(f"    Homeostatic disruption: {profile.homeostatic_disruption:.2f}")
        print(f"    Adaptation required: {profile.adaptation_requirement}")
    
    # Simulate systemic responses for all drugs
    print(f"\n🔬 Simulating systemic drug responses...")
    responses = simulator.simulate_drug_library({})  # Using mock data
    
    # Display results
    print(f"\n📊 SYSTEMIC DRUG RESPONSE RESULTS:")
    print("-" * 80)
    
    for response in responses:
        print(f"\n{response.drug_name.upper()}:")
        print(f"  Therapeutic Index: {response.overall_therapeutic_index:.2f}")
        print(f"  System Stability: {response.system_stability:.3f}")
        print(f"  Homeostatic Compensation: {response.homeostatic_compensation:.3f}")
        print(f"  System Resilience: {response.system_resilience:.3f}")
        print(f"  Clinical Phenotype: {response.clinical_phenotype}")
        print(f"  Emergent Effects: {', '.join(response.emergent_effects) if response.emergent_effects else 'None'}")
        print(f"  Confidence: {response.confidence:.3f}")
        
        # Show key organ interactions
        if response.organ_interactions:
            print(f"  Key Organ Interactions:")
            for organ, interactions in response.organ_interactions.items():
                for target_organ, strength in interactions.items():
                    if strength > 0.3:  # Only show significant interactions
                        print(f"    {organ} → {target_organ}: {strength:.3f}")
    
    # Analyze patterns
    print("\n🔍 Analyzing systemic response patterns...")
    analysis = simulator.analyze_systemic_responses(responses)
    
    print(f"\n📈 SYSTEMIC RESPONSE ANALYSIS:")
    print("-" * 50)
    
    phenotypes_analysis = analysis['clinical_phenotypes']
    print("Clinical Phenotypes:")
    for phenotype, count in phenotypes_analysis.items():
        print(f"  • {phenotype.capitalize()}: {count}")
    
    ti_analysis = analysis['therapeutic_index']
    print(f"\nTherapeutic Index:")
    print(f"  • Mean: {ti_analysis['mean']:.2f}")
    print(f"  • Maximum: {ti_analysis['max']:.2f}")
    print(f"  • Drugs with good index (>2.0): {ti_analysis['drugs_with_good_index']}")
    
    stability_analysis = analysis['system_stability']
    print(f"\nSystem Stability:")
    print(f"  • Mean stability: {stability_analysis['mean']:.3f}")
    print(f"  • Stable responses (>0.7): {stability_analysis['stable_responses']}")
    
    emergent_analysis = analysis['emergent_effects']
    print(f"\nEmergent Effects:")
    print(f"  • Total unique effects: {emergent_analysis['total_unique_effects']}")
    print(f"  • Most common effects:")
    for effect, count in emergent_analysis['most_common_effects'].items():
        print(f"    - {effect}: {count}")
    
    resilience_analysis = analysis['system_resilience']
    print(f"\nSystem Resilience:")
    print(f"  • Mean resilience: {resilience_analysis['mean']:.3f}")
    print(f"  • Resilient systems (>0.8): {resilience_analysis['resilient_systems']}")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    simulator.save_results(responses, analysis)
    simulator.visualize_systemic_responses(responses)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find best therapeutic response
    best_therapeutic = max(responses, key=lambda r: r.overall_therapeutic_index)
    print(f"• Best therapeutic index: {best_therapeutic.drug_name} ({best_therapeutic.overall_therapeutic_index:.2f})")
    
    # Find most stable system
    most_stable = max(responses, key=lambda r: r.system_stability)
    print(f"• Most stable system: {most_stable.drug_name} ({most_stable.system_stability:.3f})")
    
    # Find most resilient system
    most_resilient = max(responses, key=lambda r: r.system_resilience)
    print(f"• Most resilient system: {most_resilient.drug_name} ({most_resilient.system_resilience:.3f})")
    
    # Count beneficial phenotypes
    beneficial_phenotypes = len([r for r in responses if r.clinical_phenotype in ['therapeutic', 'adaptive']])
    print(f"• Beneficial phenotypes: {beneficial_phenotypes}/{len(responses)}")
    
    # Complex emergent effects
    complex_responses = len([r for r in responses if len(r.emergent_effects) >= 2])
    print(f"• Complex responses (≥2 emergent effects): {complex_responses}/{len(responses)}")
    
    # High confidence predictions
    high_conf = len([r for r in responses if r.confidence > 0.7])
    print(f"• High-confidence predictions: {high_conf}/{len(responses)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Systemic drug response simulation complete!")
    
    return responses

if __name__ == "__main__":
    responses = main()

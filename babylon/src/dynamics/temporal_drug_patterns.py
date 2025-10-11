"""
Temporal Drug Patterns Module
============================

Models temporal coordination patterns in drug response across multiple time
scales. Analyzes oscillatory synchronization, chronotherapy effects, and
long-term adaptation patterns in pharmacological systems.

Based on multi-scale temporal dynamics and circadian-ultradian coupling.
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
class TemporalScale:
    """Represents a temporal scale in biological systems."""
    scale_id: str
    scale_name: str
    characteristic_frequency: float  # Hz - dominant frequency
    period_range: Tuple[float, float]  # (min_period, max_period) in hours
    biological_processes: List[str]  # Processes operating at this scale
    coupling_strength: float  # Strength of coupling with other scales (0-1)
    drug_sensitivity: float  # Sensitivity to drug perturbations (0-1)

@dataclass
class DrugTemporalProfile:
    """Drug's temporal response characteristics."""
    drug_name: str
    optimal_dosing_times: List[float]  # Hours from midnight for optimal dosing
    chronotherapy_benefit: float  # 0-1 scale of benefit from timed dosing  
    temporal_scales_affected: List[str]  # Temporal scales influenced by drug
    synchronization_effects: Dict[str, float]  # scale_id -> synchronization change
    adaptation_phases: List[Tuple[float, str]]  # (duration_hours, phase_name)
    circadian_disruption: float  # 0-1 scale of circadian rhythm disruption
    ultradian_modulation: float  # Effect on ultradian rhythms

@dataclass
class TemporalPattern:
    """Result of temporal drug pattern analysis."""
    drug_name: str
    administration_time: float  # Hours from midnight
    temporal_synchronization: Dict[str, float]  # scale_id -> synchronization score
    chronotherapy_advantage: float  # Advantage of optimal timing vs random
    circadian_phase_shift: float  # Hours of phase shift induced
    ultradian_rhythm_changes: Dict[str, float]  # rhythm_type -> change
    adaptation_trajectory: List[Tuple[float, Dict[str, float]]]  # (time, state)
    multi_scale_coupling: Dict[str, Dict[str, float]]  # scale1 -> {scale2: coupling}
    temporal_biorhythms: Dict[str, List[Tuple[float, float]]]  # rhythm -> (time, value)
    optimal_dosing_window: Tuple[float, float]  # (start_hour, end_hour)
    confidence: float = 0.0

class TemporalDrugPatterns:
    """
    Models temporal coordination patterns in drug response across
    multiple biological time scales.
    
    Analyzes how drugs affect oscillatory synchronization, circadian
    rhythms, and multi-scale temporal coordination in biological systems.
    """
    
    def __init__(self):
        """Initialize temporal drug patterns analyzer."""
        self.temporal_scales = self._load_temporal_scales()
        
    def _load_temporal_scales(self) -> Dict[str, TemporalScale]:
        """Load biological temporal scales for analysis."""
        
        scales = {
            'molecular_oscillations': TemporalScale(
                scale_id='molecular_oscillations',
                scale_name='Molecular Oscillations',
                characteristic_frequency=1e13,  # ~10 THz range
                period_range=(1e-12, 1e-9),  # Femtoseconds to nanoseconds
                biological_processes=['protein_folding', 'enzyme_catalysis', 'molecular_binding'],
                coupling_strength=0.3,  # Weak coupling to larger scales
                drug_sensitivity=0.9  # High sensitivity to drug binding
            ),
            
            'cellular_oscillations': TemporalScale(
                scale_id='cellular_oscillations',
                scale_name='Cellular Oscillations',
                characteristic_frequency=1e-3,  # ~1 mHz (minutes)
                period_range=(0.1, 2.0),  # 6 minutes to 2 hours
                biological_processes=['calcium_oscillations', 'metabolic_cycles', 'gene_expression'],
                coupling_strength=0.7,  # Strong cellular coupling
                drug_sensitivity=0.8  # High drug sensitivity  
            ),
            
            'ultradian_rhythms': TemporalScale(
                scale_id='ultradian_rhythms',
                scale_name='Ultradian Rhythms',
                characteristic_frequency=2.78e-5,  # ~90-120 minute cycles
                period_range=(1.0, 4.0),  # 1-4 hours
                biological_processes=['sleep_cycles', 'hormone_pulses', 'feeding_cycles'],
                coupling_strength=0.8,  # Strong coupling
                drug_sensitivity=0.6  # Moderate drug sensitivity
            ),
            
            'circadian_rhythms': TemporalScale(
                scale_id='circadian_rhythms',
                scale_name='Circadian Rhythms',
                characteristic_frequency=1.16e-5,  # ~24 hour period
                period_range=(20.0, 28.0),  # 20-28 hours
                biological_processes=['sleep_wake_cycle', 'body_temperature', 'hormone_cycles'],
                coupling_strength=0.9,  # Very strong coupling
                drug_sensitivity=0.5  # Moderate sensitivity (robust rhythms)
            ),
            
            'infradian_rhythms': TemporalScale(
                scale_id='infradian_rhythms',
                scale_name='Infradian Rhythms',
                characteristic_frequency=3.86e-7,  # ~monthly cycles
                period_range=(168.0, 744.0),  # 1 week to 1 month
                biological_processes=['menstrual_cycles', 'seasonal_rhythms', 'immune_cycles'],
                coupling_strength=0.6,  # Moderate coupling
                drug_sensitivity=0.3  # Lower sensitivity (slow adaptation)
            ),
            
            'pharmacokinetic_rhythms': TemporalScale(
                scale_id='pharmacokinetic_rhythms',
                scale_name='Pharmacokinetic Rhythms',
                characteristic_frequency=1e-4,  # ~2-3 hour periods
                period_range=(0.5, 12.0),  # 30 minutes to 12 hours
                biological_processes=['drug_absorption', 'metabolism_cycles', 'elimination'],
                coupling_strength=0.5,  # Moderate coupling
                drug_sensitivity=1.0  # Maximum sensitivity (drug-specific)
            )
        }
        
        logger.info(f"Loaded {len(scales)} temporal scales")
        return scales
    
    def create_drug_temporal_profiles(self) -> Dict[str, DrugTemporalProfile]:
        """Create temporal response profiles for drugs."""
        
        profiles = {
            'lithium': DrugTemporalProfile(
                drug_name='lithium',
                optimal_dosing_times=[8.0, 20.0],  # Morning and evening
                chronotherapy_benefit=0.7,  # High chronotherapy benefit
                temporal_scales_affected=['circadian_rhythms', 'ultradian_rhythms', 'cellular_oscillations'],
                synchronization_effects={
                    'circadian_rhythms': 0.3,      # Stabilizes circadian rhythms
                    'ultradian_rhythms': 0.2,      # Mild ultradian stabilization
                    'cellular_oscillations': -0.1   # Slight cellular desynchronization
                },
                adaptation_phases=[
                    (72.0, 'acute_adaptation'),     # 3 days
                    (336.0, 'subacute_adaptation'), # 2 weeks  
                    (2160.0, 'chronic_adaptation')  # 3 months
                ],
                circadian_disruption=0.2,  # Mild circadian disruption initially
                ultradian_modulation=0.4   # Moderate ultradian effects
            ),
            
            'aripiprazole': DrugTemporalProfile(
                drug_name='aripiprazole',
                optimal_dosing_times=[9.0],  # Morning dosing
                chronotherapy_benefit=0.5,  # Moderate chronotherapy benefit
                temporal_scales_affected=['circadian_rhythms', 'cellular_oscillations', 'pharmacokinetic_rhythms'],
                synchronization_effects={
                    'circadian_rhythms': 0.4,       # Good circadian stabilization
                    'cellular_oscillations': 0.2,   # Mild cellular synchronization
                    'pharmacokinetic_rhythms': -0.3 # PK rhythm disruption
                },
                adaptation_phases=[
                    (168.0, 'dopamine_adaptation'),  # 1 week
                    (720.0, 'metabolic_adaptation'), # 1 month
                    (2160.0, 'receptor_adaptation')  # 3 months
                ],
                circadian_disruption=0.15, # Minimal circadian disruption
                ultradian_modulation=0.3   # Moderate ultradian effects
            ),
            
            'citalopram': DrugTemporalProfile(
                drug_name='citalopram',
                optimal_dosing_times=[7.0],  # Early morning
                chronotherapy_benefit=0.6,  # Good chronotherapy benefit
                temporal_scales_affected=['circadian_rhythms', 'ultradian_rhythms', 'cellular_oscillations'],
                synchronization_effects={
                    'circadian_rhythms': 0.5,       # Strong circadian improvement
                    'ultradian_rhythms': 0.3,       # Good ultradian synchronization
                    'cellular_oscillations': 0.1    # Mild cellular effects
                },
                adaptation_phases=[
                    (336.0, 'serotonin_adaptation'), # 2 weeks
                    (1440.0, 'mood_stabilization'),  # 2 months
                    (2880.0, 'neuroplasticity')     # 4 months
                ],
                circadian_disruption=0.1,  # Minimal disruption
                ultradian_modulation=0.5   # Strong ultradian modulation
            ),
            
            'atorvastatin': DrugTemporalProfile(
                drug_name='atorvastatin',
                optimal_dosing_times=[22.0],  # Evening dosing (cholesterol synthesis peaks at night)
                chronotherapy_benefit=0.8,   # High chronotherapy benefit
                temporal_scales_affected=['circadian_rhythms', 'pharmacokinetic_rhythms', 'cellular_oscillations'],
                synchronization_effects={
                    'circadian_rhythms': 0.1,        # Minimal circadian effects
                    'pharmacokinetic_rhythms': 0.6,  # Strong PK synchronization
                    'cellular_oscillations': 0.3     # Good cellular effects
                },
                adaptation_phases=[
                    (168.0, 'metabolic_adaptation'), # 1 week
                    (720.0, 'lipid_homeostasis'),   # 1 month
                    (2160.0, 'vascular_remodeling') # 3 months
                ],
                circadian_disruption=0.05, # Very minimal disruption
                ultradian_modulation=0.2   # Low ultradian effects
            ),
            
            'aspirin': DrugTemporalProfile(
                drug_name='aspirin',
                optimal_dosing_times=[6.0, 18.0],  # Morning and evening
                chronotherapy_benefit=0.4,  # Moderate chronotherapy benefit
                temporal_scales_affected=['pharmacokinetic_rhythms', 'cellular_oscillations'],
                synchronization_effects={
                    'pharmacokinetic_rhythms': 0.4,  # Good PK effects
                    'cellular_oscillations': -0.2,   # Some cellular disruption
                    'circadian_rhythms': 0.0         # No circadian effects
                },
                adaptation_phases=[
                    (24.0, 'antiplatelet_adaptation'), # 1 day
                    (168.0, 'anti_inflammatory'),      # 1 week
                    (720.0, 'cardioprotective')        # 1 month
                ],
                circadian_disruption=0.05, # Minimal disruption
                ultradian_modulation=0.1   # Low ultradian effects
            )
        }
        
        return profiles
    
    def calculate_temporal_synchronization(self, drug_profile: DrugTemporalProfile,
                                         administration_time: float) -> Dict[str, float]:
        """Calculate synchronization effects across temporal scales."""
        
        synchronization = {}
        
        for scale_id, scale in self.temporal_scales.items():
            if scale_id in drug_profile.temporal_scales_affected:
                # Base synchronization from drug profile
                base_sync = drug_profile.synchronization_effects.get(scale_id, 0.0)
                
                # Time-of-day modulation for circadian-coupled scales
                if scale_id == 'circadian_rhythms':
                    # Optimal times for circadian synchronization
                    optimal_times = [6.0, 18.0]  # Dawn and dusk
                    time_factor = min([abs(administration_time - t) for t in optimal_times])
                    time_penalty = time_factor / 12.0  # Normalize to 0-1
                    time_modulation = 1.0 - time_penalty
                else:
                    time_modulation = 1.0
                
                # Drug sensitivity modulation
                sensitivity_factor = scale.drug_sensitivity
                
                # Coupling strength affects synchronization propagation
                coupling_factor = scale.coupling_strength
                
                # Calculate final synchronization
                final_sync = (base_sync * sensitivity_factor * 
                             coupling_factor * time_modulation)
                
                synchronization[scale_id] = final_sync
            else:
                # Indirect effects through coupling
                indirect_sync = 0.0
                for affected_scale_id in drug_profile.temporal_scales_affected:
                    if affected_scale_id in self.temporal_scales:
                        affected_scale = self.temporal_scales[affected_scale_id]
                        coupling = scale.coupling_strength * affected_scale.coupling_strength
                        indirect_effect = drug_profile.synchronization_effects.get(affected_scale_id, 0.0)
                        indirect_sync += indirect_effect * coupling * 0.3  # Damped indirect effect
                
                synchronization[scale_id] = indirect_sync
        
        return synchronization
    
    def calculate_chronotherapy_advantage(self, drug_profile: DrugTemporalProfile,
                                        administration_time: float) -> float:
        """Calculate advantage of timed dosing vs random timing."""
        
        if not drug_profile.optimal_dosing_times:
            return 0.0
        
        # Find closest optimal dosing time
        closest_optimal = min(drug_profile.optimal_dosing_times,
                             key=lambda t: min(abs(administration_time - t),
                                              abs(administration_time - t + 24),
                                              abs(administration_time - t - 24)))
        
        # Calculate time difference
        time_diff = min(abs(administration_time - closest_optimal),
                       abs(administration_time - closest_optimal + 24),
                       abs(administration_time - closest_optimal - 24))
        
        # Convert to advantage score (0-1)
        max_advantage = drug_profile.chronotherapy_benefit
        time_penalty = time_diff / 12.0  # Normalize to 0-1 over 12 hours
        
        advantage = max_advantage * (1.0 - time_penalty)
        
        return max(0.0, advantage)
    
    def calculate_circadian_phase_shift(self, drug_profile: DrugTemporalProfile,
                                      administration_time: float,
                                      duration_days: float = 7.0) -> float:
        """Calculate circadian phase shift induced by drug."""
        
        if 'circadian_rhythms' not in drug_profile.temporal_scales_affected:
            return 0.0
        
        # Base phase shift from drug
        base_shift = drug_profile.circadian_disruption * drug_profile.synchronization_effects.get(
            'circadian_rhythms', 0.0)
        
        # Time-dependent phase shift
        # Light/dark cycle considerations
        if 6.0 <= administration_time <= 18.0:  # Daytime dosing
            light_factor = 1.2  # Enhanced phase shifting during light period
        else:  # Nighttime dosing
            light_factor = 0.8  # Reduced phase shifting during dark period
        
        # Duration-dependent accumulation
        duration_factor = min(1.0, duration_days / 7.0)  # Saturates after 1 week
        
        # Calculate total phase shift in hours
        phase_shift = base_shift * light_factor * duration_factor * 2.0  # Max 2 hour shift
        
        return phase_shift
    
    def calculate_ultradian_changes(self, drug_profile: DrugTemporalProfile) -> Dict[str, float]:
        """Calculate changes in ultradian rhythm patterns."""
        
        ultradian_changes = {}
        
        # Sleep architecture changes
        if drug_profile.drug_name in ['aripiprazole', 'citalopram']:
            ultradian_changes['rem_sleep_cycles'] = drug_profile.ultradian_modulation * 0.3
            ultradian_changes['deep_sleep_cycles'] = drug_profile.ultradian_modulation * 0.2
        
        # Hormone pulse changes
        if 'ultradian_rhythms' in drug_profile.temporal_scales_affected:
            ultradian_changes['growth_hormone_pulses'] = drug_profile.ultradian_modulation * 0.4
            ultradian_changes['cortisol_pulses'] = drug_profile.ultradian_modulation * 0.3
        
        # Metabolic rhythm changes
        if drug_profile.drug_name in ['atorvastatin', 'lithium']:
            ultradian_changes['glucose_oscillations'] = drug_profile.ultradian_modulation * 0.2
            ultradian_changes['insulin_pulses'] = drug_profile.ultradian_modulation * 0.25
        
        # Cardiovascular rhythm changes
        if drug_profile.drug_name in ['aspirin', 'atorvastatin']:
            ultradian_changes['heart_rate_variability'] = drug_profile.ultradian_modulation * 0.3
            ultradian_changes['blood_pressure_cycles'] = drug_profile.ultradian_modulation * 0.2
        
        return ultradian_changes
    
    def generate_adaptation_trajectory(self, drug_profile: DrugTemporalProfile,
                                     simulation_hours: float = 2160.0) -> List[Tuple[float, Dict[str, float]]]:
        """Generate temporal adaptation trajectory over time."""
        
        trajectory = []
        current_state = {}
        
        # Initialize baseline state
        for scale_id in self.temporal_scales.keys():
            current_state[scale_id] = 0.0  # No change initially
        
        # Add initial state
        trajectory.append((0.0, current_state.copy()))
        
        # Process each adaptation phase
        cumulative_time = 0.0
        for phase_duration, phase_name in drug_profile.adaptation_phases:
            if cumulative_time >= simulation_hours:
                break
                
            cumulative_time += phase_duration
            
            # Update state based on phase
            phase_changes = self._calculate_phase_changes(drug_profile, phase_name)
            
            for scale_id, change in phase_changes.items():
                current_state[scale_id] += change
            
            trajectory.append((cumulative_time, current_state.copy()))
        
        # Add final steady state if not reached
        if cumulative_time < simulation_hours:
            trajectory.append((simulation_hours, current_state.copy()))
        
        return trajectory
    
    def _calculate_phase_changes(self, drug_profile: DrugTemporalProfile,
                               phase_name: str) -> Dict[str, float]:
        """Calculate state changes for a specific adaptation phase."""
        
        changes = {}
        
        # Phase-specific changes
        if 'acute' in phase_name:
            # Immediate effects
            for scale_id in drug_profile.temporal_scales_affected:
                sync_effect = drug_profile.synchronization_effects.get(scale_id, 0.0)
                changes[scale_id] = sync_effect * 0.3  # 30% of final effect
        
        elif 'subacute' in phase_name or 'adaptation' in phase_name:
            # Medium-term adaptation
            for scale_id in drug_profile.temporal_scales_affected:
                sync_effect = drug_profile.synchronization_effects.get(scale_id, 0.0)
                changes[scale_id] = sync_effect * 0.5  # Additional 50% of effect
        
        elif 'chronic' in phase_name or 'stabilization' in phase_name:
            # Long-term stabilization
            for scale_id in drug_profile.temporal_scales_affected:
                sync_effect = drug_profile.synchronization_effects.get(scale_id, 0.0)
                changes[scale_id] = sync_effect * 0.2  # Final 20% of effect
        
        return changes
    
    def calculate_multi_scale_coupling(self, synchronization: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate coupling between temporal scales."""
        
        coupling = {}
        
        for scale1_id, scale1 in self.temporal_scales.items():
            coupling[scale1_id] = {}
            
            for scale2_id, scale2 in self.temporal_scales.items():
                if scale1_id == scale2_id:
                    coupling[scale1_id][scale2_id] = 1.0
                    continue
                
                # Base coupling from scale properties
                base_coupling = min(scale1.coupling_strength, scale2.coupling_strength)
                
                # Frequency relationship affects coupling
                freq_ratio = scale1.characteristic_frequency / scale2.characteristic_frequency
                if freq_ratio > 1:
                    freq_ratio = 1 / freq_ratio
                
                freq_factor = freq_ratio ** 0.5  # Square root scaling
                
                # Drug-induced synchronization affects coupling
                sync1 = synchronization.get(scale1_id, 0.0)
                sync2 = synchronization.get(scale2_id, 0.0)
                sync_effect = (abs(sync1) + abs(sync2)) * 0.5
                
                # Calculate final coupling
                final_coupling = base_coupling * freq_factor * (1.0 + sync_effect)
                coupling[scale1_id][scale2_id] = min(1.0, final_coupling)
        
        return coupling
    
    def generate_temporal_biorhythms(self, drug_profile: DrugTemporalProfile,
                                   synchronization: Dict[str, float],
                                   hours: float = 48.0) -> Dict[str, List[Tuple[float, float]]]:
        """Generate biorhythm patterns over time."""
        
        biorhythms = {}
        time_points = np.linspace(0, hours, int(hours * 4))  # 15-minute resolution
        
        for scale_id, scale in self.temporal_scales.items():
            rhythm_data = []
            sync_effect = synchronization.get(scale_id, 0.0)
            
            for t in time_points:
                # Base rhythm
                period_hours = 24.0 if scale_id == 'circadian_rhythms' else \
                              2.0 if scale_id == 'ultradian_rhythms' else \
                              0.5 if scale_id == 'cellular_oscillations' else \
                              3.0 if scale_id == 'pharmacokinetic_rhythms' else \
                              168.0  # Default weekly
                
                base_value = np.sin(2 * np.pi * t / period_hours)
                
                # Add synchronization effects
                sync_modulation = 1.0 + sync_effect * 0.3  # Up to 30% modulation
                
                # Add some noise for realism
                noise = np.random.normal(0, 0.1)
                
                final_value = base_value * sync_modulation + noise
                rhythm_data.append((t, final_value))
            
            biorhythms[scale_id] = rhythm_data
        
        return biorhythms
    
    def determine_optimal_dosing_window(self, drug_profile: DrugTemporalProfile) -> Tuple[float, float]:
        """Determine optimal dosing time window."""
        
        if not drug_profile.optimal_dosing_times:
            return (0.0, 24.0)  # No preference - any time
        
        # Find the best single dosing time
        best_time = drug_profile.optimal_dosing_times[0]
        
        # Calculate window based on chronotherapy benefit
        window_width = 12.0 * (1.0 - drug_profile.chronotherapy_benefit)  # Narrower for higher benefit
        
        start_time = best_time - window_width / 2
        end_time = best_time + window_width / 2
        
        # Handle day boundaries
        if start_time < 0:
            start_time += 24
        if end_time > 24:
            end_time -= 24
        
        return (start_time, end_time)
    
    def simulate_temporal_drug_pattern(self, drug_profile: DrugTemporalProfile,
                                     administration_time: float = 8.0,
                                     simulation_hours: float = 168.0) -> TemporalPattern:
        """Simulate complete temporal drug pattern."""
        
        # Calculate temporal synchronization
        synchronization = self.calculate_temporal_synchronization(drug_profile, administration_time)
        
        # Calculate chronotherapy advantage
        chronotherapy_advantage = self.calculate_chronotherapy_advantage(drug_profile, administration_time)
        
        # Calculate circadian phase shift
        circadian_phase_shift = self.calculate_circadian_phase_shift(
            drug_profile, administration_time, simulation_hours / 24.0
        )
        
        # Calculate ultradian changes
        ultradian_changes = self.calculate_ultradian_changes(drug_profile)
        
        # Generate adaptation trajectory
        adaptation_trajectory = self.generate_adaptation_trajectory(drug_profile, simulation_hours)
        
        # Calculate multi-scale coupling
        multi_scale_coupling = self.calculate_multi_scale_coupling(synchronization)
        
        # Generate biorhythms
        biorhythms = self.generate_temporal_biorhythms(drug_profile, synchronization, simulation_hours)
        
        # Determine optimal dosing window
        optimal_window = self.determine_optimal_dosing_window(drug_profile)
        
        # Calculate confidence
        confidence = self._calculate_temporal_confidence(drug_profile, synchronization)
        
        return TemporalPattern(
            drug_name=drug_profile.drug_name,
            administration_time=administration_time,
            temporal_synchronization=synchronization,
            chronotherapy_advantage=chronotherapy_advantage,
            circadian_phase_shift=circadian_phase_shift,
            ultradian_rhythm_changes=ultradian_changes,
            adaptation_trajectory=adaptation_trajectory,
            multi_scale_coupling=multi_scale_coupling,
            temporal_biorhythms=biorhythms,
            optimal_dosing_window=optimal_window,
            confidence=confidence
        )
    
    def _calculate_temporal_confidence(self, drug_profile: DrugTemporalProfile,
                                     synchronization: Dict[str, float]) -> float:
        """Calculate confidence in temporal pattern prediction."""
        
        base_confidence = 0.7
        
        # Higher chronotherapy benefit increases confidence
        chronotherapy_boost = drug_profile.chronotherapy_benefit * 0.2
        
        # More affected scales reduce confidence (complexity)
        scale_complexity = len(drug_profile.temporal_scales_affected) * 0.05
        
        # Strong synchronization effects increase confidence
        sync_strength = np.mean([abs(s) for s in synchronization.values()])
        sync_boost = sync_strength * 0.1
        
        # Well-studied drugs for chronotherapy
        chronotherapy_drugs = ['atorvastatin', 'lithium', 'citalopram']
        if drug_profile.drug_name in chronotherapy_drugs:
            drug_boost = 0.15
        else:
            drug_boost = 0.0
        
        final_confidence = min(0.95, base_confidence + chronotherapy_boost - 
                              scale_complexity + sync_boost + drug_boost)
        
        return max(0.3, final_confidence)
    
    def simulate_drug_library(self, administration_times: Dict[str, float] = None) -> List[TemporalPattern]:
        """Simulate temporal patterns for all drugs."""
        
        drug_profiles = self.create_drug_temporal_profiles()
        all_patterns = []
        
        # Default administration times
        if administration_times is None:
            administration_times = {
                'lithium': 20.0,      # Evening
                'aripiprazole': 9.0,  # Morning
                'citalopram': 7.0,    # Early morning
                'atorvastatin': 22.0, # Late evening
                'aspirin': 6.0        # Early morning
            }
        
        for drug_name, drug_profile in drug_profiles.items():
            logger.info(f"Simulating temporal patterns for {drug_name}")
            
            admin_time = administration_times.get(drug_name, 8.0)  # Default morning
            pattern = self.simulate_temporal_drug_pattern(drug_profile, admin_time)
            all_patterns.append(pattern)
        
        logger.info(f"Completed temporal pattern simulation for {len(all_patterns)} drugs")
        return all_patterns
    
    def analyze_temporal_patterns(self, patterns: List[TemporalPattern]) -> Dict[str, Any]:
        """Analyze patterns in temporal drug responses."""
        
        analysis = {}
        
        # Chronotherapy benefit analysis
        chronotherapy_advantages = [p.chronotherapy_advantage for p in patterns]
        analysis['chronotherapy'] = {
            'mean_advantage': np.mean(chronotherapy_advantages),
            'max_advantage': np.max(chronotherapy_advantages),
            'drugs_with_benefit': len([a for a in chronotherapy_advantages if a > 0.3])
        }
        
        # Circadian effects analysis
        phase_shifts = [abs(p.circadian_phase_shift) for p in patterns]
        analysis['circadian_effects'] = {
            'mean_phase_shift': np.mean(phase_shifts),
            'max_phase_shift': np.max(phase_shifts),
            'circadian_disruptors': len([s for s in phase_shifts if s > 1.0])
        }
        
        # Synchronization analysis
        all_sync_effects = []
        for pattern in patterns:
            all_sync_effects.extend(pattern.temporal_synchronization.values())
        
        analysis['synchronization'] = {
            'mean_effect': np.mean([abs(e) for e in all_sync_effects]),
            'synchronizing_effects': len([e for e in all_sync_effects if e > 0.2]),
            'desynchronizing_effects': len([e for e in all_sync_effects if e < -0.2])
        }
        
        # Temporal scale involvement
        scale_involvement = {}
        for pattern in patterns:
            for scale_id, effect in pattern.temporal_synchronization.items():
                if abs(effect) > 0.1:  # Significant effect
                    scale_involvement[scale_id] = scale_involvement.get(scale_id, 0) + 1
        
        analysis['scale_involvement'] = scale_involvement
        
        # Drug-specific analysis
        drug_analysis = {}
        for pattern in patterns:
            drug_analysis[pattern.drug_name] = {
                'chronotherapy_advantage': pattern.chronotherapy_advantage,
                'circadian_phase_shift': pattern.circadian_phase_shift,
                'scales_affected': len([e for e in pattern.temporal_synchronization.values() if abs(e) > 0.1]),
                'optimal_window': pattern.optimal_dosing_window,
                'confidence': pattern.confidence
            }
        
        analysis['drug_specific'] = drug_analysis
        
        return analysis
    
    def visualize_temporal_patterns(self, patterns: List[TemporalPattern],
                                  output_dir: str = "babylon_results") -> None:
        """Create visualizations of temporal drug patterns."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive temporal patterns visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Chronotherapy advantages
        drug_names = [p.drug_name for p in patterns]
        chronotherapy_advantages = [p.chronotherapy_advantage for p in patterns]
        
        bars = axes[0, 0].bar(drug_names, chronotherapy_advantages, color='lightblue', alpha=0.7)
        axes[0, 0].set_ylabel('Chronotherapy Advantage')
        axes[0, 0].set_title('Chronotherapy Benefits by Drug')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Significant Benefit')
        axes[0, 0].legend()
        
        # Color bars by advantage level
        for bar, advantage in zip(bars, chronotherapy_advantages):
            if advantage > 0.6:
                bar.set_color('darkgreen')
            elif advantage > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        # 2. Circadian phase shifts
        phase_shifts = [p.circadian_phase_shift for p in patterns]
        
        axes[0, 1].bar(drug_names, phase_shifts, color='lightcoral', alpha=0.7)
        axes[0, 1].set_ylabel('Phase Shift (hours)')
        axes[0, 1].set_title('Circadian Phase Shifts')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Significant Shift')
        axes[0, 1].axhline(y=-1.0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # 3. Temporal synchronization heatmap
        sync_matrix = []
        scale_names = list(self.temporal_scales.keys())
        
        for pattern in patterns:
            row = []
            for scale in scale_names:
                row.append(pattern.temporal_synchronization.get(scale, 0.0))
            sync_matrix.append(row)
        
        sync_matrix = np.array(sync_matrix)
        
        im = axes[0, 2].imshow(sync_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[0, 2].set_title('Temporal Synchronization Effects')
        axes[0, 2].set_xlabel('Temporal Scales')
        axes[0, 2].set_ylabel('Drugs')
        axes[0, 2].set_xticks(range(len(scale_names)))
        axes[0, 2].set_xticklabels([s.replace('_', '\n') for s in scale_names], fontsize=8, rotation=45)
        axes[0, 2].set_yticks(range(len(drug_names)))
        axes[0, 2].set_yticklabels(drug_names)
        plt.colorbar(im, ax=axes[0, 2], label='Synchronization Effect')
        
        # 4. Optimal dosing windows
        windows_start = [p.optimal_dosing_window[0] for p in patterns]
        windows_end = [p.optimal_dosing_window[1] for p in patterns]
        window_widths = []
        
        for start, end in zip(windows_start, windows_end):
            if end > start:
                width = end - start
            else:  # Crosses midnight
                width = (24 - start) + end
            window_widths.append(width)
        
        axes[1, 0].barh(drug_names, window_widths, left=windows_start, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('Time of Day (hours)')
        axes[1, 0].set_title('Optimal Dosing Windows')
        axes[1, 0].set_xlim(0, 24)
        axes[1, 0].set_xticks(range(0, 25, 4))
        
        # 5. Example biorhythm (circadian for one drug)
        example_pattern = patterns[0]  # Use first drug as example
        if 'circadian_rhythms' in example_pattern.temporal_biorhythms:
            times, values = zip(*example_pattern.temporal_biorhythms['circadian_rhythms'])
            axes[1, 1].plot(times, values, 'b-', linewidth=2, label=f'{example_pattern.drug_name} Circadian')
            axes[1, 1].set_xlabel('Time (hours)')
            axes[1, 1].set_ylabel('Rhythm Amplitude')
            axes[1, 1].set_title('Example Circadian Biorhythm')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Confidence scores
        confidences = [p.confidence for p in patterns]
        
        axes[1, 2].bar(drug_names, confidences, color='gold', alpha=0.7)
        axes[1, 2].set_ylabel('Prediction Confidence')
        axes[1, 2].set_title('Temporal Prediction Confidence')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'temporal_drug_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Temporal pattern visualizations saved to {output_path}")
    
    def save_results(self, patterns: List[TemporalPattern], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save temporal drug pattern simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed patterns as JSON
        patterns_data = []
        for pattern in patterns:
            patterns_data.append({
                'drug_name': pattern.drug_name,
                'administration_time': pattern.administration_time,
                'temporal_synchronization': pattern.temporal_synchronization,
                'chronotherapy_advantage': pattern.chronotherapy_advantage,
                'circadian_phase_shift': pattern.circadian_phase_shift,
                'ultradian_rhythm_changes': pattern.ultradian_rhythm_changes,
                'adaptation_trajectory': pattern.adaptation_trajectory,
                'multi_scale_coupling': pattern.multi_scale_coupling,
                'temporal_biorhythms': {k: v[:100] for k, v in pattern.temporal_biorhythms.items()},  # Truncate for size
                'optimal_dosing_window': pattern.optimal_dosing_window,
                'confidence': pattern.confidence
            })
        
        with open(output_path / 'temporal_drug_patterns.json', 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_data = []
        for pattern in patterns:
            csv_data.append({
                'drug_name': pattern.drug_name,
                'administration_time': pattern.administration_time,
                'chronotherapy_advantage': pattern.chronotherapy_advantage,
                'circadian_phase_shift': pattern.circadian_phase_shift,
                'optimal_window_start': pattern.optimal_dosing_window[0],
                'optimal_window_end': pattern.optimal_dosing_window[1],
                'confidence': pattern.confidence
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / 'temporal_drug_patterns.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'temporal_patterns_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Temporal drug pattern results saved to {output_path}")

def main():
    """
    Test temporal drug patterns simulation.
    
    This demonstrates how drugs affect temporal coordination across
    multiple biological time scales and chronotherapy optimization.
    """
    
    print("⏰ Testing Temporal Drug Patterns")
    print("=" * 50)
    
    # Initialize temporal patterns analyzer
    analyzer = TemporalDrugPatterns()
    
    # Display loaded temporal scales
    print(f"\n🕐 Loaded Temporal Scales:")
    for scale_id, scale in analyzer.temporal_scales.items():
        print(f"  • {scale_id}: {scale.scale_name}")
        print(f"    Frequency: {scale.characteristic_frequency:.2e} Hz")
        print(f"    Period range: {scale.period_range[0]:.1f}-{scale.period_range[1]:.1f} hours")
        print(f"    Coupling strength: {scale.coupling_strength:.1f}")
        print(f"    Drug sensitivity: {scale.drug_sensitivity:.1f}")
    
    # Create and display drug profiles
    drug_profiles = analyzer.create_drug_temporal_profiles()
    print(f"\n💊 Drug Temporal Profiles:")
    for drug_name, profile in drug_profiles.items():
        print(f"  • {drug_name}:")
        print(f"    Optimal dosing times: {profile.optimal_dosing_times}")
        print(f"    Chronotherapy benefit: {profile.chronotherapy_benefit:.2f}")
        print(f"    Circadian disruption: {profile.circadian_disruption:.2f}")
        print(f"    Ultradian modulation: {profile.ultradian_modulation:.2f}")
        print(f"    Affected scales: {profile.temporal_scales_affected}")
    
    # Simulate temporal patterns for all drugs
    print(f"\n🔬 Simulating temporal drug patterns...")
    patterns = analyzer.simulate_drug_library()
    
    # Display results
    print(f"\n📊 TEMPORAL DRUG PATTERN RESULTS:")
    print("-" * 80)
    
    for pattern in patterns:
        print(f"\n{pattern.drug_name.upper()}:")
        print(f"  Administration time: {pattern.administration_time:.1f}h")
        print(f"  Chronotherapy advantage: {pattern.chronotherapy_advantage:.3f}")
        print(f"  Circadian phase shift: {pattern.circadian_phase_shift:+.2f}h")
        print(f"  Optimal dosing window: {pattern.optimal_dosing_window[0]:.1f}-{pattern.optimal_dosing_window[1]:.1f}h")
        print(f"  Confidence: {pattern.confidence:.3f}")
        
        # Show significant synchronization effects
        print(f"  Temporal Synchronization:")
        for scale, effect in pattern.temporal_synchronization.items():
            if abs(effect) > 0.1:  # Only show significant effects
                print(f"    {scale}: {effect:+.3f}")
        
        # Show ultradian changes
        if pattern.ultradian_rhythm_changes:
            print(f"  Ultradian Changes:")
            for rhythm, change in pattern.ultradian_rhythm_changes.items():
                if abs(change) > 0.1:
                    print(f"    {rhythm}: {change:+.3f}")
    
    # Analyze patterns
    print("\n🔍 Analyzing temporal pattern trends...")
    analysis = analyzer.analyze_temporal_patterns(patterns)
    
    print(f"\n📈 TEMPORAL PATTERN ANALYSIS:")
    print("-" * 50)
    
    chronotherapy_analysis = analysis['chronotherapy']
    print("Chronotherapy Benefits:")
    print(f"  • Mean advantage: {chronotherapy_analysis['mean_advantage']:.3f}")
    print(f"  • Maximum advantage: {chronotherapy_analysis['max_advantage']:.3f}")
    print(f"  • Drugs with significant benefit: {chronotherapy_analysis['drugs_with_benefit']}")
    
    circadian_analysis = analysis['circadian_effects']
    print(f"\nCircadian Effects:")
    print(f"  • Mean phase shift: {circadian_analysis['mean_phase_shift']:.2f}h")
    print(f"  • Maximum phase shift: {circadian_analysis['max_phase_shift']:.2f}h")
    print(f"  • Circadian disruptors: {circadian_analysis['circadian_disruptors']}")
    
    sync_analysis = analysis['synchronization']
    print(f"\nSynchronization Effects:")
    print(f"  • Mean effect magnitude: {sync_analysis['mean_effect']:.3f}")
    print(f"  • Synchronizing effects: {sync_analysis['synchronizing_effects']}")
    print(f"  • Desynchronizing effects: {sync_analysis['desynchronizing_effects']}")
    
    print(f"\nTemporal Scale Involvement:")
    for scale, count in analysis['scale_involvement'].items():
        print(f"  • {scale}: {count} drugs significantly affected")
    
    print(f"\nDrug-Specific Optimal Timing:")
    for drug, metrics in analysis['drug_specific'].items():
        window_start, window_end = metrics['optimal_window']
        print(f"  • {drug}: {window_start:.1f}-{window_end:.1f}h "
              f"(advantage={metrics['chronotherapy_advantage']:.3f})")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    analyzer.save_results(patterns, analysis)
    analyzer.visualize_temporal_patterns(patterns)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find best chronotherapy candidate
    best_chronotherapy = max(patterns, key=lambda p: p.chronotherapy_advantage)
    print(f"• Best chronotherapy candidate: {best_chronotherapy.drug_name} "
          f"(advantage={best_chronotherapy.chronotherapy_advantage:.3f})")
    
    # Find strongest circadian effects
    strongest_circadian = max(patterns, key=lambda p: abs(p.circadian_phase_shift))
    print(f"• Strongest circadian effect: {strongest_circadian.drug_name} "
          f"({strongest_circadian.circadian_phase_shift:+.2f}h shift)")
    
    # Find most temporally complex drug
    most_complex = max(patterns, key=lambda p: len([e for e in p.temporal_synchronization.values() if abs(e) > 0.1]))
    complex_scales = len([e for e in most_complex.temporal_synchronization.values() if abs(e) > 0.1])
    print(f"• Most temporally complex: {most_complex.drug_name} ({complex_scales} scales affected)")
    
    # Count drugs with narrow dosing windows
    narrow_windows = len([p for p in patterns if (p.optimal_dosing_window[1] - p.optimal_dosing_window[0]) < 8.0])
    print(f"• Drugs with narrow dosing windows (<8h): {narrow_windows}/{len(patterns)}")
    
    # High confidence predictions
    high_conf = len([p for p in patterns if p.confidence > 0.8])
    print(f"• High-confidence predictions: {high_conf}/{len(patterns)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Temporal drug patterns simulation complete!")
    
    return patterns

if __name__ == "__main__":
    patterns = main()

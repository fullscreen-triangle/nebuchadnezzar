"""
Cellular Drug Response Module
============================

Models how drugs affect intracellular oscillatory dynamics after crossing
the membrane. Simulates the interaction between pharmaceutical molecules and
cellular oscillatory networks using the St. Stellas framework principles.

Based on ATP-constrained dynamics and intracellular oscillatory coupling.
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
class CellularTarget:
    """Represents a cellular target for drug action."""
    target_id: str
    target_type: str  # 'enzyme', 'receptor', 'transporter', 'ion_channel'
    baseline_activity: float  # Normal activity level (0-1)
    oscillatory_frequency: float  # Hz - target's oscillatory signature
    atp_dependency: float  # ATP consumption per activity unit
    cellular_location: str  # 'cytoplasm', 'nucleus', 'membrane', 'mitochondria'
    pathway_coupling: Dict[str, float] = field(default_factory=dict)  # Coupling to other pathways

@dataclass
class DrugCellularProfile:
    """Drug's cellular interaction profile."""
    drug_name: str
    intracellular_concentration: float  # μM - after transport
    target_affinities: Dict[str, float]  # target_id -> binding affinity (nM)
    mechanism_of_action: str  # 'inhibition', 'activation', 'modulation'
    onset_time: float  # seconds to reach equilibrium
    duration_of_action: float  # seconds of sustained effect

@dataclass
class CellularResponse:
    """Result of cellular drug response simulation."""
    drug_name: str
    target_id: str
    baseline_oscillation: float  # Hz - normal frequency
    drug_modified_oscillation: float  # Hz - frequency after drug
    amplitude_change: float  # fold change in oscillation amplitude
    atp_consumption_change: float  # fold change in ATP usage
    pathway_coupling_effects: Dict[str, float]  # effects on coupled pathways
    response_time_course: List[Tuple[float, float]]  # (time, response) pairs
    oscillatory_synchronization: float  # 0-1 sync with other cellular oscillators
    confidence: float = 0.0

class CellularDrugResponse:
    """
    Simulates drug effects on intracellular oscillatory dynamics.
    
    Models how drugs interact with cellular targets and affect the
    oscillatory networks that drive cellular function.
    """
    
    def __init__(self):
        """Initialize cellular drug response simulator."""
        self.cellular_targets = self._load_cellular_targets()
        self.atp_pool_size = 5.0  # mM - typical cellular ATP concentration
        self.temperature = 310.0  # K - physiological temperature
        self.k_B = 1.38064852e-23  # Boltzmann constant
        
    def _load_cellular_targets(self) -> Dict[str, CellularTarget]:
        """Load cellular targets for drug action."""
        
        targets = {
            'inpp1': CellularTarget(
                target_id='inpp1',
                target_type='enzyme',
                baseline_activity=0.8,
                oscillatory_frequency=7.23e13,
                atp_dependency=0.3,  # Moderate ATP dependence
                cellular_location='cytoplasm',
                pathway_coupling={
                    'inositol_metabolism': 0.9,
                    'phosphate_signaling': 0.7,
                    'calcium_signaling': 0.6
                }
            ),
            
            'gsk3b': CellularTarget(
                target_id='gsk3b',
                target_type='enzyme',
                baseline_activity=0.7,
                oscillatory_frequency=2.15e13,
                atp_dependency=0.8,  # High ATP dependence (kinase)
                cellular_location='cytoplasm',
                pathway_coupling={
                    'glycogen_synthesis': 0.8,
                    'protein_synthesis': 0.7,
                    'cell_cycle': 0.6,
                    'neuronal_plasticity': 0.9
                }
            ),
            
            'drd2_receptor': CellularTarget(
                target_id='drd2_receptor',
                target_type='receptor',
                baseline_activity=0.6,
                oscillatory_frequency=1.45e13,
                atp_dependency=0.4,  # Moderate ATP (G-protein coupling)
                cellular_location='membrane',
                pathway_coupling={
                    'camp_signaling': 0.9,
                    'dopamine_signaling': 1.0,
                    'motor_control': 0.8
                }
            ),
            
            'htr2a_receptor': CellularTarget(
                target_id='htr2a_receptor',
                target_type='receptor',
                baseline_activity=0.5,
                oscillatory_frequency=1.23e13,
                atp_dependency=0.5,
                cellular_location='membrane',
                pathway_coupling={
                    'serotonin_signaling': 1.0,
                    'mood_regulation': 0.9,
                    'sleep_wake_cycle': 0.7
                }
            ),
            
            'cyp2d6_enzyme': CellularTarget(
                target_id='cyp2d6_enzyme',
                target_type='enzyme',
                baseline_activity=0.9,
                oscillatory_frequency=3.8e13,
                atp_dependency=0.6,  # NADPH dependent
                cellular_location='membrane',  # ER membrane
                pathway_coupling={
                    'drug_metabolism': 1.0,
                    'xenobiotic_response': 0.8,
                    'oxidative_stress': 0.6
                }
            )
        }
        
        logger.info(f"Loaded {len(targets)} cellular targets")
        return targets
    
    def create_drug_cellular_profiles(self) -> Dict[str, DrugCellularProfile]:
        """Create cellular profiles for pharmaceutical molecules."""
        
        profiles = {
            'lithium': DrugCellularProfile(
                drug_name='lithium',
                intracellular_concentration=0.8,  # μM - therapeutic range
                target_affinities={
                    'inpp1': 50.0,    # nM - high affinity inhibitor
                    'gsk3b': 30.0,    # nM - high affinity inhibitor
                },
                mechanism_of_action='inhibition',
                onset_time=300.0,   # 5 minutes
                duration_of_action=14400.0  # 4 hours
            ),
            
            'aripiprazole': DrugCellularProfile(
                drug_name='aripiprazole',
                intracellular_concentration=0.1,  # μM
                target_affinities={
                    'drd2_receptor': 2.3,     # nM - partial agonist
                    'htr2a_receptor': 8.7,    # nM - antagonist
                },
                mechanism_of_action='modulation',  # Partial agonist
                onset_time=1800.0,  # 30 minutes
                duration_of_action=86400.0  # 24 hours
            ),
            
            'citalopram': DrugCellularProfile(
                drug_name='citalopram',
                intracellular_concentration=0.05,  # μM
                target_affinities={
                    'htr2a_receptor': 15.0,   # nM - weak antagonist
                    # Primary target is serotonin transporter (not modeled here)
                },
                mechanism_of_action='inhibition',
                onset_time=3600.0,  # 1 hour
                duration_of_action=43200.0  # 12 hours
            ),
            
            'atorvastatin': DrugCellularProfile(
                drug_name='atorvastatin',
                intracellular_concentration=0.02,  # μM
                target_affinities={
                    # Primary target is HMG-CoA reductase (not modeled here)
                    'cyp2d6_enzyme': 1000.0,  # nM - weak interaction
                },
                mechanism_of_action='inhibition',
                onset_time=7200.0,  # 2 hours
                duration_of_action=172800.0  # 48 hours
            )
        }
        
        return profiles
    
    def calculate_drug_target_binding(self, drug_profile: DrugCellularProfile, 
                                    target: CellularTarget) -> float:
        """Calculate drug-target binding occupancy."""
        
        if target.target_id not in drug_profile.target_affinities:
            return 0.0  # No binding
        
        # Get binding parameters
        kd = drug_profile.target_affinities[target.target_id]  # nM
        concentration = drug_profile.intracellular_concentration * 1000  # Convert μM to nM
        
        # Calculate binding occupancy using Hill equation (n=1)
        occupancy = concentration / (kd + concentration)
        
        return occupancy
    
    def calculate_oscillatory_coupling_effect(self, target: CellularTarget,
                                            drug_occupancy: float,
                                            mechanism: str) -> float:
        """Calculate effect on target's oscillatory frequency."""
        
        # Base frequency modulation based on drug occupancy and mechanism
        if mechanism == 'inhibition':
            # Inhibition typically decreases oscillatory activity
            frequency_change = -0.3 * drug_occupancy  # Up to 30% decrease
        elif mechanism == 'activation':
            # Activation increases oscillatory activity  
            frequency_change = 0.4 * drug_occupancy   # Up to 40% increase
        elif mechanism == 'modulation':
            # Partial agonists have complex effects
            frequency_change = 0.2 * drug_occupancy * (0.5 - drug_occupancy)
        else:
            frequency_change = 0.0
        
        # Apply target-specific modulation
        target_sensitivity = {
            'enzyme': 1.2,      # Enzymes are sensitive to modulation
            'receptor': 0.8,    # Receptors have built-in buffering
            'transporter': 1.0, # Moderate sensitivity
            'ion_channel': 1.5  # High sensitivity
        }
        
        sensitivity = target_sensitivity.get(target.target_type, 1.0)
        
        return frequency_change * sensitivity
    
    def calculate_atp_consumption_change(self, target: CellularTarget,
                                       frequency_change: float) -> float:
        """Calculate change in ATP consumption due to altered activity."""
        
        # ATP consumption scales with oscillatory activity
        # More activity = more ATP consumption
        base_atp_rate = target.atp_dependency * target.baseline_activity
        
        # New activity level
        new_activity = target.baseline_activity * (1.0 + frequency_change)
        new_activity = max(0.0, min(1.0, new_activity))  # Clamp to [0,1]
        
        # New ATP consumption
        new_atp_rate = target.atp_dependency * new_activity
        
        # Fold change in ATP consumption
        if base_atp_rate > 0:
            atp_change = new_atp_rate / base_atp_rate
        else:
            atp_change = 1.0
        
        return atp_change
    
    def simulate_pathway_coupling_effects(self, target: CellularTarget,
                                        frequency_change: float) -> Dict[str, float]:
        """Simulate effects on coupled pathways."""
        
        coupling_effects = {}
        
        for pathway, coupling_strength in target.pathway_coupling.items():
            # Coupled pathways are affected proportionally to coupling strength
            pathway_effect = frequency_change * coupling_strength
            
            # Add some pathway-specific modulation
            pathway_modulation = {
                'inositol_metabolism': 1.2,    # High sensitivity
                'dopamine_signaling': 1.0,     # Direct coupling
                'serotonin_signaling': 0.9,    # Moderate coupling
                'camp_signaling': 1.1,         # Amplified coupling
                'drug_metabolism': 0.8         # Buffered coupling
            }
            
            modulation = pathway_modulation.get(pathway, 1.0)
            coupling_effects[pathway] = pathway_effect * modulation
        
        return coupling_effects
    
    def calculate_response_time_course(self, drug_profile: DrugCellularProfile,
                                     max_response: float) -> List[Tuple[float, float]]:
        """Calculate time course of cellular response."""
        
        # Time points (seconds)
        time_points = np.logspace(1, 5, 50)  # 10 seconds to 100,000 seconds
        
        # Pharmacokinetic parameters
        onset_time = drug_profile.onset_time
        duration = drug_profile.duration_of_action
        
        time_course = []
        
        for t in time_points:
            if t < onset_time:
                # Rising phase - exponential approach to maximum
                response = max_response * (1 - np.exp(-t / (onset_time / 3)))
            else:
                # Decay phase - exponential decay
                decay_time = duration / 3
                response = max_response * np.exp(-(t - onset_time) / decay_time)
            
            time_course.append((t, response))
        
        return time_course
    
    def calculate_oscillatory_synchronization(self, target: CellularTarget,
                                            modified_frequency: float) -> float:
        """Calculate synchronization with other cellular oscillators."""
        
        # Cellular oscillators synchronize best when frequencies are close
        cellular_frequencies = [
            2.1e13,  # Metabolic oscillations
            1.8e13,  # Calcium oscillations
            2.5e13,  # Circadian rhythms
            1.4e13,  # Membrane potential oscillations
            3.2e13   # Protein synthesis oscillations
        ]
        
        synchronization_scores = []
        
        for cell_freq in cellular_frequencies:
            # Calculate frequency difference
            freq_diff = abs(modified_frequency - cell_freq) / cell_freq
            
            # Synchronization decreases with frequency difference
            sync_score = 1.0 / (1.0 + freq_diff)
            synchronization_scores.append(sync_score)
        
        # Overall synchronization is average of all scores
        overall_sync = np.mean(synchronization_scores)
        
        # Weight by target's baseline coupling ability
        coupling_factor = target.baseline_activity * 0.8 + 0.2
        
        return overall_sync * coupling_factor
    
    def simulate_cellular_drug_response(self, drug_profile: DrugCellularProfile,
                                      target_id: str) -> CellularResponse:
        """Simulate complete cellular response to drug."""
        
        if target_id not in self.cellular_targets:
            raise ValueError(f"Unknown target: {target_id}")
        
        target = self.cellular_targets[target_id]
        
        # Calculate drug-target binding
        binding_occupancy = self.calculate_drug_target_binding(drug_profile, target)
        
        # Calculate oscillatory effects
        frequency_change = self.calculate_oscillatory_coupling_effect(
            target, binding_occupancy, drug_profile.mechanism_of_action
        )
        
        # New oscillatory frequency
        baseline_freq = target.oscillatory_frequency
        modified_freq = baseline_freq * (1.0 + frequency_change)
        
        # Calculate amplitude change (related to activity change)
        amplitude_change = 1.0 + frequency_change * 0.8  # Amplitude follows frequency
        
        # Calculate ATP consumption change
        atp_change = self.calculate_atp_consumption_change(target, frequency_change)
        
        # Calculate pathway coupling effects
        coupling_effects = self.simulate_pathway_coupling_effects(target, frequency_change)
        
        # Calculate time course
        time_course = self.calculate_response_time_course(drug_profile, frequency_change)
        
        # Calculate synchronization
        synchronization = self.calculate_oscillatory_synchronization(target, modified_freq)
        
        # Calculate confidence
        confidence = self._calculate_response_confidence(
            drug_profile, target, binding_occupancy
        )
        
        return CellularResponse(
            drug_name=drug_profile.drug_name,
            target_id=target_id,
            baseline_oscillation=baseline_freq,
            drug_modified_oscillation=modified_freq,
            amplitude_change=amplitude_change,
            atp_consumption_change=atp_change,
            pathway_coupling_effects=coupling_effects,
            response_time_course=time_course,
            oscillatory_synchronization=synchronization,
            confidence=confidence
        )
    
    def _calculate_response_confidence(self, drug_profile: DrugCellularProfile,
                                     target: CellularTarget, 
                                     binding_occupancy: float) -> float:
        """Calculate confidence in cellular response simulation."""
        
        base_confidence = 0.7
        
        # Higher binding occupancy increases confidence
        occupancy_boost = binding_occupancy * 0.2
        
        # Well-studied targets increase confidence
        well_studied_targets = ['inpp1', 'gsk3b', 'drd2_receptor', 'htr2a_receptor']
        if target.target_id in well_studied_targets:
            target_boost = 0.15
        else:
            target_boost = 0.0
        
        # Mechanism-specific confidence
        mechanism_confidence = {
            'inhibition': 0.9,    # Well understood
            'activation': 0.8,    # Well understood
            'modulation': 0.6     # More complex
        }
        
        mechanism_factor = mechanism_confidence.get(
            drug_profile.mechanism_of_action, 0.5
        )
        
        final_confidence = min(0.95, 
                             (base_confidence + occupancy_boost + target_boost) * mechanism_factor)
        
        return final_confidence
    
    def simulate_drug_library(self) -> List[CellularResponse]:
        """Simulate cellular responses for all drug-target combinations."""
        
        drug_profiles = self.create_drug_cellular_profiles()
        responses = []
        
        for drug_name, drug_profile in drug_profiles.items():
            logger.info(f"Simulating cellular effects for {drug_name}")
            
            # Test each target that the drug can bind to
            for target_id in drug_profile.target_affinities.keys():
                if target_id in self.cellular_targets:
                    response = self.simulate_cellular_drug_response(drug_profile, target_id)
                    responses.append(response)
        
        logger.info(f"Completed cellular simulation for {len(responses)} drug-target pairs")
        return responses
    
    def analyze_cellular_patterns(self, responses: List[CellularResponse]) -> Dict[str, Any]:
        """Analyze patterns in cellular drug responses."""
        
        analysis = {}
        
        # Frequency change statistics
        freq_changes = []
        for response in responses:
            change = (response.drug_modified_oscillation - response.baseline_oscillation) / response.baseline_oscillation
            freq_changes.append(change)
        
        analysis['frequency_changes'] = {
            'mean': np.mean(freq_changes),
            'std': np.std(freq_changes),
            'range': (np.min(freq_changes), np.max(freq_changes))
        }
        
        # ATP consumption changes
        atp_changes = [r.atp_consumption_change for r in responses]
        analysis['atp_changes'] = {
            'mean': np.mean(atp_changes),
            'energy_efficient_drugs': len([c for c in atp_changes if c < 1.0]),
            'energy_costly_drugs': len([c for c in atp_changes if c > 1.2])
        }
        
        # Synchronization analysis
        sync_scores = [r.oscillatory_synchronization for r in responses]
        analysis['synchronization'] = {
            'mean': np.mean(sync_scores),
            'highly_synchronized': len([s for s in sync_scores if s > 0.7])
        }
        
        # Drug-specific analysis
        drug_effects = {}
        for response in responses:
            if response.drug_name not in drug_effects:
                drug_effects[response.drug_name] = []
            drug_effects[response.drug_name].append(response)
        
        analysis['drug_specific'] = {}
        for drug, drug_responses in drug_effects.items():
            avg_freq_change = np.mean([
                (r.drug_modified_oscillation - r.baseline_oscillation) / r.baseline_oscillation
                for r in drug_responses
            ])
            analysis['drug_specific'][drug] = {
                'targets_affected': len(drug_responses),
                'average_frequency_change': avg_freq_change,
                'average_confidence': np.mean([r.confidence for r in drug_responses])
            }
        
        return analysis
    
    def visualize_cellular_responses(self, responses: List[CellularResponse],
                                   output_dir: str = "babylon_results") -> None:
        """Create visualizations of cellular drug responses."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive cellular response visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Frequency changes by drug
        drugs = list(set(r.drug_name for r in responses))
        drug_freq_changes = {}
        
        for drug in drugs:
            drug_responses = [r for r in responses if r.drug_name == drug]
            freq_changes = [
                (r.drug_modified_oscillation - r.baseline_oscillation) / r.baseline_oscillation * 100
                for r in drug_responses
            ]
            drug_freq_changes[drug] = freq_changes
        
        # Box plot of frequency changes
        axes[0, 0].boxplot([drug_freq_changes[drug] for drug in drugs], labels=drugs)
        axes[0, 0].set_title('Oscillatory Frequency Changes by Drug')
        axes[0, 0].set_ylabel('Frequency Change (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. ATP consumption changes
        atp_changes = [r.atp_consumption_change for r in responses]
        drug_names = [r.drug_name for r in responses]
        target_names = [r.target_id for r in responses]
        
        scatter = axes[0, 1].scatter(range(len(atp_changes)), atp_changes, 
                                   c=[drugs.index(d) for d in drug_names], 
                                   cmap='viridis', alpha=0.7, s=100)
        axes[0, 1].set_title('ATP Consumption Changes')
        axes[0, 1].set_ylabel('ATP Change (fold)')
        axes[0, 1].set_xlabel('Response Index')
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Change')
        axes[0, 1].legend()
        
        # 3. Oscillatory synchronization scores
        sync_scores = [r.oscillatory_synchronization for r in responses]
        
        axes[0, 2].hist(sync_scores, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 2].set_title('Oscillatory Synchronization Distribution')
        axes[0, 2].set_xlabel('Synchronization Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Moderate Sync')
        axes[0, 2].legend()
        
        # 4. Response confidence by target type
        target_types = {}
        for response in responses:
            target = self.cellular_targets[response.target_id]
            if target.target_type not in target_types:
                target_types[target.target_type] = []
            target_types[target.target_type].append(response.confidence)
        
        type_names = list(target_types.keys())
        type_confidences = [np.mean(target_types[t]) for t in type_names]
        
        bars = axes[1, 0].bar(type_names, type_confidences, color='gold', alpha=0.7)
        axes[1, 0].set_title('Response Confidence by Target Type')
        axes[1, 0].set_ylabel('Average Confidence')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Color bars by confidence level
        for bar, conf in zip(bars, type_confidences):
            if conf > 0.8:
                bar.set_color('green')
            elif conf > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 5. Time course example (lithium on INPP1)
        lithium_inpp1 = next((r for r in responses 
                            if r.drug_name == 'lithium' and r.target_id == 'inpp1'), None)
        
        if lithium_inpp1:
            times, responses_time = zip(*lithium_inpp1.response_time_course)
            axes[1, 1].semilogx(times, responses_time, 'b-', linewidth=2, label='Lithium→INPP1')
            axes[1, 1].set_title('Example Time Course: Lithium→INPP1')
            axes[1, 1].set_xlabel('Time (seconds)')
            axes[1, 1].set_ylabel('Response (fractional change)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No lithium-INPP1\ninteraction found', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Time Course (Not Available)')
        
        # 6. Pathway coupling network effects
        all_pathways = set()
        for response in responses:
            all_pathways.update(response.pathway_coupling_effects.keys())
        
        pathway_effects = {pathway: [] for pathway in all_pathways}
        for response in responses:
            for pathway in all_pathways:
                effect = response.pathway_coupling_effects.get(pathway, 0.0)
                pathway_effects[pathway].append(effect)
        
        pathway_names = list(pathway_effects.keys())
        pathway_means = [np.mean(pathway_effects[p]) for p in pathway_names]
        
        axes[1, 2].barh(pathway_names, pathway_means, color='lightcoral', alpha=0.7)
        axes[1, 2].set_title('Average Pathway Coupling Effects')
        axes[1, 2].set_xlabel('Average Effect')
        axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cellular_drug_responses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cellular response visualizations saved to {output_path}")
    
    def save_results(self, responses: List[CellularResponse], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save cellular response simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed responses as JSON
        responses_data = []
        for response in responses:
            responses_data.append({
                'drug_name': response.drug_name,
                'target_id': response.target_id,
                'baseline_oscillation': response.baseline_oscillation,
                'drug_modified_oscillation': response.drug_modified_oscillation,
                'amplitude_change': response.amplitude_change,
                'atp_consumption_change': response.atp_consumption_change,
                'pathway_coupling_effects': response.pathway_coupling_effects,
                'response_time_course': response.response_time_course,
                'oscillatory_synchronization': response.oscillatory_synchronization,
                'confidence': response.confidence
            })
        
        with open(output_path / 'cellular_responses.json', 'w') as f:
            json.dump(responses_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_data = []
        for response in responses:
            freq_change = (response.drug_modified_oscillation - response.baseline_oscillation) / response.baseline_oscillation
            csv_data.append({
                'drug_name': response.drug_name,
                'target_id': response.target_id,
                'frequency_change_percent': freq_change * 100,
                'amplitude_change': response.amplitude_change,
                'atp_consumption_change': response.atp_consumption_change,
                'oscillatory_synchronization': response.oscillatory_synchronization,
                'confidence': response.confidence
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / 'cellular_responses.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'cellular_analysis_summary.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Cellular response results saved to {output_path}")

def main():
    """
    Test cellular drug response simulation.
    
    This demonstrates how drugs affect intracellular oscillatory dynamics
    after crossing the membrane and binding to cellular targets.
    """
    
    print("🧬 Testing Cellular Drug Response")
    print("=" * 50)
    
    # Initialize cellular response simulator
    simulator = CellularDrugResponse()
    
    # Display loaded targets
    print(f"\n🎯 Loaded Cellular Targets:")
    for target_id, target in simulator.cellular_targets.items():
        print(f"  • {target_id}: {target.target_type} at {target.cellular_location}")
        print(f"    Frequency: {target.oscillatory_frequency:.2e} Hz")
        print(f"    ATP dependency: {target.atp_dependency:.1f}")
    
    # Simulate responses for all drug-target combinations
    print(f"\n🔬 Simulating cellular drug responses...")
    responses = simulator.simulate_drug_library()
    
    # Display results
    print(f"\n📊 CELLULAR RESPONSE RESULTS:")
    print("-" * 60)
    
    for response in responses:
        freq_change = (response.drug_modified_oscillation - response.baseline_oscillation) / response.baseline_oscillation
        
        print(f"{response.drug_name.upper()} → {response.target_id.upper()}:")
        print(f"  Frequency change: {freq_change*100:+.1f}%")
        print(f"  Amplitude change: {response.amplitude_change:.2f}×")
        print(f"  ATP consumption: {response.atp_consumption_change:.2f}×")
        print(f"  Synchronization: {response.oscillatory_synchronization:.3f}")
        print(f"  Confidence: {response.confidence:.3f}")
        
        # Show top pathway effects
        top_pathways = sorted(response.pathway_coupling_effects.items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:2]
        if top_pathways:
            print(f"  Top pathway effects:")
            for pathway, effect in top_pathways:
                print(f"    {pathway}: {effect:+.3f}")
        print()
    
    # Analyze patterns
    print("🔍 Analyzing cellular response patterns...")
    analysis = simulator.analyze_cellular_patterns(responses)
    
    print(f"\n📈 CELLULAR PATTERN ANALYSIS:")
    print("-" * 40)
    
    freq_analysis = analysis['frequency_changes']
    print(f"Frequency Changes:")
    print(f"  • Mean: {freq_analysis['mean']*100:+.1f}%")
    print(f"  • Range: {freq_analysis['range'][0]*100:+.1f}% to {freq_analysis['range'][1]*100:+.1f}%")
    
    atp_analysis = analysis['atp_changes']
    print(f"\nATP Consumption:")
    print(f"  • Mean change: {atp_analysis['mean']:.2f}×")
    print(f"  • Energy efficient drugs: {atp_analysis['energy_efficient_drugs']}")
    print(f"  • Energy costly drugs: {atp_analysis['energy_costly_drugs']}")
    
    sync_analysis = analysis['synchronization']
    print(f"\nOscillatory Synchronization:")
    print(f"  • Mean score: {sync_analysis['mean']:.3f}")
    print(f"  • Highly synchronized: {sync_analysis['highly_synchronized']} responses")
    
    print(f"\nDrug-Specific Effects:")
    for drug, effects in analysis['drug_specific'].items():
        print(f"  • {drug}: {effects['targets_affected']} targets, "
              f"{effects['average_frequency_change']*100:+.1f}% freq change")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    simulator.save_results(responses, analysis)
    simulator.visualize_cellular_responses(responses)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find most effective drug-target pair
    best_response = max(responses, key=lambda r: abs(
        (r.drug_modified_oscillation - r.baseline_oscillation) / r.baseline_oscillation
    ))
    freq_change = (best_response.drug_modified_oscillation - best_response.baseline_oscillation) / best_response.baseline_oscillation
    print(f"• Strongest cellular effect: {best_response.drug_name} → {best_response.target_id} ({freq_change*100:+.1f}%)")
    
    # Find most synchronized response
    best_sync = max(responses, key=lambda r: r.oscillatory_synchronization)
    print(f"• Best synchronization: {best_sync.drug_name} → {best_sync.target_id} ({best_sync.oscillatory_synchronization:.3f})")
    
    # ATP efficiency
    efficient_responses = [r for r in responses if r.atp_consumption_change < 1.0]
    print(f"• ATP-efficient responses: {len(efficient_responses)}/{len(responses)}")
    
    # High confidence predictions
    high_conf = [r for r in responses if r.confidence > 0.8]
    print(f"• High-confidence predictions: {len(high_conf)}/{len(responses)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Cellular drug response simulation complete!")
    
    return responses

if __name__ == "__main__":
    responses = main()
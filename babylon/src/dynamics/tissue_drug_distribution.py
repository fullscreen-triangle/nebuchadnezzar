"""
Tissue Drug Distribution Module  
==============================

Models drug distribution and accumulation patterns across different tissue
types using oscillatory framework principles. Simulates how drugs traverse
tissue barriers and achieve therapeutic concentrations.

Based on tissue-specific oscillatory signatures and multi-compartment dynamics.
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
class TissueCompartment:
    """Represents a tissue compartment for drug distribution."""
    tissue_id: str
    tissue_type: str  # 'brain', 'liver', 'kidney', 'heart', 'muscle', 'fat', 'plasma'
    volume: float  # L - tissue volume
    blood_flow: float  # L/min - tissue blood flow
    permeability: float  # cm/min - vascular permeability
    binding_capacity: float  # μmol/g - protein binding capacity
    metabolism_rate: float  # 1/min - metabolic clearance rate constant
    ph: float  # Tissue pH
    lipid_content: float  # % lipid content (0-100)
    oscillatory_signature: float  # Hz - tissue-specific oscillation
    barrier_properties: Dict[str, float] = field(default_factory=dict)

@dataclass
class DrugDistributionProfile:
    """Drug-specific distribution properties."""
    drug_name: str
    plasma_protein_binding: float  # % bound to plasma proteins
    volume_of_distribution: float  # L/kg - apparent volume of distribution
    clearance_rate: float  # L/min - total body clearance
    elimination_half_life: float  # hours
    tissue_affinity_factors: Dict[str, float]  # tissue_id -> affinity multiplier
    active_transport: bool  # Whether drug uses active transport
    metabolism_pathways: List[str]  # Primary metabolic pathways

@dataclass
class DistributionResult:
    """Result of tissue drug distribution simulation."""
    drug_name: str
    tissue_id: str
    peak_concentration: float  # μM - maximum tissue concentration
    time_to_peak: float  # hours - time to reach peak concentration
    auc_tissue: float  # μM⋅h - area under concentration-time curve
    tissue_plasma_ratio: float  # Ratio of tissue to plasma concentration
    accumulation_factor: float  # Steady-state accumulation relative to single dose
    elimination_rate: float  # 1/h - tissue elimination rate constant
    oscillatory_enhancement: float  # Fold enhancement due to tissue oscillations
    barrier_penetration: float  # 0-1 scale for barrier crossing efficiency
    distribution_time_course: List[Tuple[float, float]]  # (time, concentration) pairs
    confidence: float = 0.0

class TissueDrugDistribution:
    """
    Simulates drug distribution across tissue compartments using oscillatory
    framework principles.
    
    Models how drugs distribute from plasma into tissues, accounting for
    tissue-specific barriers, binding, and oscillatory enhancement.
    """
    
    def __init__(self):
        """Initialize tissue distribution simulator."""
        self.tissue_compartments = self._load_tissue_compartments()
        self.simulation_time = 48.0  # hours - simulation duration
        self.time_resolution = 0.1  # hours - time step size
        
    def _load_tissue_compartments(self) -> Dict[str, TissueCompartment]:
        """Load tissue compartments for distribution simulation."""
        
        compartments = {
            'plasma': TissueCompartment(
                tissue_id='plasma',
                tissue_type='plasma',
                volume=3.0,  # L - plasma volume in 70kg adult
                blood_flow=5.0,  # L/min - cardiac output
                permeability=0.0,  # No barrier for plasma
                binding_capacity=600.0,  # μmol/L - albumin binding
                metabolism_rate=0.0,  # No metabolism in plasma
                ph=7.4,
                lipid_content=0.5,
                oscillatory_signature=1.2e13,
                barrier_properties={}
            ),
            
            'brain': TissueCompartment(
                tissue_id='brain',
                tissue_type='brain', 
                volume=1.4,  # L - brain volume
                blood_flow=0.75,  # L/min - cerebral blood flow
                permeability=0.001,  # cm/min - blood-brain barrier permeability
                binding_capacity=50.0,  # μmol/g - brain tissue binding
                metabolism_rate=0.02,  # 1/min - brain metabolism
                ph=7.1,
                lipid_content=60.0,  # High lipid content
                oscillatory_signature=2.3e13,
                barrier_properties={
                    'tight_junctions': 0.95,  # Very tight BBB
                    'active_efflux': 0.8,     # P-gp efflux activity
                    'specialized_transport': 0.3  # Limited transporters
                }
            ),
            
            'liver': TissueCompartment(
                tissue_id='liver',
                tissue_type='liver',
                volume=1.8,  # L - liver volume
                blood_flow=1.5,  # L/min - hepatic blood flow
                permeability=0.1,  # cm/min - high hepatic permeability
                binding_capacity=200.0,  # μmol/g - hepatic binding capacity
                metabolism_rate=0.15,  # 1/min - high metabolic activity
                ph=7.2,
                lipid_content=5.0,
                oscillatory_signature=1.8e13,
                barrier_properties={
                    'fenestrated_endothelium': 0.9,  # High permeability
                    'metabolic_capacity': 0.95,      # Very high metabolism
                    'bile_transport': 0.6            # Biliary secretion
                }
            ),
            
            'kidney': TissueCompartment(
                tissue_id='kidney',
                tissue_type='kidney',
                volume=0.3,  # L - kidney volume (both kidneys)
                blood_flow=1.2,  # L/min - renal blood flow
                permeability=0.05,  # cm/min - moderate permeability
                binding_capacity=100.0,  # μmol/g
                metabolism_rate=0.08,  # 1/min
                ph=6.8,  # Slightly acidic
                lipid_content=3.0,
                oscillatory_signature=2.1e13,
                barrier_properties={
                    'glomerular_filtration': 0.8,   # High filtration
                    'tubular_secretion': 0.7,       # Active secretion
                    'tubular_reabsorption': 0.6     # Reabsorption capacity
                }
            ),
            
            'heart': TissueCompartment(
                tissue_id='heart',
                tissue_type='heart',
                volume=0.3,  # L - heart volume
                blood_flow=0.25,  # L/min - coronary blood flow
                permeability=0.02,  # cm/min - moderate permeability
                binding_capacity=80.0,  # μmol/g
                metabolism_rate=0.05,  # 1/min
                ph=7.0,
                lipid_content=15.0,
                oscillatory_signature=2.8e13,  # High due to contractile activity
                barrier_properties={
                    'contractile_barriers': 0.6,    # Mechanical barriers
                    'high_perfusion': 0.9,         # Excellent blood supply
                    'metabolic_demand': 0.8        # High energy metabolism
                }
            ),
            
            'muscle': TissueCompartment(
                tissue_id='muscle',
                tissue_type='muscle',
                volume=28.0,  # L - skeletal muscle volume (largest compartment)
                blood_flow=1.0,  # L/min - muscle blood flow
                permeability=0.01,  # cm/min - low permeability
                binding_capacity=30.0,  # μmol/g - low binding capacity
                metabolism_rate=0.01,  # 1/min - low metabolism
                ph=7.0,
                lipid_content=10.0,
                oscillatory_signature=1.5e13,
                barrier_properties={
                    'large_volume': 0.9,           # Largest tissue compartment
                    'variable_perfusion': 0.6,    # Perfusion varies with activity
                    'slow_equilibration': 0.4     # Slow to reach equilibrium
                }
            ),
            
            'fat': TissueCompartment(
                tissue_id='fat',
                tissue_type='fat',
                volume=15.0,  # L - adipose tissue volume
                blood_flow=0.3,  # L/min - low adipose blood flow
                permeability=0.005,  # cm/min - very low permeability
                binding_capacity=10.0,  # μmol/g - very low binding
                metabolism_rate=0.005,  # 1/min - very low metabolism
                ph=7.2,
                lipid_content=85.0,  # Very high lipid content
                oscillatory_signature=0.8e13,  # Low metabolic activity
                barrier_properties={
                    'lipophilic_reservoir': 0.95,  # Excellent lipid storage
                    'slow_release': 0.2,          # Very slow drug release
                    'poor_perfusion': 0.3         # Limited blood flow
                }
            )
        }
        
        logger.info(f"Loaded {len(compartments)} tissue compartments")
        return compartments
    
    def create_drug_distribution_profiles(self) -> Dict[str, DrugDistributionProfile]:
        """Create distribution profiles for pharmaceutical molecules."""
        
        profiles = {
            'lithium': DrugDistributionProfile(
                drug_name='lithium',
                plasma_protein_binding=0.0,  # Not protein bound
                volume_of_distribution=0.7,  # L/kg - small Vd
                clearance_rate=0.02,  # L/min - renal clearance
                elimination_half_life=18.0,  # hours
                tissue_affinity_factors={
                    'brain': 0.8,   # Crosses BBB moderately
                    'kidney': 1.2,  # Accumulates in kidney
                    'heart': 0.9,   # Moderate cardiac distribution
                    'liver': 0.7,   # Limited hepatic uptake
                    'muscle': 0.6,  # Low muscle distribution
                    'fat': 0.1      # Very poor fat distribution
                },
                active_transport=True,  # Uses sodium channels
                metabolism_pathways=['renal_elimination']  # Not metabolized
            ),
            
            'aripiprazole': DrugDistributionProfile(
                drug_name='aripiprazole',
                plasma_protein_binding=99.0,  # Highly protein bound
                volume_of_distribution=4.9,  # L/kg - high Vd
                clearance_rate=0.04,  # L/min
                elimination_half_life=75.0,  # hours - very long half-life
                tissue_affinity_factors={
                    'brain': 2.5,   # High brain penetration
                    'liver': 3.0,   # High hepatic uptake
                    'kidney': 1.5,  # Moderate renal distribution
                    'heart': 1.2,   # Moderate cardiac distribution
                    'muscle': 1.8,  # Good muscle distribution
                    'fat': 5.0      # Very high fat distribution (lipophilic)
                },
                active_transport=False,
                metabolism_pathways=['cyp2d6', 'cyp3a4']
            ),
            
            'citalopram': DrugDistributionProfile(
                drug_name='citalopram',
                plasma_protein_binding=80.0,  # Moderately protein bound
                volume_of_distribution=1.2,  # L/kg
                clearance_rate=0.03,  # L/min
                elimination_half_life=35.0,  # hours
                tissue_affinity_factors={
                    'brain': 1.8,   # Good brain penetration
                    'liver': 2.2,   # High hepatic uptake
                    'kidney': 1.3,  # Moderate renal distribution
                    'heart': 1.1,   # Moderate cardiac distribution
                    'muscle': 1.0,  # Average muscle distribution
                    'fat': 2.5      # High fat distribution
                },
                active_transport=False,
                metabolism_pathways=['cyp2c19', 'cyp2d6']
            ),
            
            'atorvastatin': DrugDistributionProfile(
                drug_name='atorvastatin',
                plasma_protein_binding=98.0,  # Highly protein bound
                volume_of_distribution=4.5,  # L/kg
                clearance_rate=0.6,  # L/min - high clearance
                elimination_half_life=14.0,  # hours
                tissue_affinity_factors={
                    'liver': 20.0,  # Very high hepatic uptake (target organ)
                    'brain': 0.1,   # Poor brain penetration
                    'kidney': 1.5,  # Moderate renal distribution
                    'heart': 2.0,   # Good cardiac distribution
                    'muscle': 3.0,  # High muscle distribution (target tissue)
                    'fat': 8.0      # Very high fat distribution
                },
                active_transport=True,  # OATP transporters
                metabolism_pathways=['cyp3a4']
            ),
            
            'aspirin': DrugDistributionProfile(
                drug_name='aspirin',
                plasma_protein_binding=99.5,  # Very highly protein bound
                volume_of_distribution=0.15,  # L/kg - very small Vd
                clearance_rate=0.8,  # L/min - rapid clearance
                elimination_half_life=0.3,  # hours - very short half-life
                tissue_affinity_factors={
                    'liver': 1.5,   # Moderate hepatic distribution
                    'kidney': 2.0,  # Good renal distribution
                    'heart': 1.8,   # Good cardiac distribution (target)
                    'brain': 0.3,   # Limited brain penetration
                    'muscle': 0.8,  # Limited muscle distribution
                    'fat': 0.2      # Very poor fat distribution
                },
                active_transport=False,
                metabolism_pathways=['hepatic_hydrolysis']
            )
        }
        
        return profiles
    
    def calculate_tissue_drug_uptake(self, drug_profile: DrugDistributionProfile,
                                   tissue: TissueCompartment,
                                   plasma_concentration: float,
                                   time: float) -> float:
        """Calculate drug uptake rate into tissue compartment."""
        
        # Get tissue-specific affinity
        affinity_factor = drug_profile.tissue_affinity_factors.get(tissue.tissue_id, 1.0)
        
        # Calculate unbound drug fraction in plasma
        unbound_fraction = (100.0 - drug_profile.plasma_protein_binding) / 100.0
        unbound_plasma_conc = plasma_concentration * unbound_fraction
        
        # Permeability-limited uptake
        permeability_rate = (tissue.permeability * tissue.blood_flow * 
                           unbound_plasma_conc * affinity_factor)
        
        # Flow-limited uptake (for highly permeable tissues)
        flow_rate = tissue.blood_flow * unbound_plasma_conc * affinity_factor
        
        # Use minimum of permeability and flow limitation
        uptake_rate = min(permeability_rate, flow_rate)
        
        # Apply oscillatory enhancement
        oscillatory_enhancement = self._calculate_oscillatory_enhancement(
            drug_profile, tissue
        )
        enhanced_uptake = uptake_rate * oscillatory_enhancement
        
        return enhanced_uptake
    
    def _calculate_oscillatory_enhancement(self, drug_profile: DrugDistributionProfile,
                                         tissue: TissueCompartment) -> float:
        """Calculate oscillatory enhancement of drug uptake."""
        
        # Simple oscillatory enhancement model
        # Based on tissue metabolic activity and drug properties
        
        base_enhancement = 1.0
        
        # Higher oscillatory activity enhances uptake
        oscillation_factor = tissue.oscillatory_signature / 2.0e13  # Normalize
        
        # Lipophilic drugs benefit more from oscillatory effects
        lipophilicity_factor = 1.0
        if 'fat' in drug_profile.tissue_affinity_factors:
            fat_affinity = drug_profile.tissue_affinity_factors['fat']
            lipophilicity_factor = 1.0 + (fat_affinity - 1.0) * 0.1
        
        # Active transport benefits from oscillatory enhancement
        transport_factor = 1.2 if drug_profile.active_transport else 1.0
        
        # Combined enhancement
        total_enhancement = (base_enhancement + 
                           0.2 * oscillation_factor * 
                           lipophilicity_factor * 
                           transport_factor)
        
        return min(2.0, total_enhancement)  # Cap at 2-fold enhancement
    
    def simulate_tissue_distribution(self, drug_profile: DrugDistributionProfile,
                                   dose: float = 1.0) -> List[DistributionResult]:
        """Simulate drug distribution across all tissue compartments."""
        
        results = []
        
        # Time points for simulation
        time_points = np.arange(0, self.simulation_time + self.time_resolution, 
                               self.time_resolution)
        
        for tissue_id, tissue in self.tissue_compartments.items():
            if tissue_id == 'plasma':  # Skip plasma compartment
                continue
                
            # Initialize concentration arrays
            tissue_concentrations = []
            plasma_concentrations = []
            
            # Simulate concentration-time profile
            for t in time_points:
                # Calculate plasma concentration (simple exponential decay)
                plasma_conc = self._calculate_plasma_concentration(drug_profile, dose, t)
                plasma_concentrations.append(plasma_conc)
                
                # Calculate tissue concentration
                tissue_conc = self._calculate_tissue_concentration(
                    drug_profile, tissue, plasma_conc, t
                )
                tissue_concentrations.append(tissue_conc)
            
            # Calculate derived metrics
            peak_concentration = max(tissue_concentrations)
            time_to_peak_idx = np.argmax(tissue_concentrations)
            time_to_peak = time_points[time_to_peak_idx]
            
            # Area under curve (trapezoidal rule)
            auc_tissue = np.trapz(tissue_concentrations, time_points)
            
            # Tissue to plasma ratio at steady state
            steady_state_idx = int(len(time_points) * 0.8)  # Use last 20% of simulation
            avg_tissue_conc = np.mean(tissue_concentrations[steady_state_idx:])
            avg_plasma_conc = np.mean(plasma_concentrations[steady_state_idx:])
            tissue_plasma_ratio = avg_tissue_conc / avg_plasma_conc if avg_plasma_conc > 0 else 0
            
            # Accumulation factor (steady-state vs peak after single dose)
            single_dose_peak = tissue_concentrations[1]  # Approximate early peak
            accumulation_factor = avg_tissue_conc / single_dose_peak if single_dose_peak > 0 else 1
            
            # Elimination rate (from terminal phase)
            elimination_rate = self._calculate_elimination_rate(tissue_concentrations, time_points)
            
            # Oscillatory enhancement
            oscillatory_enhancement = self._calculate_oscillatory_enhancement(drug_profile, tissue)
            
            # Barrier penetration efficiency
            barrier_penetration = self._calculate_barrier_penetration(drug_profile, tissue)
            
            # Time course data
            time_course = list(zip(time_points, tissue_concentrations))
            
            # Confidence calculation
            confidence = self._calculate_distribution_confidence(drug_profile, tissue)
            
            result = DistributionResult(
                drug_name=drug_profile.drug_name,
                tissue_id=tissue_id,
                peak_concentration=peak_concentration,
                time_to_peak=time_to_peak,
                auc_tissue=auc_tissue,
                tissue_plasma_ratio=tissue_plasma_ratio,
                accumulation_factor=accumulation_factor,
                elimination_rate=elimination_rate,
                oscillatory_enhancement=oscillatory_enhancement,
                barrier_penetration=barrier_penetration,
                distribution_time_course=time_course,
                confidence=confidence
            )
            
            results.append(result)
        
        return results
    
    def _calculate_plasma_concentration(self, drug_profile: DrugDistributionProfile,
                                      dose: float, time: float) -> float:
        """Calculate plasma concentration using simple PK model."""
        
        # Simple one-compartment model with first-order elimination
        ke = 0.693 / drug_profile.elimination_half_life  # Elimination rate constant
        
        # Assume IV bolus with immediate distribution
        c0 = dose / drug_profile.volume_of_distribution  # Initial concentration
        
        # Exponential decay
        plasma_conc = c0 * np.exp(-ke * time)
        
        return plasma_conc
    
    def _calculate_tissue_concentration(self, drug_profile: DrugDistributionProfile,
                                      tissue: TissueCompartment,
                                      plasma_concentration: float,
                                      time: float) -> float:
        """Calculate tissue concentration based on distribution kinetics."""
        
        # Get tissue affinity
        affinity = drug_profile.tissue_affinity_factors.get(tissue.tissue_id, 1.0)
        
        # Distribution half-life (faster for more permeable tissues)
        distribution_rate = tissue.permeability * tissue.blood_flow / tissue.volume
        
        # Equilibrium concentration
        equilibrium_conc = plasma_concentration * affinity
        
        # Approach to equilibrium
        if time == 0:
            tissue_conc = 0
        else:
            # First-order approach to equilibrium
            tissue_conc = equilibrium_conc * (1 - np.exp(-distribution_rate * time))
        
        # Apply oscillatory enhancement
        oscillatory_enhancement = self._calculate_oscillatory_enhancement(drug_profile, tissue)
        enhanced_conc = tissue_conc * oscillatory_enhancement
        
        return enhanced_conc
    
    def _calculate_elimination_rate(self, concentrations: List[float], 
                                  time_points: np.ndarray) -> float:
        """Calculate elimination rate constant from concentration-time data."""
        
        # Use terminal phase (last 50% of data)
        terminal_start = len(concentrations) // 2
        terminal_concs = concentrations[terminal_start:]
        terminal_times = time_points[terminal_start:]
        
        # Linear regression on log concentrations
        if len(terminal_concs) < 2 or min(terminal_concs) <= 0:
            return 0.05  # Default elimination rate
        
        log_concs = np.log(terminal_concs)
        valid_indices = np.isfinite(log_concs)
        
        if np.sum(valid_indices) < 2:
            return 0.05
        
        slope, _ = np.polyfit(terminal_times[valid_indices], 
                            log_concs[valid_indices], 1)
        
        elimination_rate = -slope  # Rate constant (positive value)
        
        return max(0.001, elimination_rate)  # Minimum elimination rate
    
    def _calculate_barrier_penetration(self, drug_profile: DrugDistributionProfile,
                                     tissue: TissueCompartment) -> float:
        """Calculate efficiency of barrier penetration."""
        
        base_penetration = tissue.permeability / 0.1  # Normalize to liver permeability
        base_penetration = min(1.0, base_penetration)
        
        # Adjust for drug properties
        if drug_profile.active_transport and 'specialized_transport' in tissue.barrier_properties:
            # Active transport helps penetration
            transport_boost = tissue.barrier_properties['specialized_transport']
            base_penetration *= (1.0 + transport_boost)
        
        # Protein binding reduces penetration
        unbound_fraction = (100.0 - drug_profile.plasma_protein_binding) / 100.0
        penetration = base_penetration * unbound_fraction
        
        return min(1.0, penetration)
    
    def _calculate_distribution_confidence(self, drug_profile: DrugDistributionProfile,
                                         tissue: TissueCompartment) -> float:
        """Calculate confidence in distribution prediction."""
        
        base_confidence = 0.7
        
        # Well-studied tissues have higher confidence
        well_studied_tissues = ['liver', 'kidney', 'plasma', 'brain']
        if tissue.tissue_id in well_studied_tissues:
            confidence_boost = 0.15
        else:
            confidence_boost = 0.0
        
        # Drugs with known distribution have higher confidence
        if drug_profile.drug_name in ['lithium', 'atorvastatin']:
            drug_boost = 0.1
        else:
            drug_boost = 0.0
        
        # Active transport adds uncertainty
        transport_penalty = -0.05 if drug_profile.active_transport else 0.0
        
        final_confidence = min(0.95, base_confidence + confidence_boost + 
                              drug_boost + transport_penalty)
        
        return final_confidence
    
    def simulate_drug_library(self) -> List[DistributionResult]:
        """Simulate distribution for all drugs across all tissues."""
        
        drug_profiles = self.create_drug_distribution_profiles()
        all_results = []
        
        for drug_name, drug_profile in drug_profiles.items():
            logger.info(f"Simulating tissue distribution for {drug_name}")
            drug_results = self.simulate_tissue_distribution(drug_profile)
            all_results.extend(drug_results)
        
        logger.info(f"Completed tissue distribution simulation for {len(all_results)} combinations")
        return all_results
    
    def analyze_distribution_patterns(self, results: List[DistributionResult]) -> Dict[str, Any]:
        """Analyze patterns in tissue distribution results."""
        
        analysis = {}
        
        # Tissue selectivity analysis
        tissue_selectivity = {}
        for result in results:
            if result.tissue_id not in tissue_selectivity:
                tissue_selectivity[result.tissue_id] = []
            tissue_selectivity[result.tissue_id].append(result.tissue_plasma_ratio)
        
        analysis['tissue_selectivity'] = {}
        for tissue, ratios in tissue_selectivity.items():
            analysis['tissue_selectivity'][tissue] = {
                'mean_ratio': np.mean(ratios),
                'max_ratio': np.max(ratios),
                'accumulating_drugs': len([r for r in ratios if r > 2.0])
            }
        
        # Oscillatory enhancement analysis
        enhancements = [r.oscillatory_enhancement for r in results]
        analysis['oscillatory_enhancement'] = {
            'mean': np.mean(enhancements),
            'max': np.max(enhancements),
            'enhanced_count': len([e for e in enhancements if e > 1.5])
        }
        
        # Barrier penetration analysis
        penetrations = [r.barrier_penetration for r in results]
        analysis['barrier_penetration'] = {
            'mean': np.mean(penetrations),
            'poor_penetration': len([p for p in penetrations if p < 0.3]),
            'good_penetration': len([p for p in penetrations if p > 0.7])
        }
        
        # Drug-specific analysis
        drug_analysis = {}
        for result in results:
            if result.drug_name not in drug_analysis:
                drug_analysis[result.drug_name] = []
            drug_analysis[result.drug_name].append(result)
        
        analysis['drug_specific'] = {}
        for drug, drug_results in drug_analysis.items():
            # Find tissue with highest accumulation
            max_accumulation = max(r.tissue_plasma_ratio for r in drug_results)
            target_tissue = next(r.tissue_id for r in drug_results 
                               if r.tissue_plasma_ratio == max_accumulation)
            
            analysis['drug_specific'][drug] = {
                'target_tissue': target_tissue,
                'max_accumulation': max_accumulation,
                'tissues_penetrated': len([r for r in drug_results if r.barrier_penetration > 0.5])
            }
        
        return analysis
    
    def visualize_distribution_results(self, results: List[DistributionResult],
                                     output_dir: str = "babylon_results") -> None:
        """Create visualizations of tissue distribution results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive distribution visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Tissue-plasma ratios by drug
        drugs = list(set(r.drug_name for r in results))
        tissues = list(set(r.tissue_id for r in results))
        
        # Create matrix for heatmap
        ratio_matrix = np.zeros((len(drugs), len(tissues)))
        for i, drug in enumerate(drugs):
            for j, tissue in enumerate(tissues):
                result = next((r for r in results 
                             if r.drug_name == drug and r.tissue_id == tissue), None)
                if result:
                    ratio_matrix[i, j] = result.tissue_plasma_ratio
        
        im1 = axes[0, 0].imshow(ratio_matrix, cmap='Reds', aspect='auto')
        axes[0, 0].set_xticks(range(len(tissues)))
        axes[0, 0].set_xticklabels(tissues, rotation=45)
        axes[0, 0].set_yticks(range(len(drugs)))
        axes[0, 0].set_yticklabels(drugs)
        axes[0, 0].set_title('Tissue-Plasma Ratios')
        plt.colorbar(im1, ax=axes[0, 0], label='Tissue/Plasma Ratio')
        
        # 2. Oscillatory enhancement distribution
        enhancements = [r.oscillatory_enhancement for r in results]
        
        axes[0, 1].hist(enhancements, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 1].set_xlabel('Oscillatory Enhancement Factor')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Oscillatory Enhancement Distribution')
        axes[0, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='No Enhancement')
        axes[0, 1].legend()
        
        # 3. Barrier penetration by tissue
        tissue_penetrations = {}
        for tissue in tissues:
            tissue_results = [r for r in results if r.tissue_id == tissue]
            penetrations = [r.barrier_penetration for r in tissue_results]
            tissue_penetrations[tissue] = penetrations
        
        axes[0, 2].boxplot([tissue_penetrations[tissue] for tissue in tissues], 
                          labels=tissues)
        axes[0, 2].set_ylabel('Barrier Penetration')
        axes[0, 2].set_title('Barrier Penetration by Tissue')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Time to peak distribution
        times_to_peak = [r.time_to_peak for r in results]
        drug_names = [r.drug_name for r in results]
        
        drug_colors = plt.cm.Set1(np.linspace(0, 1, len(set(drug_names))))
        color_map = {drug: color for drug, color in zip(set(drug_names), drug_colors)}
        colors = [color_map[name] for name in drug_names]
        
        scatter = axes[1, 0].scatter(times_to_peak, 
                                   [r.peak_concentration for r in results],
                                   c=colors, alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Time to Peak (hours)')
        axes[1, 0].set_ylabel('Peak Concentration (μM)')
        axes[1, 0].set_title('Peak Concentration vs Time to Peak')
        axes[1, 0].set_yscale('log')
        
        # 5. Accumulation factors
        accumulation_factors = [r.accumulation_factor for r in results]
        
        axes[1, 1].hist(accumulation_factors, bins=15, alpha=0.7, color='lightgreen', 
                       edgecolor='black')
        axes[1, 1].set_xlabel('Accumulation Factor')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Tissue Accumulation Factors')
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='No Accumulation')
        axes[1, 1].legend()
        
        # 6. Example time course (lithium in brain)
        lithium_brain = next((r for r in results 
                            if r.drug_name == 'lithium' and r.tissue_id == 'brain'), None)
        
        if lithium_brain:
            times, concentrations = zip(*lithium_brain.distribution_time_course)
            axes[1, 2].plot(times, concentrations, 'b-', linewidth=2, label='Lithium→Brain')
            axes[1, 2].set_xlabel('Time (hours)')
            axes[1, 2].set_ylabel('Tissue Concentration (μM)')
            axes[1, 2].set_title('Example Time Course: Lithium→Brain')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No lithium-brain\ndata available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Time Course (Not Available)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'tissue_drug_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribution visualizations saved to {output_path}")
    
    def save_results(self, results: List[DistributionResult], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save tissue distribution simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_data = []
        for result in results:
            results_data.append({
                'drug_name': result.drug_name,
                'tissue_id': result.tissue_id,
                'peak_concentration': result.peak_concentration,
                'time_to_peak': result.time_to_peak,
                'auc_tissue': result.auc_tissue,
                'tissue_plasma_ratio': result.tissue_plasma_ratio,
                'accumulation_factor': result.accumulation_factor,
                'elimination_rate': result.elimination_rate,
                'oscillatory_enhancement': result.oscillatory_enhancement,
                'barrier_penetration': result.barrier_penetration,
                'distribution_time_course': result.distribution_time_course,
                'confidence': result.confidence
            })
        
        with open(output_path / 'tissue_distribution_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_data = []
        for result in results:
            csv_data.append({
                'drug_name': result.drug_name,
                'tissue_id': result.tissue_id,
                'peak_concentration': result.peak_concentration,
                'time_to_peak_hours': result.time_to_peak,
                'tissue_plasma_ratio': result.tissue_plasma_ratio,
                'accumulation_factor': result.accumulation_factor,
                'elimination_rate_per_hour': result.elimination_rate,
                'oscillatory_enhancement': result.oscillatory_enhancement,
                'barrier_penetration': result.barrier_penetration,
                'confidence': result.confidence
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / 'tissue_distribution_results.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'tissue_distribution_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Tissue distribution results saved to {output_path}")

def main():
    """
    Test tissue drug distribution simulation.
    
    This demonstrates how drugs distribute across different tissue
    compartments using oscillatory-enhanced distribution kinetics.
    """
    
    print("🫁 Testing Tissue Drug Distribution")
    print("=" * 50)
    
    # Initialize tissue distribution simulator
    simulator = TissueDrugDistribution()
    
    # Display loaded tissue compartments
    print(f"\n🏥 Loaded Tissue Compartments:")
    for tissue_id, tissue in simulator.tissue_compartments.items():
        print(f"  • {tissue_id}: {tissue.volume:.1f}L, Flow={tissue.blood_flow:.2f}L/min")
        print(f"    Permeability: {tissue.permeability:.3f} cm/min")
        print(f"    pH: {tissue.ph:.1f}, Lipid: {tissue.lipid_content:.1f}%")
        print(f"    Oscillation: {tissue.oscillatory_signature:.2e} Hz")
    
    # Create and display drug profiles
    drug_profiles = simulator.create_drug_distribution_profiles()
    print(f"\n💊 Drug Distribution Profiles:")
    for drug_name, profile in drug_profiles.items():
        print(f"  • {drug_name}: Vd={profile.volume_of_distribution:.1f}L/kg")
        print(f"    Protein binding: {profile.plasma_protein_binding:.1f}%")
        print(f"    Half-life: {profile.elimination_half_life:.1f}h")
        print(f"    Active transport: {profile.active_transport}")
    
    # Simulate distribution for all combinations
    print(f"\n🔬 Simulating tissue drug distribution...")
    results = simulator.simulate_drug_library()
    
    # Display results
    print(f"\n📊 TISSUE DISTRIBUTION RESULTS:")
    print("-" * 80)
    
    # Group results by drug for better display
    drug_results = {}
    for result in results:
        if result.drug_name not in drug_results:
            drug_results[result.drug_name] = []
        drug_results[result.drug_name].append(result)
    
    for drug_name, drug_dist_results in drug_results.items():
        print(f"\n{drug_name.upper()}:")
        
        # Sort by tissue-plasma ratio
        drug_dist_results.sort(key=lambda r: r.tissue_plasma_ratio, reverse=True)
        
        for result in drug_dist_results:
            print(f"  {result.tissue_id.capitalize():>8}: "
                  f"Peak={result.peak_concentration:.2f}μM "
                  f"T/P={result.tissue_plasma_ratio:.2f} "
                  f"Tpeak={result.time_to_peak:.1f}h "
                  f"Enhance={result.oscillatory_enhancement:.2f}× "
                  f"Conf={result.confidence:.3f}")
    
    # Analyze patterns
    print("\n🔍 Analyzing distribution patterns...")
    analysis = simulator.analyze_distribution_patterns(results)
    
    print(f"\n📈 DISTRIBUTION PATTERN ANALYSIS:")
    print("-" * 50)
    
    print("Tissue Selectivity:")
    for tissue, metrics in analysis['tissue_selectivity'].items():
        print(f"  • {tissue.capitalize():>8}: Mean T/P = {metrics['mean_ratio']:.2f}, "
              f"Max = {metrics['max_ratio']:.2f}, "
              f"{metrics['accumulating_drugs']} accumulating drugs")
    
    enhancement_analysis = analysis['oscillatory_enhancement']
    print(f"\nOscillatory Enhancement:")
    print(f"  • Mean enhancement: {enhancement_analysis['mean']:.2f}×")
    print(f"  • Maximum enhancement: {enhancement_analysis['max']:.2f}×")
    print(f"  • Significantly enhanced (>1.5×): {enhancement_analysis['enhanced_count']}")
    
    barrier_analysis = analysis['barrier_penetration']
    print(f"\nBarrier Penetration:")
    print(f"  • Mean penetration: {barrier_analysis['mean']:.3f}")
    print(f"  • Poor penetration (<0.3): {barrier_analysis['poor_penetration']}")
    print(f"  • Good penetration (>0.7): {barrier_analysis['good_penetration']}")
    
    print(f"\nDrug-Specific Targeting:")
    for drug, targeting in analysis['drug_specific'].items():
        print(f"  • {drug}: Targets {targeting['target_tissue']} "
              f"(T/P = {targeting['max_accumulation']:.1f}), "
              f"penetrates {targeting['tissues_penetrated']} tissues")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    simulator.save_results(results, analysis)
    simulator.visualize_distribution_results(results)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find highest tissue accumulation
    max_accumulation = max(results, key=lambda r: r.tissue_plasma_ratio)
    print(f"• Highest accumulation: {max_accumulation.drug_name} → {max_accumulation.tissue_id} "
          f"(T/P = {max_accumulation.tissue_plasma_ratio:.1f})")
    
    # Find best oscillatory enhancement
    best_enhancement = max(results, key=lambda r: r.oscillatory_enhancement)
    print(f"• Best oscillatory enhancement: {best_enhancement.drug_name} → {best_enhancement.tissue_id} "
          f"({best_enhancement.oscillatory_enhancement:.2f}×)")
    
    # Find best barrier penetration
    best_penetration = max(results, key=lambda r: r.barrier_penetration)
    print(f"• Best barrier penetration: {best_penetration.drug_name} → {best_penetration.tissue_id} "
          f"({best_penetration.barrier_penetration:.3f})")
    
    # Count rapid distribution
    rapid_distribution = len([r for r in results if r.time_to_peak < 2.0])
    print(f"• Rapid distribution (<2h): {rapid_distribution}/{len(results)}")
    
    # High confidence predictions
    high_conf = len([r for r in results if r.confidence > 0.8])
    print(f"• High-confidence predictions: {high_conf}/{len(results)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Tissue drug distribution simulation complete!")
    
    return results

if __name__ == "__main__":
    results = main()

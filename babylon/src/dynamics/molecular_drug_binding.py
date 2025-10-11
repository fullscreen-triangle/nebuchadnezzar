"""
Molecular Drug Binding Module
============================

Models molecular-level drug-target interactions using oscillatory framework
principles. Simulates binding kinetics, conformational changes, and
allosteric effects through molecular oscillatory dynamics.

Based on molecular oscillation coupling and binding site resonance theory.
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
class MolecularTarget:
    """Represents a molecular target for drug binding."""
    target_id: str
    protein_family: str  # 'gpcr', 'kinase', 'enzyme', 'transporter', 'ion_channel'
    molecular_weight: float  # kDa
    binding_site_volume: float  # Ų - cubic angstroms
    backbone_flexibility: float  # 0-1 scale
    allosteric_sites: int  # Number of allosteric sites
    native_oscillation_frequency: float  # Hz - protein breathing/dynamics
    binding_site_polarity: float  # 0-1 scale (0=hydrophobic, 1=hydrophilic)

@dataclass
class DrugMolecule:
    """Represents a drug molecule for binding simulation."""
    drug_name: str
    molecular_weight: float  # Da
    volume: float  # Ų - molecular volume
    flexibility: float  # Number of rotatable bonds
    hydrogen_bond_donors: int
    hydrogen_bond_acceptors: int
    charge_distribution: List[float]  # Partial charges on key atoms
    oscillatory_signature: float  # Hz - molecular vibrational frequency
    lipophilicity: float  # LogP
    shape_descriptor: str  # 'linear', 'bent', 'cyclic', 'branched'

@dataclass
class BindingResult:
    """Result of molecular drug-target binding simulation."""
    drug_name: str
    target_id: str
    binding_affinity: float  # nM - Kd value
    binding_kinetics: Dict[str, float]  # kon, koff rates
    conformational_change: float  # Amplitude of induced protein conformational change
    allosteric_effects: Dict[str, float]  # Effects on other binding sites
    oscillatory_coupling_strength: float  # 0-1 scale
    residence_time: float  # seconds - drug stays bound
    selectivity_profile: Dict[str, float]  # target_id -> selectivity ratio
    binding_mode: str  # 'competitive', 'allosteric', 'covalent'
    confidence: float = 0.0

class MolecularDrugBinding:
    """
    Simulates molecular-level drug-target binding using oscillatory principles.
    
    Models how drug molecules bind to protein targets through oscillatory
    coupling, conformational dynamics, and allosteric communication.
    """
    
    def __init__(self):
        """Initialize molecular binding simulator."""
        self.molecular_targets = self._load_molecular_targets()
        self.temperature = 310.0  # K - physiological temperature
        self.k_B = 1.38064852e-23  # Boltzmann constant (J/K)
        self.R = 8.314  # Gas constant (J/mol/K)
        
    def _load_molecular_targets(self) -> Dict[str, MolecularTarget]:
        """Load molecular targets for drug binding."""
        
        targets = {
            'inpp1_enzyme': MolecularTarget(
                target_id='inpp1_enzyme',
                protein_family='enzyme',
                molecular_weight=42.0,  # kDa
                binding_site_volume=1200.0,  # Ų
                backbone_flexibility=0.6,
                allosteric_sites=2,
                native_oscillation_frequency=7.23e13,
                binding_site_polarity=0.7  # Moderately polar active site
            ),
            
            'gsk3b_kinase': MolecularTarget(
                target_id='gsk3b_kinase',
                protein_family='kinase',
                molecular_weight=47.0,  # kDa
                binding_site_volume=800.0,  # Ų - ATP binding site
                backbone_flexibility=0.8,  # High flexibility for catalysis
                allosteric_sites=1,
                native_oscillation_frequency=2.15e13,
                binding_site_polarity=0.5  # Mixed polar/hydrophobic
            ),
            
            'drd2_gpcr': MolecularTarget(
                target_id='drd2_gpcr',
                protein_family='gpcr',
                molecular_weight=95.0,  # kDa (with transmembrane domains)
                binding_site_volume=600.0,  # Ų - orthosteric site
                backbone_flexibility=0.9,  # Very flexible GPCR
                allosteric_sites=3,  # Multiple allosteric sites
                native_oscillation_frequency=1.45e13,
                binding_site_polarity=0.3  # Mostly hydrophobic pocket
            ),
            
            'htr2a_gpcr': MolecularTarget(
                target_id='htr2a_gpcr', 
                protein_family='gpcr',
                molecular_weight=87.0,  # kDa
                binding_site_volume=550.0,  # Ų
                backbone_flexibility=0.85,
                allosteric_sites=2,
                native_oscillation_frequency=1.23e13,
                binding_site_polarity=0.4  # Mostly hydrophobic with polar contacts
            ),
            
            'cyp2d6_enzyme': MolecularTarget(
                target_id='cyp2d6_enzyme',
                protein_family='enzyme',
                molecular_weight=55.0,  # kDa
                binding_site_volume=2100.0,  # Ų - Large active site cavity
                backbone_flexibility=0.4,  # Rigid heme environment
                allosteric_sites=1,
                native_oscillation_frequency=3.8e13,
                binding_site_polarity=0.2  # Highly hydrophobic active site
            )
        }
        
        logger.info(f"Loaded {len(targets)} molecular targets")
        return targets
    
    def create_drug_molecules(self) -> Dict[str, DrugMolecule]:
        """Create molecular representations of pharmaceutical compounds."""
        
        molecules = {
            'lithium': DrugMolecule(
                drug_name='lithium',
                molecular_weight=73.89,  # Da - Li2CO3 formula unit
                volume=45.0,  # Ų - very small ion
                flexibility=0,  # No rotatable bonds (ion)
                hydrogen_bond_donors=0,
                hydrogen_bond_acceptors=0,
                charge_distribution=[1.0, -0.33, -0.33, -0.33],  # Li+, CO3(2-)
                oscillatory_signature=2.8e13,  # Hz
                lipophilicity=-2.3,  # Very hydrophilic
                shape_descriptor='ionic'
            ),
            
            'aripiprazole': DrugMolecule(
                drug_name='aripiprazole',
                molecular_weight=448.39,  # Da
                volume=380.0,  # Ų - large molecule
                flexibility=8,  # Multiple rotatable bonds
                hydrogen_bond_donors=2,
                hydrogen_bond_acceptors=4,
                charge_distribution=[0.15, -0.25, 0.10, -0.15, 0.20, -0.35],
                oscillatory_signature=1.85e13,  # Hz
                lipophilicity=4.3,  # Highly lipophilic
                shape_descriptor='branched'
            ),
            
            'citalopram': DrugMolecule(
                drug_name='citalopram',
                molecular_weight=324.39,  # Da
                volume=290.0,  # Ų
                flexibility=5,  # Moderate flexibility
                hydrogen_bond_donors=1,
                hydrogen_bond_acceptors=2,
                charge_distribution=[0.25, -0.20, 0.15, -0.10, -0.10],
                oscillatory_signature=2.15e13,  # Hz
                lipophilicity=3.5,  # Moderately lipophilic
                shape_descriptor='linear'
            ),
            
            'atorvastatin': DrugMolecule(
                drug_name='atorvastatin',
                molecular_weight=558.64,  # Da
                volume=450.0,  # Ų - very large molecule
                flexibility=11,  # Highly flexible
                hydrogen_bond_donors=4,
                hydrogen_bond_acceptors=7,
                charge_distribution=[-0.8, 0.1, 0.1, -0.2, 0.3, -0.15, 0.25, -0.4],
                oscillatory_signature=1.67e13,  # Hz
                lipophilicity=4.1,  # Lipophilic
                shape_descriptor='branched'
            ),
            
            'aspirin': DrugMolecule(
                drug_name='aspirin',
                molecular_weight=180.16,  # Da
                volume=140.0,  # Ų - small molecule
                flexibility=2,  # Low flexibility
                hydrogen_bond_donors=1,
                hydrogen_bond_acceptors=4,
                charge_distribution=[-0.6, 0.4, -0.2, 0.1, 0.3],
                oscillatory_signature=3.42e13,  # Hz
                lipophilicity=1.2,  # Moderately lipophilic
                shape_descriptor='cyclic'
            )
        }
        
        return molecules
    
    def calculate_shape_complementarity(self, drug: DrugMolecule, 
                                      target: MolecularTarget) -> float:
        """Calculate geometric shape complementarity between drug and target."""
        
        # Volume complementarity - drug should fit in binding site
        volume_ratio = drug.volume / target.binding_site_volume
        
        if volume_ratio > 1.0:
            # Drug too large for binding site
            volume_score = 1.0 / (1.0 + (volume_ratio - 1.0) * 2)
        else:
            # Drug smaller than binding site - good but not perfect
            volume_score = 0.7 + 0.3 * volume_ratio
        
        # Shape compatibility based on drug and target characteristics
        shape_compatibility = {
            ('ionic', 'enzyme'): 0.3,      # Ions don't fit well in enzyme sites
            ('ionic', 'gpcr'): 0.2,        # Ions don't bind GPCRs well
            ('linear', 'enzyme'): 0.8,     # Linear drugs fit enzyme channels
            ('linear', 'gpcr'): 0.6,       # Linear drugs partially fit GPCRs
            ('branched', 'enzyme'): 0.7,   # Branched drugs fit some enzymes
            ('branched', 'gpcr'): 0.9,     # Branched drugs fit GPCRs well
            ('cyclic', 'enzyme'): 0.9,     # Cyclic drugs fit enzyme pockets
            ('cyclic', 'gpcr'): 0.7        # Cyclic drugs partially fit GPCRs
        }
        
        shape_key = (drug.shape_descriptor, target.protein_family)
        shape_score = shape_compatibility.get(shape_key, 0.5)
        
        # Flexibility matching - flexible targets accommodate flexible drugs better
        flexibility_difference = abs(drug.flexibility / 10.0 - target.backbone_flexibility)
        flexibility_score = 1.0 - flexibility_difference
        
        # Overall shape complementarity
        overall_score = (volume_score * 0.5 + shape_score * 0.3 + 
                        flexibility_score * 0.2)
        
        return max(0.0, min(1.0, overall_score))
    
    def calculate_electrostatic_complementarity(self, drug: DrugMolecule,
                                              target: MolecularTarget) -> float:
        """Calculate electrostatic complementarity between drug and target."""
        
        # Drug polarity vs binding site polarity
        drug_polarity = self._estimate_drug_polarity(drug)
        polarity_difference = abs(drug_polarity - target.binding_site_polarity)
        polarity_score = 1.0 - polarity_difference
        
        # Hydrogen bonding potential
        # GPCRs and enzymes have different H-bonding requirements
        if target.protein_family == 'gpcr':
            # GPCRs prefer moderate H-bonding
            optimal_hbonds = 2
        elif target.protein_family == 'enzyme':
            # Enzymes can accommodate more H-bonds
            optimal_hbonds = 3
        else:
            optimal_hbonds = 2
        
        total_hbonds = drug.hydrogen_bond_donors + drug.hydrogen_bond_acceptors
        hbond_deviation = abs(total_hbonds - optimal_hbonds)
        hbond_score = 1.0 / (1.0 + hbond_deviation)
        
        # Charge distribution compatibility
        charge_variance = np.var(drug.charge_distribution)
        if target.protein_family == 'gpcr':
            # GPCRs prefer moderate charge distribution
            optimal_variance = 0.1
        else:
            # Enzymes can handle more charge variation
            optimal_variance = 0.15
            
        charge_score = 1.0 - abs(charge_variance - optimal_variance) / optimal_variance
        
        # Overall electrostatic complementarity
        overall_score = (polarity_score * 0.4 + hbond_score * 0.4 + 
                        charge_score * 0.2)
        
        return max(0.0, min(1.0, overall_score))
    
    def _estimate_drug_polarity(self, drug: DrugMolecule) -> float:
        """Estimate drug polarity from molecular properties."""
        
        # Lipophilicity contributes to polarity (inverse relationship)
        lipophilicity_component = (5.0 - drug.lipophilicity) / 5.0  # Normalize to 0-1
        lipophilicity_component = max(0.0, min(1.0, lipophilicity_component))
        
        # Hydrogen bonding contributes to polarity
        total_hbonds = drug.hydrogen_bond_donors + drug.hydrogen_bond_acceptors
        hbond_component = min(1.0, total_hbonds / 6.0)  # Normalize to 0-1
        
        # Charge distribution variance contributes to polarity
        charge_component = min(1.0, np.var(drug.charge_distribution) * 2)
        
        # Overall polarity estimate
        polarity = (lipophilicity_component * 0.5 + hbond_component * 0.3 + 
                   charge_component * 0.2)
        
        return polarity
    
    def calculate_oscillatory_resonance(self, drug: DrugMolecule,
                                       target: MolecularTarget) -> float:
        """Calculate oscillatory resonance between drug and target."""
        
        drug_freq = drug.oscillatory_signature
        target_freq = target.native_oscillation_frequency
        
        # Frequency difference (relative)
        freq_difference = abs(drug_freq - target_freq) / target_freq
        
        # Resonance strength (inverse relationship with frequency difference)
        resonance_strength = 1.0 / (1.0 + freq_difference)
        
        # Coupling efficiency depends on molecular properties
        # More flexible molecules couple better
        flexibility_factor = (drug.flexibility / 15.0 + target.backbone_flexibility) / 2.0
        flexibility_factor = min(1.0, flexibility_factor)
        
        # Size compatibility affects coupling
        size_ratio = min(drug.volume / target.binding_site_volume, 1.0)
        size_factor = 0.5 + 0.5 * size_ratio  # Larger drugs couple better
        
        # Overall oscillatory coupling
        coupling_strength = resonance_strength * flexibility_factor * size_factor
        
        return coupling_strength
    
    def calculate_binding_affinity(self, drug: DrugMolecule, 
                                 target: MolecularTarget) -> float:
        """Calculate binding affinity (Kd) in nM."""
        
        # Get complementarity scores
        shape_comp = self.calculate_shape_complementarity(drug, target)
        electrostatic_comp = self.calculate_electrostatic_complementarity(drug, target)
        oscillatory_coupling = self.calculate_oscillatory_resonance(drug, target)
        
        # Overall binding score
        binding_score = (shape_comp * 0.4 + electrostatic_comp * 0.4 + 
                        oscillatory_coupling * 0.2)
        
        # Convert binding score to affinity (Kd in nM)
        # High binding score = low Kd (high affinity)
        # Use exponential relationship
        kd_base = 10000.0  # nM - weak binding baseline
        kd = kd_base * np.exp(-5.0 * binding_score)
        
        return kd
    
    def calculate_binding_kinetics(self, drug: DrugMolecule, target: MolecularTarget,
                                 kd: float) -> Dict[str, float]:
        """Calculate binding kinetics (kon, koff rates)."""
        
        # Estimate kon (association rate) based on molecular properties
        # Smaller, more flexible molecules associate faster
        size_factor = 1000.0 / drug.molecular_weight  # Smaller = faster
        flexibility_factor = 1.0 + drug.flexibility / 10.0  # More flexible = faster
        
        # Base kon rate (M^-1 s^-1)
        kon_base = 1e6  # Typical protein-drug association rate
        kon = kon_base * size_factor * flexibility_factor
        
        # Calculate koff from Kd = koff/kon
        kd_molar = kd * 1e-9  # Convert nM to M
        koff = kd_molar * kon
        
        return {
            'kon': kon,      # M^-1 s^-1
            'koff': koff,    # s^-1
            'residence_time': 1.0 / koff  # seconds
        }
    
    def simulate_conformational_change(self, drug: DrugMolecule, 
                                     target: MolecularTarget,
                                     binding_strength: float) -> float:
        """Simulate drug-induced conformational change in target."""
        
        # Conformational change depends on:
        # 1. Target flexibility
        # 2. Drug binding strength  
        # 3. Oscillatory coupling
        
        oscillatory_coupling = self.calculate_oscillatory_resonance(drug, target)
        
        # Base conformational change amplitude
        base_change = binding_strength * target.backbone_flexibility
        
        # Oscillatory enhancement
        oscillatory_enhancement = 1.0 + oscillatory_coupling
        
        # Final conformational change amplitude
        conformational_amplitude = base_change * oscillatory_enhancement
        
        return min(1.0, conformational_amplitude)
    
    def simulate_allosteric_effects(self, drug: DrugMolecule, target: MolecularTarget,
                                  conformational_change: float) -> Dict[str, float]:
        """Simulate allosteric effects on other binding sites."""
        
        allosteric_effects = {}
        
        # Each allosteric site is affected by conformational changes
        for site_idx in range(target.allosteric_sites):
            site_id = f"allosteric_site_{site_idx + 1}"
            
            # Distance-dependent coupling (assume sites are distributed)
            coupling_strength = 0.5 + 0.5 * np.random.random()  # Random coupling
            
            # Effect magnitude proportional to conformational change
            effect_magnitude = conformational_change * coupling_strength
            
            # Effect can be positive or negative (cooperative or inhibitory)
            effect_sign = 1.0 if np.random.random() > 0.3 else -1.0
            
            allosteric_effects[site_id] = effect_magnitude * effect_sign
        
        return allosteric_effects
    
    def simulate_molecular_binding(self, drug: DrugMolecule, 
                                 target: MolecularTarget) -> BindingResult:
        """Simulate complete molecular binding interaction."""
        
        # Calculate binding affinity
        kd = self.calculate_binding_affinity(drug, target)
        
        # Calculate kinetics
        kinetics = self.calculate_binding_kinetics(drug, target, kd)
        
        # Calculate oscillatory coupling
        oscillatory_coupling = self.calculate_oscillatory_resonance(drug, target)
        
        # Simulate conformational change
        binding_strength = 1.0 / (1.0 + kd / 100.0)  # Strong binding = low Kd
        conformational_change = self.simulate_conformational_change(
            drug, target, binding_strength
        )
        
        # Simulate allosteric effects
        allosteric_effects = self.simulate_allosteric_effects(
            drug, target, conformational_change
        )
        
        # Determine binding mode
        binding_mode = self._determine_binding_mode(drug, target, kd)
        
        # Calculate selectivity (simplified)
        selectivity_profile = self._calculate_selectivity_profile(drug, target, kd)
        
        # Calculate confidence
        confidence = self._calculate_binding_confidence(drug, target, kd)
        
        return BindingResult(
            drug_name=drug.drug_name,
            target_id=target.target_id,
            binding_affinity=kd,
            binding_kinetics=kinetics,
            conformational_change=conformational_change,
            allosteric_effects=allosteric_effects,
            oscillatory_coupling_strength=oscillatory_coupling,
            residence_time=kinetics['residence_time'],
            selectivity_profile=selectivity_profile,
            binding_mode=binding_mode,
            confidence=confidence
        )
    
    def _determine_binding_mode(self, drug: DrugMolecule, target: MolecularTarget,
                               kd: float) -> str:
        """Determine the binding mode (competitive, allosteric, etc.)."""
        
        # Simple heuristic based on binding affinity and molecular properties
        if kd < 10.0:  # Very high affinity
            return 'competitive'  # Likely orthosteric site
        elif target.allosteric_sites > 1 and kd < 1000.0:
            return 'allosteric'   # Moderate affinity, allosteric site
        else:
            return 'competitive'  # Default to competitive
    
    def _calculate_selectivity_profile(self, drug: DrugMolecule, 
                                     current_target: MolecularTarget,
                                     current_kd: float) -> Dict[str, float]:
        """Calculate selectivity ratios against other targets."""
        
        selectivity = {}
        
        for target_id, target in self.molecular_targets.items():
            if target_id == current_target.target_id:
                continue
                
            # Estimate binding to other targets (simplified)
            other_kd = self.calculate_binding_affinity(drug, target)
            
            # Selectivity ratio = Kd_other / Kd_current
            # Higher ratio = more selective for current target
            selectivity_ratio = other_kd / current_kd
            selectivity[target_id] = selectivity_ratio
        
        return selectivity
    
    def _calculate_binding_confidence(self, drug: DrugMolecule, 
                                    target: MolecularTarget, kd: float) -> float:
        """Calculate confidence in binding prediction."""
        
        base_confidence = 0.6
        
        # Higher confidence for well-studied target families
        family_confidence = {
            'gpcr': 0.8,     # Well-studied
            'kinase': 0.9,   # Very well-studied
            'enzyme': 0.85,  # Well-studied
        }
        
        family_boost = family_confidence.get(target.protein_family, 0.5) - 0.5
        
        # Higher confidence for moderate binding affinities
        if 1.0 <= kd <= 1000.0:  # Reasonable binding range
            affinity_boost = 0.2
        elif kd > 10000.0:  # Very weak binding - less confident
            affinity_boost = -0.3
        else:  # Very strong binding - could be artifact
            affinity_boost = 0.1
        
        # Drug size affects confidence
        if 150 <= drug.molecular_weight <= 600:  # Drug-like size
            size_boost = 0.1
        else:
            size_boost = -0.1
        
        final_confidence = min(0.95, base_confidence + family_boost + 
                              affinity_boost + size_boost)
        
        return max(0.1, final_confidence)
    
    def simulate_drug_library(self) -> List[BindingResult]:
        """Simulate binding for all drug-target combinations."""
        
        drugs = self.create_drug_molecules()
        results = []
        
        for drug_name, drug in drugs.items():
            logger.info(f"Simulating molecular binding for {drug_name}")
            
            for target_id, target in self.molecular_targets.items():
                result = self.simulate_molecular_binding(drug, target)
                results.append(result)
        
        logger.info(f"Completed molecular binding simulation for {len(results)} combinations")
        return results
    
    def analyze_binding_patterns(self, results: List[BindingResult]) -> Dict[str, Any]:
        """Analyze patterns in molecular binding results."""
        
        analysis = {}
        
        # Affinity distribution
        affinities = [r.binding_affinity for r in results]
        analysis['binding_affinities'] = {
            'mean_kd': np.mean(affinities),
            'median_kd': np.median(affinities),
            'high_affinity_count': len([a for a in affinities if a < 100.0]),  # < 100 nM
            'weak_binding_count': len([a for a in affinities if a > 10000.0])  # > 10 μM
        }
        
        # Oscillatory coupling analysis
        couplings = [r.oscillatory_coupling_strength for r in results]
        analysis['oscillatory_coupling'] = {
            'mean': np.mean(couplings),
            'strong_coupling_count': len([c for c in couplings if c > 0.7])
        }
        
        # Conformational change analysis
        conf_changes = [r.conformational_change for r in results]
        analysis['conformational_changes'] = {
            'mean': np.mean(conf_changes),
            'significant_changes': len([c for c in conf_changes if c > 0.5])
        }
        
        # Drug-specific analysis
        drug_analysis = {}
        for result in results:
            if result.drug_name not in drug_analysis:
                drug_analysis[result.drug_name] = []
            drug_analysis[result.drug_name].append(result)
        
        analysis['drug_specific'] = {}
        for drug, drug_results in drug_analysis.items():
            best_affinity = min(r.binding_affinity for r in drug_results)
            avg_coupling = np.mean([r.oscillatory_coupling_strength for r in drug_results])
            
            analysis['drug_specific'][drug] = {
                'best_affinity_nM': best_affinity,
                'average_coupling': avg_coupling,
                'targets_bound': len([r for r in drug_results if r.binding_affinity < 1000.0])
            }
        
        return analysis
    
    def visualize_binding_results(self, results: List[BindingResult],
                                output_dir: str = "babylon_results") -> None:
        """Create visualizations of molecular binding results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive binding visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Binding affinity distribution
        affinities = [r.binding_affinity for r in results]
        
        axes[0, 0].hist(np.log10(affinities), bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_xlabel('Log10(Kd) [nM]')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Binding Affinity Distribution')
        axes[0, 0].axvline(x=2, color='red', linestyle='--', alpha=0.7, label='100 nM threshold')
        axes[0, 0].legend()
        
        # 2. Oscillatory coupling vs binding affinity
        couplings = [r.oscillatory_coupling_strength for r in results]
        
        scatter = axes[0, 1].scatter(np.log10(affinities), couplings, alpha=0.7, s=60)
        axes[0, 1].set_xlabel('Log10(Kd) [nM]')
        axes[0, 1].set_ylabel('Oscillatory Coupling Strength')
        axes[0, 1].set_title('Coupling vs Binding Affinity')
        
        # 3. Conformational change distribution
        conf_changes = [r.conformational_change for r in results]
        
        axes[0, 2].hist(conf_changes, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_xlabel('Conformational Change Amplitude')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Conformational Change Distribution')
        
        # 4. Drug-target binding heatmap
        drugs = list(set(r.drug_name for r in results))
        targets = list(set(r.target_id for r in results))
        
        binding_matrix = np.zeros((len(drugs), len(targets)))
        for i, drug in enumerate(drugs):
            for j, target in enumerate(targets):
                result = next((r for r in results if r.drug_name == drug and r.target_id == target), None)
                if result:
                    # Use negative log10 of Kd for heatmap (higher = stronger binding)
                    binding_matrix[i, j] = -np.log10(result.binding_affinity)
        
        im = axes[1, 0].imshow(binding_matrix, cmap='Reds', aspect='auto')
        axes[1, 0].set_xticks(range(len(targets)))
        axes[1, 0].set_xticklabels(targets, rotation=45)
        axes[1, 0].set_yticks(range(len(drugs)))
        axes[1, 0].set_yticklabels(drugs)
        axes[1, 0].set_title('Binding Affinity Heatmap')
        plt.colorbar(im, ax=axes[1, 0], label='-Log10(Kd)')
        
        # 5. Residence time analysis
        residence_times = [r.residence_time for r in results]
        drug_names = [r.drug_name for r in results]
        
        # Box plot by drug
        drug_residence = {}
        for drug in set(drug_names):
            drug_residence[drug] = [r.residence_time for r in results if r.drug_name == drug]
        
        axes[1, 1].boxplot([drug_residence[drug] for drug in drugs], labels=drugs)
        axes[1, 1].set_ylabel('Residence Time (seconds)')
        axes[1, 1].set_title('Drug Residence Times')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_yscale('log')
        
        # 6. Confidence scores
        confidences = [r.confidence for r in results]
        
        axes[1, 2].hist(confidences, bins=15, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 2].set_xlabel('Prediction Confidence')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Binding Prediction Confidence')
        axes[1, 2].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'molecular_binding_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Molecular binding visualizations saved to {output_path}")
    
    def save_results(self, results: List[BindingResult], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save molecular binding simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_data = []
        for result in results:
            results_data.append({
                'drug_name': result.drug_name,
                'target_id': result.target_id,
                'binding_affinity_nM': result.binding_affinity,
                'binding_kinetics': result.binding_kinetics,
                'conformational_change': result.conformational_change,
                'allosteric_effects': result.allosteric_effects,
                'oscillatory_coupling_strength': result.oscillatory_coupling_strength,
                'residence_time': result.residence_time,
                'selectivity_profile': result.selectivity_profile,
                'binding_mode': result.binding_mode,
                'confidence': result.confidence
            })
        
        with open(output_path / 'molecular_binding_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_data = []
        for result in results:
            csv_data.append({
                'drug_name': result.drug_name,
                'target_id': result.target_id,
                'binding_affinity_nM': result.binding_affinity,
                'log10_kd': np.log10(result.binding_affinity),
                'kon_M_per_s': result.binding_kinetics['kon'],
                'koff_per_s': result.binding_kinetics['koff'],
                'residence_time_s': result.residence_time,
                'conformational_change': result.conformational_change,
                'oscillatory_coupling': result.oscillatory_coupling_strength,
                'binding_mode': result.binding_mode,
                'confidence': result.confidence
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path / 'molecular_binding_results.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'molecular_binding_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Molecular binding results saved to {output_path}")

def main():
    """
    Test molecular drug binding simulation.
    
    This demonstrates how drugs bind to molecular targets through
    oscillatory coupling, conformational dynamics, and allosteric effects.
    """
    
    print("🧪 Testing Molecular Drug Binding")
    print("=" * 50)
    
    # Initialize molecular binding simulator
    simulator = MolecularDrugBinding()
    
    # Display loaded targets
    print(f"\n🎯 Loaded Molecular Targets:")
    for target_id, target in simulator.molecular_targets.items():
        print(f"  • {target_id}: {target.protein_family}")
        print(f"    MW: {target.molecular_weight:.1f} kDa")
        print(f"    Binding site: {target.binding_site_volume:.0f} Ų")
        print(f"    Flexibility: {target.backbone_flexibility:.1f}")
        print(f"    Frequency: {target.native_oscillation_frequency:.2e} Hz")
    
    # Create and display drug molecules
    drugs = simulator.create_drug_molecules()
    print(f"\n💊 Drug Molecules:")
    for drug_name, drug in drugs.items():
        print(f"  • {drug_name}: {drug.molecular_weight:.1f} Da")
        print(f"    Volume: {drug.volume:.0f} Ų")
        print(f"    Flexibility: {drug.flexibility} bonds")
        print(f"    LogP: {drug.lipophilicity:.1f}")
        print(f"    Frequency: {drug.oscillatory_signature:.2e} Hz")
    
    # Simulate binding for all combinations
    print(f"\n🔬 Simulating molecular drug-target binding...")
    results = simulator.simulate_drug_library()
    
    # Display results
    print(f"\n📊 MOLECULAR BINDING RESULTS:")
    print("-" * 70)
    
    # Sort results by binding affinity
    results_sorted = sorted(results, key=lambda r: r.binding_affinity)
    
    for result in results_sorted:
        print(f"{result.drug_name.upper()} → {result.target_id.upper()}:")
        print(f"  Binding affinity: {result.binding_affinity:.1f} nM")
        print(f"  Residence time: {result.residence_time:.2e} seconds")
        print(f"  Conformational change: {result.conformational_change:.3f}")
        print(f"  Oscillatory coupling: {result.oscillatory_coupling_strength:.3f}")
        print(f"  Binding mode: {result.binding_mode}")
        print(f"  Confidence: {result.confidence:.3f}")
        
        # Show kinetics
        print(f"  Kinetics: kon={result.binding_kinetics['kon']:.2e} M⁻¹s⁻¹, "
              f"koff={result.binding_kinetics['koff']:.2e} s⁻¹")
        print()
    
    # Analyze patterns
    print("🔍 Analyzing molecular binding patterns...")
    analysis = simulator.analyze_binding_patterns(results)
    
    print(f"\n📈 BINDING PATTERN ANALYSIS:")
    print("-" * 40)
    
    affinity_analysis = analysis['binding_affinities']
    print(f"Binding Affinities:")
    print(f"  • Mean Kd: {affinity_analysis['mean_kd']:.1f} nM")
    print(f"  • Median Kd: {affinity_analysis['median_kd']:.1f} nM")
    print(f"  • High affinity (< 100 nM): {affinity_analysis['high_affinity_count']}")
    print(f"  • Weak binding (> 10 μM): {affinity_analysis['weak_binding_count']}")
    
    coupling_analysis = analysis['oscillatory_coupling']
    print(f"\nOscillatory Coupling:")
    print(f"  • Mean coupling: {coupling_analysis['mean']:.3f}")
    print(f"  • Strong coupling (> 0.7): {coupling_analysis['strong_coupling_count']}")
    
    conf_analysis = analysis['conformational_changes']
    print(f"\nConformational Changes:")
    print(f"  • Mean amplitude: {conf_analysis['mean']:.3f}")
    print(f"  • Significant changes (> 0.5): {conf_analysis['significant_changes']}")
    
    print(f"\nDrug-Specific Analysis:")
    for drug, drug_stats in analysis['drug_specific'].items():
        print(f"  • {drug}: Best Kd = {drug_stats['best_affinity_nM']:.1f} nM, "
              f"{drug_stats['targets_bound']} targets bound")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    simulator.save_results(results, analysis)
    simulator.visualize_binding_results(results)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find strongest binding
    best_binding = min(results, key=lambda r: r.binding_affinity)
    print(f"• Strongest binding: {best_binding.drug_name} → {best_binding.target_id} "
          f"(Kd = {best_binding.binding_affinity:.1f} nM)")
    
    # Find best oscillatory coupling
    best_coupling = max(results, key=lambda r: r.oscillatory_coupling_strength)
    print(f"• Best oscillatory coupling: {best_coupling.drug_name} → {best_coupling.target_id} "
          f"({best_coupling.oscillatory_coupling_strength:.3f})")
    
    # Find longest residence time
    longest_residence = max(results, key=lambda r: r.residence_time)
    print(f"• Longest residence time: {longest_residence.drug_name} → {longest_residence.target_id} "
          f"({longest_residence.residence_time:.2e} seconds)")
    
    # Count high-confidence predictions
    high_conf = len([r for r in results if r.confidence > 0.8])
    print(f"• High-confidence predictions: {high_conf}/{len(results)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Molecular drug binding simulation complete!")
    
    return results

if __name__ == "__main__":
    results = main()

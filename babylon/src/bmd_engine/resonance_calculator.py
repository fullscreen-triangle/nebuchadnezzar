"""
Resonance Calculator Module
==========================

Computes detailed frequency matching quality, resonance patterns, and oscillatory
coupling strength for drug-target and drug-hole interactions. Provides precise
calculations for the BMD (Biological Maxwell Demon) therapeutic framework.

Part of the St. Stellas BMD Engine for therapeutic prediction.
"""

import numpy as np
import scipy.signal as signal
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FrequencyComponent:
    """Represents a single frequency component in oscillatory analysis."""
    frequency: float         # Hz - frequency value
    amplitude: float         # Relative amplitude (0-1)
    phase: float            # Phase angle in radians
    bandwidth: float        # Hz - frequency spread
    coherence: float        # 0-1 - phase coherence
    decay_rate: float       # 1/s - amplitude decay rate

@dataclass
class OscillatorySpectrum:
    """Complete oscillatory spectrum of a biological system or drug."""
    entity_id: str
    entity_type: str        # 'drug', 'pathway', 'hole', 'target'
    
    # Primary spectrum components
    fundamental_frequency: float    # Hz - main frequency
    harmonics: List[FrequencyComponent] = field(default_factory=list)
    subharmonics: List[FrequencyComponent] = field(default_factory=list)
    
    # Spectrum characteristics
    bandwidth: float = 0.0          # Hz - total spectral width
    spectral_density: float = 0.0   # Power distribution
    noise_floor: float = 0.01       # Background noise level
    
    # Temporal characteristics
    stability: float = 1.0          # 0-1 - frequency stability over time
    variability: float = 0.0        # 0-1 - amplitude variability
    
    # Coupling properties
    coupling_strength: float = 1.0  # 0-1 - ability to couple with others
    coupling_selectivity: float = 0.5  # 0-1 - selectivity of coupling

@dataclass
class ResonanceAnalysis:
    """Detailed analysis of resonance between two oscillatory entities."""
    entity1_id: str
    entity2_id: str
    
    # Frequency matching
    frequency_match_score: float    # 0-1 - overall frequency compatibility
    fundamental_match: float        # Direct fundamental frequency match
    harmonic_matches: Dict[str, float] = field(default_factory=dict)  # Harmonic resonances
    subharmonic_matches: Dict[str, float] = field(default_factory=dict)  # Subharmonic resonances
    
    # Resonance characteristics
    resonance_type: str = "none"    # Primary resonance mechanism
    resonance_strength: float = 0.0  # 0-1 - strength of resonance
    quality_factor: float = 1.0     # Resonance sharpness (Q factor)
    
    # Phase relationships
    phase_coherence: float = 0.0    # 0-1 - phase alignment
    phase_lock_probability: float = 0.0  # Probability of phase locking
    
    # Energy transfer
    energy_transfer_efficiency: float = 0.0  # 0-1 - energy coupling efficiency
    power_amplification: float = 1.0        # Power amplification factor
    
    # Stability and dynamics
    coupling_stability: float = 1.0  # 0-1 - stability of coupling
    bandwidth_overlap: float = 0.0   # Spectral overlap
    
    # Therapeutic relevance
    therapeutic_potential: float = 0.0  # 0-1 - potential for therapeutic effect

class ResonanceCalculator:
    """
    Advanced calculator for oscillatory resonance and frequency matching.
    
    Computes detailed resonance patterns, coupling strengths, and therapeutic
    potential for the BMD framework.
    """
    
    def __init__(self):
        """Initialize resonance calculator."""
        
        # Calculation parameters
        self.frequency_tolerance = 0.05     # ±5% for direct matching
        self.harmonic_orders = [2, 3, 4, 5, 6, 7, 8]  # Harmonic orders to check
        self.subharmonic_orders = [2, 3, 4, 5]        # Subharmonic orders
        
        # Quality thresholds
        self.min_resonance_strength = 0.1
        self.high_quality_threshold = 0.7
        self.phase_lock_threshold = 0.8
        
        logger.info("Resonance calculator initialized")
    
    def create_spectrum_from_drug(self, drug_data: Dict[str, Any]) -> OscillatorySpectrum:
        """Create oscillatory spectrum from drug properties."""
        
        drug_name = drug_data.get('name', 'unknown_drug')
        primary_freq = float(drug_data.get('primary_frequency', 1e13))
        secondary_freqs = drug_data.get('secondary_frequencies', [])
        amplitude = float(drug_data.get('oscillatory_amplitude', 0.5))
        bandwidth = float(drug_data.get('frequency_bandwidth', primary_freq * 0.1))
        
        # Create harmonics from secondary frequencies
        harmonics = []
        for i, freq in enumerate(secondary_freqs[:5]):  # Limit to 5 harmonics
            harmonic = FrequencyComponent(
                frequency=float(freq),
                amplitude=amplitude * (0.8 ** i),  # Decreasing amplitude
                phase=np.random.uniform(0, 2*np.pi),  # Random phase
                bandwidth=bandwidth * 0.5,
                coherence=0.8 - i * 0.1,  # Decreasing coherence
                decay_rate=1.0 / 3600.0   # 1-hour decay time
            )
            harmonics.append(harmonic)
        
        return OscillatorySpectrum(
            entity_id=drug_name,
            entity_type='drug',
            fundamental_frequency=primary_freq,
            harmonics=harmonics,
            bandwidth=bandwidth,
            spectral_density=amplitude,
            coupling_strength=amplitude,
            coupling_selectivity=0.6,
            stability=0.9,
            variability=0.1
        )
    
    def create_spectrum_from_hole(self, hole_data: Dict[str, Any]) -> OscillatorySpectrum:
        """Create oscillatory spectrum from oscillatory hole."""
        
        hole_id = hole_data.get('hole_id', 'unknown_hole')
        frequency = float(hole_data.get('frequency', 1e13))
        amplitude_deficit = float(hole_data.get('amplitude_deficit', 0.5))
        bandwidth = float(hole_data.get('bandwidth', frequency * 0.1))
        
        # Holes have "negative" amplitude (deficit)
        deficit_amplitude = -amplitude_deficit  # Negative indicates absence
        
        # Create hole spectrum with characteristic properties
        return OscillatorySpectrum(
            entity_id=hole_id,
            entity_type='hole',
            fundamental_frequency=frequency,
            harmonics=[],  # Holes typically don't have harmonics
            bandwidth=bandwidth,
            spectral_density=abs(deficit_amplitude),  # Use absolute value for density
            coupling_strength=amplitude_deficit,  # Stronger deficit = stronger coupling need
            coupling_selectivity=0.8,  # Holes are selective
            stability=0.95,  # Holes are usually stable
            variability=0.05
        )
    
    def calculate_fundamental_resonance(self, spectrum1: OscillatorySpectrum,
                                      spectrum2: OscillatorySpectrum) -> Tuple[float, Dict[str, Any]]:
        """Calculate resonance between fundamental frequencies."""
        
        freq1 = spectrum1.fundamental_frequency
        freq2 = spectrum2.fundamental_frequency
        
        # Frequency difference (relative)
        freq_diff = abs(freq1 - freq2) / max(freq1, freq2)
        
        # Direct resonance score
        if freq_diff <= self.frequency_tolerance:
            # Within tolerance - high resonance
            resonance_score = 1.0 - (freq_diff / self.frequency_tolerance) * 0.2
            resonance_type = "direct"
        else:
            # Outside tolerance - exponential decay
            resonance_score = np.exp(-10 * (freq_diff - self.frequency_tolerance))
            resonance_type = "weak_coupling"
        
        # Quality factor calculation
        q_factor = self._calculate_quality_factor(spectrum1, spectrum2, freq1, freq2)
        
        # Phase coherence estimation
        phase_coherence = self._estimate_phase_coherence(spectrum1, spectrum2, resonance_score)
        
        details = {
            'frequency_difference': freq_diff,
            'resonance_type': resonance_type,
            'quality_factor': q_factor,
            'phase_coherence': phase_coherence,
            'frequencies': (freq1, freq2)
        }
        
        return resonance_score, details
    
    def calculate_harmonic_resonances(self, spectrum1: OscillatorySpectrum,
                                    spectrum2: OscillatorySpectrum) -> Dict[str, float]:
        """Calculate harmonic resonance matches."""
        
        harmonic_matches = {}
        
        freq1 = spectrum1.fundamental_frequency
        freq2 = spectrum2.fundamental_frequency
        
        # Check each harmonic order
        for order in self.harmonic_orders:
            # Spectrum1 fundamental to Spectrum2 harmonic
            harmonic_freq2 = freq2 * order
            freq_diff = abs(freq1 - harmonic_freq2) / max(freq1, harmonic_freq2)
            
            if freq_diff <= self.frequency_tolerance * 2:  # Relaxed tolerance for harmonics
                resonance_strength = (1.0 - freq_diff / (self.frequency_tolerance * 2)) / order
                harmonic_matches[f"1_to_2_harmonic_{order}"] = resonance_strength
            
            # Spectrum2 fundamental to Spectrum1 harmonic
            harmonic_freq1 = freq1 * order
            freq_diff = abs(freq2 - harmonic_freq1) / max(freq2, harmonic_freq1)
            
            if freq_diff <= self.frequency_tolerance * 2:
                resonance_strength = (1.0 - freq_diff / (self.frequency_tolerance * 2)) / order
                harmonic_matches[f"2_to_1_harmonic_{order}"] = resonance_strength
        
        # Check harmonics within each spectrum
        for i, harm1 in enumerate(spectrum1.harmonics):
            for j, harm2 in enumerate(spectrum2.harmonics):
                freq_diff = abs(harm1.frequency - harm2.frequency) / max(harm1.frequency, harm2.frequency)
                
                if freq_diff <= self.frequency_tolerance:
                    resonance_strength = (1.0 - freq_diff / self.frequency_tolerance)
                    resonance_strength *= (harm1.amplitude * harm2.amplitude)  # Weight by amplitudes
                    harmonic_matches[f"harmonic_{i}_to_{j}"] = resonance_strength
        
        return harmonic_matches
    
    def calculate_subharmonic_resonances(self, spectrum1: OscillatorySpectrum,
                                       spectrum2: OscillatorySpectrum) -> Dict[str, float]:
        """Calculate subharmonic resonance matches."""
        
        subharmonic_matches = {}
        
        freq1 = spectrum1.fundamental_frequency
        freq2 = spectrum2.fundamental_frequency
        
        # Check each subharmonic order
        for order in self.subharmonic_orders:
            # Spectrum1 fundamental to Spectrum2 subharmonic
            subharmonic_freq1 = freq1 / order
            freq_diff = abs(subharmonic_freq1 - freq2) / max(subharmonic_freq1, freq2)
            
            if freq_diff <= self.frequency_tolerance * 2:  # Relaxed tolerance
                resonance_strength = (1.0 - freq_diff / (self.frequency_tolerance * 2)) / (order * 1.5)
                subharmonic_matches[f"1_sub_{order}_to_2"] = resonance_strength
            
            # Spectrum2 fundamental to Spectrum1 subharmonic
            subharmonic_freq2 = freq2 / order
            freq_diff = abs(subharmonic_freq2 - freq1) / max(subharmonic_freq2, freq1)
            
            if freq_diff <= self.frequency_tolerance * 2:
                resonance_strength = (1.0 - freq_diff / (self.frequency_tolerance * 2)) / (order * 1.5)
                subharmonic_matches[f"2_sub_{order}_to_1"] = resonance_strength
        
        return subharmonic_matches
    
    def calculate_coupling_strength(self, spectrum1: OscillatorySpectrum,
                                  spectrum2: OscillatorySpectrum,
                                  resonance_score: float) -> Dict[str, float]:
        """Calculate coupling strength between two oscillatory entities."""
        
        # Base coupling from resonance
        base_coupling = resonance_score
        
        # Coupling capacity of each entity
        coupling1 = spectrum1.coupling_strength
        coupling2 = spectrum2.coupling_strength
        
        # Effective coupling is minimum of the two (limiting factor)
        coupling_capacity = min(coupling1, coupling2)
        
        # Bandwidth overlap affects coupling
        bandwidth_overlap = self._calculate_bandwidth_overlap(spectrum1, spectrum2)
        
        # Coupling selectivity affects matching
        selectivity_factor = (spectrum1.coupling_selectivity + spectrum2.coupling_selectivity) / 2
        
        # Overall coupling strength
        total_coupling = base_coupling * coupling_capacity * bandwidth_overlap * selectivity_factor
        
        # Energy transfer efficiency
        energy_efficiency = self._calculate_energy_transfer_efficiency(
            spectrum1, spectrum2, total_coupling
        )
        
        # Power amplification for hole filling
        power_amplification = self._calculate_power_amplification(
            spectrum1, spectrum2, total_coupling
        )
        
        return {
            'total_coupling': total_coupling,
            'energy_efficiency': energy_efficiency,
            'power_amplification': power_amplification,
            'bandwidth_overlap': bandwidth_overlap,
            'coupling_capacity': coupling_capacity
        }
    
    def perform_comprehensive_resonance_analysis(self, spectrum1: OscillatorySpectrum,
                                               spectrum2: OscillatorySpectrum) -> ResonanceAnalysis:
        """Perform complete resonance analysis between two spectra."""
        
        logger.debug(f"Analyzing resonance between {spectrum1.entity_id} and {spectrum2.entity_id}")
        
        # Fundamental resonance
        fundamental_score, fundamental_details = self.calculate_fundamental_resonance(spectrum1, spectrum2)
        
        # Harmonic resonances
        harmonic_matches = self.calculate_harmonic_resonances(spectrum1, spectrum2)
        
        # Subharmonic resonances
        subharmonic_matches = self.calculate_subharmonic_resonances(spectrum1, spectrum2)
        
        # Overall frequency matching score
        frequency_match_score = self._calculate_overall_frequency_match(
            fundamental_score, harmonic_matches, subharmonic_matches
        )
        
        # Determine primary resonance type
        resonance_type, resonance_strength = self._determine_primary_resonance(
            fundamental_score, harmonic_matches, subharmonic_matches, fundamental_details
        )
        
        # Coupling analysis
        coupling_data = self.calculate_coupling_strength(spectrum1, spectrum2, frequency_match_score)
        
        # Phase relationships
        phase_coherence = fundamental_details.get('phase_coherence', 0.0)
        phase_lock_prob = self._calculate_phase_lock_probability(
            resonance_strength, phase_coherence, coupling_data['total_coupling']
        )
        
        # Stability analysis
        coupling_stability = self._calculate_coupling_stability(spectrum1, spectrum2, resonance_strength)
        
        # Therapeutic potential
        therapeutic_potential = self._calculate_therapeutic_potential(
            spectrum1, spectrum2, frequency_match_score, coupling_data, resonance_strength
        )
        
        return ResonanceAnalysis(
            entity1_id=spectrum1.entity_id,
            entity2_id=spectrum2.entity_id,
            frequency_match_score=frequency_match_score,
            fundamental_match=fundamental_score,
            harmonic_matches=harmonic_matches,
            subharmonic_matches=subharmonic_matches,
            resonance_type=resonance_type,
            resonance_strength=resonance_strength,
            quality_factor=fundamental_details.get('quality_factor', 1.0),
            phase_coherence=phase_coherence,
            phase_lock_probability=phase_lock_prob,
            energy_transfer_efficiency=coupling_data['energy_efficiency'],
            power_amplification=coupling_data['power_amplification'],
            coupling_stability=coupling_stability,
            bandwidth_overlap=coupling_data['bandwidth_overlap'],
            therapeutic_potential=therapeutic_potential
        )
    
    def _calculate_quality_factor(self, spectrum1: OscillatorySpectrum,
                                spectrum2: OscillatorySpectrum,
                                freq1: float, freq2: float) -> float:
        """Calculate quality factor of the resonance."""
        
        # Average bandwidth
        avg_bandwidth = (spectrum1.bandwidth + spectrum2.bandwidth) / 2
        
        # Average frequency
        avg_frequency = (freq1 + freq2) / 2
        
        # Q factor = frequency / bandwidth
        q_factor = avg_frequency / avg_bandwidth if avg_bandwidth > 0 else 1.0
        
        # Normalize to reasonable range
        return min(100.0, max(1.0, q_factor))
    
    def _estimate_phase_coherence(self, spectrum1: OscillatorySpectrum,
                                spectrum2: OscillatorySpectrum,
                                resonance_score: float) -> float:
        """Estimate phase coherence between oscillators."""
        
        # Base coherence from individual stabilities
        individual_coherence = (spectrum1.stability + spectrum2.stability) / 2
        
        # Resonance enhances coherence
        resonance_enhancement = resonance_score * 0.3
        
        # Coupling strength affects coherence
        coupling_effect = min(spectrum1.coupling_strength, spectrum2.coupling_strength) * 0.2
        
        total_coherence = individual_coherence + resonance_enhancement + coupling_effect
        
        return min(1.0, max(0.0, total_coherence))
    
    def _calculate_bandwidth_overlap(self, spectrum1: OscillatorySpectrum,
                                   spectrum2: OscillatorySpectrum) -> float:
        """Calculate spectral bandwidth overlap."""
        
        freq1 = spectrum1.fundamental_frequency
        freq2 = spectrum2.fundamental_frequency
        bw1 = spectrum1.bandwidth
        bw2 = spectrum2.bandwidth
        
        # Frequency ranges
        range1_start = freq1 - bw1/2
        range1_end = freq1 + bw1/2
        range2_start = freq2 - bw2/2
        range2_end = freq2 + bw2/2
        
        # Calculate overlap
        overlap_start = max(range1_start, range2_start)
        overlap_end = min(range1_end, range2_end)
        
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            total_span = max(range1_end, range2_end) - min(range1_start, range2_start)
            return overlap / total_span if total_span > 0 else 0.0
        else:
            return 0.0
    
    def _calculate_energy_transfer_efficiency(self, spectrum1: OscillatorySpectrum,
                                            spectrum2: OscillatorySpectrum,
                                            coupling_strength: float) -> float:
        """Calculate energy transfer efficiency."""
        
        # Efficiency depends on coupling strength and impedance matching
        impedance_match = self._calculate_impedance_matching(spectrum1, spectrum2)
        
        # Base efficiency from coupling
        base_efficiency = coupling_strength * 0.8
        
        # Impedance matching enhancement
        efficiency = base_efficiency * impedance_match
        
        return min(1.0, max(0.0, efficiency))
    
    def _calculate_impedance_matching(self, spectrum1: OscillatorySpectrum,
                                    spectrum2: OscillatorySpectrum) -> float:
        """Calculate impedance matching between oscillators."""
        
        # Simplified impedance based on spectral properties
        impedance1 = spectrum1.spectral_density * spectrum1.bandwidth
        impedance2 = spectrum2.spectral_density * spectrum2.bandwidth
        
        # Perfect matching when impedances are equal
        impedance_ratio = min(impedance1, impedance2) / max(impedance1, impedance2)
        
        return impedance_ratio
    
    def _calculate_power_amplification(self, spectrum1: OscillatorySpectrum,
                                     spectrum2: OscillatorySpectrum,
                                     coupling_strength: float) -> float:
        """Calculate power amplification factor."""
        
        # For drug-hole interactions, drug provides power to fill hole
        if spectrum1.entity_type == 'drug' and spectrum2.entity_type == 'hole':
            # Drug power available
            drug_power = spectrum1.spectral_density * spectrum1.coupling_strength
            
            # Hole power deficit
            hole_deficit = spectrum2.spectral_density  # Positive value of deficit
            
            # Amplification is limited by coupling efficiency
            max_amplification = drug_power / hole_deficit if hole_deficit > 0 else 1.0
            
            # Actual amplification considering coupling losses
            actual_amplification = max_amplification * coupling_strength
            
            return min(10.0, max(0.1, actual_amplification))  # Limit to reasonable range
        
        # For other interactions, amplification is more modest
        return 1.0 + coupling_strength * 0.5
    
    def _calculate_overall_frequency_match(self, fundamental_score: float,
                                         harmonic_matches: Dict[str, float],
                                         subharmonic_matches: Dict[str, float]) -> float:
        """Calculate overall frequency matching score."""
        
        # Start with fundamental match (highest weight)
        overall_score = fundamental_score * 0.6
        
        # Add best harmonic match
        if harmonic_matches:
            best_harmonic = max(harmonic_matches.values())
            overall_score += best_harmonic * 0.3
        
        # Add best subharmonic match
        if subharmonic_matches:
            best_subharmonic = max(subharmonic_matches.values())
            overall_score += best_subharmonic * 0.1
        
        return min(1.0, overall_score)
    
    def _determine_primary_resonance(self, fundamental_score: float,
                                   harmonic_matches: Dict[str, float],
                                   subharmonic_matches: Dict[str, float],
                                   fundamental_details: Dict[str, Any]) -> Tuple[str, float]:
        """Determine the primary resonance mechanism and its strength."""
        
        # Find the strongest resonance
        best_score = fundamental_score
        best_type = fundamental_details.get('resonance_type', 'direct')
        
        # Check harmonics
        if harmonic_matches:
            best_harmonic_score = max(harmonic_matches.values())
            if best_harmonic_score > best_score:
                best_score = best_harmonic_score
                # Find which harmonic
                best_harmonic_key = max(harmonic_matches, key=harmonic_matches.get)
                best_type = f"harmonic_{best_harmonic_key}"
        
        # Check subharmonics
        if subharmonic_matches:
            best_subharmonic_score = max(subharmonic_matches.values())
            if best_subharmonic_score > best_score:
                best_score = best_subharmonic_score
                # Find which subharmonic
                best_subharmonic_key = max(subharmonic_matches, key=subharmonic_matches.get)
                best_type = f"subharmonic_{best_subharmonic_key}"
        
        return best_type, best_score
    
    def _calculate_phase_lock_probability(self, resonance_strength: float,
                                        phase_coherence: float,
                                        coupling_strength: float) -> float:
        """Calculate probability of phase locking."""
        
        # Phase locking depends on all three factors
        base_probability = resonance_strength * phase_coherence * coupling_strength
        
        # Threshold behavior - needs minimum coupling for phase lock
        if coupling_strength > 0.3:
            threshold_factor = 1.0
        else:
            threshold_factor = coupling_strength / 0.3
        
        probability = base_probability * threshold_factor
        
        return min(1.0, max(0.0, probability))
    
    def _calculate_coupling_stability(self, spectrum1: OscillatorySpectrum,
                                    spectrum2: OscillatorySpectrum,
                                    resonance_strength: float) -> float:
        """Calculate stability of the coupling."""
        
        # Individual stabilities
        stability1 = spectrum1.stability
        stability2 = spectrum2.stability
        
        # Variabilities reduce stability
        variability1 = spectrum1.variability
        variability2 = spectrum2.variability
        
        # Base stability is minimum of individual stabilities
        base_stability = min(stability1, stability2)
        
        # Variability reduces stability
        variability_effect = 1.0 - max(variability1, variability2)
        
        # Strong resonance can stabilize the coupling
        resonance_stabilization = resonance_strength * 0.2
        
        total_stability = base_stability * variability_effect + resonance_stabilization
        
        return min(1.0, max(0.0, total_stability))
    
    def _calculate_therapeutic_potential(self, spectrum1: OscillatorySpectrum,
                                       spectrum2: OscillatorySpectrum,
                                       frequency_match: float,
                                       coupling_data: Dict[str, float],
                                       resonance_strength: float) -> float:
        """Calculate therapeutic potential of the interaction."""
        
        # For drug-hole interactions
        if ((spectrum1.entity_type == 'drug' and spectrum2.entity_type == 'hole') or
            (spectrum1.entity_type == 'hole' and spectrum2.entity_type == 'drug')):
            
            # Therapeutic potential depends on ability to fill the hole
            filling_potential = frequency_match * coupling_data['power_amplification'] / 10.0
            filling_potential = min(1.0, filling_potential)
            
            # Energy transfer efficiency is crucial
            efficiency_factor = coupling_data['energy_efficiency']
            
            # Resonance strength provides amplification
            resonance_factor = 1.0 + resonance_strength * 0.5
            
            therapeutic_score = filling_potential * efficiency_factor * resonance_factor
            
            return min(1.0, max(0.0, therapeutic_score))
        
        # For other interactions, therapeutic potential is lower
        return frequency_match * resonance_strength * 0.5
    
    def analyze_multi_entity_resonance(self, spectra: List[OscillatorySpectrum]) -> Dict[str, ResonanceAnalysis]:
        """Analyze resonance patterns among multiple entities."""
        
        resonance_map = {}
        
        # Analyze all pairwise interactions
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                spectrum1 = spectra[i]
                spectrum2 = spectra[j]
                
                analysis = self.perform_comprehensive_resonance_analysis(spectrum1, spectrum2)
                
                pair_key = f"{spectrum1.entity_id}_to_{spectrum2.entity_id}"
                resonance_map[pair_key] = analysis
        
        return resonance_map
    
    def optimize_frequency_for_maximum_resonance(self, target_spectrum: OscillatorySpectrum,
                                               reference_spectrum: OscillatorySpectrum) -> Dict[str, Any]:
        """Find optimal frequency adjustment for maximum resonance."""
        
        def resonance_objective(freq_multiplier):
            # Create modified spectrum with adjusted frequency
            modified_spectrum = OscillatorySpectrum(
                entity_id=f"{target_spectrum.entity_id}_modified",
                entity_type=target_spectrum.entity_type,
                fundamental_frequency=target_spectrum.fundamental_frequency * freq_multiplier,
                harmonics=target_spectrum.harmonics,
                bandwidth=target_spectrum.bandwidth,
                spectral_density=target_spectrum.spectral_density,
                coupling_strength=target_spectrum.coupling_strength,
                coupling_selectivity=target_spectrum.coupling_selectivity,
                stability=target_spectrum.stability,
                variability=target_spectrum.variability
            )
            
            # Calculate resonance
            analysis = self.perform_comprehensive_resonance_analysis(modified_spectrum, reference_spectrum)
            
            # Return negative for minimization
            return -analysis.frequency_match_score
        
        # Optimize frequency multiplier
        result = minimize_scalar(resonance_objective, bounds=(0.5, 2.0), method='bounded')
        
        optimal_multiplier = result.x
        optimal_frequency = target_spectrum.fundamental_frequency * optimal_multiplier
        max_resonance = -result.fun
        
        return {
            'optimal_frequency': optimal_frequency,
            'optimal_multiplier': optimal_multiplier,
            'max_resonance_score': max_resonance,
            'frequency_shift': optimal_frequency - target_spectrum.fundamental_frequency
        }

def main():
    """
    Test the resonance calculator functionality.
    
    Demonstrates detailed frequency matching analysis and resonance calculations
    for drug-target and drug-hole interactions.
    """
    
    print("🎵 Testing Resonance Calculator")
    print("=" * 50)
    
    # Initialize calculator
    calc = ResonanceCalculator()
    
    # Create test spectra
    print("\n📊 Creating test oscillatory spectra...")
    
    # Drug spectrum (Lithium)
    lithium_data = {
        'name': 'lithium',
        'primary_frequency': 2.8e13,
        'secondary_frequencies': [7.23e13, 2.15e13],
        'oscillatory_amplitude': 0.8,
        'frequency_bandwidth': 5e12
    }
    
    lithium_spectrum = calc.create_spectrum_from_drug(lithium_data)
    
    # Drug spectrum (Aripiprazole)
    aripiprazole_data = {
        'name': 'aripiprazole',
        'primary_frequency': 1.85e13,
        'secondary_frequencies': [1.23e13, 2.1e13],
        'oscillatory_amplitude': 0.65,
        'frequency_bandwidth': 3e12
    }
    
    aripiprazole_spectrum = calc.create_spectrum_from_drug(aripiprazole_data)
    
    # Oscillatory hole (Inositol pathway)
    hole_data = {
        'hole_id': 'inositol_hole_001',
        'frequency': 7.23e13,  # INPP1 frequency
        'amplitude_deficit': 0.7,
        'bandwidth': 5e12
    }
    
    hole_spectrum = calc.create_spectrum_from_hole(hole_data)
    
    print(f"Created spectra:")
    print(f"  • Lithium: {lithium_spectrum.fundamental_frequency:.2e} Hz")
    print(f"  • Aripiprazole: {aripiprazole_spectrum.fundamental_frequency:.2e} Hz")
    print(f"  • Inositol hole: {hole_spectrum.fundamental_frequency:.2e} Hz")
    
    # Test fundamental resonance calculation
    print(f"\n🔍 Testing fundamental resonance calculations...")
    
    # Lithium vs Inositol hole
    fund_score, fund_details = calc.calculate_fundamental_resonance(lithium_spectrum, hole_spectrum)
    
    print(f"\nLithium vs Inositol Hole (Fundamental):")
    print(f"  Resonance score: {fund_score:.3f}")
    print(f"  Resonance type: {fund_details['resonance_type']}")
    print(f"  Frequency difference: {fund_details['frequency_difference']:.3f}")
    print(f"  Quality factor: {fund_details['quality_factor']:.2f}")
    print(f"  Phase coherence: {fund_details['phase_coherence']:.3f}")
    
    # Test harmonic resonance
    print(f"\n🎼 Testing harmonic resonance analysis...")
    
    harmonic_matches = calc.calculate_harmonic_resonances(lithium_spectrum, hole_spectrum)
    
    print(f"Harmonic matches found: {len(harmonic_matches)}")
    for match_type, score in sorted(harmonic_matches.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  • {match_type}: {score:.3f}")
    
    # Test subharmonic resonance
    print(f"\n🎵 Testing subharmonic resonance analysis...")
    
    subharmonic_matches = calc.calculate_subharmonic_resonances(aripiprazole_spectrum, hole_spectrum)
    
    print(f"Subharmonic matches found: {len(subharmonic_matches)}")
    for match_type, score in sorted(subharmonic_matches.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  • {match_type}: {score:.3f}")
    
    # Test comprehensive resonance analysis
    print(f"\n🔬 Comprehensive resonance analysis...")
    
    # Lithium-hole analysis
    lithium_hole_analysis = calc.perform_comprehensive_resonance_analysis(lithium_spectrum, hole_spectrum)
    
    print(f"\nLITHIUM → INOSITOL HOLE Analysis:")
    print(f"  Overall frequency match: {lithium_hole_analysis.frequency_match_score:.3f}")
    print(f"  Primary resonance type: {lithium_hole_analysis.resonance_type}")
    print(f"  Resonance strength: {lithium_hole_analysis.resonance_strength:.3f}")
    print(f"  Phase coherence: {lithium_hole_analysis.phase_coherence:.3f}")
    print(f"  Phase lock probability: {lithium_hole_analysis.phase_lock_probability:.3f}")
    print(f"  Energy transfer efficiency: {lithium_hole_analysis.energy_transfer_efficiency:.3f}")
    print(f"  Power amplification: {lithium_hole_analysis.power_amplification:.2f}x")
    print(f"  Therapeutic potential: {lithium_hole_analysis.therapeutic_potential:.3f}")
    print(f"  Coupling stability: {lithium_hole_analysis.coupling_stability:.3f}")
    
    # Aripiprazole-hole analysis
    aripiprazole_hole_analysis = calc.perform_comprehensive_resonance_analysis(aripiprazole_spectrum, hole_spectrum)
    
    print(f"\nARIPIPRAZOLE → INOSITOL HOLE Analysis:")
    print(f"  Overall frequency match: {aripiprazole_hole_analysis.frequency_match_score:.3f}")
    print(f"  Primary resonance type: {aripiprazole_hole_analysis.resonance_type}")
    print(f"  Resonance strength: {aripiprazole_hole_analysis.resonance_strength:.3f}")
    print(f"  Therapeutic potential: {aripiprazole_hole_analysis.therapeutic_potential:.3f}")
    
    # Test multi-entity analysis
    print(f"\n🌐 Multi-entity resonance analysis...")
    
    all_spectra = [lithium_spectrum, aripiprazole_spectrum, hole_spectrum]
    multi_analysis = calc.analyze_multi_entity_resonance(all_spectra)
    
    print(f"Pairwise analyses completed: {len(multi_analysis)}")
    
    for pair_key, analysis in multi_analysis.items():
        print(f"\n{pair_key}:")
        print(f"  Frequency match: {analysis.frequency_match_score:.3f}")
        print(f"  Resonance type: {analysis.resonance_type}")
        print(f"  Therapeutic potential: {analysis.therapeutic_potential:.3f}")
    
    # Test frequency optimization
    print(f"\n⚙️  Frequency optimization analysis...")
    
    optimization_result = calc.optimize_frequency_for_maximum_resonance(
        aripiprazole_spectrum, hole_spectrum
    )
    
    print(f"Aripiprazole frequency optimization for inositol hole:")
    print(f"  Current frequency: {aripiprazole_spectrum.fundamental_frequency:.2e} Hz")
    print(f"  Optimal frequency: {optimization_result['optimal_frequency']:.2e} Hz")
    print(f"  Frequency shift: {optimization_result['frequency_shift']:.2e} Hz")
    print(f"  Optimal multiplier: {optimization_result['optimal_multiplier']:.3f}")
    print(f"  Max resonance score: {optimization_result['max_resonance_score']:.3f}")
    
    # Summary and insights
    print(f"\n💡 Key Resonance Insights:")
    print("-" * 40)
    
    # Best therapeutic match
    therapeutic_scores = {
        'lithium': lithium_hole_analysis.therapeutic_potential,
        'aripiprazole': aripiprazole_hole_analysis.therapeutic_potential
    }
    
    best_drug = max(therapeutic_scores, key=therapeutic_scores.get)
    best_score = therapeutic_scores[best_drug]
    
    print(f"• Best therapeutic match: {best_drug} (potential: {best_score:.3f})")
    
    # Resonance mechanisms
    if lithium_hole_analysis.resonance_type.startswith('direct'):
        print(f"• Lithium shows direct frequency resonance with inositol pathway")
    elif lithium_hole_analysis.resonance_type.startswith('harmonic'):
        print(f"• Lithium engages via harmonic resonance mechanism")
    
    # Phase locking potential
    if lithium_hole_analysis.phase_lock_probability > 0.7:
        print(f"• High probability of phase locking between lithium and target")
    
    # Power amplification
    if lithium_hole_analysis.power_amplification > 2.0:
        print(f"• Significant power amplification available: {lithium_hole_analysis.power_amplification:.1f}x")
    
    # Coupling stability
    if lithium_hole_analysis.coupling_stability > 0.8:
        print(f"• Stable coupling expected (stability: {lithium_hole_analysis.coupling_stability:.3f})")
    
    # Optimization opportunity
    improvement = optimization_result['max_resonance_score'] - aripiprazole_hole_analysis.frequency_match_score
    if improvement > 0.1:
        print(f"• Aripiprazole could benefit from frequency optimization (+{improvement:.3f})")
    
    print(f"\n✅ Resonance calculator testing complete!")
    print("🎵 Successfully computed detailed frequency matching and")
    print("   resonance patterns for therapeutic optimization.")
    
    return calc, multi_analysis

if __name__ == "__main__":
    calculator, analyses = main()
"""
Quantum Drug Transport Module
============================

Models drug transport at the quantum membrane level using oscillatory
framework principles. Simulates how pharmaceutical molecules cross biological
membranes through quantum-enhanced transport mechanisms.

Based on quantum membrane dynamics from the St. Stellas framework.
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
class DrugMolecule:
    """Represents a pharmaceutical molecule for transport simulation."""
    name: str
    molecular_weight: float  # Da
    lipophilicity: float     # LogP value
    charge: float           # Net charge at physiological pH
    hydrogen_bonds: int     # Number of hydrogen bond donors/acceptors
    polar_surface_area: float  # Ångström²
    oscillatory_frequency: float  # Hz - intrinsic molecular frequency
    quantum_coherence_time: float = 1e-12  # seconds

@dataclass
class MembraneProperties:
    """Properties of biological membrane for transport simulation."""
    thickness: float = 4e-9  # meters (4 nm typical)
    lipid_density: float = 1.2e18  # molecules/m²
    dielectric_constant: float = 2.5  # membrane core
    temperature: float = 310.0  # Kelvin (37°C)
    oscillatory_coupling: float = 0.8  # coupling strength with drug oscillations
    quantum_tunneling_probability: float = 0.1  # base tunneling probability

@dataclass
class TransportResult:
    """Result of quantum drug transport simulation."""
    drug_name: str
    transport_mechanism: str  # 'passive_diffusion', 'facilitated_diffusion', 'quantum_tunneling'
    permeability_coefficient: float  # cm/s
    transport_rate: float    # molecules/s/cm²
    energy_barrier: float    # kJ/mol
    quantum_enhancement_factor: float  # fold increase due to quantum effects
    oscillatory_resonance_score: float  # 0-1 score for frequency matching
    transport_time: float    # seconds to cross membrane
    confidence: float = 0.0

class QuantumDrugTransport:
    """
    Simulates drug transport across biological membranes using quantum
    oscillatory framework principles.
    
    Models both classical transport mechanisms and quantum-enhanced
    transport through oscillatory coupling and quantum tunneling.
    """
    
    def __init__(self):
        """Initialize quantum transport simulator."""
        self.membrane = MembraneProperties()
        self.transport_results = []
        
        # Physical constants
        self.k_B = 1.38064852e-23  # Boltzmann constant (J/K)
        self.h = 6.62607015e-34    # Planck constant (J⋅s)
        self.hbar = self.h / (2 * np.pi)  # Reduced Planck constant
        
    def create_drug_library(self) -> Dict[str, DrugMolecule]:
        """Create library of common pharmaceutical molecules."""
        
        drugs = {
            'lithium': DrugMolecule(
                name='lithium',
                molecular_weight=73.89,
                lipophilicity=-2.3,    # Very hydrophilic
                charge=1.0,            # Li+ ion
                hydrogen_bonds=0,      # No H-bonds as ion
                polar_surface_area=0.0, # Ion - no PSA
                oscillatory_frequency=2.8e13  # From your analysis
            ),
            
            'aripiprazole': DrugMolecule(
                name='aripiprazole',
                molecular_weight=448.39,
                lipophilicity=4.3,     # Highly lipophilic
                charge=0.0,            # Neutral at physiological pH
                hydrogen_bonds=3,      # NH and OH groups
                polar_surface_area=44.8,
                oscillatory_frequency=1.85e13
            ),
            
            'citalopram': DrugMolecule(
                name='citalopram',
                molecular_weight=324.39,
                lipophilicity=3.5,     # Moderately lipophilic
                charge=1.0,            # Protonated amine
                hydrogen_bonds=1,      # NH group
                polar_surface_area=36.2,
                oscillatory_frequency=2.15e13
            ),
            
            'atorvastatin': DrugMolecule(
                name='atorvastatin',
                molecular_weight=558.64,
                lipophilicity=4.1,     # Lipophilic
                charge=-1.0,           # Carboxylate anion
                hydrogen_bonds=6,      # Multiple OH and NH groups
                polar_surface_area=112.5,
                oscillatory_frequency=1.67e13
            ),
            
            'aspirin': DrugMolecule(
                name='aspirin',
                molecular_weight=180.16,
                lipophilicity=1.2,     # Moderately lipophilic
                charge=-1.0,           # Carboxylate anion
                hydrogen_bonds=3,      # COOH group
                polar_surface_area=63.6,
                oscillatory_frequency=3.42e13
            )
        }
        
        logger.info(f"Created drug library with {len(drugs)} molecules")
        return drugs
    
    def calculate_classical_permeability(self, drug: DrugMolecule) -> float:
        """Calculate classical membrane permeability using traditional models."""
        
        # Abraham equation for membrane permeability
        # Log P = a⋅LogP + b⋅PSA + c⋅MW + d⋅HB + e
        
        # Coefficients for blood-brain barrier permeability (similar to cell membrane)
        a = 0.15   # Lipophilicity coefficient
        b = -0.006 # Polar surface area coefficient  
        c = -0.0007 # Molecular weight coefficient
        d = -0.3   # Hydrogen bond coefficient
        e = -1.2   # Intercept
        
        log_permeability = (a * drug.lipophilicity + 
                          b * drug.polar_surface_area +
                          c * drug.molecular_weight +
                          d * drug.hydrogen_bonds + 
                          e)
        
        # Convert to permeability coefficient in cm/s
        permeability = 10**log_permeability  # cm/s
        
        return max(1e-10, permeability)  # Minimum permeability floor
    
    def calculate_energy_barrier(self, drug: DrugMolecule) -> float:
        """Calculate energy barrier for membrane crossing."""
        
        # Energy barrier depends on:
        # 1. Lipophilicity (hydrophobic interactions)
        # 2. Electrostatic interactions with membrane
        # 3. Size/steric effects
        
        # Base barrier for neutral, small molecule
        base_barrier = 25.0  # kJ/mol
        
        # Lipophilicity correction (higher LogP = lower barrier)
        lipophilic_correction = -2.5 * drug.lipophilicity
        
        # Charge penalty (charged molecules have higher barrier)
        charge_penalty = 15.0 * abs(drug.charge)
        
        # Size penalty (larger molecules have higher barrier)
        size_penalty = 0.02 * drug.molecular_weight
        
        # Hydrogen bonding penalty
        h_bond_penalty = 3.0 * drug.hydrogen_bonds
        
        total_barrier = (base_barrier + lipophilic_correction + 
                        charge_penalty + size_penalty + h_bond_penalty)
        
        return max(5.0, total_barrier)  # Minimum barrier
    
    def calculate_quantum_tunneling_probability(self, drug: DrugMolecule, 
                                              energy_barrier: float) -> float:
        """Calculate quantum tunneling probability through membrane."""
        
        # Quantum tunneling probability using WKB approximation
        # P = exp(-2 * sqrt(2m(V-E)) * a / hbar)
        
        # Effective mass for drug molecule (approximation)
        mass = drug.molecular_weight * 1.66054e-27  # kg (atomic mass unit)
        
        # Barrier height in Joules
        barrier_height = energy_barrier * 1000 / 6.022e23  # J
        
        # Assume drug has thermal energy
        thermal_energy = self.k_B * self.membrane.temperature
        
        if barrier_height <= thermal_energy:
            return 1.0  # Classical crossing
        
        # Effective barrier
        effective_barrier = barrier_height - thermal_energy
        
        # Tunneling coefficient
        tunneling_coefficient = (2 * np.sqrt(2 * mass * effective_barrier) * 
                               self.membrane.thickness / self.hbar)
        
        # Tunneling probability
        tunneling_probability = np.exp(-tunneling_coefficient)
        
        return min(1.0, tunneling_probability)
    
    def calculate_oscillatory_resonance(self, drug: DrugMolecule) -> float:
        """Calculate oscillatory resonance between drug and membrane."""
        
        # Membrane has characteristic frequencies
        membrane_frequencies = [
            2.1e13,  # Lipid hydrocarbon chain vibrations
            1.8e13,  # Phosphate group vibrations
            2.5e13,  # C-H stretching
            1.4e13   # Backbone vibrations
        ]
        
        # Find best frequency match
        drug_freq = drug.oscillatory_frequency
        
        resonance_scores = []
        for mem_freq in membrane_frequencies:
            # Calculate frequency difference
            freq_diff = abs(drug_freq - mem_freq)
            
            # Resonance quality (inverse of frequency difference)
            resonance = 1.0 / (1.0 + freq_diff / mem_freq)
            resonance_scores.append(resonance)
        
        # Best resonance match
        best_resonance = max(resonance_scores)
        
        # Apply membrane coupling strength
        effective_resonance = best_resonance * self.membrane.oscillatory_coupling
        
        return effective_resonance
    
    def calculate_quantum_enhancement(self, drug: DrugMolecule, 
                                    resonance_score: float,
                                    tunneling_probability: float) -> float:
        """Calculate quantum enhancement factor for drug transport."""
        
        # Quantum enhancement comes from:
        # 1. Oscillatory resonance (frequency matching)
        # 2. Quantum tunneling (barrier penetration)
        # 3. Quantum coherence (wave-like behavior)
        
        # Resonance enhancement
        resonance_enhancement = 1.0 + 10.0 * resonance_score
        
        # Tunneling enhancement
        tunneling_enhancement = 1.0 + 5.0 * tunneling_probability
        
        # Coherence enhancement (depends on coherence time)
        coherence_factor = min(1.0, drug.quantum_coherence_time / 1e-12)
        coherence_enhancement = 1.0 + 2.0 * coherence_factor
        
        # Total enhancement (multiplicative)
        total_enhancement = (resonance_enhancement * 
                           tunneling_enhancement * 
                           coherence_enhancement)
        
        return total_enhancement
    
    def simulate_drug_transport(self, drug: DrugMolecule) -> TransportResult:
        """Simulate complete drug transport across membrane."""
        
        # Calculate classical transport
        classical_permeability = self.calculate_classical_permeability(drug)
        energy_barrier = self.calculate_energy_barrier(drug)
        
        # Calculate quantum effects
        tunneling_probability = self.calculate_quantum_tunneling_probability(drug, energy_barrier)
        resonance_score = self.calculate_oscillatory_resonance(drug)
        quantum_enhancement = self.calculate_quantum_enhancement(
            drug, resonance_score, tunneling_probability
        )
        
        # Enhanced permeability with quantum effects
        quantum_permeability = classical_permeability * quantum_enhancement
        
        # Determine dominant transport mechanism
        if tunneling_probability > 0.1:
            mechanism = 'quantum_tunneling'
        elif resonance_score > 0.5:
            mechanism = 'facilitated_diffusion'
        else:
            mechanism = 'passive_diffusion'
        
        # Calculate transport rate (molecules/s/cm²)
        # Assuming 1 mM concentration gradient
        concentration_gradient = 1e-3 * 6.022e23 / 1e6  # molecules/cm³
        transport_rate = quantum_permeability * concentration_gradient
        
        # Calculate transport time
        transport_time = self.membrane.thickness / (quantum_permeability / 100)  # seconds
        
        # Calculate confidence based on model applicability
        confidence = self._calculate_transport_confidence(drug, mechanism)
        
        result = TransportResult(
            drug_name=drug.name,
            transport_mechanism=mechanism,
            permeability_coefficient=quantum_permeability,
            transport_rate=transport_rate,
            energy_barrier=energy_barrier,
            quantum_enhancement_factor=quantum_enhancement,
            oscillatory_resonance_score=resonance_score,
            transport_time=transport_time,
            confidence=confidence
        )
        
        return result
    
    def _calculate_transport_confidence(self, drug: DrugMolecule, mechanism: str) -> float:
        """Calculate confidence in transport simulation results."""
        
        base_confidence = 0.7
        
        # Confidence adjustments based on drug properties
        if drug.molecular_weight < 500:  # Small molecules - higher confidence
            base_confidence += 0.1
        elif drug.molecular_weight > 800:  # Large molecules - lower confidence
            base_confidence -= 0.2
        
        if abs(drug.charge) <= 1:  # Singly charged or neutral - higher confidence
            base_confidence += 0.1
        else:  # Highly charged - lower confidence
            base_confidence -= 0.15
        
        # Mechanism-specific confidence
        mechanism_confidence = {
            'passive_diffusion': 0.9,    # Well understood
            'facilitated_diffusion': 0.7, # Moderately understood
            'quantum_tunneling': 0.5     # Theoretical model
        }
        
        mechanism_factor = mechanism_confidence.get(mechanism, 0.5)
        
        final_confidence = min(0.95, base_confidence * mechanism_factor)
        return final_confidence
    
    def simulate_drug_library(self, drugs: Dict[str, DrugMolecule] = None) -> List[TransportResult]:
        """Simulate transport for entire drug library."""
        
        if drugs is None:
            drugs = self.create_drug_library()
        
        results = []
        
        for drug_name, drug in drugs.items():
            logger.info(f"Simulating transport for {drug_name}")
            result = self.simulate_drug_transport(drug)
            results.append(result)
        
        self.transport_results = results
        logger.info(f"Completed transport simulation for {len(results)} drugs")
        
        return results
    
    def analyze_transport_patterns(self, results: List[TransportResult]) -> Dict[str, Any]:
        """Analyze patterns in drug transport results."""
        
        analysis = {}
        
        # Mechanism distribution
        mechanisms = [r.transport_mechanism for r in results]
        mechanism_counts = pd.Series(mechanisms).value_counts().to_dict()
        analysis['mechanism_distribution'] = mechanism_counts
        
        # Quantum enhancement statistics
        enhancements = [r.quantum_enhancement_factor for r in results]
        analysis['quantum_enhancement'] = {
            'mean': np.mean(enhancements),
            'median': np.median(enhancements),
            'max': np.max(enhancements),
            'min': np.min(enhancements)
        }
        
        # Permeability statistics
        permeabilities = [r.permeability_coefficient for r in results]
        analysis['permeability'] = {
            'mean': np.mean(permeabilities),
            'median': np.median(permeabilities),
            'range': (np.min(permeabilities), np.max(permeabilities))
        }
        
        # Resonance statistics
        resonances = [r.oscillatory_resonance_score for r in results]
        analysis['resonance'] = {
            'mean': np.mean(resonances),
            'drugs_with_high_resonance': len([r for r in resonances if r > 0.5])
        }
        
        # Transport time statistics
        times = [r.transport_time for r in results]
        analysis['transport_time'] = {
            'mean': np.mean(times),
            'fastest': np.min(times),
            'slowest': np.max(times)
        }
        
        return analysis
    
    def visualize_transport_results(self, results: List[TransportResult],
                                  output_dir: str = "babylon_results") -> None:
        """Create visualizations of drug transport results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive transport visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Transport mechanisms
        mechanisms = [r.transport_mechanism for r in results]
        mechanism_counts = pd.Series(mechanisms).value_counts()
        
        axes[0, 0].pie(mechanism_counts.values, labels=mechanism_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Transport Mechanisms')
        
        # 2. Permeability vs Enhancement
        permeabilities = [r.permeability_coefficient for r in results]
        enhancements = [r.quantum_enhancement_factor for r in results]
        drug_names = [r.drug_name for r in results]
        
        scatter = axes[0, 1].scatter(permeabilities, enhancements, 
                                   c=[mechanisms.index(m) for m in mechanisms],
                                   cmap='viridis', alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Permeability Coefficient (cm/s)')
        axes[0, 1].set_ylabel('Quantum Enhancement Factor')
        axes[0, 1].set_title('Permeability vs Quantum Enhancement')
        axes[0, 1].set_xscale('log')
        
        # Annotate points
        for i, name in enumerate(drug_names):
            axes[0, 1].annotate(name, (permeabilities[i], enhancements[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Energy barriers
        barriers = [r.energy_barrier for r in results]
        
        bars = axes[0, 2].bar(drug_names, barriers, color='coral')
        axes[0, 2].set_title('Energy Barriers')
        axes[0, 2].set_ylabel('Energy Barrier (kJ/mol)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Color bars by barrier height
        for bar, barrier in zip(bars, barriers):
            if barrier > 40:
                bar.set_color('red')
            elif barrier > 25:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 4. Oscillatory resonance scores
        resonances = [r.oscillatory_resonance_score for r in results]
        
        axes[1, 0].bar(drug_names, resonances, color='lightblue')
        axes[1, 0].set_title('Oscillatory Resonance Scores')
        axes[1, 0].set_ylabel('Resonance Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='High Resonance')
        axes[1, 0].legend()
        
        # 5. Transport times
        times = [r.transport_time for r in results]
        
        axes[1, 1].bar(drug_names, times, color='lightgreen')
        axes[1, 1].set_title('Transport Times')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_yscale('log')
        
        # 6. Confidence scores
        confidences = [r.confidence for r in results]
        
        axes[1, 2].bar(drug_names, confidences, color='gold')
        axes[1, 2].set_title('Model Confidence')
        axes[1, 2].set_ylabel('Confidence Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Good Confidence')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'quantum_drug_transport_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create quantum enhancement correlation plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot enhancement vs resonance
        scatter = ax.scatter(resonances, enhancements, s=100, alpha=0.7, c='blue')
        
        for i, name in enumerate(drug_names):
            ax.annotate(name, (resonances[i], enhancements[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Oscillatory Resonance Score')
        ax.set_ylabel('Quantum Enhancement Factor')
        ax.set_title('Quantum Enhancement vs Oscillatory Resonance')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(resonances, enhancements, 1)
        p = np.poly1d(z)
        ax.plot(sorted(resonances), p(sorted(resonances)), "r--", alpha=0.8, label='Trend')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'quantum_enhancement_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Transport visualizations saved to {output_path}")
    
    def save_results(self, results: List[TransportResult], 
                    analysis: Dict[str, Any],
                    output_dir: str = "babylon_results") -> None:
        """Save transport simulation results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_data = []
        for result in results:
            results_data.append({
                'drug_name': result.drug_name,
                'transport_mechanism': result.transport_mechanism,
                'permeability_coefficient': result.permeability_coefficient,
                'transport_rate': result.transport_rate,
                'energy_barrier': result.energy_barrier,
                'quantum_enhancement_factor': result.quantum_enhancement_factor,
                'oscillatory_resonance_score': result.oscillatory_resonance_score,
                'transport_time': result.transport_time,
                'confidence': result.confidence
            })
        
        with open(output_path / 'quantum_transport_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(results_data)
        df.to_csv(output_path / 'quantum_transport_results.csv', index=False)
        
        # Save analysis summary
        with open(output_path / 'transport_analysis_summary.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Transport results saved to {output_path}")

def main():
    """
    Test quantum drug transport simulation.
    
    This demonstrates how pharmaceutical molecules cross biological
    membranes using quantum-enhanced transport mechanisms.
    """
    
    print("⚛️  Testing Quantum Drug Transport")
    print("=" * 50)
    
    # Initialize transport simulator
    simulator = QuantumDrugTransport()
    
    # Create drug library
    print("\n🧪 Creating pharmaceutical molecule library...")
    drugs = simulator.create_drug_library()
    
    print(f"Created library with {len(drugs)} drugs:")
    for name, drug in drugs.items():
        print(f"  • {name}: MW={drug.molecular_weight:.1f} Da, LogP={drug.lipophilicity:.1f}")
    
    # Simulate transport for all drugs
    print(f"\n🔬 Simulating quantum transport across membrane...")
    results = simulator.simulate_drug_library(drugs)
    
    # Display results
    print(f"\n📊 TRANSPORT SIMULATION RESULTS:")
    print("-" * 60)
    
    for result in results:
        print(f"{result.drug_name.upper()}:")
        print(f"  Mechanism: {result.transport_mechanism}")
        print(f"  Permeability: {result.permeability_coefficient:.2e} cm/s")
        print(f"  Energy barrier: {result.energy_barrier:.1f} kJ/mol")
        print(f"  Quantum enhancement: {result.quantum_enhancement_factor:.2f}×")
        print(f"  Resonance score: {result.oscillatory_resonance_score:.3f}")
        print(f"  Transport time: {result.transport_time:.2e} seconds")
        print(f"  Confidence: {result.confidence:.3f}")
        print()
    
    # Analyze patterns
    print("🔍 Analyzing transport patterns...")
    analysis = simulator.analyze_transport_patterns(results)
    
    print(f"\n📈 TRANSPORT PATTERN ANALYSIS:")
    print("-" * 40)
    print(f"Transport mechanisms:")
    for mechanism, count in analysis['mechanism_distribution'].items():
        print(f"  • {mechanism}: {count} drugs")
    
    print(f"\nQuantum enhancement:")
    enhancement = analysis['quantum_enhancement']
    print(f"  • Mean: {enhancement['mean']:.2f}×")
    print(f"  • Range: {enhancement['min']:.2f}× - {enhancement['max']:.2f}×")
    
    print(f"\nOscillatory resonance:")
    resonance = analysis['resonance']
    print(f"  • Mean score: {resonance['mean']:.3f}")
    print(f"  • High resonance drugs: {resonance['drugs_with_high_resonance']}")
    
    print(f"\nTransport efficiency:")
    permeability = analysis['permeability']
    print(f"  • Mean permeability: {permeability['mean']:.2e} cm/s")
    print(f"  • Range: {permeability['range'][0]:.2e} - {permeability['range'][1]:.2e} cm/s")
    
    # Save results and create visualizations
    print("\n💾 Saving results and creating visualizations...")
    simulator.save_results(results, analysis)
    simulator.visualize_transport_results(results)
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 40)
    
    # Find drug with highest quantum enhancement
    best_quantum = max(results, key=lambda r: r.quantum_enhancement_factor)
    print(f"• Highest quantum enhancement: {best_quantum.drug_name} ({best_quantum.quantum_enhancement_factor:.2f}×)")
    
    # Find drug with best resonance
    best_resonance = max(results, key=lambda r: r.oscillatory_resonance_score)
    print(f"• Best oscillatory resonance: {best_resonance.drug_name} ({best_resonance.oscillatory_resonance_score:.3f})")
    
    # Find fastest transport
    fastest = min(results, key=lambda r: r.transport_time)
    print(f"• Fastest transport: {fastest.drug_name} ({fastest.transport_time:.2e} seconds)")
    
    # Count quantum-enhanced drugs
    quantum_enhanced = len([r for r in results if r.quantum_enhancement_factor > 2.0])
    print(f"• Quantum-enhanced drugs: {quantum_enhanced}/{len(results)}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Quantum drug transport simulation complete!")
    
    return results

if __name__ == "__main__":
    results = main()

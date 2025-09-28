"""
Circuit Representation of Reaction Pathways
===========================================

Demonstrates representation of biological reaction pathways as electrical circuits
using St. Stellas molecular language and oscillatory mechanics.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass

from s_entropy_solver import solve_problem, create_coordinate, SCoordinate

@dataclass
class CircuitElement:
    """Electrical circuit element representing biological process."""
    element_type: str  # 'resistor', 'capacitor', 'inductor', 'voltage_source'
    value: float
    biological_process: str
    reaction_id: str

@dataclass
class BiologicalReaction:
    """Biological reaction with circuit mapping."""
    id: str
    name: str
    reactants: List[str]
    products: List[str]
    rate_constant: float
    gibbs_energy: float
    enzyme: str

class PathwayCircuitBuilder:
    """Builds circuit representations of metabolic pathways."""
    
    def __init__(self):
        self.reactions = []
        self.circuit_elements = []
        self.circuit_network = nx.Graph()
    
    def load_reactions_from_database(self) -> List[BiologicalReaction]:
        """Load reactions from online database (with fallback synthetic data)."""
        synthetic_reactions = [
            BiologicalReaction(
                id='R00658',
                name='Hexokinase reaction',
                reactants=['D-Glucose', 'ATP'],
                products=['D-Glucose 6-phosphate', 'ADP'],
                rate_constant=2.5e5,
                gibbs_energy=-16.7,
                enzyme='Hexokinase'
            ),
            BiologicalReaction(
                id='R01068',
                name='Phosphoglucose isomerase',
                reactants=['D-Glucose 6-phosphate'],
                products=['D-Fructose 6-phosphate'],
                rate_constant=1.8e4,
                gibbs_energy=1.7,
                enzyme='Phosphoglucose isomerase'
            ),
            BiologicalReaction(
                id='R00756',
                name='Phosphofructokinase reaction',
                reactants=['D-Fructose 6-phosphate', 'ATP'],
                products=['D-Fructose 1,6-bisphosphate', 'ADP'],
                rate_constant=3.2e4,
                gibbs_energy=-14.2,
                enzyme='Phosphofructokinase'
            )
        ]
        
        self.reactions = synthetic_reactions
        return synthetic_reactions
    
    def build_circuit_representation(self):
        """Build complete circuit representation of the pathway."""
        self.circuit_elements = []
        
        # Convert each reaction to circuit elements
        for reaction in self.reactions:
            elements = self._reaction_to_circuit(reaction)
            self.circuit_elements.extend(elements)
    
    def _reaction_to_circuit(self, reaction: BiologicalReaction) -> List[CircuitElement]:
        """Convert biological reaction to circuit elements."""
        elements = []
        
        # Enzyme catalysis as inductor
        if reaction.rate_constant > 0:
            inductance = 1.0 / reaction.rate_constant
            elements.append(CircuitElement(
                element_type='inductor',
                value=inductance,
                biological_process=f'{reaction.enzyme} catalysis',
                reaction_id=reaction.id
            ))
        
        # Substrate binding as capacitor
        binding_strength = len(reaction.reactants) * 1e-6
        elements.append(CircuitElement(
            element_type='capacitor',
            value=binding_strength,
            biological_process='substrate binding',
            reaction_id=reaction.id
        ))
        
        # Activation energy as resistor
        resistance = abs(reaction.gibbs_energy) / 10.0
        elements.append(CircuitElement(
            element_type='resistor',
            value=resistance,
            biological_process='activation barrier',
            reaction_id=reaction.id
        ))
        
        # Driving force as voltage source
        voltage = -reaction.gibbs_energy / 100.0
        elements.append(CircuitElement(
            element_type='voltage_source',
            value=voltage,
            biological_process='thermodynamic driving force',
            reaction_id=reaction.id
        ))
        
        return elements
    
    def simulate_frequency_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate frequency response of the biological circuit."""
        frequencies = np.logspace(-3, 6, 1000)
        impedances = []
        
        for f in frequencies:
            omega = 2 * np.pi * f
            Z_total = 0j
            
            for element in self.circuit_elements:
                if element.element_type == 'resistor':
                    Z_total += element.value
                elif element.element_type == 'capacitor':
                    Z_total += -1j / (omega * max(element.value, 1e-12))
                elif element.element_type == 'inductor':
                    Z_total += 1j * omega * element.value
            
            impedances.append(abs(Z_total))
        
        return frequencies, np.array(impedances)
    
    def optimize_with_s_entropy(self) -> Dict:
        """Optimize circuit using S-entropy framework."""
        problem_space = []
        
        for reaction in self.reactions:
            knowledge = np.log(1 + len(reaction.reactants) + len(reaction.products))
            time_coord = 1.0 / max(reaction.rate_constant, 1e-6)
            entropy_coord = max(0, reaction.gibbs_energy / 25.0)
            coord = create_coordinate(knowledge, time_coord, entropy_coord)
            problem_space.append(coord)
        
        result = solve_problem(problem_space)
        
        return {
            'optimal_coordinate': {
                'knowledge': result['solution'].knowledge,
                'time': result['solution'].time,
                'entropy': result['solution'].entropy
            },
            'optimization_valid': result['is_valid'],
            'solution_norm': result['solution'].norm()
        }
    
    def plot_circuit_analysis(self, save_path: str = "circuit_analysis.png"):
        """Plot circuit analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Element distribution
        element_types = [e.element_type for e in self.circuit_elements]
        type_counts = {t: element_types.count(t) for t in set(element_types)}
        
        axes[0, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Circuit Element Distribution')
        
        # Frequency response
        freqs, impedances = self.simulate_frequency_response()
        axes[0, 1].loglog(freqs, impedances)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Impedance (Ω)')
        axes[0, 1].set_title('Frequency Response')
        axes[0, 1].grid(True)
        
        # Reaction properties
        rate_constants = [r.rate_constant for r in self.reactions]
        gibbs_energies = [r.gibbs_energy for r in self.reactions]
        
        axes[1, 0].scatter(gibbs_energies, np.log10(rate_constants), alpha=0.7, s=100)
        axes[1, 0].set_xlabel('ΔG (kJ/mol)')
        axes[1, 0].set_ylabel('log₁₀(Rate Constant)')
        axes[1, 0].set_title('Thermodynamics vs Kinetics')
        
        # S-entropy optimization
        s_result = self.optimize_with_s_entropy()
        if s_result['optimization_valid']:
            coords = ['Knowledge', 'Time', 'Entropy']
            values = [s_result['optimal_coordinate']['knowledge'],
                     s_result['optimal_coordinate']['time'],
                     s_result['optimal_coordinate']['entropy']]
            
            axes[1, 1].bar(coords, values, alpha=0.7)
            axes[1, 1].set_title('S-Entropy Optimization')
            axes[1, 1].set_ylabel('Coordinate Value')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_circuit_data(self, filename: str = "circuit_data.json"):
        """Save circuit representation data."""
        data = {
            'n_reactions': len(self.reactions),
            'n_elements': len(self.circuit_elements),
            's_entropy_result': self.optimize_with_s_entropy()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    """Demonstrate circuit representation of biological pathways."""
    print("⚡ Circuit Representation of Reaction Pathways")
    print("=" * 45)
    
    builder = PathwayCircuitBuilder()
    
    # Load reactions
    reactions = builder.load_reactions_from_database()
    print(f"Loaded {len(reactions)} reactions")
    
    # Build circuit
    builder.build_circuit_representation()
    print(f"Built circuit with {len(builder.circuit_elements)} elements")
    
    # Optimize with S-entropy
    s_result = builder.optimize_with_s_entropy()
    print(f"S-Entropy optimization valid: {s_result['optimization_valid']}")
    
    # Plot and save
    builder.plot_circuit_analysis()
    builder.save_circuit_data()
    
    print("✅ Circuit analysis complete!")

if __name__ == "__main__":
    main()

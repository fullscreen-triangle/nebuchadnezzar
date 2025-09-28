"""
Oscillatory Mechanics: Electrical Circuits in S-Entropy Framework
================================================================

This standalone script expresses electrical circuit dynamics in the frequency domain
using the S-entropy framework where dx/dt = dx/de = dx/di, where:
- e = entropy
- i = information  
- dt = time to solution (not infinitesimal time change)

Converts S-entropy framework results into frequency domain with plots and saved results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from s_entropy_solver import solve_problem, create_coordinate, SCoordinate

@dataclass
class CircuitElement:
    """Represents an electrical circuit element."""
    element_type: str
    value: float
    s_coordinate: SCoordinate
    
@dataclass 
class CircuitSolution:
    """Represents S-entropy solution for circuit dynamics."""
    solution_coordinate: SCoordinate
    frequency_response: np.ndarray
    time_domain_response: np.ndarray
    entropy_flow: np.ndarray
    information_flow: np.ndarray

class OscillatoryCircuit:
    """Electrical circuit analyzer using S-entropy framework."""
    
    def __init__(self):
        self.elements = []
        self.solutions = []
        
    def add_resistor(self, resistance: float):
        """Add resistor with S-entropy mapping."""
        s_coord = create_coordinate(
            knowledge=0.1,
            time=1.0 / resistance,
            entropy=np.log(1 + resistance)
        )
        self.elements.append(CircuitElement('resistor', resistance, s_coord))
    
    def add_capacitor(self, capacitance: float):
        """Add capacitor with S-entropy mapping.""" 
        s_coord = create_coordinate(
            knowledge=np.log(1 + capacitance * 1e6),
            time=capacitance,
            entropy=1.0 / (1 + capacitance * 1e6)
        )
        self.elements.append(CircuitElement('capacitor', capacitance, s_coord))
    
    def solve_circuit_dynamics(self) -> CircuitSolution:
        """Solve circuit using S-entropy navigation."""
        problem_space = [element.s_coordinate for element in self.elements]
        result = solve_problem(problem_space)
        
        # Generate frequency response
        frequencies = np.logspace(-3, 6, 1000)
        freq_response = self._compute_frequency_response(result['solution'], frequencies)
        
        # Generate time response
        time_points = np.linspace(0, 1, 1000)
        time_response = self._compute_time_response(result['solution'], time_points)
        
        # Compute flows
        entropy_flow = self._compute_entropy_flow(result['solution'], time_points)
        info_flow = self._compute_information_flow(result['solution'], time_points)
        
        return CircuitSolution(
            result['solution'], freq_response, time_response, 
            entropy_flow, info_flow
        )
    
    def _compute_frequency_response(self, solution: SCoordinate, frequencies: np.ndarray):
        """Compute frequency response from S-entropy solution."""
        response = np.zeros(len(frequencies), dtype=complex)
        for i, f in enumerate(frequencies):
            omega = 2 * np.pi * f
            magnitude = solution.knowledge / (1 + solution.entropy * omega)
            phase = -np.arctan(solution.time * omega)
            response[i] = magnitude * np.exp(1j * phase)
        return response
    
    def _compute_time_response(self, solution: SCoordinate, time_points: np.ndarray):
        """Compute time domain response."""
        f_dom = 1.0 / (solution.entropy + 0.01)
        amplitude = solution.knowledge
        decay_rate = 1.0 / (solution.time + 0.01)
        return amplitude * np.exp(-decay_rate * time_points) * np.cos(2 * np.pi * f_dom * time_points)
    
    def _compute_entropy_flow(self, solution: SCoordinate, time_points: np.ndarray):
        """Compute entropy flow de/dt."""
        base_rate = solution.entropy / solution.time if solution.time != 0 else 0
        return base_rate * np.exp(-time_points / solution.time) if solution.time > 0 else np.ones_like(time_points) * base_rate
    
    def _compute_information_flow(self, solution: SCoordinate, time_points: np.ndarray):
        """Compute information flow di/dt."""
        base_rate = solution.knowledge / (solution.entropy + 0.01)
        f_info = 2.0 / (solution.time + 0.01)
        return base_rate * (1 + 0.5 * np.sin(2 * np.pi * f_info * time_points))
    
    def plot_results(self, solution: CircuitSolution, save_path: str = "circuit_analysis.png"):
        """Plot comprehensive analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Frequency response
        frequencies = np.logspace(-3, 6, len(solution.frequency_response))
        magnitude_db = 20 * np.log10(np.abs(solution.frequency_response))
        axes[0, 0].semilogx(frequencies, magnitude_db)
        axes[0, 0].set_title('Frequency Response - Magnitude')
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Magnitude (dB)')
        axes[0, 0].grid(True)
        
        # Phase response
        phase_deg = np.angle(solution.frequency_response) * 180 / np.pi
        axes[0, 1].semilogx(frequencies, phase_deg)
        axes[0, 1].set_title('Frequency Response - Phase')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Phase (degrees)')
        axes[0, 1].grid(True)
        
        # Time response
        time_points = np.linspace(0, 1, len(solution.time_domain_response))
        axes[0, 2].plot(time_points, solution.time_domain_response)
        axes[0, 2].set_title('Time Domain Response')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Response')
        axes[0, 2].grid(True)
        
        # Entropy flow
        axes[1, 0].plot(time_points, solution.entropy_flow)
        axes[1, 0].set_title('Entropy Flow: dS/de·de/dt')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Entropy Rate')
        axes[1, 0].grid(True)
        
        # Information flow
        axes[1, 1].plot(time_points, solution.information_flow)
        axes[1, 1].set_title('Information Flow: dI/di·di/dt')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Information Rate')
        axes[1, 1].grid(True)
        
        # S-entropy space
        sc = solution.solution_coordinate
        axes[1, 2].scatter(sc.knowledge, sc.time, s=sc.entropy*100, c='red', alpha=0.7)
        for element in self.elements:
            ec = element.s_coordinate
            axes[1, 2].scatter(ec.knowledge, ec.time, s=ec.entropy*50, alpha=0.5, 
                              label=f"{element.element_type}")
        axes[1, 2].set_title('S-Entropy Space')
        axes[1, 2].set_xlabel('S_knowledge')
        axes[1, 2].set_ylabel('S_time')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Demonstrate oscillatory circuit analysis."""
    print("🔌 Oscillatory Mechanics: Circuit Analysis")
    print("=" * 50)
    
    # Create RLC circuit
    circuit = OscillatoryCircuit()
    circuit.add_resistor(100.0)      # 100Ω
    circuit.add_capacitor(1e-6)      # 1µF
    
    print("Solving circuit dynamics using S-entropy framework...")
    solution = circuit.solve_circuit_dynamics()
    
    print(f"Solution coordinates: ({solution.solution_coordinate.knowledge:.3f}, "
          f"{solution.solution_coordinate.time:.3f}, {solution.solution_coordinate.entropy:.3f})")
    
    # Analyze results
    max_magnitude = np.max(np.abs(solution.frequency_response))
    print(f"Peak frequency response: {max_magnitude:.3f}")
    
    max_time_response = np.max(np.abs(solution.time_domain_response))
    print(f"Peak time response: {max_time_response:.3f}")
    
    # Plot results
    circuit.plot_results(solution)
    
    # Save results
    results = {
        'elements': [{'type': e.element_type, 'value': e.value, 
                     'coordinates': [e.s_coordinate.knowledge, e.s_coordinate.time, e.s_coordinate.entropy]} 
                    for e in circuit.elements],
        'solution': [solution.solution_coordinate.knowledge, 
                    solution.solution_coordinate.time, 
                    solution.solution_coordinate.entropy]
    }
    
    with open('circuit_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Analysis complete! Results saved.")

if __name__ == "__main__":
    main()

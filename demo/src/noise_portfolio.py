"""
Noise Portfolio Analysis for St. Stellas Framework
==================================================

Visualizes and analyzes the utility of noise for deriving solutions
using the St. Stellas system and Anti-Algorithm Principle.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple
from scipy import signal
from dataclasses import dataclass

from s_entropy_solver import solve_problem, create_coordinate, SCoordinate

@dataclass
class NoiseProfile:
    """Noise profile for different types of stochastic processes."""
    name: str
    signal_data: np.ndarray
    frequency: float
    amplitude: float
    entropy_content: float

class NoisePortfolio:
    """Portfolio of noise sources for solution derivation."""
    
    def __init__(self):
        self.noise_sources = []
        self.performance_metrics = {}
        
    def generate_noise_sources(self, n_samples: int = 5000) -> List[NoiseProfile]:
        """Generate various types of noise for analysis."""
        sources = []
        
        # White noise
        white_noise = np.random.normal(0, 1, n_samples)
        sources.append(NoiseProfile(
            name="White Noise",
            signal_data=white_noise,
            frequency=1.0,
            amplitude=np.std(white_noise),
            entropy_content=self._compute_entropy(white_noise)
        ))
        
        # Pink noise (1/f)
        pink_noise = self._generate_pink_noise(n_samples)
        sources.append(NoiseProfile(
            name="Pink Noise",
            signal_data=pink_noise,
            frequency=0.5,
            amplitude=np.std(pink_noise),
            entropy_content=self._compute_entropy(pink_noise)
        ))
        
        # Chaos noise
        chaos_noise = self._generate_chaos_noise(n_samples)
        sources.append(NoiseProfile(
            name="Chaos Noise",
            signal_data=chaos_noise,
            frequency=3.5,
            amplitude=np.std(chaos_noise),
            entropy_content=self._compute_entropy(chaos_noise)
        ))
        
        self.noise_sources = sources
        return sources
    
    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink noise using FFT method."""
        white = np.random.normal(0, 1, n_samples)
        freqs = np.fft.fftfreq(n_samples)
        freqs[0] = 1e-6  # Avoid division by zero
        
        fft_white = np.fft.fft(white)
        fft_pink = fft_white / np.sqrt(np.abs(freqs))
        pink = np.real(np.fft.ifft(fft_pink))
        
        return (pink - np.mean(pink)) / np.std(pink)
    
    def _generate_chaos_noise(self, n_samples: int) -> np.ndarray:
        """Generate noise from chaotic Lorenz system."""
        dt = 0.01
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        chaos_data = []
        
        for i in range(n_samples):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            chaos_data.append(x)
        
        chaos_data = np.array(chaos_data)
        return (chaos_data - np.mean(chaos_data)) / np.std(chaos_data)
    
    def _compute_entropy(self, signal: np.ndarray) -> float:
        """Compute Shannon entropy of signal."""
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-12))
    
    def test_noise_utility(self, test_problems: List[List[SCoordinate]]) -> Dict:
        """Test utility of different noise sources for problem solving."""
        results = {}
        
        for noise_source in self.noise_sources:
            source_results = []
            
            for problem in test_problems:
                # Add noise-derived perturbations
                noisy_problem = []
                for coord in problem:
                    noise_idx = np.random.randint(0, len(noise_source.signal_data))
                    noise_val = noise_source.signal_data[noise_idx] * 0.1
                    
                    noisy_coord = create_coordinate(
                        coord.knowledge + noise_val,
                        coord.time + abs(noise_val),
                        coord.entropy + noise_val**2
                    )
                    noisy_problem.append(noisy_coord)
                
                # Solve with noise
                solution = solve_problem(noisy_problem)
                
                if solution['is_valid']:
                    quality = 1.0 / (1.0 + solution['solution'].norm())
                else:
                    quality = 0.0
                
                source_results.append({
                    'quality': quality,
                    'valid': solution['is_valid']
                })
            
            # Aggregate results
            valid_solutions = [r for r in source_results if r['valid']]
            
            results[noise_source.name] = {
                'success_rate': len(valid_solutions) / len(source_results),
                'avg_quality': np.mean([r['quality'] for r in source_results]),
                'entropy_content': noise_source.entropy_content,
                'solutions': source_results
            }
        
        self.performance_metrics = results
        return results
    
    def plot_noise_analysis(self, save_path: str = "noise_analysis.png"):
        """Plot noise analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot noise signals
        for noise in self.noise_sources[:2]:
            time_axis = np.linspace(0, 1, len(noise.signal_data[:1000]))
            axes[0, 0].plot(time_axis, noise.signal_data[:1000], 
                          alpha=0.7, label=noise.name)
        axes[0, 0].set_title("Noise Signals")
        axes[0, 0].legend()
        
        # Power spectral densities
        for noise in self.noise_sources:
            freqs, psd = signal.welch(noise.signal_data, nperseg=512)
            axes[0, 1].loglog(freqs[1:], psd[1:], label=noise.name)
        axes[0, 1].set_title("Power Spectral Density")
        axes[0, 1].legend()
        
        if self.performance_metrics:
            names = list(self.performance_metrics.keys())
            success_rates = [self.performance_metrics[name]['success_rate'] for name in names]
            qualities = [self.performance_metrics[name]['avg_quality'] for name in names]
            
            # Success rates
            axes[1, 0].bar(names, success_rates, alpha=0.7)
            axes[1, 0].set_title("Success Rate by Noise Type")
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Quality scores
            axes[1, 1].bar(names, qualities, alpha=0.7, color='orange')
            axes[1, 1].set_title("Average Quality by Noise Type")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_data(self, filename: str = "noise_data.json"):
        """Save noise analysis data."""
        data = {
            'noise_sources': [
                {'name': n.name, 'entropy': n.entropy_content} 
                for n in self.noise_sources
            ],
            'performance': self.performance_metrics
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    """Demonstrate noise portfolio analysis."""
    print("🌊 Noise Portfolio Analysis")
    print("=" * 28)
    
    portfolio = NoisePortfolio()
    
    # Generate noise sources
    noise_sources = portfolio.generate_noise_sources()
    print(f"Generated {len(noise_sources)} noise sources")
    
    # Create test problems
    test_problems = []
    for i in range(5):
        problem = []
        for j in range(3):
            coord = create_coordinate(
                knowledge=np.random.exponential(2.0),
                time=np.random.exponential(1.0),
                entropy=np.random.exponential(0.5)
            )
            problem.append(coord)
        test_problems.append(problem)
    
    # Test utility
    utility_results = portfolio.test_noise_utility(test_problems)
    
    # Display results
    print("\n📊 Results:")
    for noise_name, metrics in utility_results.items():
        print(f"{noise_name}: Success={metrics['success_rate']:.3f}, "
              f"Quality={metrics['avg_quality']:.3f}")
    
    # Plot and save
    portfolio.plot_noise_analysis()
    portfolio.save_data()
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()

"""
Precision-by-Difference for Intracellular Dynamics
=================================================

Applies the precision-by-difference principle for intracellular dynamics.
Based on Sango Rine Shumba framework: ΔP_i(k) = T_ref(k) - t_i(k)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time as time_module

from s_entropy_solver import solve_problem, create_coordinate, SCoordinate

@dataclass
class CellularNode:
    """Cellular component node."""
    node_id: str
    local_time: float
    precision_metric: float
    molecular_state: Dict[str, float]
    temporal_coordinate: SCoordinate

class IntracellularPrecisionCoordinator:
    """Precision-by-difference coordinator for cellular systems."""
    
    def __init__(self, atomic_precision: float = 1e-15):
        self.atomic_precision = atomic_precision
        self.nodes = []
        
    def add_cellular_node(self, node_id: str, concentrations: Dict[str, float]):
        """Add cellular node to coordination network."""
        local_time = time_module.time() + np.random.normal(0, 1e-9)
        atomic_ref = self._get_atomic_reference()
        precision_metric = atomic_ref - local_time
        
        s_coord = create_coordinate(
            knowledge=np.log(1 + len(concentrations)),
            time=abs(precision_metric) * 1e12,
            entropy=np.std(list(concentrations.values())) + 0.01
        )
        
        node = CellularNode(node_id, local_time, precision_metric, concentrations, s_coord)
        self.nodes.append(node)
        
    def _get_atomic_reference(self) -> float:
        """Simulate atomic clock reference."""
        return time_module.time() + np.random.normal(0, self.atomic_precision)
    
    def compute_coordination(self):
        """Compute precision-by-difference coordination."""
        n = len(self.nodes)
        coordination_matrix = np.zeros((n, n))
        
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    precision_diff = abs(node_i.precision_metric - node_j.precision_metric)
                    coordination_matrix[i, j] = np.exp(-precision_diff / 1e-12)
                else:
                    coordination_matrix[i, j] = 1.0
        
        # Predict future states using S-entropy
        predictions = self._predict_states()
        
        return {
            'coordination_matrix': coordination_matrix,
            'predictions': predictions,
            'sync_quality': np.mean(coordination_matrix[coordination_matrix < 1.0])
        }
    
    def _predict_states(self, steps: int = 5) -> List[Dict[str, float]]:
        """Predict molecular states using S-entropy navigation."""
        predictions = []
        problem_space = [node.temporal_coordinate for node in self.nodes]
        
        for step in range(steps):
            future_space = []
            for coord in problem_space:
                future_coord = create_coordinate(
                    coord.knowledge * (1 + 0.01 * step),
                    coord.time + step * 0.1,
                    coord.entropy * np.exp(-0.05 * step)
                )
                future_space.append(future_coord)
            
            result = solve_problem(future_space)
            solution = result['solution']
            
            prediction = {
                'ATP': solution.knowledge * 5.0,
                'ADP': solution.time * 0.5,
                'reaction_rate': solution.knowledge / (solution.time + 0.01),
                'step': step
            }
            predictions.append(prediction)
        
        return predictions
    
    def plot_results(self, coordination, save_path: str = "precision_analysis.png"):
        """Plot coordination analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Coordination matrix
        im = axes[0, 0].imshow(coordination['coordination_matrix'], cmap='viridis')
        axes[0, 0].set_title('Coordination Matrix')
        plt.colorbar(im, ax=axes[0, 0])
        
        # Precision metrics
        precisions = [node.precision_metric for node in self.nodes]
        axes[0, 1].bar(range(len(precisions)), precisions)
        axes[0, 1].set_title('Precision Metrics')
        axes[0, 1].set_ylabel('Precision (s)')
        
        # State predictions
        if coordination['predictions']:
            steps = [p['step'] for p in coordination['predictions']]
            atp = [p['ATP'] for p in coordination['predictions']]
            rates = [p['reaction_rate'] for p in coordination['predictions']]
            
            axes[1, 0].plot(steps, atp, 'b-o', label='ATP')
            axes[1, 0].plot(steps, rates, 'r-s', label='Reaction Rate')
            axes[1, 0].set_title('Predicted States')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # S-entropy coordinates
        for node in self.nodes:
            coord = node.temporal_coordinate
            axes[1, 1].scatter(coord.knowledge, coord.time, s=coord.entropy*100, 
                              alpha=0.7, label=node.node_id)
        axes[1, 1].set_title('S-Entropy Coordinates')
        axes[1, 1].set_xlabel('Knowledge')
        axes[1, 1].set_ylabel('Time')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

def main():
    """Demonstrate precision-by-difference coordination."""
    print("⚡ Precision-by-Difference: Intracellular Coordination")
    print("=" * 50)
    
    coordinator = IntracellularPrecisionCoordinator()
    
    # Add mitochondrial components
    coordinator.add_cellular_node("outer_membrane", {"ATP": 5.0, "ADP": 0.5})
    coordinator.add_cellular_node("inner_membrane", {"ATP": 4.8, "ADP": 0.7})
    coordinator.add_cellular_node("matrix", {"ATP": 6.2, "ADP": 0.3})
    
    print(f"Added {len(coordinator.nodes)} cellular nodes")
    
    # Compute coordination
    coordination = coordinator.compute_coordination()
    
    print(f"Synchronization quality: {coordination['sync_quality']:.3f}")
    print(f"Predictions generated: {len(coordination['predictions'])}")
    
    # Plot and save results
    coordinator.plot_results(coordination)
    
    with open('precision_results.json', 'w') as f:
        json.dump({
            'sync_quality': coordination['sync_quality'],
            'node_count': len(coordinator.nodes)
        }, f, indent=2)
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()

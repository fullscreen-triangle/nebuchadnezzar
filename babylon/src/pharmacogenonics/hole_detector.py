"""
Oscillatory Hole Detection Module
=================================

Detects oscillatory holes in biological pathways created by genetic variants.
These holes represent deficits in oscillatory amplitude or frequency that can
be filled by pharmaceutical BMDs (Biological Maxwell Demons).

Based on the St. Stellas oscillatory framework for personalized medicine.
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
class OscillatoryHole:
    """Represents an oscillatory hole in a biological pathway."""
    gene_id: str
    pathway: str
    frequency: float  # Hz
    amplitude_deficit: float  # 0.0 to 1.0 (fraction of normal amplitude)
    hole_type: str  # 'expression_hole', 'coupling_hole', 'regulatory_hole'
    variant_source: str  # Which genetic variant caused this hole
    confidence: float = 0.0  # Confidence in hole detection
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PathwayOscillations:
    """Normal oscillatory patterns for a biological pathway."""
    pathway_name: str
    genes: List[str]
    base_frequencies: Dict[str, float]  # Gene -> frequency (Hz)
    coupling_strengths: Dict[str, float]  # Gene -> coupling strength
    normal_amplitudes: Dict[str, float]  # Gene -> expected amplitude

class OscillatoryHoleDetector:
    """
    Detects oscillatory holes in biological pathways based on genetic variants.
    
    Uses the oscillatory framework to identify where genetic variants create
    deficits in normal biological oscillations that could be filled by drugs.
    """
    
    def __init__(self):
        """Initialize the hole detector with pathway oscillation patterns."""
        self.pathway_oscillations = self._load_pathway_oscillations()
        self.detection_results = []
        
    def _load_pathway_oscillations(self) -> Dict[str, PathwayOscillations]:
        """Load normal oscillatory patterns for biological pathways."""
        
        # Define known oscillatory patterns for key pathways
        pathways = {
            'neurotransmitter_signaling': PathwayOscillations(
                pathway_name='neurotransmitter_signaling',
                genes=['DRD2', 'HTR2A', 'SLC6A4', 'MAOA', 'COMT'],
                base_frequencies={
                    'DRD2': 1.45e13,    # Dopamine receptor oscillations
                    'HTR2A': 1.23e13,   # Serotonin receptor oscillations  
                    'SLC6A4': 2.67e13,  # Serotonin transporter
                    'MAOA': 1.89e13,    # Monoamine oxidase
                    'COMT': 1.56e13     # Catechol-O-methyltransferase
                },
                coupling_strengths={
                    'DRD2': 0.85, 'HTR2A': 0.78, 'SLC6A4': 0.92, 
                    'MAOA': 0.67, 'COMT': 0.74
                },
                normal_amplitudes={
                    'DRD2': 1.0, 'HTR2A': 1.0, 'SLC6A4': 1.0,
                    'MAOA': 1.0, 'COMT': 1.0
                }
            ),
            
            'inositol_metabolism': PathwayOscillations(
                pathway_name='inositol_metabolism',
                genes=['INPP1', 'IMPA1', 'IMPA2', 'ITPKB'],
                base_frequencies={
                    'INPP1': 7.23e13,   # Inositol polyphosphate 1-phosphatase
                    'IMPA1': 6.45e13,   # Inositol monophosphatase 1
                    'IMPA2': 6.78e13,   # Inositol monophosphatase 2
                    'ITPKB': 5.89e13    # Inositol-trisphosphate 3-kinase B
                },
                coupling_strengths={
                    'INPP1': 0.91, 'IMPA1': 0.83, 'IMPA2': 0.79, 'ITPKB': 0.72
                },
                normal_amplitudes={
                    'INPP1': 1.0, 'IMPA1': 1.0, 'IMPA2': 1.0, 'ITPKB': 1.0
                }
            ),
            
            'GSK3_pathway': PathwayOscillations(
                pathway_name='GSK3_pathway',
                genes=['GSK3A', 'GSK3B', 'AKT1', 'CTNNB1'],
                base_frequencies={
                    'GSK3A': 2.34e13,   # Glycogen synthase kinase 3 alpha
                    'GSK3B': 2.15e13,   # Glycogen synthase kinase 3 beta
                    'AKT1': 1.87e13,    # AKT serine/threonine kinase 1
                    'CTNNB1': 1.65e13   # Catenin beta 1
                },
                coupling_strengths={
                    'GSK3A': 0.88, 'GSK3B': 0.94, 'AKT1': 0.81, 'CTNNB1': 0.76
                },
                normal_amplitudes={
                    'GSK3A': 1.0, 'GSK3B': 1.0, 'AKT1': 1.0, 'CTNNB1': 1.0
                }
            )
        }
        
        logger.info(f"Loaded oscillatory patterns for {len(pathways)} pathways")
        return pathways
    
    def detect_holes_from_variants(self, variants: List[Dict[str, Any]], 
                                 pathways: List[str] = None) -> List[OscillatoryHole]:
        """
        Detect oscillatory holes based on genetic variants.
        
        Args:
            variants: List of genetic variants (each dict should have 'gene', 'impact', etc.)
            pathways: List of pathway names to analyze (default: all pathways)
            
        Returns:
            List of detected oscillatory holes
        """
        if pathways is None:
            pathways = list(self.pathway_oscillations.keys())
            
        holes = []
        
        for pathway_name in pathways:
            if pathway_name not in self.pathway_oscillations:
                logger.warning(f"Unknown pathway: {pathway_name}")
                continue
                
            pathway = self.pathway_oscillations[pathway_name]
            pathway_holes = self._analyze_pathway_holes(variants, pathway)
            holes.extend(pathway_holes)
            
        self.detection_results = holes
        logger.info(f"Detected {len(holes)} oscillatory holes across {len(pathways)} pathways")
        
        return holes
    
    def _analyze_pathway_holes(self, variants: List[Dict[str, Any]], 
                              pathway: PathwayOscillations) -> List[OscillatoryHole]:
        """Analyze oscillatory holes in a specific pathway."""
        holes = []
        
        # Find variants affecting genes in this pathway
        pathway_variants = [v for v in variants if v.get('gene') in pathway.genes]
        
        for variant in pathway_variants:
            gene = variant['gene']
            impact = variant.get('impact', 'UNKNOWN')
            variant_id = variant.get('variant_id', f"{gene}_variant")
            
            # Calculate hole properties based on variant impact
            hole = self._calculate_hole_properties(gene, impact, variant_id, pathway)
            if hole:
                holes.append(hole)
                
        return holes
    
    def _calculate_hole_properties(self, gene: str, impact: str, variant_id: str,
                                 pathway: PathwayOscillations) -> Optional[OscillatoryHole]:
        """Calculate properties of an oscillatory hole."""
        
        if gene not in pathway.base_frequencies:
            return None
            
        base_frequency = pathway.base_frequencies[gene]
        normal_amplitude = pathway.normal_amplitudes[gene]
        coupling_strength = pathway.coupling_strengths[gene]
        
        # Map variant impact to amplitude deficit
        impact_to_deficit = {
            'HIGH': 0.85,      # High impact variants create large holes
            'MODERATE': 0.65,  # Moderate impact variants create medium holes
            'LOW': 0.35,       # Low impact variants create small holes
            'MODIFIER': 0.15,  # Modifier variants create minimal holes
            'UNKNOWN': 0.50    # Unknown impact - assume moderate
        }
        
        amplitude_deficit = impact_to_deficit.get(impact, 0.50)
        
        # Determine hole type based on gene function and impact
        hole_type = self._determine_hole_type(gene, impact, coupling_strength)
        
        # Calculate confidence based on evidence strength
        confidence = self._calculate_confidence(impact, coupling_strength)
        
        # Adjust frequency based on amplitude deficit (holes can shift frequency)
        hole_frequency = base_frequency * (1.0 + 0.1 * amplitude_deficit)
        
        return OscillatoryHole(
            gene_id=gene,
            pathway=pathway.pathway_name,
            frequency=hole_frequency,
            amplitude_deficit=amplitude_deficit,
            hole_type=hole_type,
            variant_source=variant_id,
            confidence=confidence,
            metadata={
                'base_frequency': base_frequency,
                'normal_amplitude': normal_amplitude,
                'coupling_strength': coupling_strength,
                'variant_impact': impact
            }
        )
    
    def _determine_hole_type(self, gene: str, impact: str, coupling_strength: float) -> str:
        """Determine the type of oscillatory hole."""
        
        if coupling_strength > 0.8:
            return 'coupling_hole'  # Strong coupling disruption
        elif impact in ['HIGH', 'MODERATE']:
            return 'expression_hole'  # Expression level disruption
        else:
            return 'regulatory_hole'  # Regulatory disruption
    
    def _calculate_confidence(self, impact: str, coupling_strength: float) -> float:
        """Calculate confidence in hole detection."""
        
        impact_confidence = {
            'HIGH': 0.95,
            'MODERATE': 0.80,
            'LOW': 0.60,
            'MODIFIER': 0.40,
            'UNKNOWN': 0.50
        }
        
        base_confidence = impact_confidence.get(impact, 0.50)
        
        # Higher coupling strength increases confidence
        coupling_boost = coupling_strength * 0.2
        
        return min(0.99, base_confidence + coupling_boost)
    
    def visualize_holes(self, holes: List[OscillatoryHole], 
                       output_dir: str = "babylon_results") -> None:
        """Create visualizations of detected oscillatory holes."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create hole detection summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Holes by pathway
        pathway_counts = {}
        for hole in holes:
            pathway_counts[hole.pathway] = pathway_counts.get(hole.pathway, 0) + 1
            
        axes[0, 0].bar(pathway_counts.keys(), pathway_counts.values(), color='skyblue')
        axes[0, 0].set_title('Oscillatory Holes by Pathway')
        axes[0, 0].set_ylabel('Number of Holes')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Amplitude deficit distribution
        deficits = [hole.amplitude_deficit for hole in holes]
        axes[0, 1].hist(deficits, bins=20, color='salmon', alpha=0.7)
        axes[0, 1].set_title('Amplitude Deficit Distribution')
        axes[0, 1].set_xlabel('Amplitude Deficit')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Frequency vs amplitude deficit scatter
        frequencies = [hole.frequency for hole in holes]
        axes[1, 0].scatter(frequencies, deficits, c=[hole.confidence for hole in holes], 
                          cmap='viridis', alpha=0.7)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Amplitude Deficit')
        axes[1, 0].set_title('Frequency vs Amplitude Deficit')
        axes[1, 0].set_xscale('log')
        
        # 4. Hole types
        hole_types = {}
        for hole in holes:
            hole_types[hole.hole_type] = hole_types.get(hole.hole_type, 0) + 1
            
        axes[1, 1].pie(hole_types.values(), labels=hole_types.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Hole Types Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'oscillatory_holes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed frequency spectrum plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Group holes by pathway for color coding
        pathway_colors = {'neurotransmitter_signaling': 'red', 
                         'inositol_metabolism': 'blue', 
                         'GSK3_pathway': 'green'}
        
        for hole in holes:
            color = pathway_colors.get(hole.pathway, 'black')
            ax.scatter(hole.frequency, hole.amplitude_deficit, 
                      s=hole.confidence*200, c=color, alpha=0.7,
                      label=hole.pathway if hole.pathway not in ax.get_legend_handles_labels()[1] else "")
            
            # Annotate significant holes
            if hole.amplitude_deficit > 0.7:
                ax.annotate(hole.gene_id, (hole.frequency, hole.amplitude_deficit),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude Deficit')
        ax.set_title('Oscillatory Holes Frequency Spectrum')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'oscillatory_holes_spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def save_results(self, holes: List[OscillatoryHole], 
                    output_dir: str = "babylon_results") -> None:
        """Save hole detection results to files."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON
        holes_data = []
        for hole in holes:
            holes_data.append({
                'gene_id': hole.gene_id,
                'pathway': hole.pathway,
                'frequency': hole.frequency,
                'amplitude_deficit': hole.amplitude_deficit,
                'hole_type': hole.hole_type,
                'variant_source': hole.variant_source,
                'confidence': hole.confidence,
                'metadata': hole.metadata
            })
            
        with open(output_path / 'oscillatory_holes.json', 'w') as f:
            json.dump(holes_data, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(holes_data)
        df.to_csv(output_path / 'oscillatory_holes.csv', index=False)
        
        # Save summary statistics
        summary = {
            'total_holes': len(holes),
            'pathways_affected': len(set(hole.pathway for hole in holes)),
            'genes_affected': len(set(hole.gene_id for hole in holes)),
            'average_amplitude_deficit': np.mean([hole.amplitude_deficit for hole in holes]),
            'average_confidence': np.mean([hole.confidence for hole in holes]),
            'hole_types': {hole_type: sum(1 for hole in holes if hole.hole_type == hole_type) 
                          for hole_type in set(hole.hole_type for hole in holes)}
        }
        
        with open(output_path / 'hole_detection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")

def main():
    """
    Test the oscillatory hole detector with sample genetic variants.
    
    This demonstrates how genetic variants create oscillatory holes that
    can be detected and characterized for pharmaceutical intervention.
    """
    
    print("🔬 Testing Oscillatory Hole Detection")
    print("=" * 50)
    
    # Sample genetic variants (simulating Dante Labs type data)
    sample_variants = [
        {
            'gene': 'INPP1',
            'variant_id': 'rs123456',
            'impact': 'HIGH',
            'description': 'Inositol polyphosphate 1-phosphatase variant'
        },
        {
            'gene': 'GSK3B', 
            'variant_id': 'rs789012',
            'impact': 'MODERATE',
            'description': 'Glycogen synthase kinase 3 beta variant'
        },
        {
            'gene': 'DRD2',
            'variant_id': 'rs345678',
            'impact': 'MODERATE', 
            'description': 'Dopamine receptor D2 variant'
        },
        {
            'gene': 'HTR2A',
            'variant_id': 'rs901234',
            'impact': 'LOW',
            'description': 'Serotonin receptor 2A variant'
        },
        {
            'gene': 'COMT',
            'variant_id': 'rs567890',
            'impact': 'HIGH',
            'description': 'Catechol-O-methyltransferase variant'
        }
    ]
    
    # Initialize hole detector
    detector = OscillatoryHoleDetector()
    
    # Detect oscillatory holes
    print(f"\n🔍 Analyzing {len(sample_variants)} genetic variants...")
    holes = detector.detect_holes_from_variants(sample_variants)
    
    # Display results
    print(f"\n📊 OSCILLATORY HOLES DETECTED: {len(holes)}")
    print("-" * 40)
    
    for hole in holes:
        print(f"Gene: {hole.gene_id}")
        print(f"  Pathway: {hole.pathway}")
        print(f"  Frequency: {hole.frequency:.2e} Hz")
        print(f"  Amplitude Deficit: {hole.amplitude_deficit:.3f}")
        print(f"  Hole Type: {hole.hole_type}")
        print(f"  Confidence: {hole.confidence:.3f}")
        print(f"  Variant: {hole.variant_source}")
        print()
    
    # Save results and create visualizations
    print("💾 Saving results and creating visualizations...")
    detector.save_results(holes)
    detector.visualize_holes(holes)
    
    # Summary statistics
    pathways = set(hole.pathway for hole in holes)
    avg_deficit = np.mean([hole.amplitude_deficit for hole in holes])
    avg_confidence = np.mean([hole.confidence for hole in holes])
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"  Pathways affected: {len(pathways)}")
    print(f"  Average amplitude deficit: {avg_deficit:.3f}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Results saved to: babylon_results/")
    
    print("\n✅ Oscillatory hole detection complete!")
    
    return holes

if __name__ == "__main__":
    holes = main()
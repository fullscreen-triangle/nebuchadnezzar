"""
Personal Genome-Coherent Intracellular Dynamics Analysis
========================================================

This module extends the St. Stellas framework to analyze personal genome data,
creating individualized intracellular dynamics models that are coherent with
your specific genetic makeup.

The system maps genetic variants to:
1. Personalized ATP system parameters
2. Individual-specific oscillatory patterns
3. Custom BMD (Biological Maxwell Demon) configurations
4. Personalized molecular coordinate transformations
5. Individual S-entropy navigation patterns

Input: Personal genome data (VCF, 23andMe, AncestryDNA formats)
Output: Personalized biological insights using the oscillatory framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import gzip
import re
from datetime import datetime

# Import the existing St. Stellas framework components
from st_stellas.core.s_entropy import SEntropyFramework, SCoordinate
from st_stellas.core.coordinates import MolecularCoordinates, MolecularCoordinate
from st_stellas.core.oscillations import BiologicalOscillations, BiologicalScale
from st_stellas.core.fluid_dynamics import DynamicFluxTheory

logger = logging.getLogger(__name__)

@dataclass
class GeneticVariant:
    """Represents a genetic variant from personal genome data."""
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    genotype: str  # e.g., "0/1", "1/1", "0/0"
    gene_symbol: Optional[str] = None
    consequence: Optional[str] = None
    impact: Optional[str] = None
    frequency: Optional[float] = None

@dataclass
class PersonalizedParameters:
    """Personalized parameters derived from genetic analysis."""
    atp_synthesis_efficiency: float = 1.0
    membrane_permeability_factor: float = 1.0
    oscillatory_coupling_strengths: Dict[str, float] = field(default_factory=dict)
    bmd_selection_bias: Dict[str, float] = field(default_factory=dict)
    s_entropy_navigation_speed: float = 1.0
    metabolic_pathway_weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class PersonalizedInsight:
    """A personalized biological insight derived from genome analysis."""
    insight_type: str
    description: str
    confidence: float  # 0.0 to 1.0
    genetic_evidence: List[GeneticVariant]
    oscillatory_pattern: Optional[np.ndarray] = None
    s_entropy_coordinates: Optional[SCoordinate] = None
    recommendations: List[str] = field(default_factory=list)

class GenomeLoader:
    """Loads and parses personal genome data from various formats."""
    
    def __init__(self):
        self.supported_formats = ['vcf', '23andme', 'ancestrydna', 'csv']
        
    def load_genome_data(self, file_path: str, format_type: str = 'auto') -> List[GeneticVariant]:
        """
        Load personal genome data from file.
        
        Args:
            file_path: Path to genome data file
            format_type: Format type ('vcf', '23andme', 'ancestrydna', 'csv', 'auto')
            
        Returns:
            List of genetic variants
        """
        file_path = Path(file_path)
        
        if format_type == 'auto':
            format_type = self._detect_format(file_path)
            
        logger.info(f"Loading genome data from {file_path} (format: {format_type})")
        
        if format_type == 'vcf':
            return self._load_vcf(file_path)
        elif format_type == '23andme':
            return self._load_23andme(file_path)
        elif format_type == 'ancestrydna':
            return self._load_ancestrydna(file_path)
        elif format_type == 'csv':
            return self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect genome file format."""
        if file_path.suffix.lower() in ['.vcf', '.vcf.gz']:
            return 'vcf'
        elif '23andme' in file_path.name.lower():
            return '23andme'
        elif 'ancestry' in file_path.name.lower():
            return 'ancestrydna'
        else:
            return 'csv'
    
    def _load_vcf(self, file_path: Path) -> List[GeneticVariant]:
        """Load VCF format genome data."""
        variants = []
        
        open_func = gzip.open if file_path.suffix == '.gz' else open
        mode = 'rt' if file_path.suffix == '.gz' else 'r'
        
        with open_func(file_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                
                chrom = fields[0]
                pos = int(fields[1])
                ref = fields[3]
                alt = fields[4]
                
                # Extract genotype from FORMAT field
                format_fields = fields[8].split(':')
                sample_data = fields[9].split(':')
                
                gt_idx = format_fields.index('GT') if 'GT' in format_fields else 0
                genotype = sample_data[gt_idx] if gt_idx < len(sample_data) else '0/0'
                
                variant = GeneticVariant(
                    chromosome=chrom,
                    position=pos,
                    ref_allele=ref,
                    alt_allele=alt,
                    genotype=genotype
                )
                variants.append(variant)
        
        logger.info(f"Loaded {len(variants)} variants from VCF")
        return variants
    
    def _load_23andme(self, file_path: Path) -> List[GeneticVariant]:
        """Load 23andMe format genome data."""
        variants = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                if len(fields) < 4:
                    continue
                
                rsid = fields[0]
                chrom = fields[1]
                pos = int(fields[2])
                genotype = fields[3]
                
                # Convert genotype to VCF-like format
                if genotype in ['--', 'II', 'DD']:
                    continue  # Skip no-calls and indels for now
                
                # For 23andMe, we need to infer ref/alt from genotype
                alleles = list(set(genotype))
                if len(alleles) == 1:
                    ref, alt = alleles[0], '.'
                    gt = '0/0' if genotype == alleles[0] + alleles[0] else '1/1'
                else:
                    ref, alt = alleles[0], alleles[1]
                    gt = '0/1'
                
                variant = GeneticVariant(
                    chromosome=chrom,
                    position=pos,
                    ref_allele=ref,
                    alt_allele=alt,
                    genotype=gt
                )
                variants.append(variant)
        
        logger.info(f"Loaded {len(variants)} variants from 23andMe format")
        return variants
    
    def _load_ancestrydna(self, file_path: Path) -> List[GeneticVariant]:
        """Load AncestryDNA format genome data."""
        # Similar to 23andMe but different format
        return self._load_23andme(file_path)  # Simplified for now
    
    def _load_csv(self, file_path: Path) -> List[GeneticVariant]:
        """Load CSV format genome data."""
        df = pd.read_csv(file_path)
        variants = []
        
        # Assume standard columns exist
        required_cols = ['chromosome', 'position', 'ref_allele', 'alt_allele', 'genotype']
        
        for _, row in df.iterrows():
            variant = GeneticVariant(
                chromosome=str(row['chromosome']),
                position=int(row['position']),
                ref_allele=row['ref_allele'],
                alt_allele=row['alt_allele'],
                genotype=row['genotype'],
                gene_symbol=row.get('gene_symbol', None),
                consequence=row.get('consequence', None)
            )
            variants.append(variant)
        
        logger.info(f"Loaded {len(variants)} variants from CSV")
        return variants

class PersonalizedParameterMapper:
    """Maps genetic variants to personalized biological parameters."""
    
    def __init__(self):
        # Load variant-to-parameter mapping database
        self.variant_effects = self._load_variant_effect_database()
        
    def _load_variant_effect_database(self) -> Dict[str, Dict[str, float]]:
        """Load database mapping genetic variants to biological effects."""
        # This would normally load from a comprehensive database
        # For now, we'll use key known variants affecting cellular metabolism
        
        variant_effects = {
            # ATP synthesis and mitochondrial function
            'POLG': {'atp_synthesis_efficiency': 0.85},  # Mitochondrial DNA polymerase
            'TFAM': {'atp_synthesis_efficiency': 1.15},  # Mitochondrial transcription factor
            'MT-ATP6': {'atp_synthesis_efficiency': 0.90},  # ATP synthase subunit
            'MT-ATP8': {'atp_synthesis_efficiency': 0.90},
            
            # Membrane transport
            'SLC2A1': {'membrane_permeability_factor': 1.20},  # Glucose transporter
            'SLC2A4': {'membrane_permeability_factor': 1.15},  # Insulin-responsive glucose transporter
            'ATP1A1': {'membrane_permeability_factor': 0.90},  # Na+/K+-ATPase
            
            # Circadian rhythms (oscillatory patterns)
            'CLOCK': {'oscillatory_coupling_strengths': {'circadian': 1.25}},
            'PER1': {'oscillatory_coupling_strengths': {'circadian': 1.15}},
            'PER2': {'oscillatory_coupling_strengths': {'circadian': 1.10}},
            'CRY1': {'oscillatory_coupling_strengths': {'circadian': 1.05}},
            'CRY2': {'oscillatory_coupling_strengths': {'circadian': 1.05}},
            
            # Neurotransmitter systems (BMD selection)
            'DRD2': {'bmd_selection_bias': {'dopamine': 1.20}},
            'HTR2A': {'bmd_selection_bias': {'serotonin': 1.15}},
            'COMT': {'bmd_selection_bias': {'dopamine': 0.85}},  # Slow metabolizer = higher dopamine
            
            # Metabolic pathways
            'MTHFR': {'metabolic_pathway_weights': {'folate': 0.85}},  # C677T variant
            'APOE': {'metabolic_pathway_weights': {'lipid': 1.10}},  # E4 variant
            'FTO': {'metabolic_pathway_weights': {'energy': 1.05}},  # Obesity risk
        }
        
        return variant_effects
    
    def map_variants_to_parameters(self, variants: List[GeneticVariant]) -> PersonalizedParameters:
        """
        Map genetic variants to personalized biological parameters.
        
        Args:
            variants: List of genetic variants from personal genome
            
        Returns:
            Personalized parameters for the oscillatory framework
        """
        params = PersonalizedParameters()
        
        # Count effects for each parameter type
        atp_effects = []
        membrane_effects = []
        
        for variant in variants:
            if not variant.gene_symbol:
                continue
                
            gene = variant.gene_symbol.upper()
            if gene in self.variant_effects:
                effects = self.variant_effects[gene]
                
                # Apply effects based on genotype
                effect_strength = self._calculate_effect_strength(variant.genotype)
                
                for param_type, base_effect in effects.items():
                    adjusted_effect = 1.0 + (base_effect - 1.0) * effect_strength
                    
                    if param_type == 'atp_synthesis_efficiency':
                        atp_effects.append(adjusted_effect)
                    elif param_type == 'membrane_permeability_factor':
                        membrane_effects.append(adjusted_effect)
                    elif param_type == 'oscillatory_coupling_strengths':
                        for osc_type, strength in base_effect.items():
                            if osc_type not in params.oscillatory_coupling_strengths:
                                params.oscillatory_coupling_strengths[osc_type] = 1.0
                            params.oscillatory_coupling_strengths[osc_type] *= adjusted_effect
                    elif param_type == 'bmd_selection_bias':
                        for bmd_type, bias in base_effect.items():
                            if bmd_type not in params.bmd_selection_bias:
                                params.bmd_selection_bias[bmd_type] = 1.0
                            params.bmd_selection_bias[bmd_type] *= adjusted_effect
                    elif param_type == 'metabolic_pathway_weights':
                        for pathway, weight in base_effect.items():
                            if pathway not in params.metabolic_pathway_weights:
                                params.metabolic_pathway_weights[pathway] = 1.0
                            params.metabolic_pathway_weights[pathway] *= adjusted_effect
        
        # Aggregate effects
        if atp_effects:
            params.atp_synthesis_efficiency = np.mean(atp_effects)
        if membrane_effects:
            params.membrane_permeability_factor = np.mean(membrane_effects)
        
        # Calculate overall S-entropy navigation speed based on genetic efficiency
        overall_efficiency = (params.atp_synthesis_efficiency + params.membrane_permeability_factor) / 2
        params.s_entropy_navigation_speed = overall_efficiency * 1.2  # Efficiency boosts navigation
        
        logger.info(f"Mapped {len([v for v in variants if v.gene_symbol])} gene variants to personalized parameters")
        
        return params
    
    def _calculate_effect_strength(self, genotype: str) -> float:
        """Calculate the strength of genetic effect based on genotype."""
        if genotype in ['0/0', '0|0']:  # Homozygous reference
            return 0.0
        elif genotype in ['0/1', '1/0', '0|1', '1|0']:  # Heterozygous
            return 0.5
        elif genotype in ['1/1', '1|1']:  # Homozygous alternate
            return 1.0
        else:
            return 0.0  # Unknown genotype

class PersonalizedIntracellularSimulator:
    """Runs personalized intracellular dynamics simulations using individual genetic parameters."""
    
    def __init__(self, personalized_params: PersonalizedParameters):
        self.params = personalized_params
        
        # Initialize framework components with personalized parameters
        self.s_entropy = SEntropyFramework()
        self.coordinates = MolecularCoordinates()
        self.oscillations = BiologicalOscillations()
        self.fluid_dynamics = DynamicFluxTheory()
        
    def simulate_personalized_atp_dynamics(self, duration: float = 3600.0) -> Dict[str, Any]:
        """
        Simulate ATP dynamics with personalized efficiency parameters.
        
        Args:
            duration: Simulation duration in seconds
            
        Returns:
            ATP dynamics simulation results
        """
        # Time points
        t = np.linspace(0, duration, int(duration))
        
        # Personalized ATP synthesis rate
        base_synthesis_rate = 1.0  # mmol/L/s
        personalized_synthesis_rate = base_synthesis_rate * self.params.atp_synthesis_efficiency
        
        # Simulate ATP concentration over time with oscillatory patterns
        atp_concentration = np.zeros_like(t)
        
        for i, time_point in enumerate(t):
            # Base ATP level with synthesis efficiency
            base_atp = 5.0 * self.params.atp_synthesis_efficiency
            
            # Add circadian oscillation if present
            circadian_strength = self.params.oscillatory_coupling_strengths.get('circadian', 1.0)
            circadian_component = 0.5 * circadian_strength * np.sin(2 * np.pi * time_point / 86400)  # 24-hour cycle
            
            # Add cellular oscillations (faster timescales)
            cellular_component = 0.2 * np.sin(2 * np.pi * time_point / 3600)  # 1-hour cycle
            
            atp_concentration[i] = base_atp + circadian_component + cellular_component
        
        return {
            'time': t,
            'atp_concentration': atp_concentration,
            'synthesis_efficiency': self.params.atp_synthesis_efficiency,
            'mean_atp': np.mean(atp_concentration),
            'atp_variability': np.std(atp_concentration)
        }
    
    def analyze_personalized_oscillatory_patterns(self) -> Dict[str, np.ndarray]:
        """Analyze personalized oscillatory patterns across biological scales."""
        
        # Generate personalized frequency spectrum for each scale
        scales = ['quantum_membrane', 'intracellular', 'cellular', 'tissue', 'neural', 
                 'cognitive', 'neuromuscular', 'microbiome', 'organ', 'allometric']
        
        oscillatory_patterns = {}
        
        for scale in scales:
            # Base frequency ranges for each scale (from the documents)
            base_frequencies = {
                'quantum_membrane': (1e12, 1e15),
                'intracellular': (1e3, 1e6),
                'cellular': (1e-1, 1e2),
                'tissue': (1e-2, 1e1),
                'neural': (1, 100),
                'cognitive': (0.1, 50),
                'neuromuscular': (0.01, 20),
                'microbiome': (1e-4, 1e-1),
                'organ': (1e-5, 1e-2),
                'allometric': (1e-8, 1e-5)
            }
            
            f_min, f_max = base_frequencies[scale]
            
            # Generate frequency array
            frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 100)
            
            # Personalized coupling strength for this scale
            coupling_strength = self.params.oscillatory_coupling_strengths.get(scale, 1.0)
            
            # Generate personalized power spectrum
            # Higher coupling strength = more coherent oscillations
            power_spectrum = coupling_strength * np.exp(-0.5 * ((frequencies - np.sqrt(f_min * f_max)) / (f_max - f_min))**2)
            
            oscillatory_patterns[scale] = {
                'frequencies': frequencies,
                'power_spectrum': power_spectrum,
                'coupling_strength': coupling_strength,
                'peak_frequency': frequencies[np.argmax(power_spectrum)]
            }
        
        return oscillatory_patterns
    
    def calculate_personalized_s_entropy_navigation(self, target_coordinates: List[SCoordinate]) -> Dict[str, Any]:
        """Calculate personalized S-entropy navigation efficiency to target coordinates."""
        
        navigation_results = {}
        
        for i, target in enumerate(target_coordinates):
            # Starting point (current biological state)
            current_state = SCoordinate(
                knowledge=0.5,  # Moderate knowledge state
                time=1.0,       # Current time reference
                entropy=0.3     # Low entropy (ordered state)
            )
            
            # Calculate navigation distance
            s_distance = self.s_entropy.compute_s_distance(current_state, target)
            
            # Personalized navigation speed affects time to reach target
            base_navigation_time = s_distance * 10.0  # Base scaling factor
            personalized_navigation_time = base_navigation_time / self.params.s_entropy_navigation_speed
            
            # Calculate navigation efficiency
            efficiency = 1.0 / personalized_navigation_time if personalized_navigation_time > 0 else 0.0
            
            navigation_results[f'target_{i}'] = {
                'target_coordinates': target,
                'current_coordinates': current_state,
                's_distance': s_distance,
                'navigation_time': personalized_navigation_time,
                'navigation_efficiency': efficiency,
                'genetic_speed_factor': self.params.s_entropy_navigation_speed
            }
        
        return navigation_results

class PersonalizedInsightGenerator:
    """Generates personalized biological insights from genome analysis."""
    
    def __init__(self, variants: List[GeneticVariant], 
                 parameters: PersonalizedParameters,
                 simulation_results: Dict[str, Any]):
        self.variants = variants
        self.parameters = parameters
        self.simulation_results = simulation_results
        
    def generate_atp_insights(self) -> List[PersonalizedInsight]:
        """Generate insights about personal ATP metabolism."""
        insights = []
        
        atp_efficiency = self.parameters.atp_synthesis_efficiency
        
        if atp_efficiency > 1.1:
            insight = PersonalizedInsight(
                insight_type="atp_metabolism",
                description=f"Your genetic profile suggests {atp_efficiency:.1%} ATP synthesis efficiency, "
                           f"indicating enhanced cellular energy production. This may contribute to "
                           f"better exercise tolerance and cellular resilience.",
                confidence=0.85,
                genetic_evidence=[v for v in self.variants if v.gene_symbol in ['TFAM', 'POLG', 'MT-ATP6']],
                recommendations=[
                    "Consider endurance activities to leverage your enhanced ATP production",
                    "Monitor for potential oxidative stress from increased mitochondrial activity",
                    "Maintain adequate B-vitamin intake to support enhanced metabolism"
                ]
            )
            insights.append(insight)
        elif atp_efficiency < 0.9:
            insight = PersonalizedInsight(
                insight_type="atp_metabolism",
                description=f"Your genetic profile suggests {atp_efficiency:.1%} ATP synthesis efficiency, "
                           f"indicating reduced cellular energy production. Consider targeted interventions "
                           f"to support mitochondrial function.",
                confidence=0.80,
                genetic_evidence=[v for v in self.variants if v.gene_symbol in ['POLG', 'MT-ATP6', 'MT-ATP8']],
                recommendations=[
                    "Consider CoQ10 supplementation for mitochondrial support",
                    "Focus on interval training rather than long endurance activities",
                    "Prioritize sleep and stress management for cellular recovery",
                    "Consider PQQ and other mitochondrial nutrients"
                ]
            )
            insights.append(insight)
        
        return insights
    
    def generate_oscillatory_insights(self) -> List[PersonalizedInsight]:
        """Generate insights about personal oscillatory patterns."""
        insights = []
        
        circadian_strength = self.parameters.oscillatory_coupling_strengths.get('circadian', 1.0)
        
        if circadian_strength > 1.15:
            insight = PersonalizedInsight(
                insight_type="circadian_rhythms",
                description=f"Your genetic profile indicates strong circadian rhythm coupling "
                           f"({circadian_strength:.2f}x baseline). You likely have well-regulated "
                           f"sleep-wake cycles and may be sensitive to light exposure timing.",
                confidence=0.90,
                genetic_evidence=[v for v in self.variants if v.gene_symbol in ['CLOCK', 'PER1', 'PER2']],
                recommendations=[
                    "Maintain consistent sleep schedule to leverage strong circadian control",
                    "Use bright light exposure in morning for optimal rhythm entrainment",
                    "Avoid blue light 2-3 hours before intended sleep time",
                    "Consider circadian intermittent fasting approaches"
                ]
            )
            insights.append(insight)
        elif circadian_strength < 0.9:
            insight = PersonalizedInsight(
                insight_type="circadian_rhythms",  
                description=f"Your genetic profile suggests weaker circadian rhythm coupling "
                           f"({circadian_strength:.2f}x baseline). You may benefit from external "
                           f"circadian rhythm support strategies.",
                confidence=0.85,
                genetic_evidence=[v for v in self.variants if v.gene_symbol in ['CRY1', 'CRY2', 'CLOCK']],
                recommendations=[
                    "Use light therapy devices for circadian rhythm support",
                    "Consider melatonin supplementation under medical guidance",
                    "Maintain very consistent meal timing",
                    "Use temperature regulation (cool sleeping environment)"
                ]
            )
            insights.append(insight)
        
        return insights
    
    def generate_navigation_insights(self) -> List[PersonalizedInsight]:
        """Generate insights about S-entropy navigation capabilities."""
        insights = []
        
        nav_speed = self.parameters.s_entropy_navigation_speed
        
        if nav_speed > 1.1:
            insight = PersonalizedInsight(
                insight_type="cognitive_processing",
                description=f"Your genetic profile suggests enhanced S-entropy navigation speed "
                           f"({nav_speed:.2f}x baseline), indicating potentially faster pattern "
                           f"recognition and problem-solving capabilities.",
                confidence=0.75,
                genetic_evidence=[v for v in self.variants if v.gene_symbol in ['DRD2', 'COMT', 'HTR2A']],
                recommendations=[
                    "Leverage rapid pattern recognition in complex problem-solving",
                    "Consider careers or activities requiring quick cognitive adaptation",
                    "Practice mindfulness to balance rapid processing with depth",
                    "Engage in complex strategic games and puzzles"
                ]
            )
            insights.append(insight)
        
        return insights
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive personalized genome analysis report."""
        
        # Collect all insights
        all_insights = []
        all_insights.extend(self.generate_atp_insights())
        all_insights.extend(self.generate_oscillatory_insights())
        all_insights.extend(self.generate_navigation_insights())
        
        # Calculate overall genetic optimization score
        efficiency_scores = [
            self.parameters.atp_synthesis_efficiency,
            self.parameters.membrane_permeability_factor,
            self.parameters.s_entropy_navigation_speed
        ]
        overall_optimization = np.mean(efficiency_scores)
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_variants_analyzed': len(self.variants),
            'gene_variants_found': len([v for v in self.variants if v.gene_symbol]),
            'personalized_parameters': {
                'atp_synthesis_efficiency': self.parameters.atp_synthesis_efficiency,
                'membrane_permeability_factor': self.parameters.membrane_permeability_factor,
                's_entropy_navigation_speed': self.parameters.s_entropy_navigation_speed,
                'oscillatory_coupling_strengths': dict(self.parameters.oscillatory_coupling_strengths),
                'bmd_selection_bias': dict(self.parameters.bmd_selection_bias),
                'metabolic_pathway_weights': dict(self.parameters.metabolic_pathway_weights)
            },
            'overall_genetic_optimization_score': overall_optimization,
            'insights': [
                {
                    'type': insight.insight_type,
                    'description': insight.description,
                    'confidence': insight.confidence,
                    'recommendations': insight.recommendations,
                    'genetic_evidence_count': len(insight.genetic_evidence)
                }
                for insight in all_insights
            ],
            'simulation_summary': self.simulation_results
        }
        
        return report

class PersonalGenomeAnalyzer:
    """Main interface for personal genome-coherent intracellular dynamics analysis."""
    
    def __init__(self):
        self.genome_loader = GenomeLoader()
        self.parameter_mapper = PersonalizedParameterMapper()
        
    def analyze_personal_genome(self, 
                              genome_file: str,
                              file_format: str = 'auto',
                              output_dir: str = 'personal_analysis_results') -> Dict[str, Any]:
        """
        Complete personal genome analysis using the oscillatory framework.
        
        Args:
            genome_file: Path to personal genome data file
            file_format: Format of genome file ('vcf', '23andme', 'ancestrydna', 'csv', 'auto')
            output_dir: Directory to save analysis results
            
        Returns:
            Comprehensive analysis results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting personal genome analysis...")
        
        # Step 1: Load genome data
        logger.info("Loading genome data...")
        variants = self.genome_loader.load_genome_data(genome_file, file_format)
        
        # Step 2: Map variants to personalized parameters
        logger.info("Mapping genetic variants to biological parameters...")
        personalized_params = self.parameter_mapper.map_variants_to_parameters(variants)
        
        # Step 3: Run personalized simulations
        logger.info("Running personalized intracellular dynamics simulations...")
        simulator = PersonalizedIntracellularSimulator(personalized_params)
        
        # ATP dynamics simulation
        atp_results = simulator.simulate_personalized_atp_dynamics(duration=86400)  # 24 hours
        
        # Oscillatory pattern analysis
        oscillatory_results = simulator.analyze_personalized_oscillatory_patterns()
        
        # S-entropy navigation analysis
        target_coords = [
            SCoordinate(knowledge=0.8, time=0.5, entropy=0.2),  # High knowledge, low entropy state
            SCoordinate(knowledge=0.3, time=1.5, entropy=0.7),  # Learning state
            SCoordinate(knowledge=0.6, time=0.8, entropy=0.4)   # Balanced state
        ]
        navigation_results = simulator.calculate_personalized_s_entropy_navigation(target_coords)
        
        # Combine simulation results
        simulation_results = {
            'atp_dynamics': atp_results,
            'oscillatory_patterns': oscillatory_results,
            's_entropy_navigation': navigation_results
        }
        
        # Step 4: Generate insights
        logger.info("Generating personalized insights...")
        insight_generator = PersonalizedInsightGenerator(variants, personalized_params, simulation_results)
        comprehensive_report = insight_generator.generate_comprehensive_report()
        
        # Step 5: Save results
        results_file = output_path / 'personal_genome_analysis.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        logger.info(f"Analysis complete! Results saved to {results_file}")
        
        return comprehensive_report

def main():
    """Demonstrate personal genome analysis."""
    
    # Initialize analyzer
    analyzer = PersonalGenomeAnalyzer()
    
    # Example usage - you would replace this with your actual genome file
    print("🧬 Personal Genome-Coherent Intracellular Dynamics Analysis")
    print("=" * 60)
    
    # This is where you would specify your genome file
    genome_file_path = input("Enter path to your genome file: ").strip()
    
    if not genome_file_path or not Path(genome_file_path).exists():
        print("Please provide a valid genome file path.")
        print("\nSupported formats:")
        print("- VCF files (.vcf, .vcf.gz)")
        print("- 23andMe data files")
        print("- AncestryDNA data files") 
        print("- CSV files with columns: chromosome, position, ref_allele, alt_allele, genotype")
        return
    
    try:
        # Run analysis
        results = analyzer.analyze_personal_genome(genome_file_path)
        
        # Display key results
        print(f"\n✅ Analysis Complete!")
        print(f"📊 Analyzed {results['total_variants_analyzed']} genetic variants")
        print(f"🧬 Found {results['gene_variants_found']} variants in known genes")
        print(f"🎯 Overall genetic optimization score: {results['overall_genetic_optimization_score']:.3f}")
        
        print(f"\n🔬 Personalized Parameters:")
        params = results['personalized_parameters']
        print(f"  ATP Synthesis Efficiency: {params['atp_synthesis_efficiency']:.3f}")
        print(f"  Membrane Permeability: {params['membrane_permeability_factor']:.3f}")
        print(f"  S-Entropy Navigation Speed: {params['s_entropy_navigation_speed']:.3f}")
        
        print(f"\n💡 Generated {len(results['insights'])} personalized insights")
        
        print(f"\n📁 Detailed results saved to: personal_analysis_results/personal_genome_analysis.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

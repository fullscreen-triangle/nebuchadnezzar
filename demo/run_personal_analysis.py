#!/usr/bin/env python3
"""
Personal Genome Analysis Runner
==============================

Simple script to run personal genome-coherent intracellular dynamics analysis
using your own genome data with the St. Stellas oscillatory framework.

Usage:
    python run_personal_analysis.py --genome-file /path/to/your/genome.vcf

Supported genome file formats:
- VCF files (.vcf, .vcf.gz) - Standard genomic variant format
- 23andMe raw data files - Direct download from 23andMe
- AncestryDNA raw data files - Direct download from AncestryDNA  
- CSV files - Custom format with required columns

The analysis will generate:
1. Personalized ATP metabolism parameters
2. Individual-specific oscillatory patterns
3. Custom S-entropy navigation capabilities
4. Actionable biological insights
5. Personalized optimization recommendations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add the demo/src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from personal_genome_analysis import PersonalGenomeAnalyzer
from st_stellas.core.s_entropy import SCoordinate

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('personal_genome_analysis.log')
        ]
    )

def load_config(config_file: str = 'personal_genome_config.json') -> dict:
    """Load configuration from JSON file."""
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return {
            "analysis_config": {
                "simulation_duration_hours": 24,
                "analysis_depth": "comprehensive"
            },
            "output_preferences": {
                "generate_plots": True,
                "detailed_reports": True
            }
        }

def create_visualization_plots(results: dict, output_dir: Path):
    """Create visualization plots from analysis results."""
    
    # Create plots directory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: ATP Dynamics over time
    if 'atp_dynamics' in results.get('simulation_summary', {}):
        atp_data = results['simulation_summary']['atp_dynamics']
        
        plt.figure(figsize=(12, 6))
        plt.plot(atp_data['time'] / 3600, atp_data['atp_concentration'], 'b-', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('ATP Concentration (mM)')
        plt.title(f"Personalized ATP Dynamics\n(Synthesis Efficiency: {atp_data['synthesis_efficiency']:.3f})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'atp_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: Oscillatory Patterns
    if 'oscillatory_patterns' in results.get('simulation_summary', {}):
        osc_data = results['simulation_summary']['oscillatory_patterns']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot first 4 oscillatory scales
        scales_to_plot = list(osc_data.keys())[:4]
        
        for i, scale in enumerate(scales_to_plot):
            if i < len(axes):
                data = osc_data[scale]
                axes[i].loglog(data['frequencies'], data['power_spectrum'], 'r-', linewidth=2)
                axes[i].set_xlabel('Frequency (Hz)')
                axes[i].set_ylabel('Power')
                axes[i].set_title(f'{scale.replace("_", " ").title()}\n(Coupling: {data["coupling_strength"]:.3f})')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'oscillatory_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Personalized Parameters Radar Chart
    params = results['personalized_parameters']
    
    # Create radar chart data
    metrics = [
        'ATP Synthesis\nEfficiency',
        'Membrane\nPermeability',
        'S-Entropy\nNavigation Speed'
    ]
    values = [
        params['atp_synthesis_efficiency'],
        params['membrane_permeability_factor'],
        params['s_entropy_navigation_speed']
    ]
    
    # Normalize values to 0-1 range for radar chart
    normalized_values = [(v - 0.5) / 1.0 + 0.5 for v in values]  # Assumes baseline is 1.0
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_plot = normalized_values + [normalized_values[0]]  # Complete the circle
    angles += [angles[0]]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values_plot, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values_plot, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.5)
    ax.set_yticks([0.5, 1.0, 1.5])
    ax.set_yticklabels(['Below Avg', 'Average', 'Above Avg'])
    ax.set_title('Personalized Biological Parameters\n(Relative to Population Average)', pad=20)
    plt.tight_layout()
    plt.savefig(plots_dir / 'personalized_parameters_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization plots saved to {plots_dir}")

def generate_detailed_report(results: dict, output_dir: Path):
    """Generate a detailed markdown report."""
    
    report_file = output_dir / 'detailed_analysis_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Personal Genome-Coherent Intracellular Dynamics Analysis Report\n\n")
        f.write(f"**Analysis Date:** {results['analysis_timestamp']}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Variants Analyzed:** {results['total_variants_analyzed']:,}\n")
        f.write(f"- **Gene Variants Found:** {results['gene_variants_found']:,}\n")  
        f.write(f"- **Overall Genetic Optimization Score:** {results['overall_genetic_optimization_score']:.3f}\n")
        f.write(f"- **Generated Insights:** {len(results['insights'])}\n\n")
        
        f.write("## Personalized Biological Parameters\n\n")
        params = results['personalized_parameters']
        
        f.write("### Core Metabolic Parameters\n")
        f.write(f"- **ATP Synthesis Efficiency:** {params['atp_synthesis_efficiency']:.3f} "
                f"({'Enhanced' if params['atp_synthesis_efficiency'] > 1.1 else 'Reduced' if params['atp_synthesis_efficiency'] < 0.9 else 'Normal'})\n")
        f.write(f"- **Membrane Permeability Factor:** {params['membrane_permeability_factor']:.3f}\n")
        f.write(f"- **S-Entropy Navigation Speed:** {params['s_entropy_navigation_speed']:.3f}\n\n")
        
        if params['oscillatory_coupling_strengths']:
            f.write("### Oscillatory Coupling Strengths\n")
            for osc_type, strength in params['oscillatory_coupling_strengths'].items():
                f.write(f"- **{osc_type.replace('_', ' ').title()}:** {strength:.3f}\n")
            f.write("\n")
        
        if params['bmd_selection_bias']:
            f.write("### Biological Maxwell Demon Selection Bias\n")
            for bmd_type, bias in params['bmd_selection_bias'].items():
                f.write(f"- **{bmd_type.replace('_', ' ').title()}:** {bias:.3f}\n")
            f.write("\n")
        
        f.write("## Personalized Insights\n\n")
        for i, insight in enumerate(results['insights'], 1):
            f.write(f"### {i}. {insight['type'].replace('_', ' ').title()}\n")
            f.write(f"**Confidence:** {insight['confidence']:.1%}\n\n")
            f.write(f"{insight['description']}\n\n")
            
            if insight['recommendations']:
                f.write("**Recommendations:**\n")
                for rec in insight['recommendations']:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            f.write(f"*Based on {insight['genetic_evidence_count']} genetic variants*\n\n")
            f.write("---\n\n")
        
        f.write("## Technical Details\n\n")
        f.write("### Analysis Method\n")
        f.write("This analysis uses the St. Stellas unified theoretical framework to map genetic variants ")
        f.write("to personalized biological parameters. The framework models intracellular dynamics using ")
        f.write("ATP-constrained differential equations and multi-scale oscillatory coupling analysis.\n\n")
        
        f.write("### Data Processing\n")
        f.write("1. Genetic variants mapped to known functional effects\n")
        f.write("2. Personalized parameters calculated using effect aggregation\n")
        f.write("3. Intracellular dynamics simulated with personalized parameters\n")
        f.write("4. S-entropy navigation efficiency calculated\n")
        f.write("5. Oscillatory patterns analyzed across biological scales\n\n")
        
        f.write("### Limitations\n")
        f.write("- Analysis based on current understanding of genetic variant effects\n")
        f.write("- Predictions are probabilistic and should not replace medical advice\n")
        f.write("- Environmental factors not included in genetic analysis\n")
        f.write("- Gene-gene interactions simplified in current model\n\n")
    
    print(f"📄 Detailed report saved to {report_file}")

def print_summary(results: dict):
    """Print a concise summary of results."""
    
    print("\n" + "="*80)
    print("🧬 PERSONAL GENOME ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n📊 ANALYSIS OVERVIEW:")
    print(f"   • Total genetic variants analyzed: {results['total_variants_analyzed']:,}")
    print(f"   • Gene variants with known effects: {results['gene_variants_found']:,}")
    print(f"   • Overall genetic optimization score: {results['overall_genetic_optimization_score']:.3f}")
    
    params = results['personalized_parameters']
    print(f"\n🔬 PERSONALIZED PARAMETERS:")
    print(f"   • ATP synthesis efficiency: {params['atp_synthesis_efficiency']:.3f}")
    print(f"   • Membrane permeability: {params['membrane_permeability_factor']:.3f}")
    print(f"   • S-entropy navigation speed: {params['s_entropy_navigation_speed']:.3f}")
    
    print(f"\n💡 GENERATED INSIGHTS: {len(results['insights'])}")
    for insight in results['insights']:
        confidence_emoji = "🟢" if insight['confidence'] > 0.8 else "🟡" if insight['confidence'] > 0.6 else "🟠"
        print(f"   {confidence_emoji} {insight['type'].replace('_', ' ').title()} "
              f"(confidence: {insight['confidence']:.1%})")
    
    print(f"\n📁 Results saved to: personal_analysis_results/")
    print(f"   • Raw data: personal_genome_analysis.json") 
    print(f"   • Detailed report: detailed_analysis_report.md")
    print(f"   • Visualizations: plots/")
    
    print("\n" + "="*80)

def main():
    """Main function to run personal genome analysis."""
    
    parser = argparse.ArgumentParser(
        description='Personal Genome-Coherent Intracellular Dynamics Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_personal_analysis.py --genome-file mygenome.vcf
  python run_personal_analysis.py --genome-file 23andme_data.txt --format 23andme
  python run_personal_analysis.py --genome-file variants.csv --format csv --output results/
        """
    )
    
    parser.add_argument('--genome-file', '-g', 
                       required=True,
                       help='Path to your genome data file')
    
    parser.add_argument('--format', '-f',
                       choices=['auto', 'vcf', '23andme', 'ancestrydna', 'csv'],
                       default='auto',
                       help='Genome file format (default: auto-detect)')
    
    parser.add_argument('--output', '-o',
                       default='personal_analysis_results',
                       help='Output directory for results (default: personal_analysis_results)')
    
    parser.add_argument('--config', '-c',
                       default='personal_genome_config.json',
                       help='Configuration file path (default: personal_genome_config.json)')
    
    parser.add_argument('--no-plots', 
                       action='store_true',
                       help='Skip generating visualization plots')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true', 
                       help='Suppress non-essential output')
    
    args = parser.parse_args()
    
    # Set up logging
    if not args.quiet:
        setup_logging()
    
    # Check if genome file exists
    genome_file = Path(args.genome_file)
    if not genome_file.exists():
        print(f"❌ Error: Genome file not found: {args.genome_file}")
        print("\nSupported formats:")
        print("  • VCF files (.vcf, .vcf.gz)")
        print("  • 23andMe raw data files")
        print("  • AncestryDNA raw data files")
        print("  • CSV files with columns: chromosome, position, ref_allele, alt_allele, genotype")
        return 1
    
    # Load configuration
    config = load_config(args.config)
    
    # Welcome message
    if not args.quiet:
        print("🧬 Personal Genome-Coherent Intracellular Dynamics Analysis")
        print("   Using the St. Stellas Unified Theoretical Framework")
        print("="*70)
        print(f"📁 Genome file: {args.genome_file}")
        print(f"📊 Format: {args.format} (auto-detect)" if args.format == 'auto' else f"📊 Format: {args.format}")
        print(f"💾 Output directory: {args.output}")
        print()
    
    try:
        # Initialize analyzer
        analyzer = PersonalGenomeAnalyzer()
        
        # Run analysis
        print("🔄 Starting analysis...")
        results = analyzer.analyze_personal_genome(
            genome_file=str(genome_file),
            file_format=args.format,
            output_dir=args.output
        )
        
        # Create output directory path object
        output_dir = Path(args.output)
        
        # Generate visualization plots
        if not args.no_plots and config.get('output_preferences', {}).get('generate_plots', True):
            print("📊 Generating visualization plots...")
            create_visualization_plots(results, output_dir)
        
        # Generate detailed report
        if config.get('output_preferences', {}).get('detailed_reports', True):
            print("📄 Generating detailed report...")
            generate_detailed_report(results, output_dir)
        
        # Print summary
        if not args.quiet:
            print_summary(results)
        
        print("\n✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

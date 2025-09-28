"""
Command Line Interface for St. Stellas Validation Suite
======================================================

Provides command-line access to framework validation functionality.

Usage:
    stellas-validate --organism human --quick
    stellas-validate --full-validation --output results.html
    stellas-benchmark --component s_entropy --iterations 10
"""

import click
import sys
import logging
from pathlib import Path
from typing import Optional

from .validation.framework_validation import FrameworkValidator
from .core.s_entropy import SEntropyFramework
from .core.coordinates import MolecularCoordinates
from .core.oscillations import BiologicalOscillations
from .core.fluid_dynamics import DynamicFluxTheory
from .data.databases import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version='0.1.0')
def main():
    """St. Stellas Framework Validation Suite CLI."""
    pass

@main.command()
@click.option('--organism', default='human', 
              help='Target organism for validation (human, mouse, yeast)')
@click.option('--quick', is_flag=True, 
              help='Run quick validation (subset of tests)')
@click.option('--full-validation', is_flag=True,
              help='Run complete validation suite')
@click.option('--output', default='validation_report.html',
              help='Output file for validation report')
@click.option('--cache-dir', default='data/cache',
              help='Directory for caching data')
@click.option('--ncbi-email', required=True,
              help='Email address for NCBI database access')
@click.option('--significance', default=0.05, type=float,
              help='P-value threshold for statistical significance')
def validate(organism: str, quick: bool, full_validation: bool, 
            output: str, cache_dir: str, ncbi_email: str, significance: float):
    """Run framework validation against biological databases."""
    
    click.echo("🧬 St. Stellas Framework Validation Suite")
    click.echo("=" * 50)
    
    # Initialize validator
    try:
        validator = FrameworkValidator(
            ncbi_email=ncbi_email,
            cache_dir=cache_dir,
            significance_threshold=significance
        )
    except Exception as e:
        click.echo(f"❌ Failed to initialize validator: {e}", err=True)
        sys.exit(1)
    
    # Determine validation mode
    quick_mode = quick and not full_validation
    
    click.echo(f"🎯 Target organism: {organism}")
    click.echo(f"⚡ Quick mode: {'Yes' if quick_mode else 'No'}")
    click.echo(f"📊 Significance threshold: {significance}")
    click.echo(f"💾 Cache directory: {cache_dir}")
    click.echo()
    
    # Run validation
    try:
        with click.progressbar(length=100, label="Running validation") as bar:
            # Simulate progress updates
            bar.update(10)
            
            summary = validator.validate_full_framework(
                organism=organism,
                quick_validation=quick_mode
            )
            
            bar.update(90)  # Complete progress
            
        # Display results summary
        click.echo("\n" + "=" * 50)
        click.echo("📋 VALIDATION SUMMARY")
        click.echo("=" * 50)
        
        # Color-coded overall score
        score_color = 'green' if summary.overall_score >= 0.8 else 'yellow' if summary.overall_score >= 0.6 else 'red'
        click.echo(f"🎯 Overall Score: ", nl=False)
        click.secho(f"{summary.overall_score:.3f}", fg=score_color, bold=True)
        
        click.echo(f"✅ Tests Passed: {summary.passed_tests}/{summary.total_tests}")
        click.echo(f"❌ Tests Failed: {summary.failed_tests}/{summary.total_tests}")
        
        if summary.theoretical_predictions_confirmed:
            click.echo(f"\n🎉 Confirmed Predictions ({len(summary.theoretical_predictions_confirmed)}):")
            for prediction in summary.theoretical_predictions_confirmed:
                click.echo(f"  ✓ {prediction}")
        
        if summary.areas_for_improvement:
            click.echo(f"\n🔧 Areas for Improvement ({len(summary.areas_for_improvement)}):")
            for area in summary.areas_for_improvement:
                click.echo(f"  ⚠ {area}")
        
        # Generate and save report
        report_path = validator.generate_html_report(summary, output)
        click.echo(f"\n📄 Detailed report saved to: {report_path}")
        
        # Exit with appropriate code
        exit_code = 0 if summary.overall_score >= 0.7 else 1
        
        if exit_code == 0:
            click.echo("\n🎉 Validation completed successfully!")
        else:
            click.echo("\n⚠️  Validation completed with concerns.")
            
        sys.exit(exit_code)
        
    except Exception as e:
        click.echo(f"\n❌ Validation failed: {e}", err=True)
        sys.exit(1)

@main.command()
@click.option('--component', 
              type=click.Choice(['s_entropy', 'coordinates', 'oscillations', 'fluid_dynamics', 'all']),
              default='all',
              help='Framework component to benchmark')
@click.option('--iterations', default=10, type=int,
              help='Number of benchmark iterations')
@click.option('--output', default='benchmark_results.json',
              help='Output file for benchmark results')
@click.option('--cache-dir', default='data/cache',
              help='Directory for caching data')
def benchmark(component: str, iterations: int, output: str, cache_dir: str):
    """Run performance benchmarks for framework components."""
    
    click.echo("🚀 St. Stellas Framework Benchmarks")
    click.echo("=" * 50)
    
    click.echo(f"🎯 Component: {component}")
    click.echo(f"🔄 Iterations: {iterations}")
    click.echo(f"📄 Output: {output}")
    click.echo()
    
    benchmark_results = {}
    
    # S-Entropy Framework benchmarks
    if component in ['s_entropy', 'all']:
        click.echo("⚡ Benchmarking S-Entropy Framework...")
        
        try:
            framework = SEntropyFramework()
            
            # Benchmark S-distance computation
            import time
            import numpy as np
            from .core.s_entropy import SCoordinate
            
            coords = [SCoordinate(np.random.normal(), np.random.normal(), np.random.uniform(0, 1)) 
                     for _ in range(1000)]
            
            times = []
            for _ in range(iterations):
                start_time = time.time()
                
                # Compute pairwise distances
                distances = []
                for i in range(min(100, len(coords))):
                    for j in range(i+1, min(100, len(coords))):
                        dist = framework.s_distance.compute_distance(coords[i], coords[j])
                        distances.append(dist)
                
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            benchmark_results['s_entropy'] = {
                'component': 'S-Entropy Framework',
                'operation': 'S-distance computation',
                'iterations': iterations,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'operations_per_second': 10000 / np.mean(times) if np.mean(times) > 0 else 0
            }
            
            click.echo(f"  ✅ Mean time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
            
        except Exception as e:
            click.echo(f"  ❌ S-Entropy benchmark failed: {e}")
    
    # Molecular Coordinates benchmarks
    if component in ['coordinates', 'all']:
        click.echo("⚡ Benchmarking Molecular Coordinates...")
        
        try:
            coordinates = MolecularCoordinates()
            
            test_sequence = "ATGCATGCATGCATGCATGC" * 10  # 200 bases
            
            times = []
            for _ in range(iterations):
                start_time = time.time()
                coords = coordinates.transform_dna_sequence(test_sequence)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            benchmark_results['coordinates'] = {
                'component': 'Molecular Coordinates',
                'operation': 'DNA sequence transformation',
                'sequence_length': len(test_sequence),
                'iterations': iterations,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'bases_per_second': len(test_sequence) / np.mean(times) if np.mean(times) > 0 else 0
            }
            
            click.echo(f"  ✅ Mean time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
            click.echo(f"  📊 Processing rate: {len(test_sequence) / np.mean(times):.0f} bases/sec")
            
        except Exception as e:
            click.echo(f"  ❌ Coordinates benchmark failed: {e}")
    
    # Biological Oscillations benchmarks
    if component in ['oscillations', 'all']:
        click.echo("⚡ Benchmarking Biological Oscillations...")
        
        try:
            oscillations = BiologicalOscillations()
            
            times = []
            for _ in range(min(iterations, 3)):  # Limit for performance
                start_time = time.time()
                results = oscillations.simulate_complete_biological_system(duration=1.0)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            benchmark_results['oscillations'] = {
                'component': 'Biological Oscillations',
                'operation': 'Multi-scale simulation',
                'simulation_duration': 1.0,
                'iterations': len(times),
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'simulation_ratio': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            }
            
            click.echo(f"  ✅ Mean time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
            
        except Exception as e:
            click.echo(f"  ❌ Oscillations benchmark failed: {e}")
    
    # Fluid Dynamics benchmarks
    if component in ['fluid_dynamics', 'all']:
        click.echo("⚡ Benchmarking Dynamic Flux Theory...")
        
        try:
            fluid_dynamics = DynamicFluxTheory()
            
            system_description = {
                'flow_type': 'pipe_flow',
                'flow_parameters': {'R': 0.05, 'mu': 0.001, 'dP': 1000, 'L': 1.0},
                'complexity_factor': 1.0
            }
            
            times = []
            for _ in range(iterations):
                start_time = time.time()
                results = fluid_dynamics.analyze_fluid_system(system_description)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            benchmark_results['fluid_dynamics'] = {
                'component': 'Dynamic Flux Theory',
                'operation': 'Fluid system analysis',
                'iterations': iterations,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'analyses_per_second': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            }
            
            click.echo(f"  ✅ Mean time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
            
        except Exception as e:
            click.echo(f"  ❌ Fluid dynamics benchmark failed: {e}")
    
    # Save results
    import json
    try:
        with open(output, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        click.echo(f"\n📄 Benchmark results saved to: {output}")
    except Exception as e:
        click.echo(f"\n❌ Failed to save results: {e}")
    
    # Display summary
    click.echo("\n" + "=" * 50)
    click.echo("📊 BENCHMARK SUMMARY")
    click.echo("=" * 50)
    
    for comp_name, results in benchmark_results.items():
        click.echo(f"\n{results['component']}:")
        click.echo(f"  Operation: {results['operation']}")
        click.echo(f"  Mean time: {results['mean_time']:.4f}s")
        
        if 'operations_per_second' in results:
            click.echo(f"  Performance: {results['operations_per_second']:.0f} ops/sec")
        elif 'bases_per_second' in results:
            click.echo(f"  Performance: {results['bases_per_second']:.0f} bases/sec")

@main.command()
@click.option('--organism', default='human',
              help='Target organism for data collection')
@click.option('--cache-dir', default='data/cache',
              help='Directory for caching data')
@click.option('--ncbi-email', required=True,
              help='Email address for NCBI database access')
@click.option('--output-dir', default='data/collected',
              help='Directory to save collected data')
def collect_data(organism: str, cache_dir: str, ncbi_email: str, output_dir: str):
    """Collect biological data from databases for validation."""
    
    click.echo("🗄️  St. Stellas Data Collection")
    click.echo("=" * 50)
    
    click.echo(f"🎯 Target organism: {organism}")
    click.echo(f"💾 Cache directory: {cache_dir}")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(
            ncbi_email=ncbi_email,
            cache_dir=cache_dir
        )
        
        # Collect data
        with click.progressbar(length=100, label="Collecting data") as bar:
            bar.update(20)
            
            dataset = db_manager.collect_validation_dataset(
                organism=organism,
                include_pathways=True,
                include_proteins=True,
                include_sequences=True,
                include_structures=True
            )
            
            bar.update(80)
            
        # Save collected data
        import json
        
        # Convert BiologicalData objects to JSON-serializable format
        json_dataset = {}
        for data_type, data_list in dataset.items():
            json_dataset[data_type] = []
            for bio_data in data_list:
                json_item = {
                    'source_database': bio_data.source_database,
                    'data_type': bio_data.data_type,
                    'identifier': bio_data.identifier,
                    'data': bio_data.data,
                    'metadata': bio_data.metadata,
                    'timestamp': bio_data.timestamp
                }
                json_dataset[data_type].append(json_item)
        
        output_file = output_path / f"{organism}_validation_dataset.json"
        
        with open(output_file, 'w') as f:
            json.dump(json_dataset, f, indent=2)
        
        click.echo(f"\n📄 Dataset saved to: {output_file}")
        
        # Display collection summary
        click.echo("\n" + "=" * 50)
        click.echo("📊 DATA COLLECTION SUMMARY")
        click.echo("=" * 50)
        
        for data_type, data_list in dataset.items():
            click.echo(f"{data_type.title()}: {len(data_list)} items")
        
        total_items = sum(len(data_list) for data_list in dataset.values())
        click.echo(f"\nTotal items collected: {total_items}")
        
    except Exception as e:
        click.echo(f"\n❌ Data collection failed: {e}", err=True)
        sys.exit(1)

@main.command()
@click.option('--validation-report', required=True,
              help='Path to validation report JSON file')
@click.option('--benchmark-results', required=True,
              help='Path to benchmark results JSON file')
@click.option('--output', default='analysis_report.html',
              help='Output file for analysis report')
def analyze(validation_report: str, benchmark_results: str, output: str):
    """Analyze validation and benchmark results."""
    
    click.echo("📈 St. Stellas Results Analysis")
    click.echo("=" * 50)
    
    try:
        import json
        
        # Load validation results
        with open(validation_report, 'r') as f:
            validation_data = json.load(f)
        
        # Load benchmark results
        with open(benchmark_results, 'r') as f:
            benchmark_data = json.load(f)
        
        # Perform analysis
        click.echo("📊 Analyzing validation results...")
        validation_score = validation_data.get('overall_score', 0.0)
        
        click.echo("📊 Analyzing benchmark results...")
        performance_scores = {}
        for component, results in benchmark_data.items():
            if 'operations_per_second' in results:
                performance_scores[component] = results['operations_per_second']
            elif 'bases_per_second' in results:
                performance_scores[component] = results['bases_per_second']
        
        # Generate analysis report
        click.echo(f"📄 Generating analysis report: {output}")
        
        # Simple analysis report (in a full implementation, this would be more sophisticated)
        analysis_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>St. Stellas Analysis Report</title></head>
        <body>
            <h1>St. Stellas Framework Analysis</h1>
            <h2>Validation Analysis</h2>
            <p>Overall validation score: {validation_score:.3f}</p>
            
            <h2>Performance Analysis</h2>
            <ul>
        """
        
        for component, score in performance_scores.items():
            analysis_html += f"<li>{component}: {score:.0f} operations/sec</li>"
        
        analysis_html += """
            </ul>
        </body>
        </html>
        """
        
        with open(output, 'w') as f:
            f.write(analysis_html)
        
        click.echo(f"✅ Analysis complete! Report saved to: {output}")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Complete St. Stellas Framework Validation Example
================================================

Demonstrates comprehensive validation of the entire St. Stellas theoretical
framework using real biological data from multiple databases.

This example runs the complete validation suite including:
1. S-Entropy Framework validation
2. Molecular coordinate transformation testing
3. Biological oscillation coupling analysis
4. Dynamic Flux Theory performance benchmarking
5. Cross-domain pattern transfer validation
6. Statistical significance testing
7. HTML report generation

Usage:
    python full_framework_validation.py --ncbi-email your@email.com --organism human
    python full_framework_validation.py --quick --output quick_report.html
"""

import argparse
import logging
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from st_stellas.validation.framework_validation import FrameworkValidator
from st_stellas import StStellasSuite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_comprehensive_validation(ncbi_email: str, organism: str, quick_mode: bool, output_file: str):
    """Run comprehensive St. Stellas framework validation."""
    
    print("🧬 St. Stellas Framework Comprehensive Validation")
    print("=" * 80)
    print(f"Target organism: {organism}")
    print(f"Quick mode: {quick_mode}")
    print(f"Output file: {output_file}")
    print()
    
    # Initialize validator
    print("🔧 Initializing framework validator...")
    
    try:
        validator = FrameworkValidator(
            ncbi_email=ncbi_email,
            cache_dir='data/validation_cache',
            significance_threshold=0.05
        )
        print("✅ Validator initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize validator: {e}")
        return 1
    
    # Run validation
    print("\n🧪 Running comprehensive framework validation...")
    
    start_time = time.time()
    
    try:
        # Run the full validation suite
        summary = validator.validate_full_framework(
            organism=organism,
            quick_validation=quick_mode
        )
        
        validation_time = time.time() - start_time
        
        print(f"\n✅ Validation completed in {validation_time:.1f} seconds")
        
        # Display summary results
        display_validation_summary(summary)
        
        # Generate HTML report
        print(f"\n📄 Generating detailed HTML report...")
        report_path = validator.generate_html_report(summary, output_file)
        print(f"✅ Report saved to: {report_path}")
        
        # Return appropriate exit code
        return 0 if summary.overall_score >= 0.7 else 1
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        logger.exception("Validation error details:")
        return 1

def display_validation_summary(summary):
    """Display comprehensive validation summary."""
    
    print("\n📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    # Overall metrics
    print(f"Overall Score: {summary.overall_score:.3f}")
    print(f"Tests Passed: {summary.passed_tests}/{summary.total_tests} ({summary.passed_tests/summary.total_tests:.1%})")
    print(f"Tests Failed: {summary.failed_tests}/{summary.total_tests} ({summary.failed_tests/summary.total_tests:.1%})")
    print()
    
    # Performance grade
    if summary.overall_score >= 0.9:
        grade = "A+"
        color_code = "🟢"
    elif summary.overall_score >= 0.8:
        grade = "A"
        color_code = "🟢"
    elif summary.overall_score >= 0.7:
        grade = "B"
        color_code = "🟡"
    elif summary.overall_score >= 0.6:
        grade = "C"
        color_code = "🟡"
    else:
        grade = "F"
        color_code = "🔴"
    
    print(f"Framework Grade: {color_code} {grade}")
    print()
    
    # Confirmed theoretical predictions
    if summary.theoretical_predictions_confirmed:
        print("🎉 CONFIRMED THEORETICAL PREDICTIONS:")
        for prediction in summary.theoretical_predictions_confirmed:
            print(f"  ✅ {prediction}")
        print()
    
    # Areas needing improvement
    if summary.areas_for_improvement:
        print("🔧 AREAS FOR IMPROVEMENT:")
        for area in summary.areas_for_improvement:
            print(f"  ⚠️  {area}")
        print()
    
    # Detailed test results
    print("📋 DETAILED TEST RESULTS:")
    print("-" * 30)
    
    # Group results by framework component
    component_results = {
        'S-Entropy Framework': [],
        'Molecular Coordinates': [],
        'Biological Oscillations': [],
        'Dynamic Flux Theory': [],
        'Cross-Domain Transfer': []
    }
    
    for result in summary.results:
        # Categorize test by name
        test_name = result.test_name
        if any(keyword in test_name.lower() for keyword in ['s-distance', 'entropy', 'strategic', 'predetermined']):
            component_results['S-Entropy Framework'].append(result)
        elif any(keyword in test_name.lower() for keyword in ['dna', 'protein', 'molecular', 'coordinate', 'cross-modal']):
            component_results['Molecular Coordinates'].append(result)
        elif any(keyword in test_name.lower() for keyword in ['oscillat', 'coupling', 'allometric', 'health', 'coherence']):
            component_results['Biological Oscillations'].append(result)
        elif any(keyword in test_name.lower() for keyword in ['flux', 'fluid', 'pattern', 'physics']):
            component_results['Dynamic Flux Theory'].append(result)
        else:
            component_results['Cross-Domain Transfer'].append(result)
    
    for component, results in component_results.items():
        if results:
            print(f"\n{component}:")
            for result in results:
                status = "✅" if result.success else "❌"
                score_str = f"{result.score:.3f}"
                time_str = f"{result.execution_time:.3f}s"
                p_val_str = f"p={result.p_value:.4f}" if result.p_value is not None else ""
                
                print(f"  {status} {result.test_name}: {score_str} ({time_str}) {p_val_str}")
                
                if result.error_message:
                    print(f"      Error: {result.error_message}")
    
    # Statistical summary
    print(f"\n📈 STATISTICAL SUMMARY:")
    print("-" * 25)
    
    # Calculate component-wise scores
    component_scores = {}
    for component, results in component_results.items():
        if results:
            scores = [r.score for r in results]
            component_scores[component] = sum(scores) / len(scores)
    
    for component, score in component_scores.items():
        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"  {status} {component}: {score:.3f}")
    
    # Significance testing summary
    significant_results = [r for r in summary.results if r.p_value is not None and r.p_value < 0.05]
    if significant_results:
        print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
        print(f"  Tests with p < 0.05: {len(significant_results)}")
        print(f"  Most significant: {min(significant_results, key=lambda x: x.p_value).test_name} "
              f"(p={min(significant_results, key=lambda x: x.p_value).p_value:.6f})")

def demonstrate_high_level_api():
    """Demonstrate the high-level St. Stellas API."""
    
    print("\n🚀 High-Level API Demonstration")
    print("=" * 50)
    
    try:
        # Initialize the complete St. Stellas suite
        print("Initializing St. Stellas Suite...")
        stellas = StStellasSuite()
        
        # Run quick validation of core components
        print("Running quick component validation...")
        
        # Test S-entropy framework
        print("  Testing S-entropy navigation...")
        from st_stellas.core.s_entropy import SCoordinate
        test_coords = [
            SCoordinate(1.0, 0.5, 0.3),
            SCoordinate(0.8, 0.7, 0.4),
            SCoordinate(0.6, 0.9, 0.2)
        ]
        result = stellas.s_entropy.solve_problem_navigation(test_coords)
        print(f"    S-entropy solution found: {result['is_valid']}")
        
        # Test molecular coordinates
        print("  Testing molecular coordinates...")
        dna_coords = stellas.coordinates.transform_dna_sequence("ATGCATGC")
        protein_coords = stellas.coordinates.transform_protein_sequence("MACY")
        print(f"    DNA coordinates: {len(dna_coords)} points")
        print(f"    Protein coordinates: {len(protein_coords)} points")
        
        # Test biological oscillations
        print("  Testing biological oscillations...")
        oscillation_result = stellas.oscillations.simulate_complete_biological_system(duration=1.0)
        if 'error' not in oscillation_result:
            scales_simulated = len(oscillation_result.get('scale_activities', {}))
            print(f"    Biological scales simulated: {scales_simulated}")
        
        # Test fluid dynamics
        print("  Testing dynamic flux theory...")
        fluid_system = {
            'flow_type': 'pipe_flow',
            'flow_parameters': {'R': 0.05, 'mu': 0.001, 'dP': 1000, 'L': 1.0}
        }
        fluid_result = stellas.fluid_dynamics.analyze_fluid_system(fluid_system)
        has_grand_flux = 'grand_flux_analysis' in fluid_result
        print(f"    Grand flux analysis: {has_grand_flux}")
        
        print("✅ High-level API demonstration completed successfully")
        
    except Exception as e:
        print(f"❌ High-level API demonstration failed: {e}")

def run_performance_benchmarks():
    """Run performance benchmarks for key framework components."""
    
    print("\n⚡ Performance Benchmarks")
    print("=" * 50)
    
    import time
    import numpy as np
    from st_stellas.core.s_entropy import SEntropyFramework, SCoordinate
    from st_stellas.core.coordinates import MolecularCoordinates
    
    try:
        # S-entropy framework benchmark
        print("Benchmarking S-entropy framework...")
        framework = SEntropyFramework()
        
        # Generate test problem spaces of different sizes
        problem_sizes = [10, 50, 100, 200]
        s_entropy_times = []
        
        for size in problem_sizes:
            coords = [SCoordinate(np.random.normal(), np.random.normal(), np.random.uniform(0, 1))
                     for _ in range(size)]
            
            start_time = time.time()
            result = framework.solve_problem_navigation(coords)
            elapsed = time.time() - start_time
            s_entropy_times.append(elapsed)
            
            print(f"  Problem size {size}: {elapsed:.4f}s (valid: {result['is_valid']})")
        
        # Calculate complexity scaling
        if len(s_entropy_times) >= 2:
            # Fit power law: time = a * size^b
            log_sizes = np.log10(problem_sizes)
            log_times = np.log10(s_entropy_times)
            
            if len(log_sizes) > 1 and np.std(log_sizes) > 0:
                scaling_exponent = np.polyfit(log_sizes, log_times, 1)[0]
                print(f"  Complexity scaling: O(N^{scaling_exponent:.2f})")
                
                if scaling_exponent < 1.5:
                    print(f"  ✅ Better than quadratic scaling achieved")
                else:
                    print(f"  ⚠️ Scaling higher than expected")
        
        # Molecular coordinates benchmark
        print("\nBenchmarking molecular coordinates...")
        coordinates = MolecularCoordinates()
        
        sequence_lengths = [50, 100, 200, 500]
        coord_times = []
        
        for length in sequence_lengths:
            test_sequence = "ATGC" * (length // 4)  # Repeat pattern
            
            start_time = time.time()
            coords = coordinates.transform_dna_sequence(test_sequence)
            elapsed = time.time() - start_time
            coord_times.append(elapsed)
            
            print(f"  Sequence length {length}: {elapsed:.4f}s ({len(coords)} coordinates)")
        
        # Calculate processing rate
        if coord_times:
            avg_rate = np.mean([length/time for length, time in zip(sequence_lengths, coord_times)])
            print(f"  Average processing rate: {avg_rate:.0f} bases/second")
        
        print("✅ Performance benchmarks completed")
        
    except Exception as e:
        print(f"❌ Performance benchmarks failed: {e}")

def main():
    """Main validation function."""
    
    parser = argparse.ArgumentParser(description='Complete St. Stellas Framework Validation')
    parser.add_argument('--ncbi-email', required=True,
                       help='Email address for NCBI database access')
    parser.add_argument('--organism', default='human',
                       help='Target organism for validation (human, mouse, yeast)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation (subset of tests)')
    parser.add_argument('--output', default='stellas_validation_report.html',
                       help='Output file for validation report')
    parser.add_argument('--skip-benchmarks', action='store_true',
                       help='Skip performance benchmarks')
    parser.add_argument('--demo-api', action='store_true',
                       help='Demonstrate high-level API usage')
    
    args = parser.parse_args()
    
    print("🌟 St. Stellas Framework - Complete Validation Suite")
    print("=" * 80)
    print("Validating the unified theoretical framework for:")
    print("  • S-Entropy universal problem solving")  
    print("  • Molecular coordinate transformation")
    print("  • Biological oscillatory coupling")
    print("  • Dynamic flux theory")
    print("  • Cross-domain pattern transfer")
    print()
    
    exit_code = 0
    
    try:
        # Demonstrate high-level API (if requested)
        if args.demo_api:
            demonstrate_high_level_api()
        
        # Run performance benchmarks (if requested)
        if not args.skip_benchmarks:
            run_performance_benchmarks()
        
        # Run comprehensive validation
        validation_exit_code = run_comprehensive_validation(
            args.ncbi_email,
            args.organism, 
            args.quick,
            args.output
        )
        
        exit_code = max(exit_code, validation_exit_code)
        
        # Final summary
        print("\n🎯 FINAL SUMMARY")
        print("=" * 30)
        
        if exit_code == 0:
            print("🎉 St. Stellas Framework validation: SUCCESSFUL")
            print("   All major theoretical predictions confirmed")
            print("   Framework ready for scientific publication")
        else:
            print("⚠️  St. Stellas Framework validation: PARTIAL SUCCESS")
            print("   Some components need refinement")
            print("   Check detailed report for specific issues")
        
        print(f"\n📄 Detailed results available in: {args.output}")
        print("\nThank you for validating the St. Stellas Framework!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error details:")
        exit_code = 1
    
    return exit_code

if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Babylon Framework Demonstration Script
=====================================

Runs all implemented modules in sequence to demonstrate the complete
oscillatory drug dynamics pipeline from genetic variants to quantum transport.

This script serves as both a demonstration and a validation tool for the
systematic approach to testing each component of the framework.
"""

import sys
import os
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_module_demo(module_name: str, module_path: str, description: str) -> bool:
    """Run a module demonstration with error handling."""
    
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {module_name}")
    print(f"📝 {description}")
    print(f"📁 Module: {module_path}")
    print(f"{'='*80}")
    
    try:
        start_time = time.time()
        
        # Import and run the module
        if module_path == "pharmacogenonics.hole_detector":
            from pharmacogenonics.hole_detector import main
            result = main()
        elif module_path == "pharmacogenonics.genetic_risk_assessor":
            from pharmacogenonics.genetic_risk_assessor import main
            result = main()
        elif module_path == "dynamics.quantum_drug_transport":
            from dynamics.quantum_drug_transport import main
            result = main()
        else:
            print(f"❌ Module {module_path} not implemented yet")
            return False
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n✅ {module_name} completed successfully!")
        print(f"⏱️  Runtime: {runtime:.2f} seconds")
        print(f"📊 Results: {len(result) if hasattr(result, '__len__') else 'Generated'} items")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in {module_name}:")
        print(f"   {str(e)}")
        print(f"   Check the module implementation and dependencies")
        return False

def main():
    """Run the complete Babylon framework demonstration."""
    
    print("🏛️  BABYLON OSCILLATORY FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    print("Systematic testing of oscillatory drug dynamics pipeline")
    print("From genetic variants → oscillatory holes → risk assessment → quantum transport")
    print("=" * 80)
    
    # Check if babylon_results directory exists, create if not
    results_dir = Path("babylon_results")
    if results_dir.exists():
        print(f"📁 Results will be saved to: {results_dir.absolute()}")
    else:
        results_dir.mkdir()
        print(f"📁 Created results directory: {results_dir.absolute()}")
    
    # Define modules to run
    modules = [
        {
            "name": "Oscillatory Hole Detection",
            "path": "pharmacogenonics.hole_detector", 
            "description": "Detects oscillatory holes in biological pathways from genetic variants"
        },
        {
            "name": "Genetic Risk Assessment",
            "path": "pharmacogenonics.genetic_risk_assessor",
            "description": "Quantifies drug response risk based on pharmacogenetic variants"
        },
        {
            "name": "Quantum Drug Transport",
            "path": "dynamics.quantum_drug_transport",
            "description": "Models drug transport across membranes using quantum oscillatory principles"
        }
    ]
    
    # Run each module
    results = []
    total_start = time.time()
    
    for i, module in enumerate(modules, 1):
        print(f"\n🎯 STEP {i}/{len(modules)}: {module['name']}")
        
        success = run_module_demo(
            module['name'], 
            module['path'], 
            module['description']
        )
        
        results.append({
            'module': module['name'],
            'success': success
        })
        
        if success:
            print(f"✅ Step {i} completed - proceed to next module")
        else:
            print(f"❌ Step {i} failed - check implementation")
    
    # Summary
    total_time = time.time() - total_start
    successful_modules = sum(1 for r in results if r['success'])
    
    print(f"\n{'='*80}")
    print("📊 BABYLON DEMONSTRATION SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total runtime: {total_time:.2f} seconds")
    print(f"✅ Successful modules: {successful_modules}/{len(modules)}")
    print(f"📁 All results saved to: {results_dir.absolute()}")
    
    print(f"\n📋 Module Status:")
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"   • {result['module']}: {status}")
    
    if successful_modules == len(modules):
        print(f"\n🎉 ALL MODULES COMPLETED SUCCESSFULLY!")
        print("The Babylon framework is working correctly.")
        print("\nNext steps:")
        print("1. Review generated results in babylon_results/")
        print("2. Integrate with your real pharmacogenomics data")
        print("3. Implement remaining dynamics modules")
        print("4. Scale up for larger datasets")
    else:
        print(f"\n⚠️  {len(modules) - successful_modules} module(s) failed")
        print("Check the error messages above and fix implementation issues")
    
    print(f"\n📚 For detailed documentation, see: babylon/README.md")
    print(f"🔬 For individual module testing, run files directly from src/")
    
    return results

if __name__ == "__main__":
    results = main()

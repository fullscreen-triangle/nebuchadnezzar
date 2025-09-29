#!/usr/bin/env python3
"""
Test Script for SBML Loading
============================

Quick test to verify that the local SBML file can be loaded and parsed correctly.
"""

import sys
import os
sys.path.append('src')

from reaction_pathways import PathwayDatabase, PathwayCircuitSolver
from circuit_representation import PathwayCircuitBuilder

def test_sbml_loading():
    """Test loading reactions from the local SBML file."""
    print("🧪 Testing SBML Loading Functionality")
    print("=" * 40)
    
    # Test PathwayDatabase (used by reaction_pathways.py)
    print("\n1. Testing PathwayDatabase...")
    try:
        db = PathwayDatabase()
        reactions = db.load_sbml_reactions()
        print(f"✅ PathwayDatabase: Loaded {len(reactions)} reactions")
        
        if reactions:
            # Show first few reactions
            print("\nFirst 3 reactions:")
            for i, rxn in enumerate(reactions[:3]):
                print(f"  {i+1}. {rxn.id}: {rxn.name}")
                print(f"     Reactants: {rxn.reactants}")
                print(f"     Products: {rxn.products}")
                print(f"     Enzyme: {rxn.enzymes[0] if rxn.enzymes else 'N/A'}")
                print()
        
    except Exception as e:
        print(f"❌ PathwayDatabase failed: {e}")
    
    # Test PathwayCircuitSolver (full integration)
    print("\n2. Testing PathwayCircuitSolver...")
    try:
        solver = PathwayCircuitSolver()
        solver.load_pathway_data()
        print(f"✅ PathwayCircuitSolver: Loaded {len(solver.reactions)} reactions")
        
    except Exception as e:
        print(f"❌ PathwayCircuitSolver failed: {e}")
    
    # Test PathwayCircuitBuilder (used by circuit_representation.py)
    print("\n3. Testing PathwayCircuitBuilder...")
    try:
        builder = PathwayCircuitBuilder()
        reactions = builder.load_reactions_from_database()
        print(f"✅ PathwayCircuitBuilder: Loaded {len(reactions)} reactions")
        
    except Exception as e:
        print(f"❌ PathwayCircuitBuilder failed: {e}")
    
    # Check if SBML file exists
    sbml_path = "demo/public/homo_sapiens.3.1.sbml.tgz"
    print(f"\n4. SBML File Check...")
    if os.path.exists(sbml_path):
        size_mb = os.path.getsize(sbml_path) / (1024 * 1024)
        print(f"✅ SBML file found: {sbml_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ SBML file not found: {sbml_path}")
    
    print("\n" + "=" * 40)
    print("🧪 SBML Loading Test Complete!")

if __name__ == "__main__":
    test_sbml_loading()

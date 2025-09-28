"""
St. Stellas Framework Validation Suite
=====================================

A comprehensive Python package for validating the St. Stellas unified theoretical framework
using real biological data from major databases.

This package implements and validates:
- S-Entropy Framework with tri-dimensional coordinates
- Dynamic Flux Theory with oscillatory entropy
- Molecular Coordinate Transformation systems
- Grand Unified Biological Oscillations

Author: Kundai Farai Sachikonye
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Kundai Farai Sachikonye"
__email__ = "sachikonye@wzw.tum.de"

# Core framework imports
from .core.s_entropy import SEntropyFramework, SDistanceMetric, StrategicImpossibility
from .core.coordinates import (
    MolecularCoordinates, 
    DNACoordinateTransform, 
    ProteinCoordinateTransform,
    ChemicalCoordinateTransform
)
from .core.oscillations import (
    BiologicalOscillations, 
    MultiScaleCoupling, 
    OscillatoryHierarchy
)
from .core.fluid_dynamics import (
    DynamicFluxTheory, 
    GrandFluxStandards, 
    PatternAlignment
)

# Data access imports
from .data.databases import (
    KEGGConnector, 
    PDBConnector, 
    UniProtConnector,
    NCBIConnector
)

# Validation imports
from .validation.framework_validation import FrameworkValidator
from .validation.molecular_validation import MolecularValidator
from .validation.biological_validation import BiologicalValidator

# Analysis imports
from .analysis.performance_benchmarks import PerformanceBenchmarker
from .analysis.statistical_analysis import StatisticalValidator

# Main validation class
class StStellasSuite:
    """
    Main interface for the St. Stellas Framework Validation Suite.
    
    This class provides a unified interface to validate all components
    of the St. Stellas theoretical framework against real biological data.
    """
    
    def __init__(self, config_path=None):
        """Initialize the validation suite."""
        self.s_entropy = SEntropyFramework()
        self.coordinates = MolecularCoordinates()
        self.oscillations = BiologicalOscillations()
        self.fluid_dynamics = DynamicFluxTheory()
        self.validator = FrameworkValidator()
        self.benchmarker = PerformanceBenchmarker()
        
    def validate_full_framework(self):
        """Run complete validation of all framework components."""
        results = {}
        
        # Validate S-entropy framework
        results['s_entropy'] = self.validator.validate_s_entropy_framework()
        
        # Validate molecular coordinates
        results['molecular'] = self.validator.validate_molecular_coordinates()
        
        # Validate biological oscillations  
        results['biological'] = self.validator.validate_biological_oscillations()
        
        # Validate fluid dynamics
        results['fluid'] = self.validator.validate_fluid_dynamics()
        
        # Cross-domain validation
        results['cross_domain'] = self.validator.validate_cross_domain_transfer()
        
        return results
    
    def generate_validation_report(self, output_path="validation_report.html"):
        """Generate comprehensive validation report."""
        results = self.validate_full_framework()
        return self.validator.generate_html_report(results, output_path)

# Convenience imports for easy access
__all__ = [
    # Main classes
    'StStellasSuite',
    'SEntropyFramework',
    'MolecularCoordinates', 
    'BiologicalOscillations',
    'DynamicFluxTheory',
    
    # Validators
    'FrameworkValidator',
    'MolecularValidator',
    'BiologicalValidator',
    
    # Database connectors
    'KEGGConnector',
    'PDBConnector', 
    'UniProtConnector',
    'NCBIConnector',
    
    # Analysis tools
    'PerformanceBenchmarker',
    'StatisticalValidator',
]

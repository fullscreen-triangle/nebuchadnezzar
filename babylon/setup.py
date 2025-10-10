"""
Babylon: Systematic Oscillatory Drug Dynamics Framework
======================================================

A comprehensive Python framework for systematic testing of each component
in the oscillatory drug dynamics pipeline, from genetic variants to
multi-scale therapeutic effects.

Each module is designed to be independent, testable, and well-documented
with its own main function, visualization, and result saving capabilities.
"""

from setuptools import setup, find_packages
import os

def get_version():
    return "0.1.0"

setup(
    name="babylon-oscillatory-framework",
    version=get_version(),
    author="Kundai Farai Sachikonye", 
    author_email="sachikonye@wzw.tum.de",
    description="Systematic oscillatory drug dynamics framework for personalized medicine",
    long_description="A systematic framework for testing oscillatory drug dynamics at multiple scales",
    long_description_content_type="text/markdown",
    url="https://github.com/ksachikonye/babylon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core scientific computing
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        
        # Visualization
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        
        # Data processing
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.61.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "all": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "pytest>=6.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Pharmacogenomics modules
            "babylon-holes=src.pharmacogenonics.hole_detector:main",
            "babylon-risk=src.pharmacogenonics.genetic_risk_assessor:main",
            
            # Dynamics modules
            "babylon-quantum=src.dynamics.quantum_drug_transport:main",
            # Additional modules to be implemented
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

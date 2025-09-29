#!/usr/bin/env python
"""
St. Stellas Framework Validation Suite
=====================================

A comprehensive Python package for validating the St. Stellas unified theoretical framework
using real biological data from major databases.

Author: Kundai Farai Sachikonye
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Version
def get_version():
    return "0.1.0"

setup(
    name="st-stellas-validation",
    version=get_version(),
    author="Kundai Farai Sachikonye",
    author_email="sachikonye@wzw.tum.de",
    description="Validation suite for the St. Stellas unified theoretical framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ksachikonye/st-stellas-validation",
    packages=["src"],
    package_dir={"src": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.61.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "mypy>=0.910",
        ],
        "bioinformatics": [
            "biopython>=1.79",
            "rdkit>=2022.3.1", 
            "requests>=2.25.0",
            "pyyaml>=5.4.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "molecular": [
            "mdanalysis>=2.0.0",
            "biotite>=0.35.0",
            # Note: pymol-open-source has compatibility issues with Python 3.13
            # "pymol-open-source>=2.5.0",  # Install manually if needed
        ],
        "quantum": [
            "qiskit>=0.36.0",
            "cirq>=0.14.0",
        ],
        "all": [
            "biopython>=1.79",
            "rdkit>=2022.3.1",
            "requests>=2.25.0", 
            "pyyaml>=5.4.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Entry points for the actual implemented modules
            "run-s-entropy=src.s_entropy_solver:main", 
            "run-oscillatory=src.oscillatory_mechanics:main",
            "run-precision=src.precision_by_difference:main",
            "run-spatio=src.spatio_temporal:main",
            "run-pathways=src.reaction_pathways:main",
            "run-noise=src.noise_portfolio:main",
            "run-circuits=src.circuit_representation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

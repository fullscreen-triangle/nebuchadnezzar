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

# Read version from __init__.py
def get_version():
    init_path = os.path.join("src", "st_stellas", "__init__.py")
    with open(init_path, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
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
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
        "biopython>=1.79",
        "rdkit>=2022.3.1",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "requests>=2.25.0",
        "tqdm>=4.61.0",
        "click>=8.0.0",
        "pyyaml>=5.4.0",
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
        "quantum": [
            "qiskit>=0.36.0",
            "cirq>=0.14.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "bokeh>=2.3.0",
        ],
        "molecular": [
            "mdanalysis>=2.0.0",
            "biotite>=0.35.0",
            "pymol-open-source>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stellas-validate=st_stellas.cli:main",
            "stellas-benchmark=st_stellas.benchmarks:main",
            "stellas-report=st_stellas.reporting:main",
        ],
    },
    include_package_data=True,
    package_data={
        "st_stellas": [
            "data/*.yaml",
            "data/*.json",
            "templates/*.html",
        ],
    },
    zip_safe=False,
)

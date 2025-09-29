# St. Stellas Framework Installation Guide

## Quick Installation (Minimal Dependencies)

The St. Stellas validation framework has been designed with minimal dependencies to ensure easy installation across different Python versions.

### 1. Basic Installation

```bash
# Navigate to demo folder
cd demo

# Install minimal requirements
pip install -r requirements.txt

# Or install core packages manually:
pip install numpy scipy pandas matplotlib seaborn networkx scikit-learn tqdm
```

### 2. Optional Enhanced Features

If you want additional functionality, you can install optional packages:

```bash
# For bioinformatics features
pip install biopython rdkit requests pyyaml

# For interactive visualization
pip install plotly jupyter ipywidgets

# For quantum computing validation
pip install qiskit cirq

# For molecular dynamics (may have compatibility issues with Python 3.13)
pip install mdanalysis biotite
```

### 3. Python Version Compatibility

- **Minimum**: Python 3.8
- **Recommended**: Python 3.9-3.12
- **Note**: Some packages may have issues with Python 3.13

### 4. Running the Validation Scripts

Each script can be run independently:

```bash
# Core S-entropy framework
python src/s_entropy_solver.py

# Oscillatory mechanics for circuit analysis
python src/oscillatory_mechanics.py

# Precision-by-difference temporal coordination
python src/precision_by_difference.py

# Hegel framework for cytoplasmic dynamics
python src/spatio_temporal.py

# Reaction pathway analysis with SBML data
python src/reaction_pathways.py

# Noise portfolio analysis
python src/noise_portfolio.py

# Circuit representation of biological systems
python src/circuit_representation.py
```

### 5. Using Local SBML Data

The framework now uses the local `homo_sapiens.3.1.sbml.tgz` file in `demo/public/` instead of API calls, ensuring:

- No network dependencies
- Consistent results
- Faster execution
- No rate limiting issues

### 6. Troubleshooting

**Problem**: Package installation fails with version conflicts
**Solution**: Use the minimal requirements.txt which only includes essential packages

**Problem**: "pymol-open-source" installation fails
**Solution**: This package is optional and has been removed from core requirements

**Problem**: Python 3.13 compatibility issues
**Solution**: Consider using Python 3.11 or 3.12 for maximum compatibility

### 7. Development Setup

For development work:

```bash
pip install -e .  # Editable install
pip install -r requirements.txt
```

### 8. Testing Installation

Run the test script to verify everything works:

```bash
python test_sbml_loading.py
```

This will test:

- SBML file loading
- Core framework functionality
- Module imports
- Basic validation capabilities

### 9. Quick Start

After installation, try running one of the main validation scripts:

```bash
# Quick demo of S-entropy framework
python src/s_entropy_solver.py

# Analyze biological circuits
python src/circuit_representation.py
```

Both will generate plots and save results to demonstrate the framework capabilities.

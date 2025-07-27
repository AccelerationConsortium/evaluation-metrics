#!/bin/bash
"""
Setup script for ax-platform environment on Niagara cluster.
This script sets up the Rocky Linux 9 container environment and installs ax-platform.
"""

echo "=== Setting up ax-platform environment on Niagara ==="

# Check if we're in the Rocky Linux 9 container
if [[ -z "$NIAGARA_ROCKY9_ENABLED" ]]; then
    echo "Warning: Not in Rocky Linux 9 container. Consider setting up SSH config for automatic container entry."
    echo "See: https://docs.scinet.utoronto.ca/index.php/VS_Code"
fi

# Load required modules
echo "Loading required modules..."
module load python/3.11
module load gcc/12.2.0

# Create virtual environment
VENV_DIR="$HOME/niagara_venvs/ax_env"
echo "Creating virtual environment at: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv "$VENV_DIR"
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

# Install ax-platform and dependencies
echo "Installing ax-platform and dependencies..."
echo "This may take several minutes..."

# Install PyTorch CPU version (should work in Rocky Linux 9 container)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ax-platform
pip install ax-platform

# Install submitit for job submission
pip install submitit

# Test the installation
echo "Testing ax-platform installation..."
python -c "
import torch
import ax
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Ax-platform version: {ax.__version__}')
print('✓ Installation successful!')
"

echo "=== Setup complete ==="
echo "To use this environment:"
echo "1. source $VENV_DIR/bin/activate"
echo "2. python scripts/ax_niagara_clean.py --submit"
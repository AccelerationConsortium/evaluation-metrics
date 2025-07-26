#!/bin/bash
#SBATCH --job-name=niagara_test
#SBATCH --account=def-sgbaird
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=compute
#SBATCH --output=niagara_test_%j.out
#SBATCH --error=niagara_test_%j.err

echo "=== Niagara Job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
echo "Date: $(date)"

# Load modules
module load python/3.11
module load scipy-stack

# Install ax-platform
pip install --user ax-platform

# Run test
python scripts/niagara_test_simple.py

echo "=== Job completed ==="
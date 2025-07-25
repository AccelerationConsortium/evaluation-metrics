#!/bin/bash
#SBATCH --job-name=ax_optimization
#SBATCH --account=ac-sdl
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/s/sgbaird/sgbaird/ax_optimization_%j.out
#SBATCH --error=/scratch/s/sgbaird/sgbaird/ax_optimization_%j.err

echo "=== BALAM SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Load Python module
module load python/3.11.5

echo "=== Python Environment ==="
which python
python --version

echo "=== Running optimization script ==="
cd $SLURM_SUBMIT_DIR
python scripts/simple_test_campaign.py

echo "=== Job completed ==="
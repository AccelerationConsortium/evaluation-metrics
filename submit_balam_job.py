#!/usr/bin/env python3
"""
Simple script to submit a test job to BALAM cluster.
Run this script from a BALAM login node.
"""

import subprocess
import sys
import os
from pathlib import Path

def create_test_job_script():
    """Create a simple SLURM job script for testing."""
    job_script = """#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err

echo "Starting test job on $(hostname)"
echo "Current date: $(date)"
echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo "Environment modules:"
module list

# Simple test computation
python3 -c "
import time
import sys
print('Python test starting...')
for i in range(5):
    print(f'Step {i+1}/5')
    time.sleep(1)
print('Python test completed successfully!')
print('Job finished at:', time.strftime('%Y-%m-%d %H:%M:%S'))
"

echo "Test job completed on $(hostname)"
"""
    
    with open("test_job.sh", "w") as f:
        f.write(job_script)
    
    # Make executable
    os.chmod("test_job.sh", 0o755)
    print("Created test_job.sh")

def submit_job():
    """Submit the job using sbatch."""
    try:
        result = subprocess.run(
            ["sbatch", "test_job.sh"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"Job submitted successfully!")
        print(f"Output: {result.stdout.strip()}")
        
        # Extract job ID from output like "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"Job ID: {job_id}")
        
        return job_id
        
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_job_status(job_id):
    """Check the status of the submitted job."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"\nJob status:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error checking job status: {e}")

def main():
    print("BALAM Test Job Submission Script")
    print("=" * 40)
    
    # Check if we're on a SLURM system
    if subprocess.run(["which", "sbatch"], capture_output=True).returncode != 0:
        print("Error: sbatch command not found. Are you on a SLURM system?")
        sys.exit(1)
    
    # Create the job script
    create_test_job_script()
    
    # Submit the job
    job_id = submit_job()
    
    if job_id:
        print(f"\nJob {job_id} submitted. Use 'squeue -u $USER' to check status.")
        print(f"Output will be in: test_job_{job_id}.out")
        print(f"Errors will be in: test_job_{job_id}.err")
        
        # Check initial status
        check_job_status(job_id)
        
        print(f"\nTo monitor the job:")
        print(f"  squeue -j {job_id}")
        print(f"  tail -f test_job_{job_id}.out")

if __name__ == "__main__":
    main()
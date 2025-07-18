#!/usr/bin/env python3
"""
Simple script to submit a test job to BALAM cluster using submitit.
Run this script from a BALAM login node.
"""

import os

import submitit

from test_module import test_function


def main():
    # Check for required environment variables
    if not os.getenv("SLURM_ACCOUNT"):
        print("Error: SLURM_ACCOUNT environment variable must be set")
        return
    
    # Create submitit executor
    executor = submitit.AutoExecutor(folder="./submitit_logs")
    
    # Configure SLURM parameters
    executor.update_parameters(
        slurm_job_name="balam_test",
        slurm_time=5,  # 5 minutes
        slurm_ntasks=1,
        slurm_cpus_per_task=1,
        slurm_mem="1G",
        slurm_account=os.getenv("SLURM_ACCOUNT")
    )
    
    # Submit the job
    job = executor.submit(test_function)
    print(f"Job submitted: {job.job_id}")
    
    return job



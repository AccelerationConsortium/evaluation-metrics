#!/usr/bin/env python3
"""Submit a test job to BALAM using submitit."""

import submitit
import os
from test_function import simple_test


def submit_test_job():
    """Submit a simple test job to BALAM cluster."""
    
    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
        timeout_min=5,  # Very short job
        slurm_partition="debug",  # Use debug partition for faster scheduling
        slurm_gpus_per_node=1,  # Minimum 1 GPU required on BALAM
        slurm_account=os.environ.get('SLURM_ACCOUNT'),  # Account allocation
    )
    
    print("Submitting test job to BALAM...")
    job = executor.submit(simple_test)
    print(f"Job submitted with ID: {job.job_id}")
    
    return job


job = submit_test_job()

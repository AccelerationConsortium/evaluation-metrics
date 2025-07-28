#!/usr/bin/env python3
"""Submit a test job to BALAM using submitit."""

import submitit
import os
from test_function import simple_test


def submit_test_job():
    """Submit a simple test job to Balam cluster."""
    
    # Setup scratch directory for logs
    scratch_dir = os.environ.get("SCRATCH", "/tmp")
    log_folder = f"{scratch_dir}/submitit_logs/test"
    os.makedirs(log_folder, exist_ok=True)
    
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=5,  # Very short job
        slurm_gpus_per_node=1,  # Required on Balam
        slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
        slurm_job_name="simple_test"
    )
    
    print("Submitting test job to Balam...")
    job = executor.submit(simple_test)
    print(f"Job submitted with ID: {job.job_id}")
    print(f"Log folder: {log_folder}")
    
    return job


job = submit_test_job()
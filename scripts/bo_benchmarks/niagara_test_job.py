#!/usr/bin/env python3
"""Submit a small test job to Niagara using submitit."""

import os
import submitit
from benchmark_functions import evaluate_benchmark


def simple_niagara_test():
    """A simple test function for Niagara."""
    import time
    import socket
    
    print(f"Running on {socket.gethostname()}")
    print("Testing benchmark function...")
    
    # Test the Branin function
    test_params = {"function": "branin", "x1": 0.0, "x2": 0.0}
    result = evaluate_benchmark(test_params)
    print(f"Branin(0, 0) = {result}")
    
    # Small computation
    time.sleep(5)
    
    return {"hostname": socket.gethostname(), "result": result, "status": "success"}


def submit_niagara_test():
    """Submit a test job to Niagara cluster."""
    
    # Configure submitit for Niagara
    executor = submitit.AutoExecutor(folder="submitit_logs/niagara_test")
    executor.update_parameters(
        timeout_min=10,  # Short 10-minute job
        slurm_partition="compute",  # Niagara compute partition
        slurm_additional_parameters={
            "ntasks": 1,
            "cpus-per-task": 1,
            "mem": "4G",
            "account": os.getenv("SLURM_ACCOUNT", "def-xxx"),  # Replace with your allocation
        },
    )
    
    print("Submitting test job to Niagara...")
    job = executor.submit(simple_niagara_test)
    print(f"Job submitted with ID: {job.job_id}")
    
    return job


if __name__ == "__main__":
    import sys
    
    if "--local" in sys.argv:
        # Run locally for testing
        print("Running test locally...")
        result = simple_niagara_test()
        print(f"Local test result: {result}")
    else:
        # Submit to cluster
        if not os.getenv("SLURM_ACCOUNT"):
            print("ERROR: SLURM_ACCOUNT environment variable not set")
            print("Please set your allocation: export SLURM_ACCOUNT='your-account'")
            sys.exit(1)
        
        job = submit_niagara_test()
        print(f"Niagara job submitted: {job.job_id}")
        print("Monitor with: squeue -u $USER")
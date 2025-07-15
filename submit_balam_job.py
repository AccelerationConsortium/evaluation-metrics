#!/usr/bin/env python3
"""
Simple script to submit a test job to BALAM cluster using submitit.
Run this script from a BALAM login node.
"""

import os
import sys
import time
from pathlib import Path

try:
    import submitit
except ImportError:
    print("Error: submitit not installed. Run: pip install submitit")
    sys.exit(1)


def test_function():
    """Simple test function to run on the cluster."""
    import time
    import socket
    import sys
    
    print(f"Test job starting on {socket.gethostname()}")
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Simple computation
    print("Running test computation...")
    for i in range(5):
        print(f"Step {i+1}/5")
        time.sleep(1)
    
    result = sum(range(1000))
    print(f"Computation result: {result}")
    print(f"Test job completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {"status": "success", "result": result, "hostname": socket.gethostname()}


def main():
    print("BALAM Test Job Submission Script (using submitit)")
    print("=" * 50)
    
    # Check for required environment variables
    if not os.getenv("SLURM_ACCOUNT"):
        print("Error: SLURM_ACCOUNT environment variable must be set")
        print("Run: export SLURM_ACCOUNT=your-account-name")
        sys.exit(1)
    
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
    
    print(f"Submitting job with account: {os.getenv('SLURM_ACCOUNT')}")
    print("Job parameters:")
    print("  Time: 5 minutes")
    print("  CPUs: 1")
    print("  Memory: 1G")
    
    # Submit the job
    job = executor.submit(test_function)
    
    print(f"\nJob submitted successfully!")
    print(f"Job ID: {job.job_id}")
    print(f"Log directory: ./submitit_logs")
    
    # Show how to check status
    print(f"\nTo check job status:")
    print(f"  squeue -j {job.job_id}")
    print(f"  # or in Python: job.state")
    
    print(f"\nTo get results (after completion):")
    print(f"  # job.result()  # blocks until completion")
    
    # Optionally wait a bit and show status
    print(f"\nCurrent job state: {job.state}")
    
    return job


if __name__ == "__main__":
    main()
"""Simple submission test for Niagara cluster.

This script can be run locally to test the setup, and then on Niagara to actually submit jobs.

Usage:
1. Local test (validates functions work):
   python simple_niagara_test.py --local

2. On Niagara (submit to cluster):
   export SLURM_ACCOUNT="your-account-name"
   python simple_niagara_test.py --submit

Setup on Niagara:
1. SSH to niagara.scinet.utoronto.ca
2. Load Python: module load python
3. Install dependencies: pip install --user submitit numpy
4. Set account: export SLURM_ACCOUNT="your-allocation"
5. Run this script: python simple_niagara_test.py --submit
"""

import argparse
import os
import sys
from datetime import datetime

# Add path for our benchmark functions
sys.path.insert(0, 'scripts/bo_benchmarks')

def test_benchmark_function():
    """Test that our benchmark functions work."""
    try:
        from benchmark_functions import evaluate_benchmark
        
        # Test Branin function
        test_params = {"function": "branin", "x1": 0.0, "x2": 0.0}
        result = evaluate_benchmark(test_params)
        print(f"✓ Branin test passed: {result}")
        
        # Test optimum
        opt_params = {"function": "branin", "x1": -3.141592, "x2": 12.275}
        opt_result = evaluate_benchmark(opt_params)
        print(f"✓ Branin optimum test: {opt_result:.6f} (should be ~0.398)")
        
        return True
    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        return False

def simple_evaluation_task(function_name, params):
    """Simple task that can be submitted to cluster."""
    # This function will run on the compute node
    import sys
    import os
    from datetime import datetime
    
    # Add path again (needed for cluster execution)
    sys.path.insert(0, 'scripts/bo_benchmarks')
    from benchmark_functions import evaluate_benchmark
    
    # Get cluster info
    job_id = os.getenv('SLURM_JOB_ID', 'local')
    node_name = os.getenv('SLURMD_NODENAME', 'local')
    
    print(f"Running on {node_name}, job {job_id} at {datetime.now()}")
    
    # Evaluate function
    test_params = {"function": function_name, **params}
    result = evaluate_benchmark(test_params)
    
    output = {
        "function": function_name,
        "parameters": params,
        "result": result,
        "job_id": job_id,
        "node": node_name,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Result: {output}")
    return output

def submit_to_niagara():
    """Submit test jobs to Niagara cluster."""
    try:
        import submitit
    except ImportError:
        print("Installing submitit...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "submitit"])
        import submitit
    
    # Check for SLURM account
    account = os.getenv("SLURM_ACCOUNT")
    if not account:
        print("Warning: SLURM_ACCOUNT not set. Using default 'def-sgbaird'")
        account = "def-sgbaird"
    
    print(f"Using SLURM account: {account}")
    
    # Setup executor for Niagara
    log_folder = "submitit_logs/%j"  # %j will be replaced with job ID
    
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Configure for Niagara cluster
    executor.update_parameters(
        timeout_min=15,  # 15 minute timeout
        mem_gb=4,        # 4GB memory
        cpus_per_task=1, # 1 CPU
        slurm_partition="compute",  # Niagara compute partition
        slurm_account=account,
        slurm_job_name="benchmark_test"
    )
    
    print("Executor configured for Niagara")
    
    # Define test cases
    test_cases = [
        ("branin", {"x1": 0.0, "x2": 0.0}),
        ("branin", {"x1": -3.14, "x2": 12.27}),
        ("hartmann6", {"x1": 0.2, "x2": 0.15, "x3": 0.48, "x4": 0.28, "x5": 0.31, "x6": 0.66})
    ]
    
    # Submit jobs
    jobs = []
    for i, (func_name, params) in enumerate(test_cases):
        print(f"Submitting job {i+1}: {func_name} with {params}")
        job = executor.submit(simple_evaluation_task, func_name, params)
        jobs.append(job)
        print(f"  Submitted as job ID: {job.job_id}")
    
    print(f"\n✓ Successfully submitted {len(jobs)} jobs to Niagara")
    print("Job IDs:", [job.job_id for job in jobs])
    print("\nTo check status: squeue -u $USER")
    print("To view logs: ls submitit_logs/")
    
    return jobs

def main():
    parser = argparse.ArgumentParser(description='Test Niagara cluster submission')
    parser.add_argument('--local', action='store_true', help='Run local test only')
    parser.add_argument('--submit', action='store_true', help='Submit to Niagara cluster')
    
    args = parser.parse_args()
    
    if not (args.local or args.submit):
        # Default to local test
        args.local = True
    
    print("=== Niagara Cluster Test ===")
    
    if args.local:
        print("\n1. Testing benchmark functions locally...")
        if test_benchmark_function():
            print("✓ Local tests passed - ready for cluster submission")
        else:
            print("✗ Local tests failed - fix issues before submitting")
            return 1
    
    if args.submit:
        print("\n2. Submitting to Niagara cluster...")
        try:
            jobs = submit_to_niagara()
            print("✓ Submission successful")
        except Exception as e:
            print(f"✗ Submission failed: {e}")
            return 1
    
    print("\n=== Test Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
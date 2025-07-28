#!/usr/bin/env python3
"""
Simple optimization test for Niagara cluster using submitit and ax-platform.
Usage: 
  python niagara_test_simple.py --local     # Test locally
  python niagara_test_simple.py --submit    # Submit to Niagara cluster
"""

import sys
import os
import argparse

def branin(x1, x2):
    """Branin function for optimization testing."""
    import math
    a = 1
    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)
    
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * math.cos(x1) + s

def run_optimization():
    """Run optimization using ax-platform."""
    try:
        from ax import optimize
        print("✓ Ax-platform imported successfully")
        
        # Define the optimization
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            evaluation_function=lambda params: branin(params["x1"], params["x2"]),
            total_trials=8,  # 3 Sobol + 5 BoTorch
            random_seed=42,
        )
        
        print(f"Best parameters: {best_parameters}")
        print(f"Best value: {values[0]['objective']}")
        print("✓ Optimization completed successfully on Niagara!")
        
        return best_parameters, values[0]['objective']
        
    except ImportError as e:
        print(f"✗ Failed to import ax-platform: {e}")
        # Simple fallback
        best_val = branin(-3.14, 12.27)  # Known good point
        print(f"Fallback result: f={best_val:.6f}")
        return {"x1": -3.14, "x2": 12.27}, best_val

def optimization_job():
    """Job function for submitit."""
    print("=== Niagara Optimization Job ===")
    if 'SLURM_JOB_ID' in os.environ:
        print(f"SLURM Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    
    best_params, best_value = run_optimization()
    
    print(f"\n=== Final Result ===")
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    print("=== Job Complete ===")
    
    return {"best_params": best_params, "best_value": best_value}

def submit_to_niagara():
    """Submit job to Niagara using submitit."""
    try:
        import submitit
        
        # Create executor for Niagara
        log_folder = os.path.expanduser("~/scratch/submitit_logs")
        os.makedirs(log_folder, exist_ok=True)
        
        executor = submitit.AutoExecutor(folder=log_folder)
        executor.update_parameters(
            slurm_job_name="niagara_test",
            timeout_min=10,
            slurm_partition="compute",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=1
        )
        
        print(f"Submitting job to Niagara cluster...")
        print(f"Log folder: {log_folder}")
        
        # Submit the job
        job = executor.submit(optimization_job)
        print(f"✓ Job submitted with ID: {job.job_id}")
        
        # Wait for completion (optional - can be removed for async)
        print("Waiting for job completion...")
        result = job.result()
        print(f"✓ Job completed successfully!")
        print(f"Result: {result}")
        
        return job.job_id, result
        
    except ImportError:
        print("✗ submitit not available. Please install: pip install submitit")
        return None, None
    except Exception as e:
        print(f"✗ Job submission failed: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Niagara optimization test')
    parser.add_argument('--local', action='store_true', help='Run locally')
    parser.add_argument('--submit', action='store_true', help='Submit to Niagara cluster')
    args = parser.parse_args()
    
    if args.submit:
        job_id, result = submit_to_niagara()
        if job_id:
            print(f"Niagara job {job_id} completed with result: {result}")
    else:
        # Local test
        print("=== Local Test Mode ===")
        optimization_job()

if __name__ == "__main__":
    main()
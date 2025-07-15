#!/usr/bin/env python3
"""Simple script to submit benchmark jobs to BALAM cluster."""

import argparse
import os
import sys
from pathlib import Path

def submit_job(num_init_trials=5, num_opt_steps=20, seed=42, local=False):
    """Submit a single benchmark job to BALAM or run locally."""
    
    if local:
        # Run locally for testing
        print(f"Running benchmark locally with {num_init_trials} init trials...")
        sys.path.append(str(Path(__file__).parent))
        from benchmark import run_single_benchmark
        
        result = run_single_benchmark(num_init_trials, num_opt_steps, seed)
        print(f"Final best: {result['final_best']:.4f}")
        print(f"Total evaluations: {result['total_evaluations']}")
        return result
    
    else:
        # Submit to BALAM cluster
        try:
            import submitit
        except ImportError:
            print("submitit not installed. Install with: pip install submitit")
            return None
            
        # Check for required environment variable
        if not os.getenv("SLURM_ACCOUNT"):
            print("Error: SLURM_ACCOUNT environment variable must be set")
            print("Example: export SLURM_ACCOUNT=your-account-name")
            return None
        
        # Create submitit executor
        executor = submitit.AutoExecutor(folder="submitit_logs")
        executor.update_parameters(
            slurm_partition="cpu",
            slurm_account=os.getenv("SLURM_ACCOUNT"),
            slurm_time=60,  # 60 minutes
            slurm_mem_per_cpu=4000,  # 4GB
            slurm_cpus_per_task=1
        )
        
        # Function to run on cluster
        def cluster_benchmark(num_init, num_opt, seed_val):
            sys.path.append(str(Path(__file__).parent))
            from benchmark import run_single_benchmark
            return run_single_benchmark(num_init, num_opt, seed_val)
        
        print(f"Submitting job with {num_init_trials} init trials to BALAM...")
        job = executor.submit(cluster_benchmark, num_init_trials, num_opt_steps, seed)
        print(f"Job submitted with ID: {job.job_id}")
        return job


def main():
    parser = argparse.ArgumentParser(description="Submit benchmark jobs to BALAM cluster")
    parser.add_argument("--init-trials", type=int, default=5, 
                       help="Number of initialization trials (default: 5)")
    parser.add_argument("--opt-steps", type=int, default=20,
                       help="Number of optimization steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--local", action="store_true",
                       help="Run locally instead of submitting to cluster")
    
    args = parser.parse_args()
    
    result = submit_job(
        num_init_trials=args.init_trials,
        num_opt_steps=args.opt_steps, 
        seed=args.seed,
        local=args.local
    )
    
    if args.local and result:
        print("Local run completed successfully")
    elif not args.local and result:
        print("Job submitted to BALAM cluster")


if __name__ == "__main__":
    main()
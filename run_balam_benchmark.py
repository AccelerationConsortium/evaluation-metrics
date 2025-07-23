#!/usr/bin/env python3
"""
Command-line interface for running benchmark campaigns on BALAM cluster.

Usage:
    # Test locally
    python run_balam_benchmark.py --local

    # Submit to BALAM cluster
    export SLURM_ACCOUNT="your-account"
    python run_balam_benchmark.py --partition cpu --walltime 120
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path so we can import gpcheck
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gpcheck.benchmark import BenchmarkConfig, BALAMExecutor, run_local_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run optimization benchmark campaigns")
    
    # Execution mode
    parser.add_argument("--local", action="store_true", 
                       help="Run locally instead of submitting to cluster")
    
    # Cluster configuration
    parser.add_argument("--partition", default="cpu", 
                       help="SLURM partition (default: cpu)")
    parser.add_argument("--walltime", type=int, default=180,
                       help="Walltime in minutes (default: 180)")
    parser.add_argument("--mem-per-cpu", type=int, default=4000,
                       help="Memory per CPU in MB (default: 4000)")
    
    # Benchmark configuration
    parser.add_argument("--objective", default="branin",
                       help="Objective function (default: branin)")
    parser.add_argument("--n-init", nargs="+", type=int, 
                       default=[2, 5, 10, 20, 50],
                       help="Number of initialization trials (default: 2 5 10 20 50)")
    parser.add_argument("--n-steps", type=int, default=100,
                       help="Number of optimization steps (default: 100)")
    parser.add_argument("--n-repeats", type=int, default=20,
                       help="Number of repeats per configuration (default: 20)")
    parser.add_argument("--algorithms", nargs="+", 
                       default=["GPEI", "EI", "Random"],
                       help="Algorithms to benchmark (default: GPEI EI Random)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    # Output configuration
    parser.add_argument("--output", default="benchmark_results.json",
                       help="Output file for results (default: benchmark_results.json)")
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        objective_function=args.objective,
        num_initialization_trials=args.n_init,
        num_optimization_steps=args.n_steps,
        num_repeats=args.n_repeats,
        algorithms=args.algorithms,
        seed=args.seed
    )
    
    print(f"Benchmark configuration:")
    print(f"  Objective: {config.objective_function}")
    print(f"  Initialization trials: {config.num_initialization_trials}")
    print(f"  Optimization steps: {config.num_optimization_steps}")
    print(f"  Repeats: {config.num_repeats}")
    print(f"  Algorithms: {config.algorithms}")
    print(f"  Seed: {config.seed}")
    
    if args.local:
        print("\nRunning benchmark locally...")
        results = run_local_benchmark(config)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {args.output}")
        print(f"Completed {len(results)} benchmark runs")
        
        # Print summary
        best_results = {}
        for result in results:
            key = f"{result['algorithm']}_n{result['n_init']}"
            if key not in best_results or result['best_y'] < best_results[key]:
                best_results[key] = result['best_y']
        
        print("\nBest results by configuration:")
        for key, best_y in sorted(best_results.items()):
            print(f"  {key}: {best_y:.4f}")
    
    else:
        print("\nSubmitting to BALAM cluster...")
        
        # Check environment
        if not os.getenv("SLURM_ACCOUNT"):
            print("Error: SLURM_ACCOUNT environment variable must be set")
            sys.exit(1)
        
        try:
            executor = BALAMExecutor(
                partition=args.partition,
                walltime_min=args.walltime,
                mem_per_cpu=args.mem_per_cpu
            )
            
            jobs = executor.submit_benchmark_campaign(config)
            
            print(f"Submitted {len(jobs)} jobs to BALAM cluster")
            print(f"Job IDs: {[job.job_id for job in jobs[:5]]}" + 
                  (f" ... (showing first 5)" if len(jobs) > 5 else ""))
            
            # Save job information
            job_info = {
                "config": config.__dict__,
                "job_ids": [job.job_id for job in jobs],
                "cluster_params": {
                    "partition": args.partition,
                    "walltime_min": args.walltime,
                    "mem_per_cpu": args.mem_per_cpu
                }
            }
            
            job_file = args.output.replace('.json', '_jobs.json')
            with open(job_file, 'w') as f:
                json.dump(job_info, f, indent=2)
            
            print(f"Job information saved to {job_file}")
            
        except Exception as e:
            print(f"Error submitting to cluster: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
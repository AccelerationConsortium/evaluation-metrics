#!/usr/bin/env python
"""
Example script for running benchmark campaigns on BALAM compute cluster.

This script demonstrates how to use the benchmark framework to study
the effect of initialization trials on optimization performance.

Usage:
    python run_balam_benchmark.py [--dry-run] [--local]

Environment variables needed for BALAM submission:
    SLURM_ACCOUNT: Your SLURM account allocation
    MONGODB_APP_NAME: MongoDB App Services app name  
    MONGODB_API_KEY: MongoDB API key for data storage

Environment variables needed for local testing:
    (MongoDB variables are optional for local runs)
"""

import argparse
import os
import sys

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gpcheck.benchmark import BenchmarkConfig, BALAMExecutor, run_benchmark_batch


def main():
    """Run benchmark campaign."""
    parser = argparse.ArgumentParser(description="Run BALAM benchmark campaign")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print configuration without submitting jobs"
    )
    parser.add_argument(
        "--local", 
        action="store_true", 
        help="Run locally instead of submitting to BALAM"
    )
    parser.add_argument(
        "--partition", 
        default="cpu", 
        help="SLURM partition for BALAM jobs"
    )
    parser.add_argument(
        "--walltime", 
        type=int, 
        default=60, 
        help="Job walltime in minutes"
    )
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        objective_function="branin",
        num_initialization_trials=[2, 5, 10, 20],
        num_optimization_steps=50,
        num_repeats=5,  # Reduced for demonstration
        algorithms=["GPEI", "Random"],  # Reduced for demonstration
    )
    
    print(f"Benchmark Configuration:")
    print(f"  Session ID: {config.session_id}")
    print(f"  Objective: {config.objective_function}")
    print(f"  Initialization trials: {config.num_initialization_trials}")
    print(f"  Optimization steps: {config.num_optimization_steps}")
    print(f"  Repeats per config: {config.num_repeats}")
    print(f"  Algorithms: {config.algorithms}")
    
    total_runs = len(config.algorithms) * len(config.num_initialization_trials) * config.num_repeats
    print(f"  Total runs: {total_runs}")
    
    if args.dry_run:
        print("\nDry run - no jobs submitted")
        return
    
    if args.local:
        print("\nRunning locally...")
        # Create a small batch for local testing
        parameter_sets = []
        for algorithm in config.algorithms[:1]:  # Just first algorithm
            for n_init in config.num_initialization_trials[:2]:  # Just first 2 values
                for repeat in range(min(2, config.num_repeats)):  # Just 2 repeats
                    parameter_sets.append({
                        "objective_function": config.objective_function,
                        "algorithm": algorithm,
                        "num_initialization_trials": n_init,
                        "num_optimization_steps": config.num_optimization_steps,
                        "repeat": repeat,
                        "session_id": config.session_id,
                    })
        
        results = run_benchmark_batch(parameter_sets)
        
        print(f"\nCompleted {len(results)} benchmark runs:")
        for result in results:
            print(
                f"  {result['algorithm']} with {result['num_initialization_trials']} "
                f"init trials (repeat {result['repeat']}): "
                f"final regret = {result['final_regret']:.6f}"
            )
    
    else:
        print("\nSubmitting to BALAM...")
        
        # Check required environment variables
        required_vars = ["SLURM_ACCOUNT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Error: Missing required environment variables: {missing_vars}")
            print("Please set these variables before submitting to BALAM")
            return
        
        # Create BALAM executor
        executor = BALAMExecutor(
            partition=args.partition,
            walltime_min=args.walltime,
            log_folder=f"logs/benchmark-{config.session_id}/%j",
        )
        
        # Submit jobs
        jobs = executor.submit_benchmark_campaign(config, batch_size=10)
        
        job_ids = [job.job_id for job in jobs]
        print(f"Submitted {len(jobs)} jobs with IDs: {job_ids}")
        print(f"Monitor job status with: squeue -u $USER")
        print(f"View logs in: {executor.log_folder}")
        
        # Optionally wait for results (for small test runs)
        if total_runs <= 20:
            print("\nWaiting for results (small job, will wait)...")
            try:
                results = [job.result() for job in jobs]
                all_results = []
                for batch_results in results:
                    all_results.extend(batch_results)
                
                print(f"\nCompleted {len(all_results)} benchmark runs")
                
                # Print summary by initialization trials
                for n_init in config.num_initialization_trials:
                    init_results = [r for r in all_results if r['num_initialization_trials'] == n_init]
                    if init_results:
                        avg_regret = sum(r['final_regret'] for r in init_results) / len(init_results)
                        print(f"  {n_init} init trials: avg final regret = {avg_regret:.6f}")
                        
            except Exception as e:
                print(f"Error waiting for results: {e}")
                print("Jobs may still be running - check status manually")


if __name__ == "__main__":
    main()
    """Run benchmark campaign."""
    parser = argparse.ArgumentParser(description="Run BALAM benchmark campaign")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print configuration without submitting jobs"
    )
    parser.add_argument(
        "--local", 
        action="store_true", 
        help="Run locally instead of submitting to BALAM"
    )
    parser.add_argument(
        "--partition", 
        default="cpu", 
        help="SLURM partition for BALAM jobs"
    )
    parser.add_argument(
        "--walltime", 
        type=int, 
        default=60, 
        help="Job walltime in minutes"
    )
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        objective_function="branin",
        num_initialization_trials=[2, 5, 10, 20],
        num_optimization_steps=50,
        num_repeats=5,  # Reduced for demonstration
        algorithms=["GPEI", "Random"],  # Reduced for demonstration
    )
    
    print(f"Benchmark Configuration:")
    print(f"  Session ID: {config.session_id}")
    print(f"  Objective: {config.objective_function}")
    print(f"  Initialization trials: {config.num_initialization_trials}")
    print(f"  Optimization steps: {config.num_optimization_steps}")
    print(f"  Repeats per config: {config.num_repeats}")
    print(f"  Algorithms: {config.algorithms}")
    
    total_runs = len(config.algorithms) * len(config.num_initialization_trials) * config.num_repeats
    print(f"  Total runs: {total_runs}")
    
    if args.dry_run:
        print("\nDry run - no jobs submitted")
        return
    
    if args.local:
        print("\nRunning locally...")
        # Create a small batch for local testing
        parameter_sets = []
        for algorithm in config.algorithms[:1]:  # Just first algorithm
            for n_init in config.num_initialization_trials[:2]:  # Just first 2 values
                for repeat in range(min(2, config.num_repeats)):  # Just 2 repeats
                    parameter_sets.append({
                        "objective_function": config.objective_function,
                        "algorithm": algorithm,
                        "num_initialization_trials": n_init,
                        "num_optimization_steps": config.num_optimization_steps,
                        "repeat": repeat,
                        "session_id": config.session_id,
                    })
        
        results = run_benchmark_batch(parameter_sets)
        
        print(f"\nCompleted {len(results)} benchmark runs:")
        for result in results:
            print(
                f"  {result['algorithm']} with {result['num_initialization_trials']} "
                f"init trials (repeat {result['repeat']}): "
                f"final regret = {result['final_regret']:.6f}"
            )
    
    else:
        print("\nSubmitting to BALAM...")
        
        # Check required environment variables
        required_vars = ["SLURM_ACCOUNT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Error: Missing required environment variables: {missing_vars}")
            print("Please set these variables before submitting to BALAM")
            return
        
        # Create BALAM executor
        executor = BALAMExecutor(
            partition=args.partition,
            walltime_min=args.walltime,
            log_folder=f"logs/benchmark-{config.session_id}/%j",
        )
        
        # Submit jobs
        jobs = executor.submit_benchmark_campaign(config, batch_size=10)
        
        job_ids = [job.job_id for job in jobs]
        print(f"Submitted {len(jobs)} jobs with IDs: {job_ids}")
        print(f"Monitor job status with: squeue -u $USER")
        print(f"View logs in: {executor.log_folder}")
        
        # Optionally wait for results (for small test runs)
        if total_runs <= 20:
            print("\nWaiting for results (small job, will wait)...")
            try:
                results = [job.result() for job in jobs]
                all_results = []
                for batch_results in results:
                    all_results.extend(batch_results)
                
                print(f"\nCompleted {len(all_results)} benchmark runs")
                
                # Print summary by initialization trials
                for n_init in config.num_initialization_trials:
                    init_results = [r for r in all_results if r['num_initialization_trials'] == n_init]
                    if init_results:
                        avg_regret = sum(r['final_regret'] for r in init_results) / len(init_results)
                        print(f"  {n_init} init trials: avg final regret = {avg_regret:.6f}")
                        
            except Exception as e:
                print(f"Error waiting for results: {e}")
                print("Jobs may still be running - check status manually")


if __name__ == "__main__":
    main()
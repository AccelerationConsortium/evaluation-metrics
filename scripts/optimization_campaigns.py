#!/usr/bin/env python
"""Submit batch optimization campaigns to Niagara cluster."""

import os
import sys
import submitit
from pathlib import Path

# Add the bo_benchmarks directory to Python path
bo_benchmarks_dir = Path(__file__).parent / "bo_benchmarks"
sys.path.insert(0, str(bo_benchmarks_dir))

from benchmark_functions import evaluate_benchmark
import numpy as np


def run_optimization_campaign(campaign_name, function_name, iterations, campaign_id):
    """Run a simple optimization campaign for testing."""
    print(f"Starting campaign {campaign_id}: {campaign_name}")
    print(f"Function: {function_name}, Iterations: {iterations}")

    # Define parameter bounds for each function
    bounds = {
        "branin": {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)},
        "hartmann6": {f"x{i + 1}": (0.0, 1.0) for i in range(6)},
    }

    results = []
    best_value = float("inf")
    best_params = None

    # Simple random search for demonstration
    np.random.seed(42 + campaign_id)  # Reproducible but different per campaign

    for iteration in range(iterations):
        # Generate random parameters within bounds
        params = {"function": function_name}
        for param_name, (low, high) in bounds[function_name].items():
            params[param_name] = np.random.uniform(low, high)

        # Evaluate function
        value = evaluate_benchmark(params)

        # Track best result
        if value < best_value:
            best_value = value
            best_params = params.copy()

        result = {
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "iteration": iteration,
            "parameters": params,
            "value": value,
        }
        results.append(result)

        print(f"  Iteration {iteration + 1}: {value:.6f}")

    print(f"Campaign {campaign_id} complete. Best: {best_value:.6f}")

    return {
        "campaign_name": campaign_name,
        "campaign_id": campaign_id,
        "function_name": function_name,
        "iterations": iterations,
        "results": results,
        "best_value": best_value,
        "best_params": best_params,
    }


def main():
    print("=== Submitting 3 Optimization Campaigns to Niagara ===")
    print()

    # Setup scratch directory for logs
    scratch_dir = os.environ.get("SCRATCH", "/tmp")
    log_folder = f"{scratch_dir}/submitit_logs"
    os.makedirs(log_folder, exist_ok=True)

    # Setup executor
    executor = submitit.AutoExecutor(folder=log_folder)

    # Configure for Niagara cluster
    account = os.getenv("SLURM_ACCOUNT", "def-sgbaird")
    executor.update_parameters(
        timeout_min=15,  # Minimum required time
        cpus_per_task=1,
        slurm_partition="compute",
        slurm_account=account,
        slurm_job_name=f"opt_campaign",
    )

    # Define the three campaigns
    campaigns = [
        ("branin_quick", "branin", 5, 1),
        ("hartmann6_quick", "hartmann6", 5, 2),
        ("mixed_quick_branin", "branin", 3, 3),
    ]

    jobs = []

    for campaign_name, function_name, iterations, campaign_id in campaigns:
        print(f"\nSubmitting campaign {campaign_id}: {campaign_name}")
        try:
            job = executor.submit(
                run_optimization_campaign, campaign_name, function_name, iterations, campaign_id
            )
            jobs.append(job)
            print(f"Submitted campaign {campaign_id} as job {job.job_id}")
        except Exception as e:
            print(f"Failed to submit campaign {campaign_id}: {e}")

    print(f"\n=== Successfully submitted {len(jobs)} campaigns ===")
    print("Job IDs:", [job.job_id for job in jobs])
    print("\nTo check status: squeue -u $USER")
    print(f"To view logs: ls {log_folder}/")

    return jobs


if __name__ == "__main__":
    jobs = main()

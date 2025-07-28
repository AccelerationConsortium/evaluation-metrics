#!/usr/bin/env python3
"""Single optimization campaign using Ax-platform on Niagara cluster."""

import os
import sys
import time
from pathlib import Path

# Add the bo_benchmarks directory to the path
sys.path.append(str(Path(__file__).parent / "bo_benchmarks"))

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from submitit import AutoExecutor

from benchmark_functions import evaluate_benchmark


def run_optimization_campaign():
    """Run a single optimization campaign using Ax-platform."""
    
    # Simple campaign configuration
    n_sobol_trials = 3  # Initial random samples
    n_optimization_trials = 5  # Optimization iterations
    
    # Create Ax client for Branin optimization
    ax_client = AxClient()
    ax_client.create_experiment(
        name="branin_optimization_single_campaign",
        parameters=[
            {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
        ],
        objectives={"branin_value": ObjectiveProperties(minimize=True)},
    )
    
    results = []
    
    # Run Sobol trials (initial random sampling)
    for i in range(n_sobol_trials):
        parameters, trial_index = ax_client.get_next_trial()
        parameters["function"] = "branin"  # Add function name for evaluation
        
        # Evaluate the benchmark function
        value = evaluate_benchmark(parameters)
        
        # Report back to Ax
        ax_client.complete_trial(trial_index=trial_index, raw_data={"branin_value": value})
        
        result = {
            "trial": trial_index,
            "x1": parameters["x1"],
            "x2": parameters["x2"], 
            "value": value,
            "trial_type": "sobol"
        }
        results.append(result)
        print(f"Sobol trial {i+1}/{n_sobol_trials}: {result}")
    
    # Run optimization trials (model-based)
    for i in range(n_optimization_trials):
        parameters, trial_index = ax_client.get_next_trial()
        parameters["function"] = "branin"  # Add function name for evaluation
        
        # Evaluate the benchmark function
        value = evaluate_benchmark(parameters)
        
        # Report back to Ax
        ax_client.complete_trial(trial_index=trial_index, raw_data={"branin_value": value})
        
        result = {
            "trial": trial_index,
            "x1": parameters["x1"],
            "x2": parameters["x2"],
            "value": value,
            "trial_type": "optimization"
        }
        results.append(result)
        print(f"Optimization trial {i+1}/{n_optimization_trials}: {result}")
    
    # Get best parameters
    best_parameters, best_values = ax_client.get_best_parameters()
    best_value = best_values[0]["branin_value"]
    
    print(f"\nOptimization completed!")
    print(f"Best parameters: {best_parameters}")
    print(f"Best value: {best_value}")
    print(f"Known global minimum: 0.397887")
    print(f"Gap to global minimum: {best_value - 0.397887:.6f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_file = "single_campaign_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results, best_parameters, best_value


def main():
    """Main execution function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--submit":
        # Submit to Niagara cluster
        print("Submitting optimization campaign to Niagara cluster...")
        
        # Configure submitit executor for Niagara
        log_folder = f"submitit_logs/single_campaign"
        os.makedirs(log_folder, exist_ok=True)
        
        executor = AutoExecutor(folder=log_folder)
        executor.update_parameters(
            timeout_min=15,  # 15 minute timeout
            slurm_partition="compute",
            slurm_additional_parameters={
                "ntasks": 1,
                "account": os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
            },
        )
        
        # Submit the job
        job = executor.submit(run_optimization_campaign)
        print(f"Job submitted with ID: {job.job_id}")
        
        return job
    
    else:
        # Run directly (for testing)
        print("Running optimization campaign directly...")
        return run_optimization_campaign()


if __name__ == "__main__":
    result = main()
    if hasattr(result, 'job_id'):
        print(f"Job {result.job_id} submitted successfully")
    else:
        print("Campaign completed successfully")
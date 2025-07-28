#!/usr/bin/env python3
"""
Simple optimization campaign test for BALAM cluster.
Uses ax-platform with submitit for job submission.
"""

import os
import sys
import time
from typing import Dict, Any


def branin_function(x1: float, x2: float) -> float:
    """Branin function optimization test."""
    import math
    
    a = 1
    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * math.cos(x1)
    term3 = s
    
    return term1 + term2 + term3


def run_ax_optimization():
    """Run optimization using ax-platform."""
    try:
        from ax.service.ax_client import AxClient
        import torch
        
        # Set random seed for reproducibility  
        torch.manual_seed(42)
        
        print("Starting Ax optimization...")
        
        # Create Ax client
        ax_client = AxClient()
        
        # Define search space
        ax_client.create_experiment(
            name="branin_test",
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        
        # Run optimization
        results = []
        for i in range(10):  # 10 iterations
            parameters, trial_index = ax_client.get_next_trial()
            
            # Evaluate function
            result = branin_function(parameters["x1"], parameters["x2"])
            
            # Complete trial
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            
            results.append({
                "trial": trial_index,
                "x1": parameters["x1"],
                "x2": parameters["x2"], 
                "objective": result
            })
            
            print(f"Trial {trial_index}: x1={parameters['x1']:.3f}, x2={parameters['x2']:.3f}, f={result:.6f}")
        
        # Get best parameters
        best_parameters, best_trial_index = ax_client.get_best_parameters()
        
        print(f"\nBest result: Trial {best_trial_index}")
        print(f"Best parameters: {best_parameters}")
        
        # Get best objective value
        best_objective = min(r["objective"] for r in results)
        
        print(f"Best objective: {best_objective:.6f}")
        
        return {
            "results": results,
            "best_parameters": best_parameters,
            "best_objective": best_objective,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    print("=== BALAM Optimization Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Job started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show SLURM environment
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_PROCID', 'SLURM_NODEID']
    for var in slurm_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    print("\n=== Running Ax Optimization ===")
    result = run_ax_optimization()
    
    if result["status"] == "success":
        print(f"\nOptimization completed successfully!")
        print(f"Total trials: {len(result['results'])}")
        print(f"Best objective: {result['best_objective']:.6f}")
        return_code = 0
    else:
        print(f"\nOptimization failed: {result['error']}")
        return_code = 1
    
    print(f"\nJob completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
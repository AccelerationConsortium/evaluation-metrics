#!/usr/bin/env python3
"""
Simple optimization test for Niagara cluster using ax-platform.
Usage: python niagara_test_simple.py [--local]
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

def main():
    parser = argparse.ArgumentParser(description='Niagara optimization test')
    parser.add_argument('--local', action='store_true', help='Local test mode')
    args = parser.parse_args()
    
    print("=== Niagara Optimization Test ===")
    if 'SLURM_JOB_ID' in os.environ:
        print(f"SLURM Job ID: {os.environ['SLURM_JOB_ID']}")
        print(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    
    best_params, best_value = run_optimization()
    
    print(f"\n=== Final Result ===")
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()
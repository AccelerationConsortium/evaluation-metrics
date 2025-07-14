#!/usr/bin/env python
"""
Simple test of the BALAM benchmark framework with local execution.
"""

import os
import sys

# Add src to path for direct import  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run local benchmark test."""
    print("Testing BALAM Benchmark Framework - Local Run")
    print("=" * 50)
    
    try:
        from gpcheck.benchmark import BenchmarkConfig, run_benchmark_batch
        print("✓ Successfully imported benchmark framework")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Using standalone implementation...")
        
        # Fallback to standalone test
        os.system("python test_benchmark_standalone.py")
        return
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        objective_function="branin",
        num_initialization_trials=[2, 5, 10],
        num_optimization_steps=30,
        num_repeats=3,
        algorithms=["GPEI", "Random"],
    )
    
    print(f"\nBenchmark Configuration:")
    print(f"  Session ID: {config.session_id}")
    print(f"  Objective: {config.objective_function}")
    print(f"  Initialization trials: {config.num_initialization_trials}")
    print(f"  Optimization steps: {config.num_optimization_steps}")
    print(f"  Repeats per config: {config.num_repeats}")
    print(f"  Algorithms: {config.algorithms}")
    
    total_runs = len(config.algorithms) * len(config.num_initialization_trials) * config.num_repeats
    print(f"  Total runs: {total_runs}")
    
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
    
    print(f"\nRunning {len(parameter_sets)} benchmark tests locally...")
    
    try:
        results = run_benchmark_batch(parameter_sets)
        
        print(f"\nCompleted {len(results)} benchmark runs:")
        for result in results:
            print(
                f"  {result['algorithm']} with {result['num_initialization_trials']} "
                f"init trials (repeat {result['repeat']}): "
                f"final regret = {result['final_regret']:.6f}"
            )
            
        print("\n✓ Framework is working correctly!")
        print("✓ Ready for BALAM deployment with environment variables:")
        print("  - SLURM_ACCOUNT (required for cluster submission)")
        print("  - MONGODB_APP_NAME (optional, for data storage)")
        print("  - MONGODB_API_KEY (optional, for data storage)")
        
    except Exception as e:
        print(f"✗ Error running benchmarks: {e}")
        print("Check dependencies and configuration")

if __name__ == "__main__":
    main()
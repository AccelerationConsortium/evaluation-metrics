#!/usr/bin/env python3
"""
Standalone validation script for the benchmark framework.
This script can be run independently to test core functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gpcheck.benchmark import (
    BenchmarkConfig, 
    branin_function, 
    run_single_benchmark,
    run_local_benchmark
)


def test_branin_function():
    """Test Branin function implementation."""
    print("Testing Branin function...")
    
    # Test a few points
    test_points = [
        (-3.14159, 12.275, 0.397887),  # Global minimum
        (3.14159, 2.275, 0.397887),   # Global minimum
        (0, 5, None),                  # Random point
    ]
    
    for x1, x2, expected in test_points:
        result = branin_function(x1, x2)
        print(f"  f({x1:.3f}, {x2:.3f}) = {result:.6f}")
        if expected is not None:
            assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"
    
    print("✓ Branin function tests passed")


def test_single_benchmark():
    """Test single benchmark run."""
    print("\nTesting single benchmark run...")
    
    result = run_single_benchmark(
        objective_func=branin_function,
        n_init=3,
        n_steps=5,
        algorithm="Random",
        seed=42
    )
    
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  N init: {result['n_init']}")
    print(f"  N steps: {result['n_steps']}")
    print(f"  Best value: {result['best_y']:.6f}")
    print(f"  Total evaluations: {len(result['results'])}")
    
    # Verify structure
    assert result['algorithm'] == "Random"
    assert result['n_init'] == 3
    assert result['n_steps'] == 5
    assert len(result['results']) == 8  # 3 + 5
    assert 'best_y' in result
    
    print("✓ Single benchmark test passed")


def test_local_benchmark():
    """Test local benchmark execution."""
    print("\nTesting local benchmark...")
    
    config = BenchmarkConfig(
        num_initialization_trials=[2, 3],
        num_optimization_steps=5,
        num_repeats=2,
        algorithms=["Random"],
        seed=42
    )
    
    results = run_local_benchmark(config)
    
    print(f"  Configuration: {len(config.num_initialization_trials)} init configs")
    print(f"  Algorithms: {len(config.algorithms)}")
    print(f"  Repeats: {config.num_repeats}")
    print(f"  Total runs: {len(results)}")
    
    # Check results
    expected_runs = len(config.num_initialization_trials) * len(config.algorithms) * config.num_repeats
    assert len(results) == expected_runs
    
    # Print summary
    for i, result in enumerate(results):
        print(f"  Run {i+1}: n_init={result['n_init']}, best_y={result['best_y']:.4f}")
    
    print("✓ Local benchmark test passed")


def test_configuration():
    """Test benchmark configuration."""
    print("\nTesting benchmark configuration...")
    
    # Default configuration
    config1 = BenchmarkConfig()
    print(f"  Default seed: {config1.seed}")
    assert config1.seed is not None
    
    # Custom configuration
    config2 = BenchmarkConfig(
        objective_function="branin",
        num_initialization_trials=[5, 10],
        num_optimization_steps=20,
        num_repeats=3,
        algorithms=["GPEI", "Random"],
        seed=123
    )
    
    print(f"  Custom config objective: {config2.objective_function}")
    print(f"  Custom config seed: {config2.seed}")
    assert config2.seed == 123
    
    print("✓ Configuration test passed")


def main():
    """Run all validation tests."""
    print("=" * 50)
    print("Benchmark Framework Validation")
    print("=" * 50)
    
    try:
        test_branin_function()
        test_single_benchmark()
        test_local_benchmark()
        test_configuration()
        
        print("\n" + "=" * 50)
        print("✓ All validation tests passed!")
        print("The benchmark framework is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
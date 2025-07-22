#!/usr/bin/env python
"""Test the benchmark functions locally before cluster submission."""

import numpy as np
from benchmark_functions import branin, hartmann6, evaluate_benchmark
from niagara_submitit import generate_parameter_sets, mongodb_evaluate

# Test Branin function
branin_result = branin(-np.pi, 12.275)  # Known near-optimum
print(f"Branin at near-optimum: {branin_result:.6f} (should be ~0.398)")

# Test Hartmann6 function
hartmann_result = hartmann6(np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]))
print(f"Hartmann6 at near-optimum: {hartmann_result:.6f} (should be ~-3.322)")

# Test parameter generation
branin_params = generate_parameter_sets("branin")
print(f"Generated {len(branin_params)} Branin parameter sets")

# Test local evaluation
test_params = {"function": "branin", "x1": 0.0, "x2": 0.0, "repeat": 0, "session_id": "test"}
result = mongodb_evaluate(test_params, verbose=True)
print(f"Local evaluation result: {result}")

print("Tests completed - ready for cluster submission")

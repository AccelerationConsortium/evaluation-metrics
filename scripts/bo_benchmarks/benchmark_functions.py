"""Benchmark objective functions for BO evaluation."""

import numpy as np


def branin(x1, x2):
    """Branin function (2D) with global minimum of 0.397887 at three points."""
    y = float(
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    return y


def hartmann6(x):
    """Hartmann 6D function with global minimum of -3.32237 at specific point."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    sum_term = 0
    for i in range(4):
        inner_sum = 0
        for j in range(6):
            inner_sum += A[i, j] * (x[j] - P[i, j])**2
        sum_term += alpha[i] * np.exp(-inner_sum)
    
    return -sum_term


def evaluate_benchmark(parameters):
    """Evaluate a benchmark function with given parameters."""
    function_name = parameters["function"]
    
    if function_name == "branin":
        x1 = parameters["x1"]
        x2 = parameters["x2"]
        return branin(x1, x2)
    elif function_name == "hartmann6":
        x = [parameters[f"x{i+1}"] for i in range(6)]
        return hartmann6(np.array(x))
    else:
        raise ValueError(f"Unknown function: {function_name}")

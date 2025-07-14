"""Simple test for the benchmark module."""

import math

def branin(x1, x2):
    """Branin function as implemented in honegumi."""
    y = float(
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5.0 / math.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * math.pi)) * math.cos(x1)
        + 10
    )
    return y

def test_branin_function():
    """Test the Branin function directly."""
    # Test known Branin minimum points
    # Global minima are approximately at:
    # (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)
    
    # Test a known minimum point
    y = branin(-math.pi, 12.275)
    print(f"Branin at (-π, 12.275): {y}")
    
    # Should be close to global minimum value of 0.397887
    assert y < 1.0, f"Expected Branin value < 1.0, got {y}"
    
    print("Basic Branin function test passed!")


if __name__ == "__main__":
    test_branin_function()
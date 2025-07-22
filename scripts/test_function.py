#!/usr/bin/env python3
"""Simple test function for BALAM cluster submission."""

import time


def simple_test():
    """A minimal test function."""
    print("Hello from BALAM!")
    time.sleep(10)
    result = 2 + 2
    print(f"Result: {result}")
    return result

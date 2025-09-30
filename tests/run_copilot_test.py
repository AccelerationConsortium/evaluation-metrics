#!/usr/bin/env python
"""Simple runner for copilot agent test without pytest dependency."""

import sys
from test_copilot_agent import (
    test_copilot_agent_running,
    test_basic_arithmetic,
    test_string_operations,
)


def run_tests():
    """Run all copilot agent tests."""
    tests = [
        ("test_copilot_agent_running", test_copilot_agent_running),
        ("test_basic_arithmetic", test_basic_arithmetic),
        ("test_string_operations", test_string_operations),
    ]
    
    passed = 0
    failed = 0
    
    print("Running Copilot Agent Tests")
    print("=" * 50)
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    
    print("\n✓ All tests passed! GitHub Copilot coding agent is running successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())

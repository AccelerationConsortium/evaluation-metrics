"""
Test module with functions that can be imported by submitit.
"""

def test_function():
    """Simple test function to run on the cluster."""
    import time
    result = sum(range(1000))
    return {"status": "success", "result": result}
"""Tests for the benchmark framework."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.gpcheck.benchmark import (
    branin,
    BenchmarkConfig,
    BALAMExecutor,
    MongoDBClient,
    run_single_benchmark,
    run_benchmark_batch,
)


def test_branin_function():
    """Test the Branin objective function."""
    # Test a few known points
    # Global minimum is around (pi, 2.275) ≈ (3.141592, 2.275) with value ≈ 0.397887
    result = branin(np.pi, 2.275)
    assert abs(result - 0.397887) < 0.001
    
    # Test another point
    result = branin(0, 0)
    assert isinstance(result, float)
    assert result > 0  # Branin function is always positive


def test_benchmark_config():
    """Test benchmark configuration."""
    config = BenchmarkConfig()
    
    # Test defaults
    assert config.objective_function == "branin"
    assert config.num_initialization_trials == [2, 5, 10, 20]
    assert config.num_optimization_steps == 50
    assert config.num_repeats == 10
    assert config.algorithms == ["GPEI", "EI", "Random"]
    assert config.session_id is not None
    
    # Test custom configuration
    custom_config = BenchmarkConfig(
        objective_function="hartmann6",
        num_initialization_trials=[5, 15],
        num_repeats=3,
        algorithms=["GPEI"],
    )
    
    assert custom_config.objective_function == "hartmann6"
    assert custom_config.num_initialization_trials == [5, 15]
    assert custom_config.num_repeats == 3
    assert custom_config.algorithms == ["GPEI"]


@patch.dict('os.environ', {'SLURM_ACCOUNT': 'test-account'})
def test_balam_executor():
    """Test BALAM executor initialization."""
    executor = BALAMExecutor(
        partition="cpu",
        walltime_min=30,
        mem_per_cpu=2000,
    )
    
    assert executor.partition == "cpu"
    assert executor.account == "test-account"
    assert executor.walltime_min == 30
    assert executor.mem_per_cpu == 2000


@patch.dict('os.environ', {
    'MONGODB_APP_NAME': 'test-app',
    'MONGODB_API_KEY': 'test-key'
})
def test_mongodb_client():
    """Test MongoDB client initialization."""
    client = MongoDBClient()
    
    assert client.app_name == "test-app"
    assert client.api_key == "test-key"
    assert client.database == "gp-benchmarks"
    assert client.collection == "campaigns"
    assert "test-app" in client.url


def test_mongodb_client_missing_credentials():
    """Test MongoDB client with missing credentials."""
    with pytest.raises(ValueError, match="MongoDB app name and API key must be provided"):
        MongoDBClient()


def test_run_single_benchmark():
    """Test running a single benchmark."""
    parameters = {
        "objective_function": "branin",
        "algorithm": "GPEI",
        "num_initialization_trials": 5,
        "num_optimization_steps": 20,
        "repeat": 0,
        "session_id": "test-session",
    }
    
    result = run_single_benchmark(parameters)
    
    # Check result structure
    expected_keys = [
        "objective_function", "algorithm", "num_initialization_trials",
        "num_optimization_steps", "repeat", "session_id", "final_best_value",
        "final_regret", "best_values", "global_minimum"
    ]
    
    for key in expected_keys:
        assert key in result
    
    # Check values
    assert result["objective_function"] == "branin"
    assert result["algorithm"] == "GPEI"
    assert result["num_initialization_trials"] == 5
    assert result["num_optimization_steps"] == 20
    assert result["repeat"] == 0
    assert result["session_id"] == "test-session"
    
    # Check optimization results
    assert isinstance(result["final_best_value"], float)
    assert isinstance(result["final_regret"], float)
    assert result["final_regret"] >= 0  # Regret should be non-negative
    assert len(result["best_values"]) == 20  # Should have value for each step
    assert result["global_minimum"] == 0.397887
    
    # Check that best values are monotonically decreasing (or equal)
    best_values = result["best_values"]
    for i in range(1, len(best_values)):
        assert best_values[i] <= best_values[i-1]


def test_run_single_benchmark_unknown_function():
    """Test running benchmark with unknown objective function."""
    parameters = {
        "objective_function": "unknown",
        "algorithm": "GPEI",
        "num_initialization_trials": 5,
        "num_optimization_steps": 20,
        "repeat": 0,
        "session_id": "test-session",
    }
    
    with pytest.raises(ValueError, match="Unknown objective function: unknown"):
        run_single_benchmark(parameters)


@patch('src.gpcheck.benchmark.MongoDBClient')
def test_run_benchmark_batch(mock_mongodb_class):
    """Test running a batch of benchmarks."""
    # Mock MongoDB client
    mock_client = MagicMock()
    mock_mongodb_class.return_value = mock_client
    
    parameter_batch = [
        {
            "objective_function": "branin",
            "algorithm": "GPEI",
            "num_initialization_trials": 5,
            "num_optimization_steps": 10,
            "repeat": 0,
            "session_id": "test-session",
        },
        {
            "objective_function": "branin",
            "algorithm": "Random",
            "num_initialization_trials": 10,
            "num_optimization_steps": 10,
            "repeat": 1,
            "session_id": "test-session",
        },
    ]
    
    results = run_benchmark_batch(parameter_batch)
    
    assert len(results) == 2
    assert results[0]["algorithm"] == "GPEI"
    assert results[1]["algorithm"] == "Random"
    
    # Check that MongoDB client was called
    assert mock_client.store_result.call_count == 2


@patch('src.gpcheck.benchmark.MongoDBClient')
def test_run_benchmark_batch_no_mongodb(mock_mongodb_class):
    """Test running benchmark batch when MongoDB is not available."""
    # Make MongoDB client raise ValueError
    mock_mongodb_class.side_effect = ValueError("No credentials")
    
    parameter_batch = [
        {
            "objective_function": "branin",
            "algorithm": "GPEI",
            "num_initialization_trials": 5,
            "num_optimization_steps": 10,
            "repeat": 0,
            "session_id": "test-session",
        },
    ]
    
    results = run_benchmark_batch(parameter_batch)
    
    assert len(results) == 1
    assert results[0]["algorithm"] == "GPEI"


def test_reproducibility():
    """Test that benchmark results are reproducible with same seed."""
    parameters = {
        "objective_function": "branin",
        "algorithm": "GPEI",
        "num_initialization_trials": 5,
        "num_optimization_steps": 10,
        "repeat": 42,  # Fixed seed
        "session_id": "test-session",
    }
    
    result1 = run_single_benchmark(parameters)
    result2 = run_single_benchmark(parameters)
    
    # Results should be identical with same seed
    assert result1["final_best_value"] == result2["final_best_value"]
    assert result1["best_values"] == result2["best_values"]
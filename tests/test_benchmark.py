"""
Tests for the benchmark framework.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from gpcheck.benchmark import (
    BenchmarkConfig, 
    branin_function, 
    get_objective_function, 
    run_single_benchmark,
    run_local_benchmark,
    BALAMExecutor
)


def test_benchmark_config():
    """Test BenchmarkConfig creation and defaults."""
    config = BenchmarkConfig()
    
    assert config.objective_function == "branin"
    assert config.num_initialization_trials == [2, 5, 10, 20, 50]
    assert config.num_optimization_steps == 100
    assert config.num_repeats == 20
    assert config.algorithms == ["GPEI", "EI", "Random"]
    assert config.seed is not None


def test_benchmark_config_custom():
    """Test BenchmarkConfig with custom parameters."""
    config = BenchmarkConfig(
        objective_function="custom",
        num_initialization_trials=[5, 10],
        num_optimization_steps=50,
        num_repeats=10,
        algorithms=["Random"],
        seed=42
    )
    
    assert config.objective_function == "custom"
    assert config.num_initialization_trials == [5, 10]
    assert config.num_optimization_steps == 50
    assert config.num_repeats == 10
    assert config.algorithms == ["Random"]
    assert config.seed == 42


def test_branin_function():
    """Test Branin function evaluation."""
    # Test known global minima (approximately)
    result1 = branin_function(-np.pi, 12.275)
    result2 = branin_function(np.pi, 2.275)
    result3 = branin_function(9.42478, 2.475)
    
    # Should be close to 0.397887
    assert abs(result1 - 0.397887) < 0.01
    assert abs(result2 - 0.397887) < 0.01
    assert abs(result3 - 0.397887) < 0.01


def test_get_objective_function():
    """Test objective function retrieval."""
    func = get_objective_function("branin")
    assert func == branin_function
    
    with pytest.raises(ValueError):
        get_objective_function("unknown")


def test_run_single_benchmark():
    """Test single benchmark run."""
    result = run_single_benchmark(
        objective_func=branin_function,
        n_init=5,
        n_steps=10,
        algorithm="test",
        seed=42
    )
    
    assert result["algorithm"] == "test"
    assert result["n_init"] == 5
    assert result["n_steps"] == 10
    assert result["seed"] == 42
    assert len(result["results"]) == 15  # 5 init + 10 optimization
    assert "best_y" in result
    
    # Check phases
    init_results = [r for r in result["results"] if r["phase"] == "initialization"]
    opt_results = [r for r in result["results"] if r["phase"] == "optimization"]
    assert len(init_results) == 5
    assert len(opt_results) == 10


def test_run_local_benchmark():
    """Test local benchmark execution."""
    config = BenchmarkConfig(
        num_initialization_trials=[2, 5],
        num_optimization_steps=10,
        num_repeats=2,
        algorithms=["test1", "test2"],
        seed=42
    )
    
    results = run_local_benchmark(config)
    
    # Should have 2 init configs * 2 algorithms * 2 repeats = 8 results
    assert len(results) == 8
    
    # Check that all combinations are present
    algorithms = {r["algorithm"] for r in results}
    n_inits = {r["n_init"] for r in results}
    assert algorithms == {"test1", "test2"}
    assert n_inits == {2, 5}


@patch.dict('os.environ', {}, clear=True)
def test_balam_executor_no_account():
    """Test BALAMExecutor fails without SLURM_ACCOUNT."""
    with pytest.raises(EnvironmentError):
        BALAMExecutor()


@patch.dict('os.environ', {'SLURM_ACCOUNT': 'test-account'})
def test_balam_executor_with_account():
    """Test BALAMExecutor creation with SLURM_ACCOUNT."""
    executor = BALAMExecutor(
        partition="gpu",
        walltime_min=120,
        mem_per_cpu=8000
    )
    
    assert executor.partition == "gpu"
    assert executor.walltime_min == 120
    assert executor.mem_per_cpu == 8000


@patch.dict('os.environ', {'SLURM_ACCOUNT': 'test-account'})
@patch('gpcheck.benchmark.submitit')
def test_submit_benchmark_campaign(mock_submitit):
    """Test benchmark campaign submission."""
    # Mock submitit
    mock_executor = Mock()
    mock_job = Mock()
    mock_job.job_id = "12345"
    mock_executor.submit.return_value = mock_job
    mock_submitit.AutoExecutor.return_value = mock_executor
    
    executor = BALAMExecutor()
    config = BenchmarkConfig(
        num_initialization_trials=[2],
        num_optimization_steps=5,
        num_repeats=1,
        algorithms=["test"],
        seed=42
    )
    
    jobs = executor.submit_benchmark_campaign(config)
    
    # Should submit 1 job (1 init config * 1 algorithm * 1 repeat)
    assert len(jobs) == 1
    assert jobs[0] == mock_job
    
    # Check executor was configured
    mock_submitit.AutoExecutor.assert_called_once()
    mock_executor.update_parameters.assert_called_once()
    mock_executor.submit.assert_called_once()


def test_balam_executor_no_submitit():
    """Test BALAMExecutor fails gracefully without submitit."""
    with patch.dict('os.environ', {'SLURM_ACCOUNT': 'test-account'}):
        with patch('gpcheck.benchmark.submitit', None):
            executor = BALAMExecutor()
            config = BenchmarkConfig()
            
            with pytest.raises(ImportError):
                executor.submit_benchmark_campaign(config)
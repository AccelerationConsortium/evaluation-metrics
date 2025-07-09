from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from gpcheck.data import create_hartmann_dataset
from gpcheck.metrics import calculate_metrics
from gpcheck.models import GPConfig, GPModel


# core dataset fixture
@pytest.fixture(scope="session")
def sample_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """standard dataset for most tests: 3D, 15 train, 8 test."""
    return create_hartmann_dataset(seed=42, dim=3, n_train=15, n_test=8)


# core model configuration
@pytest.fixture
def default_config() -> GPConfig:
    """default GPConfig for testing."""
    return GPConfig(seed=42)


# fitted model with sample dataset
@pytest.fixture
def fitted_model(
    sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any],
    default_config: GPConfig,
) -> Tuple[GPModel, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """fitted GP model with sample dataset."""
    train_X, train_Y, test_X, test_Y, f_true = sample_dataset
    model = GPModel(default_config)
    model.fit(train_X, train_Y)
    return model, train_X, train_Y, test_X, test_Y


# metrics computed from fitted model
@pytest.fixture
def sample_metrics(
    fitted_model: Tuple[GPModel, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Dict[str, Any]:
    """sample metrics computed from fitted model."""
    model, train_X, train_Y, test_X, test_Y = fitted_model
    return calculate_metrics(model, train_X, train_Y, test_X, test_Y)


# helper functions
def create_dataset_with_params(
    seed: int = 42, dim: int = 3, n_train: int = 10, n_test: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """helper function to create dataset with custom parameters."""
    return create_hartmann_dataset(seed=seed, dim=dim, n_train=n_train, n_test=n_test)


def assert_valid_dataset(
    train_X: torch.Tensor, train_Y: torch.Tensor, test_X: torch.Tensor, test_Y: torch.Tensor
) -> None:
    """assert that a dataset has valid structure and values."""
    # check types
    assert isinstance(train_X, torch.Tensor)
    assert isinstance(train_Y, torch.Tensor)
    assert isinstance(test_X, torch.Tensor)
    assert isinstance(test_Y, torch.Tensor)

    # check shapes are consistent
    assert train_X.shape[0] == train_Y.shape[0]
    assert test_X.shape[0] == test_Y.shape[0]
    assert train_X.shape[1] == test_X.shape[1]
    assert train_Y.shape[1] == test_Y.shape[1] == 1

    # check dtypes
    assert train_X.dtype == torch.float64
    assert train_Y.dtype == torch.float64
    assert test_X.dtype == torch.float64
    assert test_Y.dtype == torch.float64

    # check bounds for Hartmann function
    assert torch.all(train_X >= 0.0) and torch.all(train_X <= 1.0)
    assert torch.all(test_X >= 0.0) and torch.all(test_X <= 1.0)


def assert_valid_metrics(metrics: Dict[str, Any]) -> None:
    """assert that metrics have valid structure and values."""
    # check required keys
    required_keys = [
        "r2",
        "lengthscales",
        "importance",
        "imp_cumsum",
        "loo_nll",
        "rank_tau",
        "predictions",
    ]
    assert all(key in metrics for key in required_keys)

    # check predictions structure
    pred_keys = ["mean", "std", "true"]
    assert all(key in metrics["predictions"] for key in pred_keys)

    # check value ranges
    assert metrics["r2"] <= 1.0
    assert metrics["loo_nll"] > 0
    assert -1.0 <= metrics["rank_tau"] <= 1.0
    assert np.all(metrics["lengthscales"] > 0)
    assert np.isclose(np.sum(metrics["importance"]), 1.0)

    # check no NaN values
    assert not np.isnan(metrics["r2"])
    assert not np.isnan(metrics["loo_nll"])
    assert not np.isnan(metrics["rank_tau"])
    assert not np.any(np.isnan(metrics["lengthscales"]))
    assert not np.any(np.isnan(metrics["importance"]))

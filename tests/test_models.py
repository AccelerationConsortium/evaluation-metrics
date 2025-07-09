from typing import Any, Tuple

from botorch.models import SingleTaskGP
import numpy as np
import pytest
import torch

from gpcheck.models import GPConfig, GPModel


class TestGPConfig:
    """test the GPConfig class."""

    def test_defaults(self) -> None:
        """test GPConfig default values."""
        config = GPConfig()

        assert config.seed == 0
        assert config.ard_num_dims is None
        assert config.use_standardize is True
        assert config.model_class == SingleTaskGP
        assert isinstance(config.device, torch.device)

    def test_custom_values(self) -> None:
        """test GPConfig with custom values."""
        custom_device = torch.device("cpu")
        config = GPConfig(seed=42, device=custom_device, ard_num_dims=3, use_standardize=False)

        assert config.seed == 42
        assert config.device == custom_device
        assert config.ard_num_dims == 3
        assert config.use_standardize is False


class TestGPModel:
    """test the GPModel class."""

    def test_initialization(self) -> None:
        """test GPModel initialization."""
        config = GPConfig()
        model = GPModel(config)

        assert model.config == config
        assert not hasattr(model, "model")

    def test_fit_and_predict(
        self, sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> None:
        """test model fitting and prediction."""
        train_X, train_Y, test_X, test_Y, f_true = sample_dataset

        config = GPConfig(seed=42)
        model = GPModel(config)

        # should not be fitted initially
        assert not hasattr(model, "model")

        # fit the model
        model.fit(train_X, train_Y)

        # should be fitted now
        assert hasattr(model, "model")
        assert model.model is not None
        assert isinstance(model.model, SingleTaskGP)

        # make predictions
        mean, std = model.predict(test_X)

        # check outputs
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        assert mean.shape == (test_X.shape[0],)
        assert std.shape == (test_X.shape[0],)
        assert torch.all(std > 0)  # standard deviation should be positive

    def test_lengthscales(
        self, sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> None:
        """test getting lengthscales after fitting."""
        train_X, train_Y, test_X, test_Y, f_true = sample_dataset

        config = GPConfig(seed=42)
        model = GPModel(config)
        model.fit(train_X, train_Y)

        # get lengthscales
        lengthscales, importance = model.get_lengthscales()

        # check outputs
        assert isinstance(lengthscales, np.ndarray)
        assert isinstance(importance, np.ndarray)
        assert lengthscales.shape == (train_X.shape[1],)
        assert importance.shape == (train_X.shape[1],)
        assert np.all(lengthscales > 0)  # should be positive
        assert np.isclose(np.sum(importance), 1.0)  # should sum to 1

    def test_predict_before_fit(
        self, sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> None:
        """test that prediction fails before fitting."""
        train_X, train_Y, test_X, test_Y, f_true = sample_dataset

        config = GPConfig()
        model = GPModel(config)

        with pytest.raises((ValueError, AttributeError)):
            model.predict(test_X)

    def test_lengthscales_before_fit(self) -> None:
        """test that getting lengthscales fails before fitting."""
        config = GPConfig()
        model = GPModel(config)

        with pytest.raises((ValueError, AttributeError)):
            model.get_lengthscales()

    def test_reproducibility(
        self, sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> None:
        """test that model results are reproducible."""
        train_X, train_Y, test_X, test_Y, f_true = sample_dataset

        config = GPConfig(seed=42)

        # create two models with same config
        model1 = GPModel(config)
        model1.fit(train_X, train_Y)
        mean1, std1 = model1.predict(test_X)

        model2 = GPModel(config)
        model2.fit(train_X, train_Y)
        mean2, std2 = model2.predict(test_X)

        # results should be identical
        torch.testing.assert_close(mean1, mean2)
        torch.testing.assert_close(std1, std2)

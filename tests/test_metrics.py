from typing import Any, Tuple

import numpy as np
import torch

from gpcheck.metrics import calculate_metrics, importance_concentration, loo_pseudo_likelihood


class TestLOOPseudoLikelihood:
    """test the loo_pseudo_likelihood function."""

    def test_basic_functionality(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that loo_pseudo_likelihood returns reasonable values."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        loo_nll = loo_pseudo_likelihood(model.model, train_X, train_Y)

        # check output type and positivity
        assert isinstance(loo_nll, (float, np.floating))
        assert loo_nll > 0
        assert not np.isnan(loo_nll)
        assert not np.isinf(loo_nll)

    def test_deterministic_behavior(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that loo_pseudo_likelihood is deterministic."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        loo_nll1 = loo_pseudo_likelihood(model.model, train_X, train_Y)
        loo_nll2 = loo_pseudo_likelihood(model.model, train_X, train_Y)

        assert loo_nll1 == loo_nll2


class TestImportanceConcentration:
    """test the importance_concentration function."""

    def test_basic_functionality(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that importance_concentration returns reasonable values."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        # get importance from model
        _, importance = model.get_lengthscales()
        imp_cumsum = importance_concentration(importance)

        # check output types and shapes
        assert isinstance(importance, np.ndarray)
        assert isinstance(imp_cumsum, dict)
        assert importance.shape == (train_X.shape[1],)
        assert len(imp_cumsum) == train_X.shape[1]

        # check properties
        assert np.all(importance >= 0)
        assert np.isclose(np.sum(importance), 1.0)
        assert np.all(list(imp_cumsum.values())[0] >= 0)
        assert np.isclose(list(imp_cumsum.values())[-1], 1.0)

        # check monotonic increase
        cumsum_values = list(imp_cumsum.values())
        assert all(cumsum_values[i] >= cumsum_values[i - 1] for i in range(1, len(cumsum_values)))

    def test_deterministic_behavior(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that importance_concentration is deterministic."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        # get importance from model
        _, importance = model.get_lengthscales()
        imp_cumsum1 = importance_concentration(importance)
        imp_cumsum2 = importance_concentration(importance)

        assert imp_cumsum1 == imp_cumsum2


class TestCalculateMetrics:
    """test the calculate_metrics function."""

    def test_basic_functionality(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that calculate_metrics returns all required metrics."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)

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

    def test_value_ranges(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that calculated metrics have reasonable value ranges."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)

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

    def test_predictions_structure(
        self, fitted_model: Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """test that predictions have correct structure."""
        model, train_X, train_Y, test_X, test_Y = fitted_model

        metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)
        predictions = metrics["predictions"]

        # check shapes
        assert predictions["mean"].shape == (test_X.shape[0],)
        assert predictions["std"].shape == (test_X.shape[0],)
        assert predictions["true"].shape == (test_X.shape[0],)

        # check types
        assert isinstance(predictions["mean"], np.ndarray)
        assert isinstance(predictions["std"], np.ndarray)
        assert isinstance(predictions["true"], np.ndarray)

        # check that std is positive
        assert np.all(predictions["std"] > 0)

    def test_reproducibility(
        self, sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> None:
        """test that calculate_metrics is reproducible."""
        train_X, train_Y, test_X, test_Y, f_true = sample_dataset

        from gpcheck.models import GPConfig, GPModel

        config = GPConfig(seed=42)

        # create two models with same config
        model1 = GPModel(config)
        model1.fit(train_X, train_Y)
        metrics1 = calculate_metrics(model1, train_X, train_Y, test_X, test_Y)

        model2 = GPModel(config)
        model2.fit(train_X, train_Y)
        metrics2 = calculate_metrics(model2, train_X, train_Y, test_X, test_Y)

        # results should be identical
        assert metrics1["r2"] == metrics2["r2"]
        assert metrics1["loo_nll"] == metrics2["loo_nll"]
        assert metrics1["rank_tau"] == metrics2["rank_tau"]
        np.testing.assert_array_equal(metrics1["lengthscales"], metrics2["lengthscales"])
        np.testing.assert_array_equal(metrics1["importance"], metrics2["importance"])

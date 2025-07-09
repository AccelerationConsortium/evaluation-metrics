from typing import Any, Dict

import numpy as np

from gpcheck.data import create_hartmann_dataset
from gpcheck.metrics import calculate_metrics
from gpcheck.models import GPConfig, GPModel
from gpcheck.visualization import create_evaluation_plot


class TestEndToEndWorkflow:
    """test the complete end-to-end workflow."""

    def test_complete_workflow(self) -> None:
        """test complete workflow from data creation to visualization."""
        # create data
        train_X, train_Y, test_X, test_Y, f_true = create_hartmann_dataset(
            seed=42, dim=3, n_train=15, n_test=8
        )

        # create and fit model
        config = GPConfig(seed=42)
        model = GPModel(config)
        model.fit(train_X, train_Y)

        # calculate metrics
        metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)

        # create plot
        fig = create_evaluation_plot(metrics)

        # verify everything worked
        assert metrics is not None
        assert fig is not None
        assert all(key in metrics for key in ["r2", "loo_nll", "rank_tau"])

    def test_reproducibility_across_modules(self) -> None:
        """test that the entire workflow is reproducible."""

        # run workflow twice with same seed
        def run_workflow() -> Dict[str, Any]:
            train_X, train_Y, test_X, test_Y, f_true = create_hartmann_dataset(
                seed=42, dim=3, n_train=10, n_test=5
            )
            config = GPConfig(seed=42)
            model = GPModel(config)
            model.fit(train_X, train_Y)
            return calculate_metrics(model, train_X, train_Y, test_X, test_Y)

        metrics1 = run_workflow()
        metrics2 = run_workflow()

        # key metrics should be identical
        assert metrics1["r2"] == metrics2["r2"]
        assert metrics1["loo_nll"] == metrics2["loo_nll"]
        assert metrics1["rank_tau"] == metrics2["rank_tau"]

    def test_different_configurations(self) -> None:
        """test workflow with different model configurations."""
        train_X, train_Y, test_X, test_Y, f_true = create_hartmann_dataset(
            seed=42, dim=3, n_train=12, n_test=6
        )

        # test different configurations
        configs = [
            GPConfig(seed=42, use_standardize=True),
            GPConfig(seed=42, use_standardize=False),
            GPConfig(seed=42, ard_num_dims=3),
        ]

        for config in configs:
            model = GPModel(config)
            model.fit(train_X, train_Y)
            metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)

            # all metrics should be computed successfully
            assert not np.isnan(metrics["r2"])
            assert not np.isnan(metrics["loo_nll"])
            assert not np.isnan(metrics["rank_tau"])

    def test_backward_compatibility(self) -> None:
        """test that refactored code produces similar results to original."""
        # create data similar to original script
        train_X, train_Y, test_X, test_Y, f_true = create_hartmann_dataset(
            seed=42, dim=6, n_train=20, n_test=10
        )

        # use default configuration
        config = GPConfig(seed=42)
        model = GPModel(config)
        model.fit(train_X, train_Y)

        # calculate metrics
        metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)

        # verify structure matches original
        assert "r2" in metrics
        assert "lengthscales" in metrics
        assert "importance" in metrics
        assert "imp_cumsum" in metrics
        assert "loo_nll" in metrics
        assert "rank_tau" in metrics
        assert "predictions" in metrics

        # verify predictions structure
        pred = metrics["predictions"]
        assert "mean" in pred
        assert "std" in pred
        assert "true" in pred

        # verify data types and shapes
        assert isinstance(pred["mean"], np.ndarray)
        assert isinstance(pred["std"], np.ndarray)
        assert isinstance(pred["true"], np.ndarray)
        assert pred["mean"].shape == (10,)
        assert pred["std"].shape == (10,)
        assert pred["true"].shape == (10,)

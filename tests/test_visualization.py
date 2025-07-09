from typing import Any, Dict, Tuple

import plotly.graph_objects as go
import pytest
import torch

from gpcheck.metrics import calculate_metrics
from gpcheck.models import GPConfig, GPModel
from gpcheck.visualization import create_evaluation_plot


class TestCreateEvaluationPlot:
    """test the create_evaluation_plot function."""

    def test_basic_functionality(self, sample_metrics: Dict[str, Any]) -> None:
        """test that create_evaluation_plot creates a valid plotly figure."""
        fig = create_evaluation_plot(sample_metrics)

        # check that it returns a plotly figure
        assert isinstance(fig, go.Figure)

        # check that it has data
        assert hasattr(fig, "data")
        assert fig.data is not None

        # check that it has layout
        assert fig.layout is not None
        assert fig.layout.title is not None

    def test_with_custom_title(self, sample_metrics: Dict[str, Any]) -> None:
        """test create_evaluation_plot with custom title."""
        custom_title = "Custom Test Title"
        fig = create_evaluation_plot(sample_metrics, title=custom_title)

        assert isinstance(fig, go.Figure)
        assert custom_title in fig.layout.title.text

    def test_handles_different_data_sizes(
        self, sample_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]
    ) -> None:
        """test that create_evaluation_plot handles different data sizes."""

        train_X, train_Y, test_X, test_Y, f_true = sample_dataset

        # create model and metrics
        config = GPConfig(seed=42)
        model = GPModel(config)
        model.fit(train_X, train_Y)
        metrics = calculate_metrics(model, train_X, train_Y, test_X, test_Y)

        # create plot
        fig = create_evaluation_plot(metrics)

        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "data")
        assert fig.data is not None

    def test_error_handling(self) -> None:
        """test that create_evaluation_plot handles invalid inputs gracefully."""
        # test with empty metrics
        with pytest.raises((KeyError, ValueError, TypeError)):
            create_evaluation_plot({})

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_evaluation_plot(
    metrics: Dict[str, Any],
    title: str = "GP model evaluation metrics",
) -> go.Figure:
    """
    Create a comprehensive evaluation plot with multiple subplots.

    This replicates the original plotting functionality from evaluation_metrics.py

    Args:
        metrics: Dictionary containing metrics from calculate_metrics()
        title: Title for the overall plot

    Returns:
        Plotly figure with evaluation plots

    Maybe better to turn "metrics" into a Pydantic model? Not sure.
    """
    predictions = metrics["predictions"]
    test_np = predictions["true"]
    mean_np = predictions["mean"]
    std_np = predictions["std"]
    imp = metrics["importance"]
    ls = metrics["lengthscales"]
    r2 = metrics["r2"]
    rank_tau = metrics["rank_tau"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"prediction vs true (R²={r2:.4f})",
            "feature importance",
            "lengthscales",
            f"rank correlation (τ={rank_tau:.4f})",
        ),
    )

    # prediction vs true values
    fig.add_trace(
        go.Scatter(
            x=test_np,
            y=mean_np,
            mode="markers",
            error_y=dict(array=std_np, visible=True),
            name="predictions",
        ),
        row=1,
        col=1,
    )
    min_val = min(test_np.min(), mean_np.min())
    max_val = max(test_np.max(), mean_np.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(dash="dash"),
            name="perfect prediction",
        ),
        row=1,
        col=1,
    )

    # feature importance
    fig.add_trace(
        go.Bar(x=[f"x{i + 1}" for i in range(len(ls))], y=imp, name="importance"), row=1, col=2
    )

    # lengthscales
    fig.add_trace(
        go.Bar(x=[f"x{i + 1}" for i in range(len(ls))], y=ls, name="lengthscales"), row=2, col=1
    )

    # rank correlation
    fig.add_trace(
        go.Scatter(
            x=np.argsort(test_np).argsort(),
            y=np.argsort(mean_np).argsort(),
            mode="markers",
            name="rank correlation",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[0, len(test_np) - 1],
            y=[0, len(test_np) - 1],
            mode="lines",
            line=dict(dash="dash"),
            name="perfect rank",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=800, width=1200, showlegend=False, title_text=title, title_x=0.5)

    fig.update_xaxes(title_text="true value", row=1, col=1)
    fig.update_yaxes(title_text="predicted value", row=1, col=1)
    fig.update_xaxes(title_text="feature", row=1, col=2)
    fig.update_yaxes(title_text="normalized importance (1/ℓ)", row=1, col=2)
    fig.update_xaxes(title_text="feature", row=2, col=1)
    fig.update_yaxes(title_text="lengthscale (ℓ)", row=2, col=1)
    fig.update_xaxes(title_text="true value rank", row=2, col=2)
    fig.update_yaxes(title_text="predicted value rank", row=2, col=2)

    return fig

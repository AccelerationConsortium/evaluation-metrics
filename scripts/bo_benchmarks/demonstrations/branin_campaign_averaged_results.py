#!/usr/bin/env python3
"""
Branin optimization campaign with 10 repeat runs and averaged evaluation metrics.
This script runs multiple campaigns with different seeds and plots the average behavior.
"""

from pathlib import Path
import time

from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
import torch

from gpcheck.metrics import loo_pseudo_likelihood
from gpcheck.models import GPConfig, GPModel

obj1_name = "branin"


def branin(x1, x2):
    y = float(
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    return y


def interval_score(y_true, lower, upper, alpha=0.05):
    """
    Calculate interval score for uncertainty quantification evaluation.

    For 95% confidence intervals (alpha=0.05), the formula is:
    f_interval = (upper - lower) + (2/alpha) * max(0, lower - y) + (2/alpha) * max(0, y - upper)

    Args:
        y_true: True values
        lower: Lower confidence bounds
        upper: Upper confidence bounds
        alpha: Confidence level (0.05 for 95% CI)

    Returns:
        Array of interval scores (lower is better)
    """
    width = upper - lower
    penalty = (2 / alpha) * (np.maximum(0, lower - y_true) + np.maximum(0, y_true - upper))
    return width + penalty


def calculate_iteration_metrics(df, objective_name, trial_index):
    """Calculate evaluation metrics for a given trial."""
    if trial_index < 3:  # Need at least 3 points for meaningful metrics
        return None

    # Prepare data for GP modeling (use all points up to current trial)
    current_data = df.iloc[: trial_index + 1].copy()
    X = current_data[["x1", "x2"]].values
    y = current_data[objective_name].values.astype(float)

    # Create train-test split (80-20)
    n_train = max(2, int(0.8 * len(X)))  # Ensure at least 2 training points
    indices = np.arange(len(X))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # If no test data available, use last 20% or at least 1 point
    if len(test_indices) == 0:
        n_test = max(1, len(X) - n_train)
        train_indices = indices[:-n_test]
        test_indices = indices[-n_test:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Fit GP model
    config = GPConfig()
    model = GPModel(config)

    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64).unsqueeze(
        -1
    )  # Add output dimension
    model.fit(X_train_tensor, y_train_tensor)

    # Predictions on test set for R²
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    y_pred_test_output = model.predict(X_test_tensor)
    if isinstance(y_pred_test_output, tuple):
        y_pred_test = y_pred_test_output[0].detach().numpy().flatten()
    else:
        y_pred_test = y_pred_test_output.detach().numpy().flatten()
    gp_r2 = r2_score(y_test, y_pred_test)

    # Calculate uncertainty on test set for interval score
    y_pred_test_with_var, y_var_test = model.predict(X_test_tensor, return_std=True)
    if isinstance(y_pred_test_with_var, tuple):
        y_pred_test_with_var = y_pred_test_with_var[0].detach().numpy().flatten()
        y_var_test = y_var_test[0].detach().numpy().flatten()
    else:
        y_pred_test_with_var = y_pred_test_with_var.detach().numpy().flatten()
        y_var_test = y_var_test.detach().numpy().flatten()
    y_std_test = np.sqrt(y_var_test)

    # 95% confidence intervals
    lower = y_pred_test_with_var - 1.96 * y_std_test
    upper = y_pred_test_with_var + 1.96 * y_std_test
    interval_score_val = np.mean(interval_score(y_test, lower, upper))

    # Predictions on all data for ranking correlation
    X_tensor = torch.tensor(X, dtype=torch.float64)
    y_pred_all_output = model.predict(X_tensor)
    if isinstance(y_pred_all_output, tuple):
        y_pred_all = y_pred_all_output[0].detach().numpy().flatten()
    else:
        y_pred_all = y_pred_all_output.detach().numpy().flatten()

    # Calculate ranking correlation (Kendall's tau)
    rank_tau, _ = kendalltau(y, y_pred_all)
    if np.isnan(rank_tau):
        rank_tau = 0.0

    # Calculate LOO negative log-likelihood
    loo_nll = loo_pseudo_likelihood(X, y, model.kernel, model.noise_variance)

    # Calculate feature importance (inverse length scales)
    try:
        # Access the model's kernel lengthscale
        if hasattr(model.model, "covar_module") and hasattr(
            model.model.covar_module, "base_kernel"
        ):
            lengthscale = model.model.covar_module.base_kernel.lengthscale.detach().numpy()
        elif hasattr(model.model, "covar_module") and hasattr(
            model.model.covar_module, "lengthscale"
        ):
            lengthscale = model.model.covar_module.lengthscale.detach().numpy()
        else:
            # Fallback: assume default lengthscales
            lengthscale = np.ones(X.shape[1])

        if lengthscale.ndim == 0:
            lengthscale = np.array([lengthscale.item()] * X.shape[1])
        elif lengthscale.ndim == 2:
            lengthscale = lengthscale.flatten()

        inv_lengthscales = 1.0 / (lengthscale + 1e-8)  # Add small value to avoid division by zero
    except Exception:
        inv_lengthscales = np.ones(X.shape[1])

    imp_mean = np.mean(inv_lengthscales)
    imp_std = np.std(inv_lengthscales)

    print(
        f"Trial {trial_index}: GP R² = {gp_r2:.4f}, Rank τ = {rank_tau:.4f}, "
        f"LOO NLL = {loo_nll:.4f}"
    )
    print(
        f"Trial {trial_index}: Interval Score = {interval_score_val:.4f}, Imp Std = {imp_std:.4f}"
    )

    return {
        "trial": trial_index,
        "gp_r2": gp_r2,
        "rank_tau": rank_tau,
        "loo_nll": loo_nll,
        "interval_score": interval_score_val,
        "imp_mean": imp_mean,
        "imp_std": imp_std,
        "ax_cv_r2": None,  # Will be filled by separate function
        "ax_cv_interval_score": None,
    }


def calculate_ax_cross_validation_metrics(cv_client, trial_index):
    """Calculate Ax cross-validation metrics."""
    if trial_index < 3:
        return None, None

    try:
        # Get model bridge
        if hasattr(cv_client._generation_strategy, "_curr"):
            bridge = cv_client._generation_strategy._curr.model
        else:
            return None, None

        # Perform cross-validation
        cv_results = cross_validate(bridge)

        # Create cross-validation plot to extract data
        fig = interact_cross_validation_plotly(cv_results)

        # Extract observed vs predicted values from the plot
        y_act = fig["data"][0].y  # Observed values
        y_pred = fig["data"][0].x  # Predicted values (cross-validation)

        # Calculate R² for cross-validation
        ax_cv_r2 = r2_score(y_act, y_pred)

        # Extract uncertainty data from the plot
        assert fig["data"][1].error_y is not None, (
            "Uncertainty data not available from Ax cross-validation"
        )
        uncertainty = fig["data"][1].error_y.array

        # Convert standard errors to 95% confidence intervals for interval score calculation
        lower_bounds = y_pred - 1.96 * uncertainty
        upper_bounds = y_pred + 1.96 * uncertainty
        ax_cv_interval_score = np.mean(interval_score(y_act, lower_bounds, upper_bounds))

        print(
            f"Trial {trial_index}: Ax CV R² = {ax_cv_r2:.4f}, "
            f"Ax CV Interval Score = {ax_cv_interval_score:.4f}"
        )

        return ax_cv_r2, ax_cv_interval_score

    except Exception as e:
        print(f"Trial {trial_index}: Could not calculate Ax CV metrics: {e}")
        return None, None


def run_single_campaign(seed, num_trials=50):
    """Run a single optimization campaign with the given seed."""
    print(f"\n=== Running Campaign with Seed {seed} ===")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create Ax client for the main optimization
    ax_client = AxClient()
    ax_client.create_experiment(
        name=f"branin_experiment_seed_{seed}",
        parameters=[
            {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
        ],
        objectives={obj1_name: ObjectiveProperties(minimize=True)},
    )

    # Create separate Ax client for cross-validation metrics only
    cv_generation_strategy = GenerationStrategy(
        [
            GenerationStep(model=Models.SOBOL, num_trials=3),
            GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-1),
        ]
    )
    cv_client = AxClient(generation_strategy=cv_generation_strategy)
    cv_client.create_experiment(
        name=f"branin_cv_experiment_seed_{seed}",
        parameters=[
            {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
        ],
        objectives={obj1_name: ObjectiveProperties(minimize=True)},
    )

    # Data collection
    data = []
    all_metrics = []

    # Run optimization trials
    for i in range(num_trials):
        # Get next trial parameters from main client
        parameters, trial_index = ax_client.get_next_trial()

        # Evaluate objective function
        result = branin(parameters["x1"], parameters["x2"])

        # Store data
        data.append(
            {
                "trial": trial_index,
                "x1": parameters["x1"],
                "x2": parameters["x2"],
                obj1_name: result,
            }
        )

        # Complete trial in main client
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)

        # Mirror data to CV client for cross-validation analysis
        _, cv_trial_index = cv_client.attach_trial(parameters)
        cv_client.complete_trial(trial_index=cv_trial_index, raw_data=result)

        # Calculate metrics iteration-by-iteration (starting from trial 3)
        df = pd.DataFrame(data)

        if trial_index >= 2:  # Calculate metrics starting from trial 3 (index 2)
            metrics = calculate_iteration_metrics(df, obj1_name, trial_index)
            if metrics:
                # Calculate Ax cross-validation metrics
                ax_cv_r2, ax_cv_interval_score = calculate_ax_cross_validation_metrics(
                    cv_client, trial_index
                )
                metrics["ax_cv_r2"] = ax_cv_r2
                metrics["ax_cv_interval_score"] = ax_cv_interval_score
                all_metrics.append(metrics)

    # Create final DataFrame
    df = pd.DataFrame(data)

    return df, all_metrics


def run_averaged_campaigns(num_campaigns=10, num_trials=50):
    """Run multiple campaigns and compute averaged metrics."""

    start_time = time.time()

    all_campaign_results = []

    for campaign_idx in range(num_campaigns):
        seed = campaign_idx + 1000  # Use seeds 1000, 1001, 1002, etc.
        df, metrics = run_single_campaign(seed, num_trials)

        campaign_result = {"seed": seed, "df": df, "metrics": metrics}
        all_campaign_results.append(campaign_result)

        elapsed = time.time() - start_time
        print(f"Completed campaign {campaign_idx + 1}/{num_campaigns} in {elapsed:.1f}s")

    # Now compute averaged metrics
    print(f"\n=== Computing Averaged Results from {num_campaigns} Campaigns ===")

    # Find common trial indices across all campaigns
    all_trial_indices = set()
    for result in all_campaign_results:
        trial_indices = {m["trial"] for m in result["metrics"]}
        all_trial_indices.update(trial_indices)

    common_trial_indices = sorted(all_trial_indices)

    # Average metrics across campaigns
    averaged_metrics = []

    for trial_idx in common_trial_indices:
        trial_metrics = []
        best_so_far_values = []

        for result in all_campaign_results:
            # Find metrics for this trial
            trial_metric = None
            for m in result["metrics"]:
                if m["trial"] == trial_idx:
                    trial_metric = m
                    break

            if trial_metric:
                trial_metrics.append(trial_metric)

                # Get best-so-far value for this trial
                df = result["df"]
                best_so_far = np.minimum.accumulate(df[obj1_name].values)
                if trial_idx < len(best_so_far):
                    best_so_far_values.append(best_so_far[trial_idx])

        if trial_metrics:
            # Compute averages (excluding None values)
            def safe_mean(values):
                valid_values = [v for v in values if v is not None and not np.isnan(v)]
                return np.mean(valid_values) if valid_values else None

            averaged_metric = {
                "trial": trial_idx,
                "best_so_far": np.mean(best_so_far_values) if best_so_far_values else None,
                "gp_r2": safe_mean([m["gp_r2"] for m in trial_metrics]),
                "rank_tau": safe_mean([m["rank_tau"] for m in trial_metrics]),
                "loo_nll": safe_mean([m["loo_nll"] for m in trial_metrics]),
                "interval_score": safe_mean([m["interval_score"] for m in trial_metrics]),
                "imp_std": safe_mean([m["imp_std"] for m in trial_metrics]),
                "ax_cv_r2": safe_mean([m["ax_cv_r2"] for m in trial_metrics]),
                "ax_cv_interval_score": safe_mean(
                    [m["ax_cv_interval_score"] for m in trial_metrics]
                ),
            }
            averaged_metrics.append(averaged_metric)

    return averaged_metrics, all_campaign_results


# Main execution
if __name__ == "__main__":
    print("Starting 10-campaign averaged evaluation...")
    start_time = time.time()

    # Run 2 campaigns and compute averages (for testing)
    averaged_metrics, all_results = run_averaged_campaigns(num_campaigns=2, num_trials=15)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")

    # Create the averaged plot
    objective_name = obj1_name

    if averaged_metrics:
        trial_nums = [m["trial"] for m in averaged_metrics]
        best_so_far_values = [m["best_so_far"] for m in averaged_metrics]
        gp_r2_values = [m["gp_r2"] for m in averaged_metrics]
        rank_tau_values = [m["rank_tau"] for m in averaged_metrics]
        loo_values = [m["loo_nll"] for m in averaged_metrics]
        imp_std_values = [m["imp_std"] for m in averaged_metrics]

        # Extract Ax CV R² and interval scores (filter None/NaN)
        ax_cv_r2_values = [
            m["ax_cv_r2"]
            for m in averaged_metrics
            if (m["ax_cv_r2"] is not None and np.isfinite(m["ax_cv_r2"]))
        ]
        interval_score_values = [
            m["interval_score"] for m in averaged_metrics if m["interval_score"] is not None
        ]
        ax_cv_interval_score_values = [
            m["ax_cv_interval_score"]
            for m in averaged_metrics
            if (m["ax_cv_interval_score"] is not None and np.isfinite(m["ax_cv_interval_score"]))
        ]

        ax_cv_trial_nums = [
            m["trial"]
            for m in averaged_metrics
            if (m["ax_cv_r2"] is not None and np.isfinite(m["ax_cv_r2"]))
        ]
        interval_trial_nums = [
            m["trial"] for m in averaged_metrics if m["interval_score"] is not None
        ]
        ax_cv_interval_trial_nums = [
            m["trial"]
            for m in averaged_metrics
            if (m["ax_cv_interval_score"] is not None and np.isfinite(m["ax_cv_interval_score"]))
        ]

        # Create single plot (no top subplot since we're only showing averaged metrics)
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=150)

        # Create 7 y-axes: 1 left + 6 right
        # Left axis: Best-so-far trace
        # Right axes: R² (shared), Rank τ, LOO NLL, GP Interval Score, Ax CV Interval Score, Imp Std

        ax_r2 = ax.twinx()  # Right axis 1: R² values (0-1)
        ax_tau = ax.twinx()  # Right axis 2: Rank correlation τ (-1 to 1)
        ax_loo = ax.twinx()  # Right axis 3: LOO NLL
        ax_gp_int = ax.twinx()  # Right axis 4: GP Interval Score
        ax_ax_int = ax.twinx()  # Right axis 5: Ax CV Interval Score
        ax_imp_std = ax.twinx()  # Right axis 6: Importance Std Dev

        # Position the additional y-axes
        ax_tau.spines["right"].set_position(("outward", 60))
        ax_loo.spines["right"].set_position(("outward", 120))
        ax_gp_int.spines["right"].set_position(("outward", 180))
        ax_ax_int.spines["right"].set_position(("outward", 240))
        ax_imp_std.spines["right"].set_position(("outward", 300))

        # Plot best-so-far trace on the left (main) y-axis
        line_best = ax.plot(
            trial_nums,
            best_so_far_values,
            color="gray",
            linewidth=6,
            alpha=0.3,
            label="Best so far (avg)",
            linestyle="-",
            zorder=1,
        )

        # Plot R² values on right axis 1
        line_gp_r2 = ax_r2.plot(
            trial_nums, gp_r2_values, "r-", label="GP R² (avg)", marker="o", linewidth=2, zorder=3
        )
        lines = line_best + line_gp_r2

        # Plot Ax CV R² if available (also on right axis 1)
        if ax_cv_r2_values:
            line_ax_r2 = ax_r2.plot(
                ax_cv_trial_nums,
                ax_cv_r2_values,
                "orange",
                label="Ax CV R² (avg)",
                marker="d",
                linewidth=2,
                linestyle="--",
                zorder=3,
            )
            lines += line_ax_r2

        # Set R² axis range dynamically to include negative values if present
        r2_all = [r for r in gp_r2_values if not np.isnan(r)]
        if ax_cv_r2_values:
            r2_all += [r for r in ax_cv_r2_values if not np.isnan(r)]

        if r2_all:  # Only set limits if we have valid R² values
            r2_min = min(r2_all)
            r2_max = max(r2_all)
            margin = max(1e-3, 0.05 * (r2_max - r2_min if r2_max > r2_min else 1.0))
            ax_r2.set_ylim(r2_min - margin, r2_max + margin)
        else:
            ax_r2.set_ylim(-1, 1)  # Default range if no valid R² values

        # Plot rank tau correlation on right axis 2 (fixed -1 to 1 range)
        line_tau = ax_tau.plot(
            trial_nums,
            rank_tau_values,
            "g-",
            label="Rank τ (avg)",
            marker="s",
            linewidth=2,
            zorder=3,
        )
        lines += line_tau
        ax_tau.set_ylim(-1, 1)

        # Plot LOO NLL on right axis 3 (auto range)
        line_loo = ax_loo.plot(
            trial_nums, loo_values, "b-", label="LOO NLL (avg)", marker="^", linewidth=2, zorder=3
        )
        lines += line_loo

        # Plot GP interval score on right axis 4 (auto range)
        if interval_score_values:
            line_gp_int = ax_gp_int.plot(
                interval_trial_nums,
                interval_score_values,
                "purple",
                label="GP Interval Score (avg)",
                marker="v",
                linewidth=2,
                linestyle=":",
                zorder=3,
            )
            lines += line_gp_int

        # Plot Ax CV interval score on right axis 5 (auto range)
        if ax_cv_interval_score_values:
            line_ax_int = ax_ax_int.plot(
                ax_cv_interval_trial_nums,
                ax_cv_interval_score_values,
                "magenta",
                label="Ax CV Interval Score (avg)",
                marker="*",
                linewidth=2,
                linestyle="-.",
                zorder=3,
            )
            lines += line_ax_int

        # Plot importance std dev on right axis 6 (auto range)
        line_imp_std = ax_imp_std.plot(
            trial_nums,
            imp_std_values,
            "pink",
            label="Importance Std (avg)",
            marker="p",
            linewidth=2,
            linestyle="-.",
            zorder=3,
        )
        lines += line_imp_std

        # Set axis labels and colors
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(f"Best {objective_name}", color="gray")
        ax_r2.set_ylabel("R² Values", color="red")
        ax_tau.set_ylabel("Rank Correlation (τ)", color="green")
        ax_loo.set_ylabel("LOO NLL", color="blue")
        ax_gp_int.set_ylabel("GP Interval Score", color="purple")
        ax_ax_int.set_ylabel("Ax CV Interval Score", color="magenta")
        ax_imp_std.set_ylabel("Importance Std", color="pink")

        # Set tick colors to match y-axis colors
        ax.tick_params(axis="y", labelcolor="gray")
        ax_r2.tick_params(axis="y", labelcolor="red")
        ax_tau.tick_params(axis="y", labelcolor="green")
        ax_loo.tick_params(axis="y", labelcolor="blue")
        ax_gp_int.tick_params(axis="y", labelcolor="purple")
        ax_ax_int.tick_params(axis="y", labelcolor="magenta")
        ax_imp_std.tick_params(axis="y", labelcolor="pink")

        ax.set_title("Averaged Evaluation Metrics vs Trial (10 Campaigns)")
        ax.set_xlim(-10, None)  # Add whitespace for legend

        # Place legend inside plot with white background and transparency
        labels = [line.get_label() for line in lines]
        legend = ax.legend(
            lines, labels, loc="upper left", framealpha=0.8, fancybox=False, shadow=False
        )
        legend.get_frame().set_facecolor("white")

        print(f"\nAveraged LOO Negative Log-Likelihood values: {loo_values}")
        print(f"Averaged Rank tau correlation values: {rank_tau_values}")
        if ax_cv_r2_values:
            print(f"Averaged Ax Cross Validation R² values: {ax_cv_r2_values}")
        if interval_score_values:
            print(f"Averaged GP Interval Score values: {interval_score_values}")
        if ax_cv_interval_score_values:
            print(f"Averaged Ax CV Interval Score values: {ax_cv_interval_score_values}")
        print(f"Averaged Importance Std values: {imp_std_values}")

    else:
        ax.text(
            0.5,
            0.5,
            "Not enough data\nfor metrics calculation",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Averaged Evaluation Metrics")

    plt.tight_layout()
    # Save matplotlib figure to the same directory as this script
    _out_path = Path(__file__).with_name("branin_campaign_averaged_results.png")
    _out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(_out_path), dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nCompleted {len(averaged_metrics)} averaged metric calculations from 10 campaigns")
    print(f"Results saved to: {_out_path}")

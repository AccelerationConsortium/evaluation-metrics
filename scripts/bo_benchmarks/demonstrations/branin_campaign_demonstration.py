#!/usr/bin/env python3
"""
Branin optimization campaign with 50 trials and iteration-by-iteration evaluation metrics calculation.
This script demonstrates proper use of ax-platform with GP evaluation metrics calculated after each trial.
"""

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import torch
from gpcheck.models import GPModel, GPConfig
from gpcheck.metrics import loo_pseudo_likelihood, importance_concentration
from sklearn.metrics import r2_score
from scipy.stats import kendalltau

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
    below_penalty = np.maximum(0, lower - y_true)
    above_penalty = np.maximum(0, y_true - upper)
    return width + (2 / alpha) * below_penalty + (2 / alpha) * above_penalty


def normal_95ci(mean, std):
    """Convert mean and standard deviation to 95% confidence intervals."""
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    return lower, upper


def calculate_ax_cross_validation_metrics(ax_client, ax_client_cv, trial_index):
    """
    Calculate metrics using a separate Ax client dedicated to cross-validation.
    This client uses BOTORCH_MODULAR immediately. Since we only attach completed
    trials to this client (no suggestions requested from it), we may need to
    trigger a one-off generation to force-fit the model before running CV.
    """
    # Mirror all completed trials from the main client onto the CV client (attach only missing ones)
    df_main = ax_client.get_trials_data_frame()
    df_cv = ax_client_cv.get_trials_data_frame()
    # Use rounded parameter tuples as unique keys to avoid relying on arm_name
    existing_params = set()
    if not df_cv.empty:
        for _, r in df_cv.iterrows():
            existing_params.add((round(float(r["x1"]), 6), round(float(r["x2"]), 6)))
    for _, row in df_main.sort_values("trial_index").iterrows():
        key = (round(float(row["x1"]), 6), round(float(row["x2"]), 6))
        if key in existing_params:
            continue
        params = {"x1": float(row["x1"]), "x2": float(row["x2"]) }
        _, t_idx_cv = ax_client_cv.attach_trial(parameters=params)
        ax_client_cv.complete_trial(trial_index=t_idx_cv, raw_data=float(row[obj1_name]))
        existing_params.add(key)

    # Need at least 2 points for LOO CV to be meaningful
    df_cv = ax_client_cv.get_trials_data_frame()
    if len(df_cv) < 2:
        return None, None

    # Ensure the CV model is fit to the latest data on every call by triggering a refit
    # via a one-off generation, then abandon that transient trial.
    _ , tmp_trial_idx = ax_client_cv.get_next_trial()
    ax_client_cv.abandon_trial(tmp_trial_idx)
    model_bridge = ax_client_cv.generation_strategy.model

    # Cross-validate and compute R²
    cv = cross_validate(model=model_bridge, folds=-1)
    fig = interact_cross_validation_plotly(cv)
    # In Ax, data[0] is diagonal, data[1] is the error scatter trace for first metric
    trace = fig.data[1]
    y_act = np.array(trace.x)  # type: ignore[attr-defined]
    y_pred = np.array(trace.y)  # type: ignore[attr-defined]
    print(f"  Ax CV points: {len(y_act)}")
    ax_cv_r2 = r2_score(y_act, y_pred)
    print(f"  Ax CV R²: {ax_cv_r2:.4f}")

    # Compute interval score using uncertainty bars from the CV plot (LOO predictions)
    # Pull error bars (95% CI half-width) from the CV plot trace and compute interval score
    err = trace.error_y  # type: ignore[attr-defined]
    assert getattr(err, "array", None) is not None, "Error bars not available from Ax cross-validation"
    err_array = np.asarray(err.array, dtype=float)
    lower = y_pred - err_array
    upper = y_pred + err_array
    interval_scores = interval_score(y_act, lower, upper)
    ax_cv_interval_score = float(np.mean(interval_scores))
    print(f"  Ax CV Interval Score: {ax_cv_interval_score:.4f}")

    # Create CV plots directory
    cv_dir = Path(__file__).parent / "cv_plots"
    cv_dir.mkdir(parents=True, exist_ok=True)

    # Save the individual iteration files (for GIF creation later)
    cv_html_path = cv_dir / f"branin_cv_plot_trial_{trial_index}.html"
    fig.write_html(str(cv_html_path), include_plotlyjs="cdn")
    print(f"  Saved Ax CV Plotly HTML: {cv_html_path}")

    # Also save the CV data for slider creation
    cv_data = {
        'trial': trial_index,
        'x': y_act.tolist(),
        'y': y_pred.tolist(), 
        'error_y': err_array.tolist()
    }
    import json
    cv_data_path = cv_dir / f"branin_cv_data_trial_{trial_index}.json"
    with open(cv_data_path, 'w') as f:
        json.dump(cv_data, f)
    print(f"  Saved CV data: {cv_data_path}")

    # Save a simple parity plot with error bars as PNG using Matplotlib (for GIF creation)
    cv_out_path = cv_dir / f"branin_cv_plot_trial_{trial_index}.png"
    plt.figure(figsize=(5, 5), dpi=150)
    vmin = float(min(np.min(y_act), np.min(y_pred)))
    vmax = float(max(np.max(y_act), np.max(y_pred)))
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = vmin - pad, vmax + pad
    plt.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')
    plt.errorbar(y_act, y_pred, yerr=err_array, fmt='o', color='#1f77b4',
                 ecolor='lightgray', elinewidth=1, capsize=2, alpha=0.8)
    plt.xlabel('Observed')
    plt.ylabel('Predicted (LOO)')
    plt.title(f'Ax CV (trial {trial_index})')
    plt.tight_layout()
    plt.savefig(cv_out_path)
    plt.close()
    print(f"  Saved Ax CV Matplotlib PNG: {cv_out_path}")

    return ax_cv_r2, ax_cv_interval_score


def calculate_iteration_metrics(ax_client, ax_client_cv, trial_index, x1, x2, result):
    """Calculate evaluation metrics after each trial using both GP model and Ax cross validation."""
    # Skip if not enough trials
    if trial_index < 3:
        return None
    
    # Get all completed trials data
    df = ax_client.get_trials_data_frame()
    if len(df) < 3:
        return None
    
    # ===== GP-based metrics (gpcheck) =====
    # Prepare data for GP model with train-test split
    all_X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
    all_Y = torch.tensor(df[obj1_name].values, dtype=torch.float32).unsqueeze(-1)
    
    # Use 80-20 train-test split for realistic R² evaluation
    n_total = len(all_X)
    n_train = max(2, int(0.8 * n_total))  # Ensure at least 2 training points
    
    # Use fixed random indices for reproducible splits
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_X = all_X[train_indices]
    train_Y = all_Y[train_indices]
    test_X = all_X[test_indices] if len(test_indices) > 0 else train_X  # Fallback to train if no test data
    test_Y = all_Y[test_indices] if len(test_indices) > 0 else train_Y

    # Fit GP model using gpcheck
    config = GPConfig(seed=42)
    gp_model = GPModel(config)
    gp_model.fit(train_X, train_Y)

    # Calculate GP-based metrics on test set
    mean_pred, std_pred = gp_model.predict(test_X)

    gp_r2 = r2_score(test_Y.detach().cpu().numpy(), mean_pred.detach().cpu().numpy())

    ls, imp = gp_model.get_lengthscales()
    imp_dict = importance_concentration(imp)
    
    # Calculate mean and std dev of feature importances (inverse lengthscales)
    imp_mean = np.mean(imp)
    imp_std = np.std(imp)

    loo = loo_pseudo_likelihood(gp_model.model, train_X, train_Y)

    rank_tau = kendalltau(test_Y.squeeze().detach().cpu().numpy(), mean_pred.detach().cpu().numpy())[0]

    # Calculate interval score using GP predictions on test set
    y_true = test_Y.squeeze().detach().cpu().numpy()
    y_pred_mean = mean_pred.detach().cpu().numpy()
    y_pred_std = std_pred.detach().cpu().numpy()

    # Convert to 95% confidence intervals
    lower, upper = normal_95ci(y_pred_mean, y_pred_std)
    interval_scores = interval_score(y_true, lower, upper)
    mean_interval_score = np.mean(interval_scores)

    # ===== Ax cross validation metrics =====
    ax_cv_r2, ax_cv_interval_score = calculate_ax_cross_validation_metrics(ax_client, ax_client_cv, trial_index)

    metrics = {
        'trial': trial_index,
        'gp_r2': gp_r2,
        'ax_cv_r2': ax_cv_r2,
        'interval_score': mean_interval_score,
        'ax_cv_interval_score': ax_cv_interval_score,
        'imp_cumsum': imp_dict,
        'imp_mean': imp_mean,
        'imp_std': imp_std,
        'loo_nll': loo,
        'rank_tau': rank_tau,
        'lengthscales': ls,
        'importance': imp
    }

    # Print metrics for this iteration
    print(f"  GP R²: {gp_r2:.4f}, LOO NLL: {loo:.2f}")
    print(f"  GP Interval Score: {mean_interval_score:.4f}")
    print(f"  Importance Mean: {imp_mean:.4f}, Importance Std: {imp_std:.4f}")

    return metrics


# Create main ax client for optimization
ax_client = AxClient()

ax_client.create_experiment(
    parameters=[
        {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
        {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
    ],
    objectives={
        obj1_name: ObjectiveProperties(minimize=True),
    },
)

# Create separate AxClient for cross-validation with custom generation strategy
# This client mirrors the main client's data and uses BOTORCH_MODULAR immediately (no Sobol init)
gs_cv = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # Use BOTORCH_MODULAR for all trials
        )
    ]
)

ax_client_cv = AxClient(generation_strategy=gs_cv)
ax_client_cv.create_experiment(
    parameters=[
        {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
        {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
    ],
    objectives={
        obj1_name: ObjectiveProperties(minimize=True),
    },
)

# Track metrics per iteration
all_metrics = []

# Run 50 trials with iteration-by-iteration metrics calculation
print("Running Branin optimization campaign with 50 trials...")
for i in range(50):
    parameterization, trial_index = ax_client.get_next_trial()

    # extract parameters
    x1 = parameterization["x1"]
    x2 = parameterization["x2"]

    results = branin(x1, x2)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)
    
    print(f"Trial {trial_index}: x1={x1:.3f}, x2={x2:.3f}, branin={results:.3f}")
    
    # Calculate metrics after each trial (from trial 3 onward)
    iteration_metrics = calculate_iteration_metrics(ax_client, ax_client_cv, trial_index, x1, x2, results)
    if iteration_metrics:
        all_metrics.append(iteration_metrics)

best_parameters, metrics = ax_client.get_best_parameters()
print(f"\nBest parameters: {best_parameters}")
print(f"Best objective: {metrics}")

# Get results data
objectives = ax_client.objective_names
df = ax_client.get_trials_data_frame()

print(f"\nDataframe shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

if all_metrics:
    print("\n=== Final Evaluation Metrics ===")
    final_metrics = all_metrics[-1]
    print(f"GP R²: {final_metrics['gp_r2']:.4f}")
    if final_metrics['ax_cv_r2'] is not None:
        print(f"Ax Cross Validation R²: {final_metrics['ax_cv_r2']:.4f}")
    if final_metrics['interval_score'] is not None:
        print(f"GP Mean Interval Score: {final_metrics['interval_score']:.4f}")
    if final_metrics['ax_cv_interval_score'] is not None:
        print(f"Ax CV Mean Interval Score: {final_metrics['ax_cv_interval_score']:.4f}")
    print(f"Rank correlation (τ): {final_metrics['rank_tau']:.4f}")
    print(f"Leave-One-Out Negative Log-Likelihood: {final_metrics['loo_nll']:.2f}")
    print(f"Importance concentration: {final_metrics['imp_cumsum']}")
    print(f"Importance Mean: {final_metrics['imp_mean']:.4f}")
    print(f"Importance Std Dev: {final_metrics['imp_std']:.4f}")

# Create stacked vertical plots as requested
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), dpi=150)  # Increased width by 1.5x

# Plot 1: Standard "best so far" trace - use string name instead of list
objective_name = objectives[0]  # Get the first (and only) objective name as string
ax1.scatter(df.index, df[objective_name], ec="k", fc="none", label="Observed")
ax1.plot(
    df.index,
    np.minimum.accumulate(df[objective_name]),
    color="#0033FF",
    lw=2,
    label="Best to Trial",
)
ax1.set_xlabel("Trial Number")
ax1.set_ylabel(f"{objective_name} objective")
ax1.set_title("Optimization Progress")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Evaluation metrics over trials (iteration-by-iteration results)
if all_metrics:
    trial_nums = [m['trial'] for m in all_metrics]
    gp_r2_values = [m['gp_r2'] for m in all_metrics]
    rank_tau_values = [m['rank_tau'] for m in all_metrics] 
    loo_values = [m['loo_nll'] for m in all_metrics]
    imp_mean_values = [m['imp_mean'] for m in all_metrics]
    imp_std_values = [m['imp_std'] for m in all_metrics]
    
    # Extract Ax CV R² and interval scores (filter None/NaN)
    ax_cv_r2_values = [m['ax_cv_r2'] for m in all_metrics if (m['ax_cv_r2'] is not None and np.isfinite(m['ax_cv_r2']))]
    interval_score_values = [m['interval_score'] for m in all_metrics if m['interval_score'] is not None]
    ax_cv_interval_score_values = [m['ax_cv_interval_score'] for m in all_metrics if (m['ax_cv_interval_score'] is not None and np.isfinite(m['ax_cv_interval_score']))]
    
    ax_cv_trial_nums = [m['trial'] for m in all_metrics if (m['ax_cv_r2'] is not None and np.isfinite(m['ax_cv_r2']))]
    interval_trial_nums = [m['trial'] for m in all_metrics if m['interval_score'] is not None]
    ax_cv_interval_trial_nums = [m['trial'] for m in all_metrics if (m['ax_cv_interval_score'] is not None and np.isfinite(m['ax_cv_interval_score']))]
    
    # Create 8 y-axes as requested: 1 left + 7 right
    # Left axis: Best-so-far trace
    # Right axes: R² (shared), Rank τ, LOO NLL, GP Interval Score, Ax CV Interval Score, Imp Mean, Imp Std
    
    ax2_r2 = ax2.twinx()       # Right axis 1: R² values (0-1)
    ax2_tau = ax2.twinx()      # Right axis 2: Rank correlation τ (-1 to 1)  
    ax2_loo = ax2.twinx()      # Right axis 3: LOO NLL
    ax2_gp_int = ax2.twinx()   # Right axis 4: GP Interval Score
    ax2_ax_int = ax2.twinx()   # Right axis 5: Ax CV Interval Score
    ax2_imp_mean = ax2.twinx() # Right axis 6: Importance Mean
    ax2_imp_std = ax2.twinx()  # Right axis 7: Importance Std Dev
    
    # Position the additional y-axes
    ax2_tau.spines['right'].set_position(('outward', 60))
    ax2_loo.spines['right'].set_position(('outward', 120))
    ax2_gp_int.spines['right'].set_position(('outward', 180))
    ax2_ax_int.spines['right'].set_position(('outward', 240))
    ax2_imp_mean.spines['right'].set_position(('outward', 300))
    ax2_imp_std.spines['right'].set_position(('outward', 360))
    
    # Plot best-so-far trace on the left (main) y-axis
    best_so_far = np.minimum.accumulate(df[objective_name])
    line_best = ax2.plot(df.index, best_so_far, color='gray', linewidth=6, alpha=0.3, 
                         label='Best so far', linestyle='-', zorder=1)
    
    # Plot R² values on right axis 1
    line_gp_r2 = ax2_r2.plot(trial_nums, gp_r2_values, 'r-', label='GP R²', marker='o', linewidth=2, zorder=3)
    lines = line_best + line_gp_r2
    
    # Plot Ax CV R² if available (also on right axis 1)
    if ax_cv_r2_values:
        line_ax_r2 = ax2_r2.plot(ax_cv_trial_nums, ax_cv_r2_values, 'orange', label='Ax CV R²', marker='d', 
                              linewidth=2, linestyle='--', zorder=3)
        lines += line_ax_r2
    
    # Set R² axis range dynamically to include negative values if present
    r2_all = [r for r in gp_r2_values if not np.isnan(r)]
    if ax_cv_r2_values:
        r2_all += [r for r in ax_cv_r2_values if not np.isnan(r)]
    
    if r2_all:  # Only set limits if we have valid R² values
        r2_min = min(r2_all)
        r2_max = max(r2_all)
        margin = max(1e-3, 0.05 * (r2_max - r2_min if r2_max > r2_min else 1.0))
        ax2_r2.set_ylim(r2_min - margin, r2_max + margin)
    else:
        ax2_r2.set_ylim(-1, 1)  # Default range if no valid R² values
    
    # Plot rank tau correlation on right axis 2 (fixed -1 to 1 range)
    line_tau = ax2_tau.plot(trial_nums, rank_tau_values, 'g-', label='Rank τ', marker='s', linewidth=2, zorder=3)
    lines += line_tau
    ax2_tau.set_ylim(-1, 1)
    
    # Plot LOO NLL on right axis 3 (auto range)
    line_loo = ax2_loo.plot(trial_nums, loo_values, 'b-', label='LOO NLL', marker='^', linewidth=2, zorder=3)
    lines += line_loo
    
    # Plot GP interval score on right axis 4 (auto range)
    if interval_score_values:
        line_gp_int = ax2_gp_int.plot(interval_trial_nums, interval_score_values, 'purple', label='GP Interval Score', 
                              marker='v', linewidth=2, linestyle=':', zorder=3)
        lines += line_gp_int
    
    # Plot Ax CV interval score on right axis 5 (auto range)
    if ax_cv_interval_score_values:
        line_ax_int = ax2_ax_int.plot(ax_cv_interval_trial_nums, ax_cv_interval_score_values, 'magenta', 
                              label='Ax CV Interval Score', marker='*', linewidth=2, linestyle='-.', zorder=3)
        lines += line_ax_int
    
    # Plot importance mean on right axis 6 (auto range)
    line_imp_mean = ax2_imp_mean.plot(trial_nums, imp_mean_values, 'brown', 
                              label='Importance Mean', marker='h', linewidth=2, linestyle=':', zorder=3)
    lines += line_imp_mean
    
    # Plot importance std dev on right axis 7 (auto range)
    line_imp_std = ax2_imp_std.plot(trial_nums, imp_std_values, 'pink', 
                              label='Importance Std', marker='p', linewidth=2, linestyle='-.', zorder=3)
    lines += line_imp_std
        
    # Set axis labels and colors
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel(f"Best {objective_name}", color='gray')
    ax2_r2.set_ylabel("R² Values", color='red')
    ax2_tau.set_ylabel("Rank Correlation (τ)", color='green')
    ax2_loo.set_ylabel("LOO NLL", color='blue')
    ax2_gp_int.set_ylabel("GP Interval Score", color='purple')
    ax2_ax_int.set_ylabel("Ax CV Interval Score", color='magenta')
    ax2_imp_mean.set_ylabel("Importance Mean", color='brown')
    ax2_imp_std.set_ylabel("Importance Std", color='pink')
    
    # Set tick colors to match y-axis colors
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2_r2.tick_params(axis='y', labelcolor='red')
    ax2_tau.tick_params(axis='y', labelcolor='green')
    ax2_loo.tick_params(axis='y', labelcolor='blue')
    ax2_gp_int.tick_params(axis='y', labelcolor='purple')
    ax2_ax_int.tick_params(axis='y', labelcolor='magenta')
    ax2_imp_mean.tick_params(axis='y', labelcolor='brown')
    ax2_imp_std.tick_params(axis='y', labelcolor='pink')
    
    ax2.set_title("Evaluation Metrics vs Trial (Iteration-by-Iteration)")
    
    # Place legend inside plot with white background and transparency
    labels = [l.get_label() for l in lines]
    legend = ax2.legend(lines, labels, loc='upper left', framealpha=0.8, fancybox=False, shadow=False)
    legend.get_frame().set_facecolor('white')
    
    print(f"\nLOO Negative Log-Likelihood values: {loo_values}")
    print(f"Rank tau correlation values: {rank_tau_values}")
    if ax_cv_r2_values:
        print(f"Ax Cross Validation R² values: {ax_cv_r2_values}")
    if interval_score_values:
        print(f"GP Interval Score values: {interval_score_values}")
    if ax_cv_interval_score_values:
        print(f"Ax CV Interval Score values: {ax_cv_interval_score_values}")
    print(f"Importance Mean values: {imp_mean_values}")
    print(f"Importance Std values: {imp_std_values}")
    
else:
    ax2.text(0.5, 0.5, 'Not enough data\nfor metrics calculation', 
            ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Evaluation Metrics")

plt.tight_layout()
# Save matplotlib figure to the same directory as this script
_out_path = Path(__file__).with_name('branin_campaign_demonstration_results.png')
_out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(_out_path), dpi=150, bbox_inches='tight')
plt.show()

# Create and save Plotly version
if all_metrics:
    # Create subplots with secondary y-axes
    plotly_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Optimization Progress', 'Evaluation Metrics vs Trial'],
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Top plot: Optimization progress
    plotly_fig.add_trace(
        go.Scatter(x=df.index, y=df[objective_name], mode='markers', 
                  name='Observed', marker=dict(color='black', symbol='circle-open')),
        row=1, col=1
    )
    best_so_far = np.minimum.accumulate(df[objective_name])
    plotly_fig.add_trace(
        go.Scatter(x=df.index, y=best_so_far, mode='lines', 
                  name='Best to Trial', line=dict(color='#0033FF', width=2)),
        row=1, col=1
    )
    
    # Bottom plot: Metrics
    # Best-so-far trace (gray, semi-transparent)
    plotly_fig.add_trace(
        go.Scatter(x=df.index, y=best_so_far, mode='lines', 
                  name='Best so far', line=dict(color='gray', width=6), opacity=0.3),
        row=2, col=1
    )
    
    # GP R² (red)
    plotly_fig.add_trace(
        go.Scatter(x=trial_nums, y=gp_r2_values, mode='lines+markers', 
                  name='GP R²', line=dict(color='red'), marker=dict(symbol='circle')),
        row=2, col=1, secondary_y=True
    )
    
    # Ax CV R² (orange, if available)
    if ax_cv_r2_values:
        plotly_fig.add_trace(
            go.Scatter(x=ax_cv_trial_nums, y=ax_cv_r2_values, mode='lines+markers', 
                      name='Ax CV R²', line=dict(color='orange', dash='dash'), 
                      marker=dict(symbol='diamond')),
            row=2, col=1, secondary_y=True
        )
    
    # Other metrics (will be on secondary y-axis, but we'll handle scaling)
    colors = ['green', 'blue', 'purple', 'magenta', 'brown', 'pink']
    metrics_data = [
        (trial_nums, rank_tau_values, 'Rank τ', 'square'),
        (trial_nums, loo_values, 'LOO NLL', 'triangle-up'),
        (interval_trial_nums, interval_score_values, 'GP Interval Score', 'triangle-down'),
        (ax_cv_interval_trial_nums, ax_cv_interval_score_values, 'Ax CV Interval Score', 'star'),
        (trial_nums, imp_mean_values, 'Importance Mean', 'hexagon'),
        (trial_nums, imp_std_values, 'Importance Std', 'pentagon')
    ]
    
    for i, (x_data, y_data, name, symbol) in enumerate(metrics_data):
        if y_data:  # Only plot if data exists
            plotly_fig.add_trace(
                go.Scatter(x=x_data, y=y_data, mode='lines+markers', 
                          name=name, line=dict(color=colors[i]), 
                          marker=dict(symbol=symbol), yaxis='y3'),
                row=2, col=1
            )
    
    # Update layout
    plotly_fig.update_layout(
        title="Branin Campaign Results",
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    # Update axes
    plotly_fig.update_xaxes(title_text="Trial Number", row=1, col=1)
    plotly_fig.update_yaxes(title_text=f"{objective_name} objective", row=1, col=1)
    plotly_fig.update_xaxes(title_text="Trial Number", row=2, col=1)
    plotly_fig.update_yaxes(title_text=f"Best {objective_name}", row=2, col=1)
    plotly_fig.update_yaxes(title_text="Metrics", secondary_y=True, row=2, col=1)
    
    # Save Plotly HTML
    _html_path = Path(__file__).with_name('branin_campaign_demonstration_results.html')
    plotly_fig.write_html(str(_html_path), include_plotlyjs="cdn")
    print(f"Saved Plotly HTML: {_html_path}")

# Create GIF and slider plots from CV plots
cv_dir = Path(__file__).parent / "cv_plots"
if cv_dir.exists() and all_metrics:
    try:
        import imageio
        from PIL import Image
        
        # Create GIF from PNG files
        png_files = sorted(cv_dir.glob("branin_cv_plot_trial_*.png"))
        if png_files:
            images = []
            for png_file in png_files:
                images.append(imageio.imread(png_file))
            
            gif_path = cv_dir / "branin_cv_evolution.gif"
            imageio.mimsave(gif_path, images, duration=3.2)  # 4x slower (was 0.8, now 3.2)
            print(f"Created GIF: {gif_path}")
            
            # Try to create plotly slider figure
            try:
                # Create plotly figure with slider from the saved CV data JSON files
                slider_fig = go.Figure()
                
                # Load data for each trial from the saved JSON files
                trial_data = []
                for trial_idx in [m['trial'] for m in all_metrics if m['ax_cv_r2'] is not None]:
                    json_file = cv_dir / f"branin_cv_data_trial_{trial_idx}.json"
                    if json_file.exists():
                        try:
                            import json
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            trial_data.append(data)
                        except Exception as e:
                            print(f"Could not load JSON data from trial {trial_idx}: {e}")
                            continue
                
                if trial_data:
                    # Add diagonal reference line
                    all_x = []
                    all_y = []
                    for trial in trial_data:
                        all_x.extend(trial['x'])
                        all_y.extend(trial['y'])
                    
                    if all_x and all_y:
                        vmin = min(min(all_x), min(all_y))
                        vmax = max(max(all_x), max(all_y))
                        pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
                        lo, hi = vmin - pad, vmax + pad
                        
                        # Add diagonal line (always visible)
                        slider_fig.add_trace(
                            go.Scatter(
                                x=[lo, hi], y=[lo, hi],
                                mode='lines',
                                line=dict(color='black', dash='dash'),
                                name='y = x',
                                showlegend=True
                            )
                        )
                    
                    # Add data for each trial
                    for i, trial in enumerate(trial_data):
                        if trial['x'] and trial['y']:
                            # Add scatter plot with error bars
                            error_y = None
                            if trial['error_y']:
                                error_y = dict(
                                    type='data',
                                    array=trial['error_y'],
                                    visible=True,
                                    color='lightgray'
                                )
                            
                            slider_fig.add_trace(
                                go.Scatter(
                                    x=trial['x'], 
                                    y=trial['y'],
                                    mode='markers',
                                    marker=dict(color='#1f77b4', size=6),
                                    error_y=error_y,
                                    name=f'Trial {trial["trial"]}',
                                    visible=(i == 0)  # Only first trial visible initially
                                )
                            )
                    
                    # Create slider steps
                    steps = []
                    for i, trial in enumerate(trial_data):
                        # Make diagonal line always visible, toggle data traces
                        visibility = [True]  # Diagonal line always visible
                        visibility.extend([False] * len(trial_data))  # All data traces hidden
                        visibility[i + 1] = True  # Show current trial's data
                        
                        step = dict(
                            method="update",
                            args=[{"visible": visibility}],
                            label=f"Trial {trial['trial']}"
                        )
                        steps.append(step)
                    
                    sliders = [dict(
                        active=0,
                        currentvalue={"prefix": "Showing Trial: "},
                        pad={"t": 50},
                        steps=steps
                    )]
                    
                    slider_fig.update_layout(
                        sliders=sliders,
                        title="Cross-Validation Evolution (Interactive Slider)",
                        xaxis_title="Observed",
                        yaxis_title="Predicted (LOO)",
                        showlegend=True
                    )
                    
                    slider_html_path = cv_dir / "branin_cv_evolution_slider.html" 
                    slider_fig.write_html(str(slider_html_path), include_plotlyjs="cdn")
                    print(f"Created Plotly slider figure: {slider_html_path}")
                else:
                    print("No valid trial data found for slider figure")
                    
            except Exception as e:
                print(f"Could not create plotly slider figure: {e}")
                print("Falling back to individual HTML files")
                
    except ImportError:
        print("imageio not available - cannot create GIF")
        print("Falling back to individual PNG files")
    except Exception as e:
        print(f"Error creating GIF/slider plots: {e}")
        print("Falling back to individual PNG/HTML files")

print(f"\nCompleted {len(all_metrics)} iteration(s) with evaluation metrics")

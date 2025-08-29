#!/usr/bin/env python3
"""
Branin optimization campaign with 10 trials and iteration-by-iteration evaluation metrics calculation.
This script demonstrates proper use of ax-platform with GP evaluation metrics calculated after each trial.
"""

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
import matplotlib.pyplot as plt
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


def calculate_ax_cross_validation_metrics(ax_client):
    """Calculate metrics using Ax's built-in cross validation."""
    try:
        # Skip if using Sobol (first 5 trials) - no model available yet
        if ax_client.generation_strategy.current_step_index == 0:
            return None, None
            
        # Get the model bridge from the generation strategy 
        model_bridge = ax_client.generation_strategy.model
        if model_bridge is None:
            return None, None
        
        # Run cross validation using Ax's built-in function with the model bridge
        cv = cross_validate(model=model_bridge, folds=-1)  # Leave-one-out CV
        fig = interact_cross_validation_plotly(cv)
        y_act = fig["data"][1].x
        y_pred = fig["data"][1].y
        
        # Calculate R² using Ax's cross validation predictions
        ax_cv_r2 = r2_score(y_act, y_pred)
        print(f"  Ax CV R²: {ax_cv_r2:.4f}")
        
        # For interval score calculation, we need uncertainty estimates
        # We'll use the residuals to estimate uncertainty
        residuals = np.array(y_act) - np.array(y_pred)
        pred_std = np.std(residuals)  # Simple estimate of prediction uncertainty
        
        # Convert to 95% confidence intervals using the estimated std
        lower = np.array(y_pred) - 1.96 * pred_std
        upper = np.array(y_pred) + 1.96 * pred_std
        
        interval_scores = interval_score(np.array(y_act), lower, upper)
        ax_cv_interval_score = np.mean(interval_scores)
        print(f"  Ax CV Interval Score: {ax_cv_interval_score:.4f}")
        
        return ax_cv_r2, ax_cv_interval_score
        
    except Exception as e:
        print(f"Ax cross validation failed: {e}")
        return None, None


def calculate_iteration_metrics(ax_client, trial_index):
    """Calculate evaluation metrics after each trial using both GP model and Ax cross validation."""
    # Skip if not enough trials
    if trial_index < 3:
        return None
    
    # Get all completed trials data
    df = ax_client.get_trials_data_frame()
    if len(df) < 3:
        return None
    
    try:
        # ===== GP-based metrics (gpcheck) =====
        # Prepare data for GP model
        train_X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
        train_Y = torch.tensor(df[obj1_name].values, dtype=torch.float32).unsqueeze(-1)
        
        # Fit GP model using gpcheck
        config = GPConfig(seed=42)
        gp_model = GPModel(config)
        gp_model.fit(train_X, train_Y)
        
        # Calculate GP-based metrics
        mean_pred, std_pred = gp_model.predict(train_X)
        
        gp_r2 = r2_score(train_Y.detach().cpu().numpy(), mean_pred.detach().cpu().numpy())
        
        ls, imp = gp_model.get_lengthscales()
        imp_dict = importance_concentration(imp)
        
        loo = loo_pseudo_likelihood(gp_model.model, train_X, train_Y)
        
        rank_tau = kendalltau(train_Y.squeeze().detach().cpu().numpy(), mean_pred.detach().cpu().numpy())[0]
        
        # Calculate interval score using GP predictions
        y_true = train_Y.squeeze().detach().cpu().numpy()
        y_pred_mean = mean_pred.detach().cpu().numpy()
        y_pred_std = std_pred.detach().cpu().numpy()
        
        # Convert to 95% confidence intervals
        lower, upper = normal_95ci(y_pred_mean, y_pred_std)
        interval_scores = interval_score(y_true, lower, upper)
        mean_interval_score = np.mean(interval_scores)
        
        # ===== Ax cross validation metrics =====
        ax_cv_r2, ax_cv_interval_score = calculate_ax_cross_validation_metrics(ax_client)
        
        metrics = {
            'trial': trial_index,
            'gp_r2': gp_r2,  # GP-based R²
            'ax_cv_r2': ax_cv_r2,  # Ax cross validation R²
            'interval_score': mean_interval_score,  # Mean interval score (GP-based)
            'ax_cv_interval_score': ax_cv_interval_score,  # Mean interval score (Ax CV)
            'imp_cumsum': imp_dict,
            'loo_nll': loo,
            'rank_tau': rank_tau,
            'lengthscales': ls,
            'importance': imp
        }
        
        # Print metrics for this iteration
        print(f"  GP R²: {gp_r2:.4f}, LOO NLL: {loo:.2f}")
        print(f"  GP Interval Score: {mean_interval_score:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Metrics calculation failed: {e}")
        return None


# Create ax client
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

# Track metrics per iteration
all_metrics = []

# Run 10 trials with iteration-by-iteration metrics calculation
print("Running Branin optimization campaign with 10 trials...")
for i in range(10):
    parameterization, trial_index = ax_client.get_next_trial()

    # extract parameters
    x1 = parameterization["x1"]
    x2 = parameterization["x2"]

    results = branin(x1, x2)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)
    
    print(f"Trial {trial_index}: x1={x1:.3f}, x2={x2:.3f}, branin={results:.3f}")
    
    # Calculate metrics after each trial (from trial 3 onward)
    iteration_metrics = calculate_iteration_metrics(ax_client, trial_index)
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

# Create stacked vertical plots as requested
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), dpi=150)

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
    
    # Extract Ax CV R² and interval scores (may have None values)
    ax_cv_r2_values = [m['ax_cv_r2'] for m in all_metrics if m['ax_cv_r2'] is not None]
    interval_score_values = [m['interval_score'] for m in all_metrics if m['interval_score'] is not None]
    ax_cv_interval_score_values = [m['ax_cv_interval_score'] for m in all_metrics if m['ax_cv_interval_score'] is not None]
    
    ax_cv_trial_nums = [m['trial'] for m in all_metrics if m['ax_cv_r2'] is not None]
    interval_trial_nums = [m['trial'] for m in all_metrics if m['interval_score'] is not None]
    ax_cv_interval_trial_nums = [m['trial'] for m in all_metrics if m['ax_cv_interval_score'] is not None]
    
    # Use multiple y-axes for different scales
    ax2_twin1 = ax2.twinx()  # R² values will be on the right
    ax2_twin2 = ax2.twinx()
    ax2_twin3 = ax2.twinx()
    ax2_twin2.spines['right'].set_position(('outward', 60))
    ax2_twin3.spines['right'].set_position(('outward', 120))
    
    # Plot best-so-far trace on the left (main) y-axis
    best_so_far = np.minimum.accumulate(df[objective_name])
    line_best = ax2.plot(df.index, best_so_far, color='gray', linewidth=6, alpha=0.3, 
                         label='Best so far', linestyle='-', zorder=1)
    
    # Plot R² values on the right y-axis (ax2_twin1)
    line1 = ax2_twin1.plot(trial_nums, gp_r2_values, 'r-', label='GP R²', marker='o', linewidth=2, zorder=3)
    lines = line_best + line1
    
    # Plot Ax CV R² if available (also on right y-axis)
    if ax_cv_r2_values:
        line4 = ax2_twin1.plot(ax_cv_trial_nums, ax_cv_r2_values, 'orange', label='Ax CV R²', marker='d', 
                              linewidth=2, linestyle='--', zorder=3)
        lines += line4
    
    # Set appropriate y-axis limits for R² values
    r2_values_all = gp_r2_values + (ax_cv_r2_values if ax_cv_r2_values else [])
    if r2_values_all:
        r2_min, r2_max = min(r2_values_all), max(r2_values_all)
        r2_range = r2_max - r2_min
        ax2_twin1.set_ylim(r2_min - 0.1 * r2_range, r2_max + 0.1 * r2_range)
    
    # Plot other metrics on additional y-axes
    line2 = ax2_twin2.plot(trial_nums, rank_tau_values, 'g-', label='Rank τ', marker='s', linewidth=2, zorder=3)
    line3 = ax2_twin3.plot(trial_nums, loo_values, 'b-', label='LOO NLL', marker='^', linewidth=2, zorder=3)
    lines += line2 + line3
    
    # Plot GP-based interval score if available  
    if interval_score_values:
        line5 = ax2_twin3.plot(interval_trial_nums, interval_score_values, 'purple', label='GP Interval Score', 
                              marker='v', linewidth=2, linestyle=':', zorder=3)
        lines += line5
    
    # Plot Ax CV interval score if available
    if ax_cv_interval_score_values:
        line6 = ax2_twin3.plot(ax_cv_interval_trial_nums, ax_cv_interval_score_values, 'magenta', 
                              label='Ax CV Interval Score', marker='*', linewidth=2, linestyle='-.', zorder=3)
        lines += line6
        
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel(f"Best {objective_name}", color='gray')
    ax2_twin1.set_ylabel("R² Values", color='r')
    ax2_twin2.set_ylabel("Rank Correlation (τ)", color='g')
    ax2_twin3.set_ylabel("LOO NLL / Interval Score", color='b')
    
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2_twin1.tick_params(axis='y', labelcolor='r')
    ax2_twin2.tick_params(axis='y', labelcolor='g')
    ax2_twin3.tick_params(axis='y', labelcolor='b')
    
    ax2.set_title("Evaluation Metrics vs Trial (Iteration-by-Iteration)")
    
    # Combined legend
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    print(f"\nLOO Negative Log-Likelihood values: {loo_values}")
    print(f"Rank tau correlation values: {rank_tau_values}")
    if ax_cv_r2_values:
        print(f"Ax Cross Validation R² values: {ax_cv_r2_values}")
    if interval_score_values:
        print(f"GP Interval Score values: {interval_score_values}")
    if ax_cv_interval_score_values:
        print(f"Ax CV Interval Score values: {ax_cv_interval_score_values}")
    
else:
    ax2.text(0.5, 0.5, 'Not enough data\nfor metrics calculation', 
            ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Evaluation Metrics")

plt.tight_layout()
plt.savefig('/home/runner/work/evaluation-metrics/evaluation-metrics/scripts/bo_benchmarks/demonstrations/branin_campaign_demonstration_results.png', 
            dpi=150, bbox_inches='tight')
plt.show()

print(f"\nCompleted {len(all_metrics)} iteration(s) with evaluation metrics")
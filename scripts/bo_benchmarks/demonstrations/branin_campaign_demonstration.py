#!/usr/bin/env python3
"""
Branin optimization campaign with 10 trials and iteration-by-iteration evaluation metrics calculation.
This script demonstrates proper use of ax-platform with GP evaluation metrics calculated after each trial.
"""

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.adapter.cross_validation import cross_validate
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


def calculate_iteration_metrics(ax_client, trial_index):
    """Calculate evaluation metrics after each trial using GP model."""
    # Skip if not enough trials
    if trial_index < 3:
        return None
    
    # Get all completed trials data
    df = ax_client.get_trials_data_frame()
    if len(df) < 3:
        return None
    
    # Calculate GP model metrics using gpcheck
    try:
        # Prepare data for GP model
        train_X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
        train_Y = torch.tensor(df[obj1_name].values, dtype=torch.float32).unsqueeze(-1)
        
        # Fit GP model using gpcheck
        config = GPConfig(seed=42)
        gp_model = GPModel(config)
        gp_model.fit(train_X, train_Y)
        
        # Calculate metrics
        mean_pred, std_pred = gp_model.predict(train_X)
        
        r2 = r2_score(train_Y.detach().cpu().numpy(), mean_pred.detach().cpu().numpy())
        
        ls, imp = gp_model.get_lengthscales()
        imp_dict = importance_concentration(imp)
        
        loo = loo_pseudo_likelihood(gp_model.model, train_X, train_Y)
        
        rank_tau = kendalltau(train_Y.squeeze().detach().cpu().numpy(), mean_pred.detach().cpu().numpy())[0]
        
        metrics = {
            'trial': trial_index,
            'r2': r2,
            'imp_cumsum': imp_dict,
            'loo_nll': loo,
            'rank_tau': rank_tau,
            'lengthscales': ls,
            'importance': imp
        }
        
        return metrics
        
    except Exception as e:
        print(f"GP metrics calculation failed: {e}")
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
        print(f"  R²: {iteration_metrics['r2']:.4f}, LOO NLL: {iteration_metrics['loo_nll']:.2f}")

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
    print(f"R²: {final_metrics['r2']:.4f}")
    print(f"Rank correlation (τ): {final_metrics['rank_tau']:.4f}")
    print(f"Leave-One-Out Negative Log-Likelihood: {final_metrics['loo_nll']:.2f}")
    print(f"Importance concentration: {final_metrics['imp_cumsum']}")

# Create stacked vertical plots as requested
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), dpi=150)

# Plot 1: Standard "best so far" trace
ax1.scatter(df.index, df[objectives], ec="k", fc="none", label="Observed")
ax1.plot(
    df.index,
    np.minimum.accumulate(df[objectives]),
    color="#0033FF",
    lw=2,
    label="Best to Trial",
)
ax1.set_xlabel("Trial Number")
ax1.set_ylabel(f"{obj1_name} objective")
ax1.set_title("Optimization Progress")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Evaluation metrics over trials (iteration-by-iteration results)
if all_metrics:
    trial_nums = [m['trial'] for m in all_metrics]
    r2_values = [m['r2'] for m in all_metrics]
    rank_tau_values = [m['rank_tau'] for m in all_metrics] 
    loo_values = [m['loo_nll'] for m in all_metrics]
    
    # Use secondary y-axes for different scales
    ax2_twin1 = ax2.twinx()
    ax2_twin2 = ax2.twinx()
    ax2_twin2.spines['right'].set_position(('outward', 60))
    
    line1 = ax2.plot(trial_nums, r2_values, 'r-', label='R²', marker='o', linewidth=2)
    line2 = ax2_twin1.plot(trial_nums, rank_tau_values, 'g-', label='Rank τ', marker='s', linewidth=2)
    line3 = ax2_twin2.plot(trial_nums, loo_values, 'b-', label='LOO NLL', marker='^', linewidth=2)
    
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("R²", color='r')
    ax2_twin1.set_ylabel("Rank Correlation (τ)", color='g')
    ax2_twin2.set_ylabel("LOO Negative Log-Likelihood", color='b')
    
    ax2.tick_params(axis='y', labelcolor='r')
    ax2_twin1.tick_params(axis='y', labelcolor='g')
    ax2_twin2.tick_params(axis='y', labelcolor='b')
    
    ax2.set_title("Evaluation Metrics vs Trial (Iteration-by-Iteration)")
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    print(f"\nLOO Negative Log-Likelihood values: {loo_values}")
    
else:
    ax2.text(0.5, 0.5, 'Not enough data\nfor metrics calculation', 
            ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Evaluation Metrics")

plt.tight_layout()
plt.savefig('/home/runner/work/evaluation-metrics/evaluation-metrics/scripts/bo_benchmarks/demonstrations/branin_campaign_demonstration_results.png', 
            dpi=150, bbox_inches='tight')
plt.show()

print(f"\nCompleted {len(all_metrics)} iteration(s) with evaluation metrics")
#!/usr/bin/env python3
"""
Branin optimization campaign with 10 trials and evaluation metrics calculation.
Requested by @sgbaird to demonstrate evaluation metrics from iteration 3 onward.
"""

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt
import torch

# Import evaluation metrics
from evaluation_metrics import GPModel, GPConfig


obj1_name = "branin"


def branin(x1, x2):
    y = float(
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    return y


def calculate_evaluation_metrics(df, iteration_start=3):
    """Calculate evaluation metrics from iteration_start onward."""
    if len(df) < iteration_start + 1:
        return None
    
    # Get data from iteration_start onward
    df_subset = df.iloc[iteration_start:].copy()
    
    # Prepare training data
    train_X = torch.tensor(df_subset[['x1', 'x2']].values, dtype=torch.float32)
    train_Y = torch.tensor(df_subset[obj1_name].values, dtype=torch.float32).unsqueeze(-1)
    
    # Fit GP model
    config = GPConfig(seed=42)
    gp_model = GPModel(config)
    gp_model.fit(train_X, train_Y)
    
    # Calculate metrics using test data (just use same data for demo)
    mean_pred, std_pred = gp_model.predict(train_X)
    
    from sklearn.metrics import r2_score
    from scipy.stats import kendalltau
    from evaluation_metrics import loo_pseudo_likelihood, importance_concentration
    
    r2 = r2_score(train_Y.detach().cpu().numpy(), mean_pred.detach().cpu().numpy())
    
    ls, imp = gp_model.get_lengthscales()
    imp_dict = importance_concentration(imp)
    
    loo = loo_pseudo_likelihood(gp_model.model, train_X, train_Y)
    
    rank_tau = kendalltau(train_Y.squeeze().detach().cpu().numpy(), mean_pred.detach().cpu().numpy())[0]
    
    metrics = {
        'r2': r2,
        'imp_cumsum': imp_dict,
        'loo_nll': loo,
        'rank_tau': rank_tau,
        'lengthscales': ls,
        'importance': imp
    }
    
    return metrics


def main():
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

    # Run 10 trials (changed from 19 to 10 as requested)
    print("Running Branin optimization campaign with 10 trials...")
    for i in range(10):
        parameterization, trial_index = ax_client.get_next_trial()

        # extract parameters
        x1 = parameterization["x1"]
        x2 = parameterization["x2"]

        results = branin(x1, x2)
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
        
        print(f"Trial {trial_index}: x1={x1:.3f}, x2={x2:.3f}, branin={results:.3f}")

    best_parameters, metrics = ax_client.get_best_parameters()
    print(f"\nBest parameters: {best_parameters}")
    print(f"Best objective: {metrics}")

    # Get results data
    objectives = ax_client.objective_names
    df = ax_client.get_trials_data_frame()
    
    print(f"\nDataframe shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())

    # Calculate evaluation metrics from iteration 3 onward
    print("\nCalculating evaluation metrics from iteration 3 onward...")
    eval_metrics = calculate_evaluation_metrics(df, iteration_start=3)
    
    if eval_metrics:
        print("\n=== Evaluation Metrics (from iteration 3 onward) ===")
        print(f"R²: {eval_metrics['r2']:.4f}")
        print(f"Rank correlation (τ): {eval_metrics['rank_tau']:.4f}")
        print(f"Leave-One-Out Negative Log-Likelihood: {eval_metrics['loo_nll']:.4f}")
        print(f"Importance concentration: {eval_metrics['imp_cumsum']}")
        print(f"Lengthscales: {eval_metrics['lengthscales']}")
        print(f"Feature importance: {eval_metrics['importance']}")
    else:
        print("Not enough data for evaluation metrics calculation")

    # Plot results as requested
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=150)
    
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
    
    # Plot 2: Evaluation metrics over trials (from iteration 3)
    if eval_metrics:
        # Calculate metrics for each subset from iteration 3 onward
        r2_values = []
        rank_tau_values = []
        loo_values = []
        trial_nums = []
        
        for i in range(3, len(df)):
            subset_metrics = calculate_evaluation_metrics(df, iteration_start=i)
            if subset_metrics:
                r2_values.append(subset_metrics['r2'])
                rank_tau_values.append(subset_metrics['rank_tau'])
                loo_values.append(subset_metrics['loo_nll'])
                trial_nums.append(i)
        
        # Plot metrics
        ax2_twin1 = ax2.twinx()
        ax2_twin2 = ax2.twinx()
        ax2_twin2.spines['right'].set_position(('outward', 60))
        
        line1 = ax2.plot(trial_nums, r2_values, 'r-', label='R²', marker='o')
        line2 = ax2_twin1.plot(trial_nums, rank_tau_values, 'g-', label='Rank τ', marker='s')
        line3 = ax2_twin2.plot(trial_nums, loo_values, 'b-', label='LOO NLL', marker='^')
        
        ax2.set_xlabel("Starting Trial Number")
        ax2.set_ylabel("R²", color='r')
        ax2_twin1.set_ylabel("Rank Correlation (τ)", color='g')
        ax2_twin2.set_ylabel("LOO Negative Log-Likelihood", color='b')
        
        ax2.tick_params(axis='y', labelcolor='r')
        ax2_twin1.tick_params(axis='y', labelcolor='g')
        ax2_twin2.tick_params(axis='y', labelcolor='b')
        
        ax2.set_title("Evaluation Metrics vs Starting Trial")
        
        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
    else:
        ax2.text(0.5, 0.5, 'Not enough data\nfor metrics calculation', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Evaluation Metrics")
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/evaluation-metrics/evaluation-metrics/branin_campaign_results.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return df, eval_metrics


if __name__ == "__main__":
    df, metrics = main()
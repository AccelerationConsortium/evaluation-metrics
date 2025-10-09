#!/usr/bin/env python3
"""
Run 10 repeat benchmark campaigns to determine optimal number of initialization points for Branin function.
This script creates individual campaign plots and a composite analysis plot.
"""

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import torch
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import logging
import datetime
import imageio.v2 as imageio

# Add the bo_benchmarks directory to Python path
bo_benchmarks_dir = Path(__file__).parent / "bo_benchmarks"
sys.path.insert(0, str(bo_benchmarks_dir))

try:
    from gpcheck.models import GPModel, GPConfig
    from gpcheck.metrics import loo_pseudo_likelihood, importance_concentration
    from sklearn.metrics import r2_score
    from scipy.stats import kendalltau
    HAS_GPCHECK = True
except ImportError:
    HAS_GPCHECK = False
    print("Warning: gpcheck not available, will use simplified metrics")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

obj1_name = "branin"


def setup_logging(run_timestamp):
    """Setup comprehensive logging to file and console."""
    # Create logs directory with timestamp (separate from results)
    logs_dir = Path(__file__).parent / "branin_evaluation_logs" / f"run_{run_timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logger
    log_file = logs_dir / "branin_evaluation.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('branin_evaluation')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, logs_dir


def branin(x1, x2):
    """Branin optimization function."""
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
    """
    if not HAS_GPCHECK:
        return None, None
        
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

    try:
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
        ax_cv_r2 = r2_score(y_act, y_pred)

        # Compute interval score using uncertainty bars from the CV plot (LOO predictions)
        # Pull error bars (95% CI half-width) from the CV plot trace and compute interval score
        err = trace.error_y  # type: ignore[attr-defined]
        if getattr(err, "array", None) is not None:
            err_array = np.asarray(err.array, dtype=float)
            lower = y_pred - err_array
            upper = y_pred + err_array
            interval_scores = interval_score(y_act, lower, upper)
            ax_cv_interval_score = float(np.mean(interval_scores))
        else:
            ax_cv_interval_score = None

        return ax_cv_r2, ax_cv_interval_score
    except Exception as e:
        print(f"  Warning: Ax CV calculation failed: {e}")
        return None, None


def calculate_iteration_metrics(ax_client, ax_client_cv, trial_index):
    """Calculate evaluation metrics after each trial using both GP model and Ax cross validation."""
    if not HAS_GPCHECK:
        return None
        
    # Skip if not enough trials
    if trial_index < 3:
        return None
    
    # Get all completed trials data
    df = ax_client.get_trials_data_frame()
    if len(df) < 3:
        return None
    
    try:
        # ===== GP-based metrics (gpcheck) =====
        # Prepare data for GP model with train-test split
        all_X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
        all_Y = torch.tensor(df[obj1_name].values, dtype=torch.float32).unsqueeze(-1)
        
        # Use 80-20 train-test split for realistic R² evaluation
        n_total = len(all_X)
        n_train = max(2, int(0.8 * n_total))  # Ensure at least 2 training points
        
        # Use fixed random indices for reproducible splits
        np.random.seed(42 + trial_index)  # Different seed per trial for variation
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_X = all_X[train_indices]
        train_Y = all_Y[train_indices]
        test_X = all_X[test_indices] if len(test_indices) > 0 else train_X  # Fallback to train if no test data
        test_Y = all_Y[test_indices] if len(test_indices) > 0 else train_Y

        # Fit GP model using gpcheck
        config = GPConfig(seed=42 + trial_index)
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
            'gp_interval_score': mean_interval_score,
            'ax_cv_interval_score': ax_cv_interval_score,
            'imp_mean': imp_mean,
            'imp_std': imp_std,
            'loo_nll': loo,
            'rank_tau': rank_tau,
        }

        return metrics
        
    except Exception as e:
        print(f"  Warning: Metrics calculation failed for trial {trial_index}: {e}")
        return None



def run_single_campaign(campaign_id, num_trials=30, num_init_trials=5):
    """Run a single optimization campaign with metrics collection and return results."""
    logger = logging.getLogger('branin_evaluation')
    logger.info(f"Starting campaign {campaign_id} (init={num_init_trials}, trials={num_trials})...")
    
    # Create main ax client for optimization with proper generation strategy
    # Use Sobol for initialization, then GPEI
    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL, 
                num_trials=num_init_trials,
                min_trials_observed=num_init_trials,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,  # Continue indefinitely with GPEI
            ),
        ]
    )
    
    ax_client = AxClient(generation_strategy=gs, verbose_logging=False)
    ax_client.create_experiment(
        parameters=[
            {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
        ],
        objectives={obj1_name: ObjectiveProperties(minimize=True)},
    )
    
    # Create separate AxClient for cross-validation with custom generation strategy
    if HAS_GPCHECK:
        gs_cv = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # Use BOTORCH_MODULAR for all trials
                )
            ]
        )
        ax_client_cv = AxClient(generation_strategy=gs_cv, verbose_logging=False)
        ax_client_cv.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            objectives={obj1_name: ObjectiveProperties(minimize=True)},
        )
    else:
        ax_client_cv = None
    
    # Storage for results
    campaign_results = {
        'campaign_id': campaign_id,
        'num_init_trials': int(num_init_trials),  # Add initialization count for analysis
        'trials': [],
        'best_values': [],
        'objective_values': [],
        'parameters': [],
        'metrics': {
            'gp_r2': [],
            'ax_cv_r2': [],
            'rank_tau': [],
            'loo_nll': [],
            'gp_interval_score': [],
            'ax_cv_interval_score': [],
            'importance_std': []
        }
    }
    
    # Run optimization trials
    for trial_idx in range(num_trials):
        # Get next trial parameters
        parameters, trial_index = ax_client.get_next_trial()
        
        # Evaluate Branin function
        objective_value = branin(parameters["x1"], parameters["x2"])
        
        # Complete trial
        ax_client.complete_trial(trial_index=trial_index, raw_data=objective_value)
        
        # Store trial data
        campaign_results['trials'].append(trial_index)
        campaign_results['parameters'].append(parameters)
        campaign_results['objective_values'].append(objective_value)
        
        # Calculate best value so far
        try:
            best_trial = ax_client.get_best_trial()
            if best_trial is not None and len(best_trial) > 1:
                best_value = best_trial[1][obj1_name]
            else:
                # Fallback: calculate best from data
                data = ax_client.experiment.fetch_data().df
                if not data.empty:
                    best_value = data['mean'].min()
                else:
                    best_value = objective_value
        except:
            # Fallback: calculate best from stored values
            if campaign_results['best_values']:
                best_value = min(min(campaign_results['best_values']), objective_value)
            else:
                best_value = objective_value
        
        campaign_results['best_values'].append(best_value)
        
        # Calculate metrics if available
        if HAS_GPCHECK and trial_index >= 2:  # Need at least 3 trials for metrics
            metrics = calculate_iteration_metrics(ax_client, ax_client_cv, trial_index)
            if metrics:
                campaign_results['metrics']['gp_r2'].append(metrics.get('gp_r2', np.nan))
                campaign_results['metrics']['ax_cv_r2'].append(metrics.get('ax_cv_r2', np.nan))
                campaign_results['metrics']['rank_tau'].append(metrics.get('rank_tau', np.nan))
                campaign_results['metrics']['loo_nll'].append(metrics.get('loo_nll', np.nan))
                campaign_results['metrics']['gp_interval_score'].append(metrics.get('gp_interval_score', np.nan))
                campaign_results['metrics']['ax_cv_interval_score'].append(metrics.get('ax_cv_interval_score', np.nan))
                campaign_results['metrics']['importance_std'].append(metrics.get('imp_std', np.nan))
            else:
                # Add NaN values if metrics calculation failed
                for metric_name in campaign_results['metrics'].keys():
                    campaign_results['metrics'][metric_name].append(np.nan)
        else:
            # Add NaN values for early trials
            for metric_name in campaign_results['metrics'].keys():
                campaign_results['metrics'][metric_name].append(np.nan)
        
        if trial_idx % 10 == 0:
            logger.info(f"  Campaign {campaign_id}: Trial {trial_idx}, Current: {objective_value:.6f}, Best: {best_value:.6f}")
    
    logger.info(f"Campaign {campaign_id} completed. Final best: {campaign_results['best_values'][-1]:.6f}")
    return campaign_results


def create_individual_campaign_plot(campaign_results, output_dir):
    """Create individual campaign plot similar to branin_campaign_demonstration_results.png."""
    campaign_id = campaign_results['campaign_id']
    trials = campaign_results['trials']
    best_values = campaign_results['best_values']
    objective_values = campaign_results['objective_values']
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Optimization progress showing both raw and best-so-far
    # Raw objective values
    ax1.scatter(trials, objective_values, alpha=0.6, s=30, color='lightblue', label=f'Raw {obj1_name}', zorder=1)
    # Best-so-far trace
    ax1.plot(trials, best_values, 'r-', linewidth=2, label=f'Best {obj1_name}', marker='o', markersize=4, zorder=2)
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel(f"{obj1_name}")
    ax1.set_title(f"Campaign {campaign_id}: Optimization Progress")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom subplot: Convergence metrics (simplified)
    # Show optimization efficiency metrics
    if len(best_values) > 1:
        # Calculate improvement per trial
        improvements = []
        for i in range(1, len(best_values)):
            improvement = best_values[i-1] - best_values[i]  # Positive means improvement
            improvements.append(improvement)
        
        improvement_trials = trials[1:]
        
        ax2.bar(improvement_trials, improvements, alpha=0.7, color='green', 
                label='Improvement per Trial', width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel("Improvement")
        ax2.set_title(f"Campaign {campaign_id}: Optimization Efficiency")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add cumulative improvement on second y-axis
        ax2_cum = ax2.twinx()
        cumulative_improvement = np.cumsum(improvements)
        ax2_cum.plot(improvement_trials, cumulative_improvement, 'orange', 
                     label='Cumulative Improvement', marker='s', linewidth=2)
        ax2_cum.set_ylabel("Cumulative Improvement", color='orange')
        ax2_cum.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends and place them to avoid overlap
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_cum.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
    else:
        ax2.text(0.5, 0.5, 'Not enough data\nfor efficiency analysis', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f"Campaign {campaign_id}: Optimization Efficiency")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'branin_campaign_{campaign_id}_results.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def create_plotly_campaign_plot(campaign_results, campaign_dir):
    """Create interactive Plotly version of the campaign plot."""
    campaign_id = campaign_results['campaign_id']
    trials = campaign_results['trials']
    best_values = campaign_results['best_values']
    objective_values = campaign_results['objective_values']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Campaign {campaign_id}: Optimization Progress', 
                       f'Campaign {campaign_id}: Optimization Efficiency'),
        vertical_spacing=0.1
    )
    
    # Top subplot: Optimization progress
    # Raw objective values
    fig.add_trace(
        go.Scatter(
            x=trials, y=objective_values,
            mode='markers',
            name=f'Raw {obj1_name}',
            marker=dict(color='lightblue', size=8, opacity=0.6),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Best-so-far trace
    fig.add_trace(
        go.Scatter(
            x=trials, y=best_values,
            mode='lines+markers',
            name=f'Best {obj1_name}',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Bottom subplot: Efficiency metrics
    if len(best_values) > 1:
        improvements = []
        for i in range(1, len(best_values)):
            improvement = best_values[i-1] - best_values[i]
            improvements.append(improvement)
        
        improvement_trials = trials[1:]
        cumulative_improvement = np.cumsum(improvements)
        
        # Improvement bars
        fig.add_trace(
            go.Bar(
                x=improvement_trials, y=improvements,
                name='Improvement per Trial',
                marker=dict(color='green', opacity=0.7),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Cumulative improvement line
        fig.add_trace(
            go.Scatter(
                x=improvement_trials, y=cumulative_improvement,
                mode='lines+markers',
                name='Cumulative Improvement',
                line=dict(color='orange', width=2),
                marker=dict(size=6),
                yaxis='y2',
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'Campaign {campaign_id}: Branin Optimization Results',
        height=800,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Trial Number", row=1, col=1)
    fig.update_yaxes(title_text=f"{obj1_name}", row=1, col=1)
    fig.update_xaxes(title_text="Trial Number", row=2, col=1)
    fig.update_yaxes(title_text="Improvement", row=2, col=1)
    
    # Save interactive HTML
    html_path = campaign_dir / f'branin_campaign_{campaign_id}_results.html'
    fig.write_html(str(html_path))
    
    return html_path


def create_campaign_gif(campaign_results, campaign_dir):
    """Create GIF showing optimization progress over time."""
    import math
    
    campaign_id = campaign_results['campaign_id']
    trials = campaign_results['trials']
    best_values = campaign_results['best_values']
    objective_values = campaign_results['objective_values']
    parameters = campaign_results['parameters']
    
    frames = []
    temp_dir = campaign_dir / 'temp_frames'
    temp_dir.mkdir(exist_ok=True)
    
    # Pre-calculate consistent axis limits for the progress plot
    all_obj_vals = objective_values
    y_min = min(all_obj_vals)
    y_max = max(all_obj_vals)
    y_margin = (y_max - y_min) * 0.1  # 10% margin
    y_min_plot = y_min - y_margin
    y_max_plot = y_max + y_margin
    
    # Pre-calculate consistent colorbar limits
    obj_min = min(all_obj_vals)
    obj_max = max(all_obj_vals)
    
    # Pre-calculate consistent tick marks for better scale consistency
    from matplotlib.ticker import MaxNLocator
    
    # Create frames for each trial
    for frame_idx in range(1, len(trials), 2):  # Start from trial 1, every 2nd frame
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left subplot: Parameter space visualization
        x1_vals = [p['x1'] for p in parameters[:frame_idx+1]]
        x2_vals = [p['x2'] for p in parameters[:frame_idx+1]]
        obj_vals = objective_values[:frame_idx+1]
        
        # Create a scatter plot colored by objective value with consistent colorbar
        scatter = ax1.scatter(x1_vals, x2_vals, c=obj_vals, cmap='viridis_r', 
                             s=50, alpha=0.8, edgecolors='black', linewidth=0.5,
                             vmin=obj_min, vmax=obj_max)  # Fixed colorbar range
        
        # Highlight the best point
        best_idx = np.argmin(obj_vals)
        ax1.scatter(x1_vals[best_idx], x2_vals[best_idx], c='red', s=150, 
                   marker='*', edgecolors='black', linewidth=1, label='Best Point')
        
        ax1.set_xlim(-5, 10)
        ax1.set_ylim(0, 10)
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title(f'Campaign {campaign_id}: Parameter Space (Trial {frame_idx})')
        ax1.legend(loc='upper left')  # Fixed legend position
        ax1.grid(True, alpha=0.3)
        
        # Set consistent tick marks for x1 and x2 axes
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Add colorbar with consistent range
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label(f'{obj1_name} Value')
        
        # Right subplot: Optimization progress with fixed y-axis
        ax2.plot(trials[:frame_idx+1], best_values[:frame_idx+1], 'r-', 
                linewidth=2, marker='o', markersize=4, label=f'Best {obj1_name}')
        ax2.scatter(trials[:frame_idx+1], objective_values[:frame_idx+1], 
                   alpha=0.6, s=30, color='lightblue', label=f'Raw {obj1_name}')
        
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel(f'{obj1_name}')
        ax2.set_title(f'Campaign {campaign_id}: Progress (Trial {frame_idx})')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')  # Fixed legend position
        ax2.set_xlim(0, len(trials))
        ax2.set_ylim(y_min_plot, y_max_plot)  # Fixed y-axis limits
        
        # Set consistent tick marks for progress plot
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        plt.tight_layout()
        
        # Save frame
        frame_path = temp_dir / f'frame_{frame_idx:03d}.png'
        plt.savefig(str(frame_path), dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(str(frame_path))
    
    # Create GIF using PIL with explicit per-frame delays
    gif_path = campaign_dir / f'branin_campaign_{campaign_id}_evolution.gif'
    
    if frames:
        from PIL import Image
        
        # Load frames as PIL Images (use palette mode for GIF)
        pil_frames = [Image.open(fp).convert("P", palette=Image.ADAPTIVE) for fp in frames]

        # 2000 ms per frame, hold last for 1000 ms  
        durations_ms = [2000] * len(pil_frames)
        if durations_ms:
            durations_ms[-1] = 1000

        # Save with explicit per-frame delays; disposal=2 avoids artifacts; optimize=False preserves timing
        pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations_ms,   # list of per-frame durations in ms
            loop=0,                  # infinite loop
            disposal=2,              # restore to background before next frame
            optimize=False           # don't re-optimize away delays
        )

        # Clean up the temporary PNGs
        for fp in frames:
            os.remove(fp)
        temp_dir.rmdir()
    
    return gif_path


def create_composite_plot(all_campaigns, output_dir):
    """Create composite plot with modified requirements."""
    
    # Extract data from all campaigns
    max_trials = max(len(campaign['best_values']) for campaign in all_campaigns)
    trial_range = range(max_trials)
    
    # Collect best values for each trial across campaigns
    best_values_matrix = []
    for campaign in all_campaigns:
        best_vals = campaign['best_values']
        # Pad shorter campaigns with their final value
        if len(best_vals) < max_trials:
            best_vals = best_vals + [best_vals[-1]] * (max_trials - len(best_vals))
        best_values_matrix.append(best_vals)
    
    best_values_matrix = np.array(best_values_matrix)
    
    # Calculate mean and std for best-so-far trace
    mean_best = np.mean(best_values_matrix, axis=0)
    std_best = np.std(best_values_matrix, axis=0)
    
    # Extract and average evaluation metrics across campaigns
    all_gp_r2 = []
    all_ax_cv_r2 = []
    all_rank_tau = []
    all_loo_nll = []
    all_gp_interval = []
    all_ax_cv_interval = []
    all_importance_std = []
    
    for campaign in all_campaigns:
        metrics = campaign.get('metrics', {})
        
        # Pad metrics to max_trials length
        for metric_name in ['gp_r2', 'ax_cv_r2', 'rank_tau', 'loo_nll', 'gp_interval_score', 'ax_cv_interval_score', 'importance_std']:
            if metric_name in metrics:
                metric_vals = metrics[metric_name]
                if len(metric_vals) < max_trials:
                    # Pad with NaN for missing values
                    metric_vals = metric_vals + [np.nan] * (max_trials - len(metric_vals))
                
                if metric_name == 'gp_r2':
                    all_gp_r2.append(metric_vals)
                elif metric_name == 'ax_cv_r2':
                    all_ax_cv_r2.append(metric_vals)
                elif metric_name == 'rank_tau':
                    all_rank_tau.append(metric_vals)
                elif metric_name == 'loo_nll':
                    all_loo_nll.append(metric_vals)
                elif metric_name == 'gp_interval_score':
                    all_gp_interval.append(metric_vals)
                elif metric_name == 'ax_cv_interval_score':
                    all_ax_cv_interval.append(metric_vals)
                elif metric_name == 'importance_std':
                    all_importance_std.append(metric_vals)
    
    # Create composite plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top subplot: Optimization progress (mean + uncertainty bands only)
    ax1.plot(trial_range, mean_best, 'r-', linewidth=3, label=f'Mean Best {obj1_name}')
    ax1.fill_between(trial_range, 
                     mean_best - std_best, 
                     mean_best + std_best, 
                     alpha=0.3, color='red', label='±1 Std Dev')
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel(f"Best {obj1_name}")
    ax1.set_title("Branin Optimization: Average Progress Across 10 Repeat Campaigns")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom subplot: Average evaluation metrics (no uncertainty bands)
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("Metric Values")
    ax2.set_title("Average Evaluation Metrics Across 10 Repeat Campaigns")
    ax2.grid(True, alpha=0.3)
    
    # Plot metrics with different colors and markers - single axis approach for better visibility
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'magenta', 'brown']
    markers = ['o', 'd', 's', '^', 'v', 'h', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    plotted_lines = []
    
    # Plot averaged metrics
    metric_names = ['gp_r2', 'ax_cv_r2', 'rank_tau', 'loo_nll', 'gp_interval_score', 'ax_cv_interval_score', 'importance_std']
    metric_data = [all_gp_r2, all_ax_cv_r2, all_rank_tau, all_loo_nll, all_gp_interval, all_ax_cv_interval, all_importance_std]
    
    for i, (metric_name, data) in enumerate(zip(metric_names, metric_data)):
        if data and len(data) > 0:  # Check if data exists and is not empty
            mean_metric = np.nanmean(data, axis=0)
            valid_indices = ~np.isnan(mean_metric)
            if np.any(valid_indices):
                # Scale metrics to [0, 1] for better visualization on same axis
                valid_values = mean_metric[valid_indices]
                if len(valid_values) > 1:
                    metric_min = np.min(valid_values)
                    metric_max = np.max(valid_values)
                    if metric_max > metric_min:
                        scaled_values = (valid_values - metric_min) / (metric_max - metric_min)
                    else:
                        scaled_values = np.ones_like(valid_values) * 0.5
                else:
                    scaled_values = np.array([0.5])
                
                trial_subset = np.array(trial_range)[valid_indices]
                line = ax2.plot(trial_subset, scaled_values, 
                               color=colors[i], marker=markers[i], linestyle=linestyles[i],
                               label=f'{metric_name.replace("_", " ").title()}', 
                               linewidth=2, markersize=4, alpha=0.8)
                plotted_lines.extend(line)
    
    # Add legend
    if plotted_lines:
        ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Normalized Metric Values")
    
    plt.tight_layout()
    
    # Save composite plot
    composite_path = output_dir / 'branin_repeat_campaigns_composite_results.png'
    plt.savefig(str(composite_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    return composite_path


# These functions have been replaced with lightweight sanity_check_plots


def main():
    """Main execution function."""
    print("=== Exhaustive Branin Evaluation: Init Counts 2-30 ===")
    print("This will run campaigns with different initialization counts and evaluate at all budget levels")
    print()
    
    # Create timestamped run directory (preserve all previous runs)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent / "branin_exhaustive_evaluation_results"
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging (separate from results)
    logger, logs_dir = setup_logging(timestamp)
    logger.info("=== Starting Exhaustive Branin Evaluation ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    
    # Parameters for exhaustive evaluation
    init_counts = range(2, 31)  # 2 to 30 initialization points
    num_repeats = 10  # Number of repeat campaigns per init count
    max_trials = 30  # Run all campaigns to 30 trials
    
    # Storage for all results
    all_results = {}  # init_count -> list of campaign results
    
    # Run campaigns for each initialization count
    for init_count in init_counts:
        logger.info(f"\n=== Testing {init_count} initialization points ===")
        
        # Create subdirectory for this init count
        init_dir = output_dir / f"init_{init_count}"
        init_dir.mkdir(exist_ok=True)
        
        init_campaigns = []
        
        for repeat_id in range(1, num_repeats + 1):
            try:
                campaign_id = f"{init_count}_{repeat_id}"
                campaign_results = run_single_campaign(
                    campaign_id, 
                    num_trials=max_trials, 
                    num_init_trials=init_count
                )
                init_campaigns.append(campaign_results)
                
                # Save individual campaign immediately
                campaign_file = init_dir / f"campaign_{campaign_id}.json"
                with open(campaign_file, 'w') as f:
                    # Convert numpy types for JSON serialization
                    json_safe_results = {}
                    for key, value in campaign_results.items():
                        if isinstance(value, dict):
                            json_safe_results[key] = {
                                k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in val]
                                if isinstance(val, list) else val
                                for k, val in value.items()
                            }
                        elif isinstance(value, list):
                            json_safe_results[key] = [
                                float(v) if isinstance(v, (np.floating, np.integer)) else v
                                for v in value
                            ]
                        else:
                            json_safe_results[key] = value
                    json.dump(json_safe_results, f, indent=2)
                
                logger.info(f"Campaign {campaign_id}: Final best = {campaign_results['best_values'][-1]:.6f}")
                logger.info(f"Saved campaign data: {campaign_file}")
                
            except Exception as e:
                logger.error(f"Error in campaign {campaign_id}: {e}")
                continue
        
        # Store results for this init count
        all_results[init_count] = init_campaigns
        logger.info(f"Completed {len(init_campaigns)}/{num_repeats} campaigns for init_count={init_count}")
        
        # Save intermediate results after each init_count completion
        intermediate_path = output_dir / f"intermediate_results_init_{init_count}.json"
        with open(intermediate_path, 'w') as f:
            json.dump({
                'completed_init_counts': list(all_results.keys()),
                'current_init_count': init_count,
                'campaigns_completed': len(init_campaigns),
                'parameters': {
                    'init_counts': list(init_counts),
                    'num_repeats': num_repeats,
                    'max_trials': max_trials
                }
            }, f, indent=2)
        logger.info(f"Saved intermediate progress: {intermediate_path}")
    
    # Generate truncated budget analysis
    logger.info("\n=== Analyzing results across all budgets ===")
    budget_analysis = analyze_budget_truncation(all_results, max_trials)
    
    # Save comprehensive results
    results_path = output_dir / "exhaustive_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'budget_analysis': budget_analysis,
            'parameters': {
                'init_counts': list(init_counts),
                'num_repeats': num_repeats,
                'max_trials': max_trials
            }
        }, f, indent=2)
    logger.info(f"Saved comprehensive results: {results_path}")
    
    # Create sanity check plots
    create_sanity_check_plots(budget_analysis, output_dir, all_results)
    
    logger.info(f"\n=== Evaluation Complete ===")
    logger.info(f"Tested init counts: {min(init_counts)}-{max(init_counts)}")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"Logs saved in: {logs_dir}")


def analyze_budget_truncation(all_results, max_trials):
    """Analyze performance at all budget levels using truncation."""
    budget_analysis = {}
    
    for budget in range(1, max_trials + 1):
        budget_analysis[budget] = {}
        
        for init_count, campaigns in all_results.items():
            if not campaigns:
                continue
                
            # Extract performance at this budget level (truncated best_values)
            budget_performances = []
            for campaign in campaigns:
                if len(campaign['best_values']) >= budget:
                    budget_performances.append(campaign['best_values'][budget - 1])
            
            if budget_performances:
                budget_analysis[budget][init_count] = {
                    'mean': float(np.mean(budget_performances)),
                    'std': float(np.std(budget_performances)),
                    'min': float(np.min(budget_performances)),
                    'max': float(np.max(budget_performances)),
                    'count': len(budget_performances)
                }
    
    return budget_analysis


def create_sanity_check_plots(budget_analysis, output_dir, all_results=None):
    """Create simple plots for sanity checking results."""
    
    # Plot 1: Final performance (budget=30) vs init count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data for final budget
    final_budget = max(budget_analysis.keys())
    init_counts = []
    mean_performance = []
    std_performance = []
    
    for init_count in sorted(budget_analysis[final_budget].keys()):
        data = budget_analysis[final_budget][init_count]
        init_counts.append(init_count)
        mean_performance.append(data['mean'])
        std_performance.append(data['std'])
    
    # Left plot: Mean performance with error bars
    ax1.errorbar(init_counts, mean_performance, yerr=std_performance, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax1.set_xlabel('Number of Initialization Points')
    ax1.set_ylabel('Final Best Objective Value')
    ax1.set_title(f'Final Performance vs Init Count (Budget={final_budget})')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Performance across different budgets (heatmap-style)
    budgets = [5, 10, 15, 20, 25, 30]  # Sample of budgets for clarity
    init_sample = [2, 5, 10, 15, 20, 25, 30]  # Sample of init counts
    
    performance_matrix = []
    for budget in budgets:
        if budget in budget_analysis:
            row = []
            for init_count in init_sample:
                if init_count in budget_analysis[budget]:
                    row.append(budget_analysis[budget][init_count]['mean'])
                else:
                    row.append(np.nan)
            performance_matrix.append(row)
    
    if performance_matrix:
        im = ax2.imshow(performance_matrix, cmap='viridis_r', aspect='auto')
        ax2.set_xlabel('Init Count')
        ax2.set_ylabel('Budget')
        ax2.set_title('Mean Performance Across Budgets')
        ax2.set_xticks(range(len(init_sample)))
        ax2.set_xticklabels(init_sample)
        ax2.set_yticks(range(len(budgets)))
        ax2.set_yticklabels(budgets)
        plt.colorbar(im, ax=ax2, label='Mean Best Objective')
    
    plt.tight_layout()
    plot_path = output_dir / 'sanity_check_plots.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sanity check plots: {plot_path}")
    
    # Plot 2: Simple convergence curves for different init counts
    if all_results is None:
        print("Warning: all_results not provided, skipping convergence curves")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Show convergence for a few representative init counts
    representative_inits = [2, 5, 10, 15, 20, 30]
    colors = plt.cm.tab10(np.linspace(0, 1, len(representative_inits)))
    
    for i, init_count in enumerate(representative_inits):
        if init_count not in all_results or not all_results[init_count]:
            continue
            
        # Average best_values across repeats for this init count
        max_len = max(len(campaign['best_values']) for campaign in all_results[init_count])
        avg_curve = []
        
        for trial_idx in range(max_len):
            trial_values = []
            for campaign in all_results[init_count]:
                if len(campaign['best_values']) > trial_idx:
                    trial_values.append(campaign['best_values'][trial_idx])
            if trial_values:
                avg_curve.append(np.mean(trial_values))
        
        if avg_curve:
            ax.plot(range(1, len(avg_curve) + 1), avg_curve, 
                   color=colors[i], linewidth=2, marker='o', markersize=4,
                   label=f'Init={init_count}')
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Best Objective Value')
    ax.set_title('Average Convergence Curves by Init Count')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    convergence_path = output_dir / 'convergence_curves.png'
    plt.savefig(str(convergence_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence curves: {convergence_path}")


# Remove unused global variable
# all_results = {}  # This is now handled in main()


if __name__ == "__main__":
    main()
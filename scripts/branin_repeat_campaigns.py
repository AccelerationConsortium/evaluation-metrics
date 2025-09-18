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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import torch
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import imageio

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


def branin(x1, x2):
    """Branin optimization function."""
    y = float(
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    return y


def interval_score(y_true, lower, upper, alpha=0.05):
    """Calculate interval score for uncertainty quantification evaluation."""
    width = upper - lower
    lower_penalty = 2 / alpha * np.maximum(0, lower - y_true)
    upper_penalty = 2 / alpha * np.maximum(0, y_true - upper)
    return width + lower_penalty + upper_penalty


def normal_95ci(mean, std):
    """Convert normal distribution parameters to 95% confidence intervals."""
    return mean - 1.96 * std, mean + 1.96 * std


def calculate_ax_cross_validation_metrics(ax_client, ax_client_cv, trial_index):
    """Calculate Ax cross-validation R² and interval score."""
    if trial_index < 5:  # Need at least 5 trials for meaningful CV
        return np.nan, np.nan
    
    try:
        # Ensure the CV client has the same trials as the main client
        data = ax_client.experiment.fetch_data().df
        if data.empty:
            return np.nan, np.nan
            
        # Create parameters and complete trials in CV client if needed
        for i in range(trial_index + 1):
            if i >= len(ax_client_cv.experiment.trials):
                trial = ax_client.experiment.trials[i]
                params = trial.arm.parameters
                trial_data = data[data['trial_index'] == i]
                if not trial_data.empty:
                    objective_value = trial_data['mean'].iloc[0]
                    # Generate and complete trial in CV client
                    _, cv_trial_index = ax_client_cv.get_next_trial(fixed_features=params)
                    ax_client_cv.complete_trial(trial_index=cv_trial_index, raw_data=objective_value)
        
        # Perform cross-validation if we have a fitted model
        if hasattr(ax_client_cv.generation_strategy, 'model') and ax_client_cv.generation_strategy.model is not None:
            cv_results = cross_validate(ax_client_cv.generation_strategy.model)
            
            # Calculate R² from CV predictions
            y_true = []
            y_pred = []
            lower_bounds = []
            upper_bounds = []
            
            for result in cv_results:
                y_true.append(result.observed.data.mean)
                y_pred.append(result.predicted.mean)
                # Calculate 95% CI from predicted mean and std
                pred_std = np.sqrt(result.predicted.covariance)
                lower, upper = normal_95ci(result.predicted.mean, pred_std)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            # Calculate metrics
            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
            interval_scores = interval_score(np.array(y_true), np.array(lower_bounds), np.array(upper_bounds))
            mean_interval_score = np.mean(interval_scores)
            
            return r2, mean_interval_score
        else:
            return np.nan, np.nan
        
    except Exception as e:
        # print(f"CV metrics calculation failed: {e}")
        return np.nan, np.nan


def run_single_campaign(campaign_id, num_trials=50, num_init_trials=5):
    """Run a single optimization campaign and return results."""
    print(f"Starting campaign {campaign_id}...")
    
    # Create main ax client for optimization
    ax_client = AxClient(verbose_logging=False)  # Reduce logging
    ax_client.create_experiment(
        parameters=[
            {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
        ],
        objectives={obj1_name: ObjectiveProperties(minimize=True)},
    )
    
    # Storage for results
    campaign_results = {
        'campaign_id': campaign_id,
        'trials': [],
        'best_values': [],
        'objective_values': [],
        'parameters': []
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
        
        if trial_idx % 10 == 0:
            print(f"  Campaign {campaign_id}: Trial {trial_idx}, Current: {objective_value:.6f}, Best: {best_value:.6f}")
    
    print(f"Campaign {campaign_id} completed. Final best: {campaign_results['best_values'][-1]:.6f}")
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
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_cum.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
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
    campaign_id = campaign_results['campaign_id']
    trials = campaign_results['trials']
    best_values = campaign_results['best_values']
    objective_values = campaign_results['objective_values']
    parameters = campaign_results['parameters']
    
    frames = []
    temp_dir = campaign_dir / 'temp_frames'
    temp_dir.mkdir(exist_ok=True)
    
    # Create frames for each trial
    for frame_idx in range(5, len(trials), 2):  # Start from trial 5, every 2nd frame
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left subplot: Parameter space visualization
        x1_vals = [p['x1'] for p in parameters[:frame_idx+1]]
        x2_vals = [p['x2'] for p in parameters[:frame_idx+1]]
        obj_vals = objective_values[:frame_idx+1]
        
        # Create a scatter plot colored by objective value
        scatter = ax1.scatter(x1_vals, x2_vals, c=obj_vals, cmap='viridis_r', 
                             s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Highlight the best point
        best_idx = np.argmin(obj_vals)
        ax1.scatter(x1_vals[best_idx], x2_vals[best_idx], c='red', s=150, 
                   marker='*', edgecolors='black', linewidth=1, label='Best Point')
        
        ax1.set_xlim(-5, 10)
        ax1.set_ylim(0, 10)
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_title(f'Campaign {campaign_id}: Parameter Space (Trial {frame_idx})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label(f'{obj1_name} Value')
        
        # Right subplot: Optimization progress
        ax2.plot(trials[:frame_idx+1], best_values[:frame_idx+1], 'r-', 
                linewidth=2, marker='o', markersize=4, label=f'Best {obj1_name}')
        ax2.scatter(trials[:frame_idx+1], objective_values[:frame_idx+1], 
                   alpha=0.6, s=30, color='lightblue', label=f'Raw {obj1_name}')
        
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel(f'{obj1_name}')
        ax2.set_title(f'Campaign {campaign_id}: Progress (Trial {frame_idx})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, len(trials))
        
        plt.tight_layout()
        
        # Save frame
        frame_path = temp_dir / f'frame_{frame_idx:03d}.png'
        plt.savefig(str(frame_path), dpi=100, bbox_inches='tight')
        plt.close()
        
        frames.append(str(frame_path))
    
    # Create GIF
    gif_path = campaign_dir / f'branin_campaign_{campaign_id}_evolution.gif'
    
    if frames:
        with imageio.get_writer(str(gif_path), mode='I', duration=0.8) as writer:
            for frame_path in frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        # Clean up temporary frames
        for frame_path in frames:
            os.remove(frame_path)
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
    
    # Bottom subplot: Average convergence characteristics (no uncertainty bands)
    # Calculate average improvement rates and cumulative improvements
    all_improvements = []
    all_cumulative_improvements = []
    
    for campaign in all_campaigns:
        best_vals = campaign['best_values']
        if len(best_vals) > 1:
            improvements = []
            for i in range(1, len(best_vals)):
                improvement = best_vals[i-1] - best_vals[i]  # Positive means improvement
                improvements.append(improvement)
            
            # Pad to max_trials-1 if needed
            if len(improvements) < max_trials - 1:
                improvements = improvements + [0] * (max_trials - 1 - len(improvements))
            
            all_improvements.append(improvements)
            all_cumulative_improvements.append(np.cumsum(improvements))
    
    if all_improvements:
        all_improvements = np.array(all_improvements)
        all_cumulative_improvements = np.array(all_cumulative_improvements)
        
        # Calculate averages
        mean_improvements = np.mean(all_improvements, axis=0)
        mean_cumulative = np.mean(all_cumulative_improvements, axis=0)
        
        improvement_trials = list(range(1, max_trials))
        
        # Plot average improvements
        ax2.bar(improvement_trials, mean_improvements, alpha=0.7, color='green', 
                label='Avg Improvement per Trial', width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel("Average Improvement")
        ax2.set_title("Average Optimization Efficiency Across 10 Repeat Campaigns")
        ax2.grid(True, alpha=0.3)
        
        # Add cumulative improvement on second y-axis
        ax2_cum = ax2.twinx()
        ax2_cum.plot(improvement_trials, mean_cumulative, 'orange', 
                     label='Avg Cumulative Improvement', marker='s', linewidth=2)
        ax2_cum.set_ylabel("Average Cumulative Improvement", color='orange')
        ax2_cum.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_cum.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
    else:
        ax2.text(0.5, 0.5, 'Not enough data\nfor efficiency analysis', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Average Optimization Efficiency Across 10 Repeat Campaigns")
    
    plt.tight_layout()
    
    # Save composite plot
    composite_path = output_dir / 'branin_repeat_campaigns_composite_results.png'
    plt.savefig(str(composite_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    return composite_path


def main():
    """Main execution function."""
    print("=== Running 10 Repeat Branin Campaigns ===")
    print("This will run 10 independent campaigns to determine optimal initialization points")
    print()
    
    # Setup output directory
    output_dir = Path(__file__).parent / "branin_repeat_campaign_results"
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each campaign
    for i in range(1, 11):
        campaign_dir = output_dir / f"cv_plots_{i}"
        campaign_dir.mkdir(exist_ok=True)
    
    # Run campaigns sequentially (parallel execution can be memory intensive)
    all_campaigns = []
    
    for campaign_id in range(1, 11):
        try:
            campaign_results = run_single_campaign(campaign_id, num_trials=30)  # Reduced for faster execution
            all_campaigns.append(campaign_results)
            
            # Create individual campaign plot
            individual_plot_path = create_individual_campaign_plot(campaign_results, output_dir)
            print(f"  Saved individual plot: {individual_plot_path}")
            
            # Create individual campaign directory and additional outputs
            campaign_dir = output_dir / f"cv_plots_{campaign_id}"
            
            # Create Plotly HTML version
            try:
                html_path = create_plotly_campaign_plot(campaign_results, campaign_dir)
                print(f"  Saved HTML plot: {html_path}")
            except Exception as e:
                print(f"  Warning: Could not create HTML plot: {e}")
            
            # Create campaign GIF
            try:
                gif_path = create_campaign_gif(campaign_results, campaign_dir)
                print(f"  Saved GIF: {gif_path}")
            except Exception as e:
                print(f"  Warning: Could not create GIF: {e}")
            
            # Save campaign data as JSON
            campaign_data_path = campaign_dir / f"branin_campaign_{campaign_id}_data.json"
            with open(campaign_data_path, 'w') as f:
                # Convert numpy types to standard Python types for JSON serialization
                json_safe_results = {}
                for key, value in campaign_results.items():
                    if isinstance(value, list):
                        json_safe_results[key] = [
                            float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for v in value
                        ]
                    else:
                        json_safe_results[key] = value
                
                json.dump(json_safe_results, f, indent=2)
            print(f"  Saved campaign data: {campaign_data_path}")
            
        except Exception as e:
            print(f"Error in campaign {campaign_id}: {e}")
            continue
    
    if all_campaigns:
        # Create composite plot
        composite_plot_path = create_composite_plot(all_campaigns, output_dir)
        print(f"\nSaved composite plot: {composite_plot_path}")
        
        # Save summary statistics
        summary_path = output_dir / "campaign_summary.json"
        final_best_values = [campaign['best_values'][-1] for campaign in all_campaigns]
        summary = {
            'num_campaigns': len(all_campaigns),
            'final_best_values': final_best_values,
            'mean_final_best': float(np.mean(final_best_values)),
            'std_final_best': float(np.std(final_best_values)),
            'min_final_best': float(np.min(final_best_values)),
            'max_final_best': float(np.max(final_best_values))
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary statistics: {summary_path}")
        
        print(f"\n=== Campaign Summary ===")
        print(f"Completed {len(all_campaigns)} campaigns")
        print(f"Mean final best value: {summary['mean_final_best']:.6f} ± {summary['std_final_best']:.6f}")
        print(f"Best overall result: {summary['min_final_best']:.6f}")
        print(f"Results saved in: {output_dir}")
    
    else:
        print("No campaigns completed successfully!")


if __name__ == "__main__":
    main()
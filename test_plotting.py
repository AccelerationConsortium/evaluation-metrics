#!/usr/bin/env python3
"""
Simple test of the plotting logic to verify the issue mentioned in the comment.
This script creates fake data to test the plotting structure.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Create fake data similar to what the real script would generate
obj1_name = "branin"

# Simulate 10 trials of data
np.random.seed(42)
trial_data = {
    'x1': np.random.uniform(-5, 10, 10),
    'x2': np.random.uniform(0, 10, 10),
    obj1_name: np.random.uniform(1, 300, 10)  # Random branin-like values
}

# Make it decreasing on average (like optimization would)
trial_data[obj1_name] = np.sort(trial_data[obj1_name])[::-1]  # Start high, go lower
trial_data[obj1_name][5:] = np.random.uniform(0.4, 10, 5)  # Better values later

df = pd.DataFrame(trial_data)
df.index.name = 'trial'

# Simulate metrics data (from trial 3 onward)
all_metrics = []
for i in range(3, 10):
    all_metrics.append({
        'trial': i,
        'gp_r2': 0.99 + 0.01 * np.random.random(),
        'ax_cv_r2': 0.3 + 0.4 * np.random.random(),
        'interval_score': 30 - 2 * i + np.random.random(),
        'ax_cv_interval_score': 25 - 1.5 * i + np.random.random(),
        'loo_nll': 20000 - 2000 * i + 1000 * np.random.random(),
        'rank_tau': 0.999 + 0.001 * np.random.random()
    })

print("Testing plotting logic...")
print(f"DataFrame shape: {df.shape}")
print(f"Metrics count: {len(all_metrics)}")

# Create stacked vertical plots as in the original
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), dpi=150)

# ===== TOP PLOT (ax1) =====
# Plot 1: Standard "best so far" trace - THIS SHOULD HAVE BOTH SCATTER AND LINE
objectives = [obj1_name]
print(f"Plotting top plot with objectives: {objectives}")

ax1.scatter(df.index, df[objectives], ec="k", fc="none", label="Observed")
print(f"Added scatter plot with {len(df)} points")

ax1.plot(
    df.index,
    np.minimum.accumulate(df[objectives]),
    color="#0033FF",
    lw=2,
    label="Best to Trial",
)
print("Added best-so-far line to top plot")

ax1.set_xlabel("Trial Number")
ax1.set_ylabel(f"{obj1_name} objective")
ax1.set_title("Optimization Progress")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ===== BOTTOM PLOT (ax2) =====
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
    ax2_twin1 = ax2.twinx()
    ax2_twin2 = ax2.twinx()
    ax2_twin3 = ax2.twinx()
    ax2_twin2.spines['right'].set_position(('outward', 60))
    ax2_twin3.spines['right'].set_position(('outward', 120))
    
    # Plot main metrics first
    line1 = ax2.plot(trial_nums, gp_r2_values, 'r-', label='GP R²', marker='o', linewidth=2, zorder=3)
    line2 = ax2_twin1.plot(trial_nums, rank_tau_values, 'g-', label='Rank τ', marker='s', linewidth=2, zorder=3)
    line3 = ax2_twin2.plot(trial_nums, loo_values, 'b-', label='LOO NLL', marker='^', linewidth=2, zorder=3)
    
    # Plot Ax CV R² if available
    lines = line1 + line2 + line3
    if ax_cv_r2_values:
        line4 = ax2.plot(ax_cv_trial_nums, ax_cv_r2_values, 'orange', label='Ax CV R²', marker='d', 
                        linewidth=2, linestyle='--', zorder=3)
        lines += line4
    
    # Set appropriate y-axis limits for R² values before adding best-so-far trace
    r2_values_all = gp_r2_values + (ax_cv_r2_values if ax_cv_r2_values else [])
    if r2_values_all:
        r2_min, r2_max = min(r2_values_all), max(r2_values_all)
        r2_range = r2_max - r2_min
        ax2.set_ylim(r2_min - 0.1 * r2_range, r2_max + 0.1 * r2_range)
        print(f"Set R² y-axis limits: {r2_min - 0.1 * r2_range:.3f} to {r2_max + 0.1 * r2_range:.3f}")
    
    # Add overlaid best-so-far trace (faded gray, semi-transparent) without affecting y-limits
    best_so_far = np.minimum.accumulate(df[objectives])
    ax2.plot(df.index, best_so_far, color='gray', linewidth=6, alpha=0.3, 
             label='Best so far', linestyle='-', zorder=1)
    print("Added faded gray best-so-far overlay to bottom plot")
    
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
        
    # Add best-so-far line to the legend
    best_line = [plt.Line2D([0], [0], color='gray', linewidth=6, alpha=0.3, label='Best so far')]
    lines = best_line + lines
        
    ax2_twin3.set_ylabel("Interval Score", color='purple')
    ax2_twin3.tick_params(axis='y', labelcolor='purple')
    
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel("R² Values", color='r')
    ax2_twin1.set_ylabel("Rank Correlation (τ)", color='g')
    ax2_twin2.set_ylabel("LOO Negative Log-Likelihood", color='b')
    
    ax2.tick_params(axis='y', labelcolor='r')
    ax2_twin1.tick_params(axis='y', labelcolor='g')
    ax2_twin2.tick_params(axis='y', labelcolor='b')
    
    ax2.set_title("Evaluation Metrics vs Trial (Iteration-by-Iteration)")
    
    # Combined legend
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
else:
    ax2.text(0.5, 0.5, 'Not enough data\nfor metrics calculation', 
            ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title("Evaluation Metrics")

plt.tight_layout()
plt.savefig('/home/runner/work/evaluation-metrics/evaluation-metrics/test_plotting_output.png', 
            dpi=150, bbox_inches='tight')

print("Plot saved to test_plotting_output.png")
print("\nCHECKING PLOT CONTENTS:")
print("Top plot (ax1) should have:")
print("  - Scatter points (black circles)")  
print("  - Best-so-far line (blue line)")
print("Bottom plot (ax2) should have:")
print("  - Various metrics with different colors")
print("  - Faded gray best-so-far overlay")

# Check if both elements are in the top plot
ax1_children = ax1.get_children()
scatter_count = 0
line_count = 0
for child in ax1_children:
    if hasattr(child, 'get_facecolors'):
        scatter_count += 1
    elif hasattr(child, 'get_color') and hasattr(child, 'get_linewidth'):
        line_count += 1

print(f"\nTop plot analysis:")
print(f"  Scatter plots found: {scatter_count}")
print(f"  Line plots found: {line_count}")

# Check the bottom plot for gray overlay
ax2_children = ax2.get_children()
gray_overlay_found = False
for child in ax2_children:
    if hasattr(child, 'get_color') and hasattr(child, 'get_alpha'):
        try:
            if 'gray' in str(child.get_color()).lower() and child.get_alpha() == 0.3:
                gray_overlay_found = True
                break
        except:
            pass

print(f"Bottom plot analysis:")
print(f"  Gray overlay found: {gray_overlay_found}")
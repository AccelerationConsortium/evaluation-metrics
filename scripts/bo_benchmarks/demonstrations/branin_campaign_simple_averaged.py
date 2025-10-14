#!/usr/bin/env python3
"""
Simple 10-repeat Branin campaigns for averaged results.
This script runs the working single campaign 10 times and averages the results.
"""

import numpy as np
import pandas as pd
import subprocess
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import re


def run_single_campaign_subprocess(seed):
    """Run a single campaign using subprocess to avoid model conflicts."""
    script_path = Path(__file__).parent / "branin_campaign_demonstration.py"

    # Create a temporary script that sets the seed
    temp_script = f"""
import os
import sys
sys.path.insert(0, '{Path(__file__).parent.parent.parent}')

# Set the seed
import numpy as np
import torch
np.random.seed({seed})
torch.manual_seed({seed})

# Import and run the original script content but capture results
exec(open(r'{script_path}').read())

# Extract metrics from all_metrics variable
import json
with open('/tmp/metrics_seed_{seed}.json', 'w') as f:
    json.dump(all_metrics, f)
"""

    temp_file = f"/tmp/campaign_seed_{seed}.py"
    with open(temp_file, "w") as f:
        f.write(temp_script)

    try:
        # Run the script
        result = subprocess.run(["python", temp_file], capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            # Load the metrics
            with open(f"/tmp/metrics_seed_{seed}.json", "r") as f:
                metrics = json.load(f)
            return metrics
        else:
            print(f"Campaign {seed} failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Campaign {seed} timed out")
        return None
    except Exception as e:
        print(f"Campaign {seed} error: {e}")
        return None
    finally:
        # Clean up
        if Path(temp_file).exists():
            Path(temp_file).unlink()
        metrics_file = Path(f"/tmp/metrics_seed_{seed}.json")
        if metrics_file.exists():
            metrics_file.unlink()


def main():
    print("Starting 10-repeat campaign averaging...")
    start_time = time.time()

    all_campaign_metrics = []

    # Run 10 campaigns
    for i in range(10):
        seed = 1000 + i
        print(f"\nRunning campaign {i + 1}/10 with seed {seed}...")

        metrics = run_single_campaign_subprocess(seed)
        if metrics:
            all_campaign_metrics.append(metrics)
            print(f"Campaign {i + 1} completed successfully with {len(metrics)} metrics")
        else:
            print(f"Campaign {i + 1} failed")

    if not all_campaign_metrics:
        print("No successful campaigns!")
        return

    print(f"\nSuccessfully completed {len(all_campaign_metrics)} campaigns")

    # Compute averaged metrics
    trial_indices = set()
    for campaign_metrics in all_campaign_metrics:
        for metric in campaign_metrics:
            trial_indices.add(metric["trial"])

    trial_indices = sorted(trial_indices)

    averaged_metrics = []
    for trial_idx in trial_indices:
        trial_data = []

        for campaign_metrics in all_campaign_metrics:
            for metric in campaign_metrics:
                if metric["trial"] == trial_idx:
                    trial_data.append(metric)
                    break

        if trial_data:
            # Average the metrics
            def safe_mean(values):
                valid_values = [v for v in values if v is not None and not np.isnan(v)]
                return np.mean(valid_values) if valid_values else None

            averaged_metric = {
                "trial": trial_idx,
                "gp_r2": safe_mean([m["gp_r2"] for m in trial_data]),
                "rank_tau": safe_mean([m["rank_tau"] for m in trial_data]),
                "loo_nll": safe_mean([m["loo_nll"] for m in trial_data]),
                "interval_score": safe_mean([m["interval_score"] for m in trial_data]),
                "imp_std": safe_mean([m["imp_std"] for m in trial_data]),
                "ax_cv_r2": safe_mean(
                    [m["ax_cv_r2"] for m in trial_data if m["ax_cv_r2"] is not None]
                ),
                "ax_cv_interval_score": safe_mean(
                    [
                        m["ax_cv_interval_score"]
                        for m in trial_data
                        if m["ax_cv_interval_score"] is not None
                    ]
                ),
            }
            averaged_metrics.append(averaged_metric)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print(f"Averaged across {len(all_campaign_metrics)} campaigns")

    # Create the plot (single plot without top subplot as requested)
    if averaged_metrics:
        trial_nums = [m["trial"] for m in averaged_metrics]
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

        # Create single plot (no best-so-far subplot since we don't have the objective values)
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), dpi=150)

        # Create 6 y-axes: 1 left + 5 right (no best-so-far for averaged results)
        # Left axis: GP R²
        # Right axes: Rank τ, LOO NLL, GP Interval Score, Ax CV Interval Score, Imp Std

        ax_tau = ax.twinx()  # Right axis 1: Rank correlation τ (-1 to 1)
        ax_loo = ax.twinx()  # Right axis 2: LOO NLL
        ax_gp_int = ax.twinx()  # Right axis 3: GP Interval Score
        ax_ax_int = ax.twinx()  # Right axis 4: Ax CV Interval Score
        ax_imp_std = ax.twinx()  # Right axis 5: Importance Std Dev

        # Position the additional y-axes
        ax_tau.spines["right"].set_position(("outward", 60))
        ax_loo.spines["right"].set_position(("outward", 120))
        ax_gp_int.spines["right"].set_position(("outward", 180))
        ax_ax_int.spines["right"].set_position(("outward", 240))
        ax_imp_std.spines["right"].set_position(("outward", 300))

        # Plot GP R² on left (main) y-axis
        line_gp_r2 = ax.plot(
            trial_nums, gp_r2_values, "r-", label="GP R² (avg)", marker="o", linewidth=2, zorder=3
        )
        lines = line_gp_r2

        # Plot Ax CV R² if available (also on left axis)
        if ax_cv_r2_values:
            line_ax_r2 = ax.plot(
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
            ax.set_ylim(r2_min - margin, r2_max + margin)
        else:
            ax.set_ylim(-1, 1)  # Default range if no valid R² values

        # Plot rank tau correlation on right axis 1 (fixed -1 to 1 range)
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

        # Plot LOO NLL on right axis 2 (auto range)
        line_loo = ax_loo.plot(
            trial_nums, loo_values, "b-", label="LOO NLL (avg)", marker="^", linewidth=2, zorder=3
        )
        lines += line_loo

        # Plot GP interval score on right axis 3 (auto range)
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

        # Plot Ax CV interval score on right axis 4 (auto range)
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

        # Plot importance std dev on right axis 5 (auto range)
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
        ax.set_ylabel("R² Values", color="red")
        ax_tau.set_ylabel("Rank Correlation (τ)", color="green")
        ax_loo.set_ylabel("LOO NLL", color="blue")
        ax_gp_int.set_ylabel("GP Interval Score", color="purple")
        ax_ax_int.set_ylabel("Ax CV Interval Score", color="magenta")
        ax_imp_std.set_ylabel("Importance Std", color="pink")

        # Set tick colors to match y-axis colors
        ax.tick_params(axis="y", labelcolor="red")
        ax_tau.tick_params(axis="y", labelcolor="green")
        ax_loo.tick_params(axis="y", labelcolor="blue")
        ax_gp_int.tick_params(axis="y", labelcolor="purple")
        ax_ax_int.tick_params(axis="y", labelcolor="magenta")
        ax_imp_std.tick_params(axis="y", labelcolor="pink")

        ax.set_title(
            f"Averaged Evaluation Metrics vs Trial ({len(all_campaign_metrics)} Campaigns)"
        )
        ax.set_xlim(-10, None)  # Add whitespace for legend

        # Place legend inside plot with white background and transparency
        labels = [l.get_label() for l in lines]
        legend = ax.legend(
            lines, labels, loc="upper left", framealpha=0.8, fancybox=False, shadow=False
        )
        legend.get_frame().set_facecolor("white")

        print(f"\nAveraged metrics summary:")
        print(f"Trial range: {min(trial_nums)} to {max(trial_nums)}")
        print(f"Final GP R²: {gp_r2_values[-1]:.4f}")
        if ax_cv_r2_values:
            print(f"Final Ax CV R²: {ax_cv_r2_values[-1]:.4f}")
        print(f"Final Rank τ: {rank_tau_values[-1]:.4f}")
        print(f"Final LOO NLL: {loo_values[-1]:.4f}")
        if interval_score_values:
            print(f"Final GP Interval Score: {interval_score_values[-1]:.4f}")
        if ax_cv_interval_score_values:
            print(f"Final Ax CV Interval Score: {ax_cv_interval_score_values[-1]:.4f}")
        print(f"Final Importance Std: {imp_std_values[-1]:.4f}")

        plt.tight_layout()
        # Save matplotlib figure to the same directory as this script
        _out_path = Path(__file__).with_name("branin_campaign_averaged_results.png")
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(_out_path), dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\nResults saved to: {_out_path}")
    else:
        print("No averaged metrics to plot")


if __name__ == "__main__":
    main()

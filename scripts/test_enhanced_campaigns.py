#!/usr/bin/env python3
"""
Test script for branin repeat campaigns with enhanced features.
Run 2 campaigns to test HTML and GIF generation.
"""

import sys
from pathlib import Path

# Add the scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from branin_repeat_campaigns import run_single_campaign, create_individual_campaign_plot, create_plotly_campaign_plot, create_campaign_gif
import json

def main():
    print("=== Testing Enhanced Campaign Features ===")
    
    # Setup test output directory
    output_dir = Path(__file__).parent / "test_campaign_results"
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for test campaigns
    for i in range(1, 3):
        campaign_dir = output_dir / f"cv_plots_{i}"
        campaign_dir.mkdir(exist_ok=True)
    
    # Run 2 test campaigns
    for campaign_id in range(1, 3):
        try:
            print(f"\nRunning test campaign {campaign_id}...")
            campaign_results = run_single_campaign(campaign_id, num_trials=15)  # Smaller for testing
            
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
                            float(v) if isinstance(v, (list, tuple)) else (
                                float(v) if hasattr(v, '__float__') else v
                            )
                            for v in value
                        ]
                    else:
                        json_safe_results[key] = value
                
                json.dump(json_safe_results, f, indent=2)
            print(f"  Saved campaign data: {campaign_data_path}")
            
        except Exception as e:
            print(f"Error in test campaign {campaign_id}: {e}")
            continue
    
    print("\n=== Test Complete ===")
    print(f"Check results in: {output_dir}")

if __name__ == "__main__":
    main()
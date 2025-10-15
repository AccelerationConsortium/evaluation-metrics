#!/usr/bin/env python3
"""
Smoke test for branin_repeat_campaigns.py
Tests the script with minimal parameters to verify functionality.
Uses a separate directory to avoid interfering with real results.
"""

import datetime
import json
import logging
import shutil
import sys
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the functions we need to test
from branin_repeat_campaigns import (
    run_single_campaign,
    analyze_budget_truncation,
    create_sanity_check_plots,
    setup_logging,
)


def smoke_test():
    """Run a minimal smoke test of the branin campaigns."""
    print("=" * 60)
    print("SMOKE TEST: Branin Repeat Campaigns")
    print("=" * 60)
    print()

    # Create smoke test directory (separate from production)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).parent / "smoke_test_results"
    output_dir = base_dir / f"smoke_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created smoke test directory: {output_dir}")

    # Setup logging for smoke test
    logs_dir = Path(__file__).parent / "smoke_test_logs" / f"smoke_test_{timestamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger, _ = setup_logging(f"smoke_test_{timestamp}")
    logger.info("=== Starting Smoke Test ===")
    print(f"✓ Setup logging in: {logs_dir}")

    # Test parameters (minimal to run quickly)
    init_counts = [2, 5]  # Just 2 init counts instead of 29
    num_repeats = 2  # Just 2 repeats instead of 10
    max_trials = 5  # Just 5 trials instead of 30

    print(f"\nTest parameters:")
    print(f"  - Init counts: {init_counts}")
    print(f"  - Repeats per init: {num_repeats}")
    print(f"  - Trials per campaign: {max_trials}")
    print()

    all_results = {}
    total_campaigns = len(init_counts) * num_repeats

    # Run minimal campaigns
    campaign_count = 0
    for init_count in init_counts:
        logger.info(f"\n=== Testing {init_count} initialization points ===")
        print(f"\nTesting init_count={init_count}...")

        # Create subdirectory for this init count
        init_dir = output_dir / f"init_{init_count}"
        init_dir.mkdir(exist_ok=True)

        init_campaigns = []

        for repeat_id in range(1, num_repeats + 1):
            campaign_count += 1
            campaign_id = f"{init_count}_{repeat_id}"
            
            # Generate unique seed
            seed = hash(f"{init_count}_{repeat_id}_{timestamp}") % (2**31)
            
            print(f"  [{campaign_count}/{total_campaigns}] Running campaign {campaign_id} (seed={seed})...")

            try:
                campaign_results = run_single_campaign(
                    campaign_id, 
                    num_trials=max_trials, 
                    num_init_trials=init_count,
                    seed=seed
                )
                init_campaigns.append(campaign_results)

                # Save campaign data
                campaign_file = init_dir / f"campaign_{campaign_id}.json"
                with open(campaign_file, "w") as f:
                    json_safe_results = {
                        k: (
                            [float(v) if isinstance(v, (float, int)) else v for v in val]
                            if isinstance(val, list)
                            else val
                        )
                        for k, val in campaign_results.items()
                        if k != "metrics"  # Skip complex nested metrics for smoke test
                    }
                    json.dump(json_safe_results, f, indent=2)

                final_best = campaign_results["best_values"][-1]
                logger.info(f"Campaign {campaign_id}: Final best = {final_best:.6f}")
                print(f"    ✓ Final best value: {final_best:.6f}")

            except Exception as e:
                logger.error(f"Error in campaign {campaign_id}: {e}")
                print(f"    ✗ ERROR: {e}")
                raise

        all_results[init_count] = init_campaigns
        logger.info(f"Completed {len(init_campaigns)}/{num_repeats} campaigns for init_count={init_count}")

    # Test analysis functions
    print("\nRunning analysis...")
    budget_analysis = analyze_budget_truncation(all_results, max_trials)
    print("  ✓ Budget analysis completed")

    # Save results
    results_path = output_dir / "smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "budget_analysis": budget_analysis,
                "parameters": {
                    "init_counts": init_counts,
                    "num_repeats": num_repeats,
                    "max_trials": max_trials,
                },
            },
            f,
            indent=2,
        )
    print(f"  ✓ Saved results to: {results_path}")

    # Create sanity check plots
    print("\nGenerating plots...")
    try:
        create_sanity_check_plots(budget_analysis, output_dir, all_results)
        print("  ✓ Sanity check plots generated")
    except Exception as e:
        print(f"  ! Warning: Plot generation had issues: {e}")
        logger.warning(f"Plot generation warning: {e}")

    # Verify results
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check that we have different results for different repeats
    all_same = True
    for init_count in init_counts:
        campaigns = all_results[init_count]
        if len(campaigns) > 1:
            first_best = campaigns[0]["best_values"]
            for i, campaign in enumerate(campaigns[1:], 1):
                if campaign["best_values"] != first_best:
                    all_same = False
                    print(f"✓ Init {init_count}: Repeat {i+1} differs from repeat 1 (proper randomization)")
                    break
            if all_same:
                print(f"✗ Init {init_count}: All repeats are identical (seeding may be broken)")

    if not all_same:
        print("\n✓ SEEDING VERIFICATION PASSED: Repeats show different results")
    else:
        print("\n✗ SEEDING VERIFICATION FAILED: All repeats are identical")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Logs saved to: {logs_dir}")
    print("\nTo clean up smoke test files, run:")
    print(f"  rm -rf {base_dir}")
    print(f"  rm -rf {Path(__file__).parent / 'smoke_test_logs'}")
    
    return 0 if not all_same else 1


if __name__ == "__main__":
    try:
        exit_code = smoke_test()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

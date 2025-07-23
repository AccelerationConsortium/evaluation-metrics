#!/usr/bin/env python
"""Submit batch jobs for three simple optimization campaigns.

This script submits 3 different optimization campaigns to test the framework:
1. Branin function optimization (5 iterations)
2. Hartmann6 function optimization (5 iterations) 
3. Mixed campaign with both functions (3 iterations each)

Usage:
    python scripts/batch_optimization_campaigns.py --local    # Test locally
    python scripts/batch_optimization_campaigns.py --submit   # Submit to cluster
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir / "bo_benchmarks"))

def run_optimization_campaign(campaign_name, function_name, iterations, campaign_id):
    """Run a simple optimization campaign for testing."""
    from benchmark_functions import evaluate_benchmark
    import numpy as np
    
    print(f"Starting campaign {campaign_id}: {campaign_name}")
    print(f"Function: {function_name}, Iterations: {iterations}")
    
    # Define parameter bounds for each function
    bounds = {
        "branin": {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)},
        "hartmann6": {f"x{i+1}": (0.0, 1.0) for i in range(6)}
    }
    
    results = []
    best_value = float('inf')
    best_params = None
    
    # Simple random search for demonstration
    np.random.seed(42 + campaign_id)  # Reproducible but different per campaign
    
    for iteration in range(iterations):
        # Generate random parameters within bounds
        params = {"function": function_name}
        for param_name, (low, high) in bounds[function_name].items():
            params[param_name] = np.random.uniform(low, high)
        
        # Evaluate function
        value = evaluate_benchmark(params)
        
        # Track best result
        if value < best_value:
            best_value = value
            best_params = params.copy()
        
        result = {
            "campaign_id": campaign_id,
            "campaign_name": campaign_name,
            "iteration": iteration,
            "parameters": params,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        print(f"  Iteration {iteration+1}: {value:.6f}")
    
    print(f"Campaign {campaign_id} complete. Best: {best_value:.6f}")
    print(f"  Best params: {best_params}")
    
    return {
        "campaign_name": campaign_name,
        "campaign_id": campaign_id,
        "function_name": function_name,
        "iterations": iterations,
        "results": results,
        "best_value": best_value,
        "best_params": best_params
    }

def submit_campaign_to_cluster(campaign_name, function_name, iterations, campaign_id):
    """Submit a single campaign as a cluster job."""
    try:
        import submitit
    except ImportError:
        print("Installing submitit...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "submitit"])
        import submitit
    
    # Setup executor
    log_folder = f"submitit_logs/campaigns/{campaign_name}_%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Configure for cluster (using Niagara defaults)
    account = os.getenv("SLURM_ACCOUNT", "def-sgbaird")
    executor.update_parameters(
        timeout_min=10,  # Short timeout for simple campaigns
        cpus_per_task=1,
        slurm_partition="compute",
        slurm_account=account,
        slurm_job_name=f"opt_campaign_{campaign_id}"
    )
    
    # Submit job
    job = executor.submit(run_optimization_campaign, campaign_name, function_name, iterations, campaign_id)
    print(f"Submitted campaign {campaign_id} as job {job.job_id}")
    
    return job

def test_campaigns_locally():
    """Test all three campaigns locally."""
    print("=== Testing 3 Optimization Campaigns Locally ===\n")
    
    # Define the three campaigns
    campaigns = [
        ("branin_quick", "branin", 5, 1),
        ("hartmann6_quick", "hartmann6", 5, 2), 
        ("mixed_quick_branin", "branin", 3, 3)
    ]
    
    all_results = []
    
    for campaign_name, function_name, iterations, campaign_id in campaigns:
        print(f"\n--- Campaign {campaign_id}: {campaign_name} ---")
        result = run_optimization_campaign(campaign_name, function_name, iterations, campaign_id)
        all_results.append(result)
    
    print("\n=== All Campaigns Complete ===")
    for result in all_results:
        print(f"Campaign {result['campaign_id']} ({result['campaign_name']}): "
              f"Best = {result['best_value']:.6f}")
    
    return all_results

def submit_campaigns_to_cluster():
    """Submit all three campaigns to the cluster."""
    print("=== Submitting 3 Optimization Campaigns to Cluster ===\n")
    
    # Check SLURM account
    account = os.getenv("SLURM_ACCOUNT")
    if not account:
        print("Warning: SLURM_ACCOUNT not set. Using default 'def-sgbaird'")
        account = "def-sgbaird"
    print(f"Using SLURM account: {account}")
    
    # Define the three campaigns
    campaigns = [
        ("branin_quick", "branin", 5, 1),
        ("hartmann6_quick", "hartmann6", 5, 2),
        ("mixed_quick_branin", "branin", 3, 3)
    ]
    
    jobs = []
    
    for campaign_name, function_name, iterations, campaign_id in campaigns:
        print(f"\nSubmitting campaign {campaign_id}: {campaign_name}")
        try:
            job = submit_campaign_to_cluster(campaign_name, function_name, iterations, campaign_id)
            jobs.append(job)
        except Exception as e:
            print(f"Failed to submit campaign {campaign_id}: {e}")
    
    print(f"\n=== Successfully submitted {len(jobs)} campaigns ===")
    print("Job IDs:", [job.job_id for job in jobs])
    print("\nTo check status: squeue -u $USER")
    print("To view logs: ls submitit_logs/campaigns/")
    
    return jobs

def main():
    parser = argparse.ArgumentParser(description='Run 3 optimization campaigns')
    parser.add_argument('--local', action='store_true', help='Run campaigns locally')
    parser.add_argument('--submit', action='store_true', help='Submit campaigns to cluster')
    
    args = parser.parse_args()
    
    if not (args.local or args.submit):
        # Default to local test
        args.local = True
    
    if args.local:
        print("Running campaigns locally...")
        results = test_campaigns_locally()
        print("\n✓ Local testing complete")
    
    if args.submit:
        print("Submitting campaigns to cluster...")
        try:
            jobs = submit_campaigns_to_cluster()
            print("✓ Cluster submission complete")
        except Exception as e:
            print(f"✗ Cluster submission failed: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
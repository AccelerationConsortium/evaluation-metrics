# BALAM Benchmark Scripts

This directory contains scripts for running optimization benchmarks on the BALAM compute cluster.

## Scripts

### `benchmark.py`
Core benchmark functionality implementing the Branin function optimization with GP-based Bayesian optimization. Includes metadata collection with predicted means and uncertainties for all data points.

### `submit_balam.py`
Simple script to submit benchmark jobs to BALAM cluster or run locally for testing.

## Usage

### Local Testing
```bash
# Run a quick local test
python scripts/submit_balam.py --local --init-trials 3 --opt-steps 5

# Run with different parameters
python scripts/submit_balam.py --local --init-trials 10 --opt-steps 20 --seed 123
```

### BALAM Cluster Submission
```bash
# Set required environment variable
export SLURM_ACCOUNT=your-account-name

# Submit job to cluster
python scripts/submit_balam.py --init-trials 5 --opt-steps 20
```

## Parameters

- `--init-trials`: Number of random initialization trials before switching to GP optimization (default: 5)
- `--opt-steps`: Number of optimization steps after initialization (default: 20)  
- `--seed`: Random seed for reproducibility (default: 42)
- `--local`: Run locally instead of submitting to cluster

## Output

The benchmark collects comprehensive metadata including:
- Optimization trajectory (best value over time)
- All evaluated points (X, y values)
- GP predicted means and uncertainties for all data points at each step
- Final results and configuration parameters

This supports studying the effect of initialization trial count on optimization performance.
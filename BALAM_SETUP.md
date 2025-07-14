# BALAM Benchmark Framework Setup

This document describes how to set up and use the benchmark framework for running optimization campaigns on the BALAM compute cluster.

## Overview

The framework enables studying the effect of initialization trials on optimization performance by:

1. Running benchmark campaigns with various numbers of initialization trials
2. Testing multiple optimization algorithms (GPEI, EI, Random, etc.)
3. Submitting jobs to BALAM via submitit
4. Storing results in MongoDB for analysis
5. Generating performance comparisons

## BALAM Cluster Setup

### 1. Account and Environment

Ensure you have access to BALAM and know your account allocation:

```bash
# Check your allocations
myallocation

# Load required modules (example - adjust for BALAM specifics)
module load python/3.12
module load cuda/11.8  # if using GPU partitions
```

### 2. Environment Variables

Set the following environment variables in your BALAM session:

```bash
# Required for SLURM job submission
export SLURM_ACCOUNT="your-account-name"

# Optional for MongoDB data storage (recommended for large campaigns)
export MONGODB_APP_NAME="your-mongodb-app"
export MONGODB_API_KEY="your-api-key"
```

### 3. Python Environment

Install the package and dependencies:

```bash
# Clone the repository
git clone https://github.com/AccelerationConsortium/evaluation-metrics.git
cd evaluation-metrics

# Install dependencies (recommend using virtual environment)
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage Examples

### 1. Local Testing

Test the framework locally before submitting to the cluster:

```bash
# Run a small local test
python run_balam_benchmark.py --local

# Dry run to check configuration
python run_balam_benchmark.py --dry-run
```

### 2. BALAM Submission

Submit a benchmark campaign to BALAM:

```bash
# Submit to CPU partition
python run_balam_benchmark.py --partition cpu --walltime 120

# Submit to GPU partition (if using GPU-accelerated algorithms)
python run_balam_benchmark.py --partition gpu --walltime 60
```

### 3. Programmatic Usage

```python
from src.gpcheck.benchmark import BenchmarkConfig, BALAMExecutor

# Create configuration
config = BenchmarkConfig(
    objective_function="branin",
    num_initialization_trials=[2, 5, 10, 20, 50],
    num_optimization_steps=100,
    num_repeats=20,
    algorithms=["GPEI", "EI", "Random"]
)

# Submit to BALAM
executor = BALAMExecutor(
    partition="cpu",
    walltime_min=180,
    mem_per_cpu=4000
)

jobs = executor.submit_benchmark_campaign(config)
```

## Configuration Options

### BenchmarkConfig Parameters

- `objective_function`: Function to optimize ("branin", future: "hartmann6")
- `num_initialization_trials`: List of initialization trial counts to test
- `num_optimization_steps`: Total optimization steps per run
- `num_repeats`: Number of repeated runs per configuration
- `algorithms`: Optimization algorithms to compare

### BALAMExecutor Parameters

- `partition`: SLURM partition ("cpu", "gpu", etc.)
- `account`: SLURM account (from environment if not specified)
- `walltime_min`: Maximum job runtime in minutes
- `mem_per_cpu`: Memory per CPU in MB
- `cpus_per_task`: Number of CPUs per task
- `log_folder`: Directory for job logs

## BALAM-Specific Considerations

### Partition Selection

Choose appropriate partitions based on your needs:

- `cpu`: General CPU computing
- `gpu`: GPU-accelerated workloads
- `express`: Short jobs (< 30 min)
- `large`: Large memory jobs

### Resource Allocation

Typical resource requirements:

```python
# For CPU-only optimization
executor = BALAMExecutor(
    partition="cpu",
    walltime_min=120,
    mem_per_cpu=4000,
    cpus_per_task=1
)

# For GPU-accelerated optimization
executor = BALAMExecutor(
    partition="gpu", 
    walltime_min=60,
    mem_per_cpu=8000,
    cpus_per_task=4
)
```

### Job Dependencies

For large campaigns, the framework can set up job dependencies:

```python
# Jobs will wait for data collection to complete
jobs = executor.submit_benchmark_campaign(config, batch_size=20)
```

## MongoDB Integration

### Setup

1. Create MongoDB Atlas cluster or use existing instance
2. Set up MongoDB App Services with Data API
3. Create database `gp-benchmarks` with collection `campaigns`
4. Generate API key and set environment variables

### Data Schema

Results are stored with the following structure:

```json
{
  "objective_function": "branin",
  "algorithm": "GPEI",
  "num_initialization_trials": 10,
  "num_optimization_steps": 50,
  "repeat": 0,
  "session_id": "uuid-string",
  "final_best_value": 0.4123,
  "final_regret": 0.0145,
  "best_values": [15.2, 8.1, 3.4, ...],
  "global_minimum": 0.397887,
  "timestamp": 1642789123.456,
  "date": "2024-01-21 15:32:03.456789"
}
```

## Monitoring and Analysis

### Job Status

Monitor job progress:

```bash
# Check job status
squeue -u $USER

# View specific job logs
cat logs/benchmark-{session-id}/{job-id}/*.out
```

### Data Collection

After jobs complete, analyze results:

```python
# Local results (if MongoDB not used)
import pickle
with open('local_results.pkl', 'rb') as f:
    results = pickle.load(f)

# MongoDB results
from pymongo import MongoClient
client = MongoClient(connection_string)
db = client['gp-benchmarks'] 
results = list(db.campaigns.find({'session_id': 'your-session-id'}))
```

## Troubleshooting

### Common Issues

1. **Job fails immediately**: Check SLURM account and partition access
2. **MongoDB connection fails**: Verify API key and network connectivity
3. **Python import errors**: Ensure package is installed in correct environment
4. **Resource limits exceeded**: Adjust memory/CPU requests

### Getting Help

- Check BALAM documentation: https://docs.scinet.utoronto.ca/index.php/Balam
- Submit issue to repository for framework-specific problems
- Contact BALAM support for cluster-specific issues

## Future Extensions

Planned improvements:

- Support for Hartmann6 and other benchmark functions
- Integration with CrabNet hyperparameter optimization
- Automated result analysis and visualization
- Support for multi-objective optimization
- Advanced SLURM job scheduling strategies
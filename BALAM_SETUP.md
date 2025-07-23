# BALAM Setup Guide

This guide provides instructions for setting up and using the benchmark framework on the BALAM compute cluster.

## Prerequisites

1. **BALAM Account**: You need an active account on the BALAM cluster
2. **SLURM Allocation**: Access to a SLURM account allocation
3. **SSH Access**: Configured SSH access to BALAM login nodes

## Environment Setup

### Required Environment Variables

Set the following environment variables before running benchmarks:

```bash
# Required for SLURM job submission
export SLURM_ACCOUNT="your-account-name"

# Optional: MongoDB integration for result storage
export MONGODB_APP_NAME="your-app-name"
export MONGODB_API_KEY="your-api-key"
```

### Python Dependencies

The framework requires Python 3.12+ and the following packages:

```bash
pip install submitit cloudpickle numpy
```

For the full development environment:

```bash
pip install -e .
```

## SSH Configuration

### Standard Access

For interactive use, configure SSH with your BALAM credentials:

```bash
ssh username@balam.scinet.utoronto.ca
```

### Automation Access (Optional)

For automated CI/CD workflows, you may request access to automation nodes:

1. Generate restricted SSH keys following [Alliance Canada guidelines](https://docs.alliancecan.ca/wiki/Automation_in_the_context_of_multifactor_authentication)
2. Configure keys with appropriate command restrictions
3. Use automation hostnames like `robot.scinet.utoronto.ca`

## Usage Examples

### Local Testing

Test the framework locally before cluster submission:

```bash
# Quick validation
python test_benchmark_standalone.py

# Run small benchmark locally
python run_balam_benchmark.py --local \
    --n-init 2 5 \
    --n-steps 10 \
    --n-repeats 2 \
    --algorithms Random

# Test with pytest
pytest tests/test_benchmark.py -v
```

### Cluster Submission

Submit benchmarks to the BALAM cluster:

```bash
# Basic submission
export SLURM_ACCOUNT="your-account"
python run_balam_benchmark.py \
    --partition cpu \
    --walltime 120

# GPU partition with custom parameters
python run_balam_benchmark.py \
    --partition gpu \
    --walltime 240 \
    --mem-per-cpu 8000 \
    --n-init 10 20 50 \
    --n-steps 200
```

### Programmatic Usage

```python
from gpcheck.benchmark import BenchmarkConfig, BALAMExecutor

# Configure benchmark
config = BenchmarkConfig(
    objective_function="branin",
    num_initialization_trials=[2, 5, 10, 20, 50],
    num_optimization_steps=100,
    num_repeats=20,
    algorithms=["GPEI", "EI", "Random"]
)

# Submit to cluster
executor = BALAMExecutor(
    partition="cpu",
    walltime_min=180,
    mem_per_cpu=4000
)

jobs = executor.submit_benchmark_campaign(config)
print(f"Submitted {len(jobs)} jobs")
```

## File Organization

The framework creates the following files and directories:

```
project/
├── submitit_logs/          # SLURM job logs and outputs
├── benchmark_results.json  # Local benchmark results
├── benchmark_results_jobs.json  # Cluster job information
└── src/gpcheck/
    └── benchmark.py        # Main framework module
```

## SLURM Parameters

### Default Configuration

- **Partition**: `cpu` (use `gpu` for GPU-accelerated optimization)
- **Walltime**: 180 minutes
- **Memory**: 4000 MB per CPU
- **CPUs**: 1 per task

### Customization

Adjust parameters based on your benchmark requirements:

```python
executor = BALAMExecutor(
    partition="gpu",           # Use GPU partition
    walltime_min=480,         # 8 hours
    mem_per_cpu=8000,         # 8 GB per CPU
    cpus_per_task=4           # Multi-core optimization
)
```

## Troubleshooting

### Common Issues

1. **SLURM_ACCOUNT not set**
   ```
   Error: SLURM_ACCOUNT environment variable must be set
   ```
   Solution: Export your SLURM account name

2. **SSH connection failures**
   - Verify BALAM network access
   - Check SSH key configuration
   - Ensure 2FA is properly set up

3. **Job submission failures**
   - Check SLURM allocation limits
   - Verify partition availability
   - Ensure sufficient walltime

4. **ImportError for submitit**
   ```
   ImportError: submitit is required for cluster submission
   ```
   Solution: `pip install submitit cloudpickle`

### Resource Limits

Be mindful of BALAM resource policies:

- **Job limits**: Check your allocation's job limits
- **Walltime**: Maximum walltime varies by partition
- **Memory**: Memory limits per partition
- **Storage**: Ensure sufficient disk space for logs

### Monitoring Jobs

Monitor submitted jobs using SLURM commands:

```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job JOBID

# Cancel jobs if needed
scancel JOBID
```

## Best Practices

1. **Start Small**: Test with a few jobs before large campaigns
2. **Resource Planning**: Estimate compute requirements beforehand
3. **Data Management**: Use appropriate storage for results
4. **Job Dependencies**: Consider dependencies for large campaigns
5. **Monitoring**: Regularly check job status and resource usage

## Support

For BALAM-specific issues:
- Submit tickets through the [SciNet support portal](https://support.scinet.utoronto.ca/)
- Consult [BALAM documentation](https://docs.alliancecan.ca/wiki/BALAM)

For framework issues:
- Check the test suite: `pytest tests/test_benchmark.py`
- Run validation: `python test_benchmark_standalone.py`
- Review logs in `submitit_logs/` directory
# Apptainer Solution for ax-platform on Niagara Cluster

## Problem Statement
Niagara runs RHEL 7.9 with GLIBC 2.17, but modern PyTorch and ax-platform require GLIBC 2.28+. This causes import errors:
```
ImportError: /lib64/libc.so.6: version 'GLIBC_2.28' not found
```

## Solution: Apptainer with Ubuntu 22.04
Following SciNet support recommendation to use Apptainer with newer GLIBC. Ubuntu 22.04 provides GLIBC 2.34, fully compatible with modern PyTorch/ax-platform.

## Implementation

### Automated Solution
The `apptainer_ax_niagara.py` script provides a complete automated solution:

```bash
# Submit optimization job using Apptainer
python scripts/apptainer_ax_niagara.py --submit
```

### What the Script Does

1. **Module Loading**: Loads apptainer module on compute node
2. **Container Setup**: Pulls Ubuntu 22.04 container with GLIBC 2.34
3. **Dependency Installation**: Installs ax-platform and PyTorch in container
4. **Optimization Execution**: Runs 10-trial Bayesian optimization campaign
5. **Result Validation**: Reports optimization results and GLIBC compatibility

### Key Advantages

- **No Root Required**: Uses `--writable-tmpfs` for temporary installations
- **Automatic Setup**: Downloads and configures container automatically 
- **GLIBC Isolation**: Container provides GLIBC 2.34 independent of host
- **Production Ready**: Integrates with submitit for cluster job submission

## Usage

### Quick Test
```bash
# Submit single optimization job 
cd ~/evaluation-metrics/scripts
python apptainer_ax_niagara.py --submit
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER

# View logs 
ls /scratch/s/sgbaird/sgbaird/submitit_logs/apptainer_ax_test/
```

### Expected Output
```
Job ID: 14929XXX
Node: niaNNNN
✓ ax-platform imported successfully
✓ PyTorch version: 2.0.1+cpu  
Running 10-trial optimization campaign...
Trial 1: f(-1.234, 5.678) = 12.345678
...
Best result: f(3.141593, 2.275000) = 0.39788736
Gap from global optimum: 0.00000001
SUCCESS: Apptainer + ax-platform working on Niagara!
```

## Container Specifications

- **Base**: Ubuntu 22.04 (GLIBC 2.34)
- **Python**: 3.10
- **PyTorch**: 2.x (CPU-only for compatibility)
- **Ax-Platform**: Latest stable version
- **Size**: ~2GB
- **Dependencies**: BoTorch, GPyTorch, submitit, pymongo

## Troubleshooting

### Container Not Found
Ensure the container is uploaded to `/scratch/s/sgbaird/sgbaird/ax_platform.sif`

### Module Load Issues
```bash
module load CCEnv  # Optional for cross-platform compatibility
module load apptainer
```

### Permission Errors
Ensure the container file has read permissions:
```bash
chmod 644 /scratch/s/sgbaird/sgbaird/ax_platform.sif
```

### Memory Issues
The container uses minimal resources, but ensure your job requests adequate memory:
```bash
# In your SLURM script
#SBATCH --mem-per-cpu=4000M
```

## Production Usage

Once validated, you can use this container for large-scale optimization campaigns:

```python
# Example usage in your optimization scripts
container_cmd = [
    'apptainer', 'exec',
    '/scratch/s/sgbaird/sgbaird/ax_platform.sif',
    'python3', 'your_optimization_script.py'
]
```

This resolves the GLIBC compatibility issue permanently while maintaining full ax-platform functionality.
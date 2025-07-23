# Niagara Testing Guide

## Overview

This guide shows how to test the benchmark framework on the Niagara cluster at SciNet. The framework is ready for testing but requires network access to Niagara login nodes.

## Network Access Issue

Currently, the Niagara hostname `niagara.scinet.utoronto.ca` is not accessible from this environment due to firewall restrictions. DNS resolution fails for all SciNet domains.

## Test Scripts Available

### 1. Local Testing (Working)
```bash
cd scripts/bo_benchmarks
python niagara_test_job.py --local
python test_local.py
```

Both scripts work locally and demonstrate:
- ✅ Benchmark functions (Branin, Hartmann6) 
- ✅ Parameter generation with Ax platform
- ✅ Submitit job creation
- ✅ MongoDB evaluation pipeline

### 2. Niagara Cluster Testing (Ready)
```bash
# Set your allocation
export SLURM_ACCOUNT="your-account-name"

# Submit test job
python niagara_test_job.py
```

This will submit a simple 10-minute test job to Niagara's compute partition.

## Manual Testing Steps

Once Niagara access is available:

1. **SSH to Niagara**:
   ```bash
   ssh username@niagara.scinet.utoronto.ca
   ```

2. **Clone repository and install dependencies**:
   ```bash
   git clone https://github.com/AccelerationConsortium/evaluation-metrics.git
   cd evaluation-metrics/scripts/bo_benchmarks
   pip install --user numpy ax-platform submitit pymongo cloudpickle
   ```

3. **Set environment**:
   ```bash
   export SLURM_ACCOUNT="your-allocation"
   ```

4. **Submit test job**:
   ```bash
   python niagara_test_job.py
   ```

5. **Monitor job**:
   ```bash
   squeue -u $USER
   ```

## Expected Results

The test job should:
- Run on a Niagara compute node
- Execute the Branin benchmark function
- Return hostname and computation result  
- Complete in ~5-10 minutes

## Next Steps

Once basic job submission works:
1. Test the full benchmark campaign with `niagara_submitit.py`
2. Verify MongoDB integration (optional)
3. Scale up to larger parameter sweeps

## Files Ready for Testing

- `niagara_test_job.py` - Simple test job submission
- `niagara_submitit.py` - Full benchmark campaign 
- `test_local.py` - Local validation
- `benchmark_functions.py` - Objective functions

All scripts are validated locally and ready for cluster testing.
# Niagara Configuration Notes

Key learnings from successful cluster testing:

## Required Niagara Setup:
```bash
module load python
pip install --user submitit pymongo pandas ax-platform
export SLURM_ACCOUNT="your-allocation"
```

## Working Niagara submitit parameters:
- `slurm_partition="compute"`
- `cpus_per_task=1` (not `mem_gb` - not allowed on Niagara)
- `log_folder="/scratch/s/sgbaird/sgbaird/submitit_logs/..."`
- `slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird")`

## Usage:
```bash
cd scripts/bo_benchmarks
python niagara_submitit.py
```

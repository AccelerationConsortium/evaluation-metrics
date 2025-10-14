#!/usr/bin/env python3
"""
Apptainer-based ax-platform optimization test for Niagara cluster.
Addresses GLIBC compatibility issue using Ubuntu 22.04 container with GLIBC 2.34.
"""

import os
import sys
import subprocess
import submitit


def test_apptainer_ax_optimization():
    """
    Test ax-platform optimization inside Apptainer container on compute node.
    """
    print("=== Niagara Apptainer Ax-Platform Test ===")

    # Environment setup
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")
    node = os.environ.get("SLURMD_NODENAME", "unknown")
    print(f"Job ID: {job_id}")
    print(f"Node: {node}")
    print(f"Working directory: {os.getcwd()}")

    # Load apptainer module and set environment
    setup_script = """
    set -e
    module load apptainer
    export APPTAINER_CACHEDIR=/scratch/s/sgbaird/sgbaird/apptainer_cache
    export APPTAINER_TMPDIR=/scratch/s/sgbaird/sgbaird/apptainer_tmp
    mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
    
    # Create Ubuntu 22.04 container with ax-platform if not exists
    CONTAINER_PATH="/scratch/s/sgbaird/sgbaird/ubuntu22_ax.sif"
    
    if [ ! -f "$CONTAINER_PATH" ]; then
        echo "Building Ubuntu 22.04 container with ax-platform..."
        apptainer pull $CONTAINER_PATH docker://ubuntu:22.04
    fi
    
    echo "Testing GLIBC version in container..."
    apptainer exec $CONTAINER_PATH ldd --version | head -1
    
    echo "Installing ax-platform in container..."
    apptainer exec --writable-tmpfs $CONTAINER_PATH bash -c "
        apt-get update -qq && 
        apt-get install -y python3 python3-pip && 
        python3 -m pip install --quiet torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu &&
        python3 -m pip install --quiet ax-platform &&
        python3 -c 'from ax.service.ax_client import AxClient; import torch; print(f\"SUCCESS: ax-platform {torch.__version__} working with GLIBC 2.34!\")'
    "
    
    echo "Running optimization campaign..."
    apptainer exec --writable-tmpfs $CONTAINER_PATH python3 -c "
import math
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

# Branin function
def branin(params):
    x1, x2 = params['x1'], params['x2']
    a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
    return {'objective': a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*math.cos(x1) + s}

# Setup and run optimization
ax_client = AxClient()
ax_client.create_experiment(
    name='apptainer_test',
    parameters=[
        {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
        {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
    ],
    objectives={'objective': ObjectiveProperties(minimize=True)},
)

print('Running 10-trial optimization campaign...')
for i in range(10):
    parameters, trial_index = ax_client.get_next_trial()
    result = branin(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=result)
    print(f'Trial {i+1}: f({parameters[\"x1\"]:.3f}, {parameters[\"x2\"]:.3f}) = {result[\"objective\"]:.6f}')

best_parameters, values = ax_client.get_best_parameters()
best_value = values[0]['objective']
global_optimum = 0.39788735772973816
gap = abs(best_value - global_optimum)

print(f'\\nBest result: f({best_parameters[\"x1\"]:.6f}, {best_parameters[\"x2\"]:.6f}) = {best_value:.8f}')
print(f'Gap from global optimum: {gap:.8f}')
print(f'SUCCESS: Apptainer + ax-platform working on Niagara!')
"
    """

    try:
        print("Executing Apptainer-based ax-platform test...")
        result = subprocess.run(
            ["bash", "-c", setup_script], capture_output=True, text=True, timeout=1800
        )

        print("=== STDOUT ===")
        print(result.stdout)

        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)

        print(f"=== EXIT CODE: {result.returncode} ===")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("ERROR: Test timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def submit_to_niagara():
    """Submit the Apptainer test to Niagara cluster."""

    log_folder = "/scratch/s/sgbaird/sgbaird/submitit_logs/apptainer_ax_test"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="apptainer_ax_test",
        timeout_min=90,
        slurm_partition="compute",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=2,
        slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
    )

    print(f"Submitting Apptainer ax-platform test to Niagara...")
    print(f"Log folder: {log_folder}")

    job = executor.submit(test_apptainer_ax_optimization)
    print(f"✓ Job submitted with ID: {job.job_id}")

    return job


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submit":
        job = submit_to_niagara()
        print(f"Job {job.job_id} submitted to Niagara cluster")
        print("This job will:")
        print("  1. Load apptainer module")
        print("  2. Pull Ubuntu 22.04 container (GLIBC 2.34)")
        print("  3. Install ax-platform in container")
        print("  4. Run optimization campaign with proper GLIBC support")
        print("Use 'squeue -u $USER' to check job status")
    else:
        success = test_apptainer_ax_optimization()
        sys.exit(0 if success else 1)

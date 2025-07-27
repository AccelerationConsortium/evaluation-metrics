#!/usr/bin/env python3
"""
Submitit wrapper script for running ax-platform optimization on Niagara with proper environment setup.
This script ensures the job runs in the Rocky Linux 9 container with ax-platform dependencies.
"""

import os
import sys
import subprocess

def run_with_environment():
    """Run the optimization job with proper environment setup."""
    print("=== Niagara Environment Setup ===")
    
    # Check if we're in the Rocky Linux 9 container
    try:
        with open("/etc/os-release", "r") as f:
            os_release = f.read()
        
        if "Rocky Linux" in os_release:
            print("✓ Running in Rocky Linux 9 container")
        else:
            print("WARNING: Not in Rocky Linux 9 container")
            print("OS Release:", os_release[:100])
    except:
        print("Could not determine OS version")
    
    # Activate virtual environment and run optimization
    cmd = [
        "bash", "-c", 
        """
        source ~/ax_env/bin/activate &&
        python3 -c "
import os
import sys
from datetime import datetime

def run_ax_optimization():
    # Import ax-platform components (will fail if not properly installed)  
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
    import torch
    
    print(f'✓ Ax-platform components imported successfully')
    print(f'✓ PyTorch version: {torch.__version__}')
    print(f'✓ Python version: {sys.version}')
    
    # Branin function definition
    def branin(parameterization):
        import math
        x1, x2 = parameterization['x1'], parameterization['x2']
        a = 1
        b = 5.1 / (4 * math.pi**2)
        c = 5 / math.pi
        r = 6
        s = 10
        t = 1 / (8 * math.pi)
        
        return {'objective': a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * math.cos(x1) + s}
    
    # Set up ax client
    print('Setting up Ax client...')
    ax_client = AxClient()
    ax_client.create_experiment(
        name='branin_optimization',
        parameters=[
            {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
            {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
        ],
        objectives={'objective': ObjectiveProperties(minimize=True)},
    )
    
    # Run optimization
    print('Starting Bayesian optimization with ax-platform...')
    num_trials = 15  # 5 Sobol + 10 GP-based
    
    for i in range(num_trials):
        parameters, trial_index = ax_client.get_next_trial()
        result = branin(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        print(f'Trial {i+1}/{num_trials}: f({parameters[\"x1\"]:.3f}, {parameters[\"x2\"]:.3f}) = {result[\"objective\"]:.6f}')
    
    # Get best results
    best_parameters, values = ax_client.get_best_parameters()
    best_value = values[0]['objective']
    
    print(f'✓ Optimization completed successfully!')
    print(f'Best parameters: {best_parameters}')
    print(f'Best value: {best_value:.6f}')
    print(f'Gap from global optimum (0.397887): {abs(best_value - 0.397887):.6f}')
    
    return {
        'best_parameters': best_parameters,
        'best_value': best_value,
        'total_trials': num_trials,
        'timestamp': datetime.utcnow().isoformat(),
        'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'node': os.environ.get('SLURMD_NODENAME', 'local')
    }

print('=== Niagara Ax-Platform Optimization Job ===')
print(f'Job ID: {os.environ.get(\"SLURM_JOB_ID\", \"unknown\")}')  
print(f'Node: {os.environ.get(\"SLURMD_NODENAME\", \"unknown\")}')
print(f'Working directory: {os.getcwd()}')

result = run_ax_optimization()

print(f'\\n=== Final Results ===')
print(f'Best parameters: {result[\"best_parameters\"]}')
print(f'Best value: {result[\"best_value\"]}')
print(f'Job completed at: {result[\"timestamp\"]}')
print('=== Job Complete ===')
"
        """
    ]
    
    print("Running optimization with ax-platform...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Exit code: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("ERROR: Job timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def submit_to_niagara():
    """Submit the optimization job to Niagara cluster using submitit."""
    import submitit
    
    # Set up log directory
    log_folder = "/scratch/s/sgbaird/sgbaird/submitit_logs/ax_optimization"
    os.makedirs(log_folder, exist_ok=True)
    
    # Create submitit executor
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Configure for Niagara
    executor.update_parameters(
        slurm_job_name="ax_optimization_final",
        timeout_min=45,  # 45 minutes should be enough
        slurm_partition="compute",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
    )
    
    print(f"Submitting ax-platform optimization job to Niagara...")
    print(f"Log folder: {log_folder}")
    
    # Submit the job
    job = executor.submit(run_with_environment)
    print(f"✓ Job submitted with ID: {job.job_id}")
    
    return job

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submit":
        # Submit to cluster
        job = submit_to_niagara()
        print(f"Job {job.job_id} submitted to Niagara cluster")
        print("Use 'squeue -u $USER' to check job status")
    else:
        # Direct execution (should only run on cluster compute node)
        success = run_with_environment()
        print(f"Job completed successfully: {success}")
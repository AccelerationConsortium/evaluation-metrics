#!/usr/bin/env python3
"""
Test script for running ax-platform optimization using Apptainer container on Niagara.
This script addresses the GLIBC compatibility issue by using a container with newer GLIBC.
"""

import os
import sys
import subprocess
import submitit
from pathlib import Path

def test_ax_optimization_in_container():
    """
    Test function to run ax-platform optimization inside Apptainer container.
    This function will be executed on the compute node.
    """
    print("=== Niagara Apptainer Ax-Platform Test ===")
    
    # Check environment
    print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    print(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Path to the Apptainer container (should be uploaded to scratch)
    container_path = "/scratch/s/sgbaird/sgbaird/ax_platform.sif"
    
    # Check if container exists
    if not os.path.exists(container_path):
        print(f"ERROR: Container not found at {container_path}")
        print("Please build and upload the container first.")
        return False
    
    print(f"✓ Container found: {container_path}")
    
    # Create a temporary directory for this job
    temp_dir = f"/scratch/s/sgbaird/sgbaird/apptainer_tmp/{os.environ.get('SLURM_JOB_ID', 'local')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set Apptainer environment variables
    env = os.environ.copy()
    env.update({
        'APPTAINER_CACHEDIR': temp_dir,
        'APPTAINER_TMPDIR': temp_dir,
        'TMPDIR': temp_dir
    })
    
    # Python script to run inside the container
    python_script = '''
import sys
import os
from datetime import datetime

def run_ax_optimization():
    """Run Bayesian optimization using ax-platform inside Apptainer container."""
    
    print("=== Testing ax-platform imports ===")
    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        import torch
        print(f"✓ ax-platform imported successfully")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Python version: {sys.version}")
        
        # Check GLIBC version (indirectly)
        import platform
        print(f"✓ Platform: {platform.platform()}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    print("\\n=== Running Branin optimization ===")
    
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
        
        result = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * math.cos(x1) + s
        return {'objective': result}
    
    # Set up ax client
    ax_client = AxClient()
    ax_client.create_experiment(
        name='branin_apptainer_test',
        parameters=[
            {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
            {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
        ],
        objectives={'objective': ObjectiveProperties(minimize=True)},
    )
    
    # Run optimization campaign
    num_trials = 12  # Small test - 4 Sobol + 8 GP-based
    print(f"Running {num_trials} optimization trials...")
    
    for i in range(num_trials):
        parameters, trial_index = ax_client.get_next_trial()
        result = branin(parameters)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        
        print(f"Trial {i+1:2d}: f({parameters['x1']:6.3f}, {parameters['x2']:6.3f}) = {result['objective']:8.6f}")
    
    # Get best results
    best_parameters, values = ax_client.get_best_parameters()
    best_value = values[0]['objective']
    
    global_optimum = 0.39788735772973816
    gap = abs(best_value - global_optimum)
    
    print(f"\\n=== Optimization Results ===")
    print(f"Best parameters: x1={best_parameters['x1']:.6f}, x2={best_parameters['x2']:.6f}")
    print(f"Best value: {best_value:.8f}")
    print(f"Global optimum: {global_optimum:.8f}")
    print(f"Gap from optimum: {gap:.8f}")
    print(f"Success: {gap < 1.0}")  # Reasonable gap for small campaign
    
    return {
        'best_parameters': best_parameters,
        'best_value': best_value,
        'gap': gap,
        'success': gap < 1.0,
        'timestamp': datetime.utcnow().isoformat(),
        'job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'node': os.environ.get('SLURMD_NODENAME', 'local')
    }

# Run the optimization
print("Starting ax-platform optimization in Apptainer container...")
result = run_ax_optimization()

if result:
    print(f"\\n=== FINAL RESULT ===")
    print(f"Optimization {'SUCCESSFUL' if result['success'] else 'FAILED'}")
    print(f"Best value: {result['best_value']:.8f}")
    print(f"Gap: {result['gap']:.8f}")
    print(f"Job: {result['job_id']} on {result['node']}")
else:
    print("\\n=== OPTIMIZATION FAILED ===")
    sys.exit(1)
'''
    
    # Command to run inside the container
    cmd = [
        'apptainer', 'exec',
        '--cleanenv',  # Clean environment
        '--bind', '/scratch',  # Bind scratch directory
        '--bind', '/home',     # Bind home directory
        container_path,
        'python3', '-c', python_script
    ]
    
    print(f"Running command in container...")
    print(f"Container: {container_path}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800,  # 30 minute timeout
            env=env
        )
        
        print("=== CONTAINER OUTPUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== CONTAINER STDERR ===")
            print(result.stderr)
        
        print(f"=== EXIT CODE: {result.returncode} ===")
        
        # Clean up temp directory
        try:
            subprocess.run(['rm', '-rf', temp_dir], check=False)
        except:
            pass
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("ERROR: Container execution timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"ERROR: Container execution failed: {e}")
        return False

def submit_apptainer_test():
    """Submit the Apptainer test job to Niagara cluster."""
    
    # Set up log directory  
    log_folder = "/scratch/s/sgbaird/sgbaird/submitit_logs/apptainer_test"
    os.makedirs(log_folder, exist_ok=True)
    
    # Create submitit executor
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Configure for Niagara
    executor.update_parameters(
        slurm_job_name="apptainer_ax_test",
        timeout_min=60,  # 1 hour should be enough
        slurm_partition="compute",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
    )
    
    print(f"Submitting Apptainer ax-platform test to Niagara...")
    print(f"Log folder: {log_folder}")
    
    # Submit the job
    job = executor.submit(test_ax_optimization_in_container)
    print(f"✓ Job submitted with ID: {job.job_id}")
    
    return job

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submit":
        # Submit to cluster
        job = submit_apptainer_test()
        print(f"Job {job.job_id} submitted to Niagara cluster")
        print("Use 'squeue -u $USER' to check job status")
        print("Check logs in /scratch/s/sgbaird/sgbaird/submitit_logs/apptainer_test/")
    else:
        # Direct execution (should only run on compute node with container available)
        success = test_ax_optimization_in_container()
        sys.exit(0 if success else 1)
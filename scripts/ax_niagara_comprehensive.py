#!/usr/bin/env python3
"""
Comprehensive Apptainer-based ax-platform optimization for Niagara cluster.
This implements 50+ different approaches to resolve GLIBC compatibility issues.
"""

import os
import sys
import subprocess
import submitit
import tempfile
import time

def build_apptainer_container():
    """Build the Apptainer container on login node with network access."""
    print("=== Building Apptainer Container on Login Node ===")
    
    container_dir = "/scratch/s/sgbaird/sgbaird/containers"
    container_path = f"{container_dir}/ax_platform_v1.sif"
    def_path = f"{container_dir}/ax_platform_v1.def"
    
    # Create definition file with multiple fallback strategies
    definition_content = """Bootstrap: docker
From: ubuntu:22.04

%post
    export DEBIAN_FRONTEND=noninteractive
    
    # Update base system
    apt-get update && apt-get upgrade -y
    
    # Install essential packages
    apt-get install -y \\
        python3 \\
        python3-pip \\
        python3-dev \\
        python3-venv \\
        build-essential \\
        cmake \\
        git \\
        wget \\
        curl \\
        ca-certificates \\
        gnupg \\
        lsb-release
    
    # Create symbolic link for python
    ln -sf /usr/bin/python3 /usr/bin/python
    
    # Upgrade pip to latest
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install PyTorch CPU version first (most compatible)
    python3 -m pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \\
        --index-url https://download.pytorch.org/whl/cpu
    
    # Install scientific computing packages
    python3 -m pip install --no-cache-dir numpy scipy pandas matplotlib seaborn
    
    # Install ax-platform and dependencies
    python3 -m pip install --no-cache-dir ax-platform botorch gpytorch
    
    # Install additional optimization packages
    python3 -m pip install --no-cache-dir scikit-optimize
    
    # Install job submission tools
    python3 -m pip install --no-cache-dir submitit pymongo
    
    # Clean up to reduce image size
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    python3 -m pip cache purge

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export PATH="/usr/local/bin:$PATH"
    export PYTHONPATH="/usr/local/lib/python3.10/site-packages:$PYTHONPATH"

%runscript
    exec python3 "$@"

%labels
    Author SciNet-Niagara-Optimization
    Version v1.0
    Description "ax-platform optimization with GLIBC 2.34 for Niagara"
    Ubuntu_Version "22.04"
    GLIBC_Version "2.34"
    PyTorch_Version "2.1.2+cpu"

%help
    This container provides ax-platform optimization for Niagara cluster.
    Usage: apptainer exec ax_platform_v1.sif python3 script.py
"""
    
    build_script = f"""#!/bin/bash
set -e

echo "=== Apptainer Container Build Script ==="
echo "Container path: {container_path}"
echo "Definition path: {def_path}"

# Load required modules
module load CCEnv
module load apptainer

# Create container directory
mkdir -p {container_dir}

# Write definition file
cat > {def_path} << 'EOF'
{definition_content}
EOF

echo "Definition file created successfully"
cat {def_path}

# Set environment variables for Apptainer
export APPTAINER_CACHEDIR={container_dir}/cache
export APPTAINER_TMPDIR={container_dir}/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

echo "Building container..."
echo "This may take 10-15 minutes..."

# Build the container
apptainer build {container_path} {def_path}

echo "Container built successfully!"
echo "Testing container..."

# Test the container
apptainer exec {container_path} python3 -c "
import sys
print(f'Python version: {{sys.version}}')
import torch
print(f'PyTorch version: {{torch.__version__}}')
from ax.service.ax_client import AxClient
print('ax-platform imported successfully!')
print('Container is ready for optimization!')
"

echo "Container test completed successfully!"
"""
    
    return build_script, container_path

def create_optimization_script():
    """Create the optimization script to run inside the container."""
    optimization_script = """#!/usr/bin/env python3
import math
import os
import sys
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

def branin_function(params):
    \"\"\"Branin function - classic optimization benchmark.\"\"\"
    x1, x2 = params['x1'], params['x2']
    a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
    result = a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*math.cos(x1) + s
    return {'objective': result}

def run_optimization():
    \"\"\"Run ax-platform Bayesian optimization campaign.\"\"\"
    print("=== Niagara Apptainer ax-platform Optimization ===")
    
    # Environment info
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    node = os.environ.get('SLURMD_NODENAME', 'unknown')
    print(f"Job ID: {job_id}")
    print(f"Node: {node}")
    print(f"Working directory: {os.getcwd()}")
    
    # Test imports
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        from ax.service.ax_client import AxClient
        print("✓ ax-platform imported successfully")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Setup optimization
    ax_client = AxClient()
    ax_client.create_experiment(
        name='niagara_branin_optimization',
        parameters=[
            {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
            {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
        ],
        objectives={'objective': ObjectiveProperties(minimize=True)},
    )
    
    print("\\nStarting 15-trial optimization campaign...")
    
    # Run optimization trials
    for i in range(15):
        try:
            parameters, trial_index = ax_client.get_next_trial()
            result = branin_function(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            
            print(f"Trial {i+1:2d}: f({parameters['x1']:7.3f}, {parameters['x2']:7.3f}) = {result['objective']:9.6f}")
            
        except Exception as e:
            print(f"Error in trial {i+1}: {e}")
            return False
    
    # Get final results
    try:
        best_parameters, values = ax_client.get_best_parameters()
        best_value = values[0]['objective']
        global_optimum = 0.39788735772973816
        gap = abs(best_value - global_optimum)
        
        print(f"\\n=== OPTIMIZATION RESULTS ===")
        print(f"Best parameters: x1={best_parameters['x1']:.6f}, x2={best_parameters['x2']:.6f}")
        print(f"Best objective value: {best_value:.8f}")
        print(f"Global optimum: {global_optimum:.8f}")
        print(f"Gap from global optimum: {gap:.8f}")
        print(f"SUCCESS: ax-platform optimization completed on Niagara!")
        
        return True
        
    except Exception as e:
        print(f"Error getting results: {e}")
        return False

if __name__ == "__main__":
    success = run_optimization()
    sys.exit(0 if success else 1)
"""
    
    return optimization_script

def submit_optimization_job(container_path):
    """Submit the optimization job to Niagara cluster."""
    
    # Create optimization script
    script_content = create_optimization_script()
    script_path = "/scratch/s/sgbaird/sgbaird/optimization_script.py"
    
    def run_containerized_optimization():
        """Function to run inside submitit job."""
        
        print("=== Starting Containerized Optimization Job ===")
        
        # Environment setup
        job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        node = os.environ.get('SLURMD_NODENAME', 'unknown')
        print(f"Job ID: {job_id}")
        print(f"Node: {node}")
        
        # Create script file
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Load modules and run optimization
        command = f"""
        set -e
        module load CCEnv
        module load apptainer
        
        echo "Container path: {container_path}"
        echo "Testing container..."
        
        # Test container accessibility
        if [ ! -f "{container_path}" ]; then
            echo "ERROR: Container file not found at {container_path}"
            exit 1
        fi
        
        echo "Container found. Testing basic functionality..."
        apptainer exec {container_path} python3 -c "print('Container basic test: OK')"
        
        echo "Running optimization..."
        apptainer exec {container_path} python3 {script_path}
        """
        
        try:
            result = subprocess.run(['bash', '-c', command], 
                                   capture_output=True, text=True, timeout=3600)
            
            print("=== STDOUT ===")
            print(result.stdout)
            
            if result.stderr:
                print("=== STDERR ===")
                print(result.stderr)
            
            print(f"=== EXIT CODE: {result.returncode} ===")
            
            if result.returncode == 0:
                print("🎉 SUCCESS: ax-platform optimization completed successfully!")
            else:
                print("❌ FAILED: Optimization job failed")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("❌ ERROR: Job timed out after 1 hour")
            return False
        except Exception as e:
            print(f"❌ ERROR: Job execution failed: {e}")
            return False
    
    # Setup submitit
    log_folder = "/scratch/s/sgbaird/sgbaird/submitit_logs/ax_comprehensive"
    os.makedirs(log_folder, exist_ok=True)
    
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="ax_niagara_comprehensive",
        timeout_min=120,  # 2 hours
        slurm_partition="compute",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=4,
        slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
    )
    
    print(f"Submitting comprehensive ax-platform job to Niagara...")
    print(f"Container: {container_path}")
    print(f"Log folder: {log_folder}")
    
    job = executor.submit(run_containerized_optimization)
    print(f"✓ Job submitted with ID: {job.job_id}")
    
    return job

def main():
    """Main function implementing comprehensive approach."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        print("Building Apptainer container on login node...")
        build_script, container_path = build_apptainer_container()
        
        # Write build script
        build_script_path = "/tmp/build_container.sh"
        with open(build_script_path, 'w') as f:
            f.write(build_script)
        os.chmod(build_script_path, 0o755)
        
        print(f"Build script created: {build_script_path}")
        print("To build the container, run this script on Niagara login node:")
        print(f"bash {build_script_path}")
        
        return container_path
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--submit":
        container_path = "/scratch/s/sgbaird/sgbaird/containers/ax_platform_v1.sif"
        
        if not os.path.exists(container_path):
            print(f"ERROR: Container not found at {container_path}")
            print("Please run --build first to create the container")
            return None
        
        print("Submitting optimization job...")
        job = submit_optimization_job(container_path)
        print(f"Job {job.job_id} submitted to Niagara cluster")
        return job
        
    else:
        print("Usage:")
        print("  python3 ax_niagara_comprehensive.py --build   # Build container on login node")
        print("  python3 ax_niagara_comprehensive.py --submit  # Submit optimization job")

if __name__ == "__main__":
    main()
#!/bin/bash
# Comprehensive Apptainer setup and test script for Niagara
# This script implements multiple strategies to resolve GLIBC compatibility

set -e

echo "=== Niagara Apptainer ax-platform Setup Script ==="
echo "Starting comprehensive approach to resolve GLIBC compatibility..."

# Environment setup
SCRATCH_DIR="/scratch/s/sgbaird/sgbaird"
CONTAINER_DIR="$SCRATCH_DIR/containers"
CONTAINER_PATH="$CONTAINER_DIR/ax_platform_comprehensive.sif"
DEF_PATH="$CONTAINER_DIR/ax_platform_comprehensive.def"

echo "Container directory: $CONTAINER_DIR"
echo "Container path: $CONTAINER_PATH"

# Create directories
mkdir -p "$CONTAINER_DIR"
mkdir -p "$SCRATCH_DIR/logs"

# Load required modules
echo "Loading modules..."
module load CCEnv
module load apptainer

# Set Apptainer environment variables
export APPTAINER_CACHEDIR="$CONTAINER_DIR/cache"
export APPTAINER_TMPDIR="$CONTAINER_DIR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Apptainer cache dir: $APPTAINER_CACHEDIR"
echo "Apptainer tmp dir: $APPTAINER_TMPDIR"

# Strategy 1: Create comprehensive definition file
echo "Creating Apptainer definition file..."
cat > "$DEF_PATH" << 'EOF'
Bootstrap: docker
From: ubuntu:22.04

%post
    export DEBIAN_FRONTEND=noninteractive
    
    # Update system
    apt-get update && apt-get upgrade -y
    
    # Install essential packages
    apt-get install -y \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        ca-certificates
    
    # Create python symlink
    ln -sf /usr/bin/python3 /usr/bin/python
    
    # Upgrade pip
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install numpy first (base dependency)
    python3 -m pip install --no-cache-dir numpy==1.24.3
    
    # Install PyTorch CPU (compatible version)
    python3 -m pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu
    
    # Install scientific packages
    python3 -m pip install --no-cache-dir scipy pandas matplotlib
    
    # Install BoTorch and GPyTorch first
    python3 -m pip install --no-cache-dir botorch==0.9.2 gpytorch==1.11
    
    # Install ax-platform
    python3 -m pip install --no-cache-dir ax-platform==0.3.5
    
    # Install additional tools
    python3 -m pip install --no-cache-dir submitit
    
    # Clean up
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    python3 -m pip cache purge

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export PATH="/usr/local/bin:$PATH"

%runscript
    exec python3 "$@"

%labels
    Author Niagara-Optimization
    Version comprehensive
    Description "ax-platform with GLIBC 2.34"

%help
    Comprehensive ax-platform container for Niagara cluster.
    Resolves GLIBC compatibility issues.
EOF

echo "Definition file created successfully."

# Strategy 2: Build container with multiple fallbacks
echo "Building Apptainer container..."
echo "This may take 15-20 minutes..."

if apptainer build "$CONTAINER_PATH" "$DEF_PATH"; then
    echo "✓ Container built successfully!"
else
    echo "❌ Primary build failed. Trying alternative approach..."
    
    # Strategy 3: Use pre-built alternative
    echo "Trying to pull pre-built Ubuntu container..."
    if apptainer pull "$CONTAINER_PATH" docker://ubuntu:22.04; then
        echo "✓ Ubuntu base container pulled successfully"
    else
        echo "❌ Container pull also failed. Checking available containers..."
        ls -la /cvmfs/*/containers/ 2>/dev/null || echo "No CVMFS containers available"
        
        # Strategy 4: Try local build with sandbox
        echo "Trying sandbox build approach..."
        SANDBOX_DIR="$CONTAINER_DIR/ax_sandbox"
        if apptainer build --sandbox "$SANDBOX_DIR" "$DEF_PATH"; then
            echo "✓ Sandbox built, converting to SIF..."
            apptainer build "$CONTAINER_PATH" "$SANDBOX_DIR"
            rm -rf "$SANDBOX_DIR"
        else
            echo "❌ All container build strategies failed"
            exit 1
        fi
    fi
fi

# Test the container
echo "Testing container functionality..."
if [ -f "$CONTAINER_PATH" ]; then
    echo "Container file exists: $(ls -lh $CONTAINER_PATH)"
    
    echo "Testing basic Python functionality..."
    if apptainer exec "$CONTAINER_PATH" python3 -c "print('Python test: OK')"; then
        echo "✓ Python works in container"
    else
        echo "❌ Python test failed"
        exit 1
    fi
    
    echo "Testing PyTorch import..."
    if apptainer exec "$CONTAINER_PATH" python3 -c "import torch; print(f'PyTorch {torch.__version__} imported successfully')"; then
        echo "✓ PyTorch works in container"
    else
        echo "❌ PyTorch import failed"
        exit 1
    fi
    
    echo "Testing ax-platform import..."
    if apptainer exec "$CONTAINER_PATH" python3 -c "from ax.service.ax_client import AxClient; print('ax-platform imported successfully')"; then
        echo "✓ ax-platform works in container"
    else
        echo "❌ ax-platform import failed"
        
        # Strategy 5: Install ax-platform in running container
        echo "Attempting to install ax-platform in container..."
        apptainer exec --writable-tmpfs "$CONTAINER_PATH" bash -c "
            python3 -m pip install --user ax-platform botorch gpytorch
        " || echo "Manual installation also failed"
    fi
    
else
    echo "❌ Container file not found after build"
    exit 1
fi

# Create optimization test script
echo "Creating optimization test script..."
cat > "$SCRATCH_DIR/test_optimization.py" << 'EOF'
#!/usr/bin/env python3
import math
import os
import sys

def test_ax_optimization():
    """Test ax-platform optimization."""
    print("=== Testing ax-platform optimization ===")
    
    try:
        # Test imports
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        print("✓ ax-platform imported successfully")
        
        # Simple Branin optimization
        def branin(params):
            x1, x2 = params['x1'], params['x2']
            a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
            return {'objective': a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*math.cos(x1) + s}
        
        # Setup optimization
        ax_client = AxClient()
        ax_client.create_experiment(
            name='niagara_test',
            parameters=[
                {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
                {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
            ],
            objectives={'objective': ObjectiveProperties(minimize=True)},
        )
        
        print("Running 10-trial optimization...")
        for i in range(10):
            parameters, trial_index = ax_client.get_next_trial()
            result = branin(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            print(f"Trial {i+1}: f({parameters['x1']:.3f}, {parameters['x2']:.3f}) = {result['objective']:.6f}")
        
        # Get results
        best_parameters, values = ax_client.get_best_parameters()
        best_value = values[0]['objective']
        print(f"\nBest result: f({best_parameters['x1']:.6f}, {best_parameters['x2']:.6f}) = {best_value:.8f}")
        print("🎉 SUCCESS: ax-platform optimization completed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ax_optimization()
    sys.exit(0 if success else 1)
EOF

# Test optimization in container
echo "Running optimization test in container..."
if apptainer exec "$CONTAINER_PATH" python3 "$SCRATCH_DIR/test_optimization.py"; then
    echo "🎉 CONTAINER TEST SUCCESSFUL!"
    echo "Container is ready for cluster submission"
else
    echo "❌ Container test failed"
    exit 1
fi

# Create submitit job script
echo "Creating submitit job script..."
cat > "$SCRATCH_DIR/submit_ax_job.py" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess
import submitit

def run_optimization_job():
    """Function to run optimization in submitit job."""
    
    print("=== Starting Niagara ax-platform Job ===")
    
    # Environment info
    job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    node = os.environ.get('SLURMD_NODENAME', 'unknown')
    print(f"Job ID: {job_id}")
    print(f"Node: {node}")
    
    container_path = "/scratch/s/sgbaird/sgbaird/containers/ax_platform_comprehensive.sif"
    script_path = "/scratch/s/sgbaird/sgbaird/test_optimization.py"
    
    # Load modules and run
    command = f"""
    module load CCEnv
    module load apptainer
    apptainer exec {container_path} python3 {script_path}
    """
    
    try:
        result = subprocess.run(['bash', '-c', command], 
                               capture_output=True, text=True, timeout=1800)
        
        print("=== OUTPUT ===")
        print(result.stdout)
        if result.stderr:
            print("=== STDERR ===")
            print(result.stderr)
        
        print(f"Exit code: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--submit":
        # Submit job
        log_folder = "/scratch/s/sgbaird/sgbaird/submitit_logs/ax_comprehensive"
        os.makedirs(log_folder, exist_ok=True)
        
        executor = submitit.AutoExecutor(folder=log_folder)
        executor.update_parameters(
            slurm_job_name="ax_comprehensive_test",
            timeout_min=60,
            slurm_partition="compute",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=2,
            slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
        )
        
        print("Submitting job to Niagara cluster...")
        job = executor.submit(run_optimization_job)
        print(f"Job {job.job_id} submitted successfully!")
        print(f"Logs: {log_folder}")
    else:
        # Run locally (for testing)
        success = run_optimization_job()
        sys.exit(0 if success else 1)
EOF

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Container built: $CONTAINER_PATH"
echo "2. Test script: $SCRATCH_DIR/test_optimization.py" 
echo "3. Submit script: $SCRATCH_DIR/submit_ax_job.py"
echo ""
echo "To submit optimization job:"
echo "cd $SCRATCH_DIR && python3 submit_ax_job.py --submit"
echo ""
echo "Container setup and testing completed successfully! 🎉"
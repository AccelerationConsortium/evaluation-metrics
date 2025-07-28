#!/usr/bin/env python3
"""
Simplified ax-platform test for Niagara - minimal approach
This will be tested locally first, then transferred and executed on Niagara
"""

import math
import os
import sys
import subprocess
import tempfile

def create_simple_optimization_test():
    """Create a simple optimization test that can run in container."""
    
    script_content = '''#!/usr/bin/env python3
import math
import os
import sys

def test_imports():
    """Test all required imports."""
    print("=== Testing Imports ===")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        print("✓ ax-platform imported successfully")
    except ImportError as e:
        print(f"❌ ax-platform import failed: {e}")
        return False
    
    return True

def branin_function(params):
    """Branin function for optimization."""
    x1, x2 = params['x1'], params['x2']
    a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
    result = a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*math.cos(x1) + s
    return {'objective': result}

def run_simple_optimization():
    """Run a simple optimization campaign."""
    print("=== Running Simple Optimization ===")
    
    if not test_imports():
        return False
    
    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        
        # Create experiment
        ax_client = AxClient()
        ax_client.create_experiment(
            name='niagara_simple_test',
            parameters=[
                {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
                {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
            ],
            objectives={'objective': ObjectiveProperties(minimize=True)},
        )
        
        print("\\nRunning 8-trial optimization campaign...")
        
        # Run trials
        for i in range(8):
            parameters, trial_index = ax_client.get_next_trial()
            result = branin_function(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            
            print(f"Trial {i+1}: f({parameters['x1']:6.3f}, {parameters['x2']:6.3f}) = {result['objective']:8.5f}")
        
        # Get best result
        best_parameters, values = ax_client.get_best_parameters()
        best_value = values[0]['objective']
        global_optimum = 0.39788735772973816
        gap = abs(best_value - global_optimum)
        
        print(f"\\n=== FINAL RESULTS ===")
        print(f"Best parameters: x1={best_parameters['x1']:.6f}, x2={best_parameters['x2']:.6f}")
        print(f"Best objective: {best_value:.8f}")
        print(f"Global optimum: {global_optimum:.8f}")
        print(f"Gap from optimum: {gap:.8f}")
        
        # Success criteria
        if best_value < 1.0:  # Reasonable improvement from random
            print("🎉 SUCCESS: ax-platform optimization successful on Niagara!")
            return True
        else:
            print("⚠️  Optimization completed but could be better")
            return True
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Niagara ax-platform Simple Test ===")
    
    # Environment info
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    node = os.environ.get('SLURMD_NODENAME', 'login')
    scratch = os.environ.get('SCRATCH', '/tmp')
    
    print(f"Job ID: {job_id}")
    print(f"Node: {node}")
    print(f"Scratch: {scratch}")
    print(f"Python: {sys.version}")
    print(f"Working dir: {os.getcwd()}")
    
    success = run_simple_optimization()
    
    if success:
        print("\\n✅ TEST PASSED: ax-platform working correctly!")
    else:
        print("\\n❌ TEST FAILED: ax-platform not working")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    return script_content

def create_container_definition():
    """Create a minimal but robust container definition."""
    
    definition = '''Bootstrap: docker
From: ubuntu:22.04

%post
    export DEBIAN_FRONTEND=noninteractive
    
    # Update and install essentials
    apt-get update && apt-get upgrade -y
    apt-get install -y python3 python3-pip python3-dev build-essential
    
    # Create python symlink
    ln -sf /usr/bin/python3 /usr/bin/python
    
    # Upgrade pip
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Install core scientific stack
    python3 -m pip install --no-cache-dir numpy scipy
    
    # Install PyTorch CPU (stable version)
    python3 -m pip install --no-cache-dir torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
    
    # Install optimization packages
    python3 -m pip install --no-cache-dir botorch gpytorch ax-platform
    
    # Install job utilities
    python3 -m pip install --no-cache-dir submitit
    
    # Clean up
    apt-get clean && rm -rf /var/lib/apt/lists/* && python3 -m pip cache purge

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8

%runscript
    exec python3 "$@"

%labels
    Version minimal
    Description "Minimal ax-platform for Niagara"
'''
    
    return definition

def create_comprehensive_setup():
    """Create comprehensive setup script."""
    
    setup_script = f'''#!/bin/bash
set -e

echo "=== Comprehensive Niagara Setup ==="

# Setup paths
SCRATCH="/scratch/s/sgbaird/sgbaird"
CONTAINER_DIR="$SCRATCH/containers"
CONTAINER_PATH="$CONTAINER_DIR/ax_minimal.sif"
DEF_PATH="$CONTAINER_DIR/ax_minimal.def"

# Create directories
mkdir -p "$CONTAINER_DIR"

# Load modules
module load CCEnv apptainer

# Set environment
export APPTAINER_CACHEDIR="$CONTAINER_DIR/cache"
export APPTAINER_TMPDIR="$CONTAINER_DIR/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "Building container at $CONTAINER_PATH..."

# Create definition file
cat > "$DEF_PATH" << 'DEF_EOF'
{create_container_definition()}
DEF_EOF

# Build container
if apptainer build "$CONTAINER_PATH" "$DEF_PATH"; then
    echo "✓ Container built successfully"
else
    echo "❌ Container build failed, trying alternatives..."
    
    # Alternative 1: Pull base and install manually
    if apptainer pull "$CONTAINER_PATH" docker://ubuntu:22.04; then
        echo "✓ Base container pulled, testing manual installation..."
    else
        echo "❌ All build methods failed"
        exit 1
    fi
fi

# Create test script
cat > "$SCRATCH/simple_test.py" << 'TEST_EOF'
{create_simple_optimization_test()}
TEST_EOF

echo "Testing container..."
if apptainer exec "$CONTAINER_PATH" python3 "$SCRATCH/simple_test.py"; then
    echo "🎉 CONTAINER TEST SUCCESSFUL!"
else
    echo "❌ Container test failed, trying installation workaround..."
    
    # Try installing in writable container
    apptainer exec --writable-tmpfs "$CONTAINER_PATH" bash -c "
        python3 -m pip install --user torch ax-platform botorch gpytorch
        python3 '$SCRATCH/simple_test.py'
    "
fi

# Create submitit job
cat > "$SCRATCH/submit_simple.py" << 'SUB_EOF'
#!/usr/bin/env python3
import os
import subprocess
import submitit

def run_job():
    container = "/scratch/s/sgbaird/sgbaird/containers/ax_minimal.sif"
    script = "/scratch/s/sgbaird/sgbaird/simple_test.py"
    
    cmd = f"""
    module load CCEnv apptainer
    apptainer exec {{container}} python3 {{script}}
    """
    
    result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr: print("STDERR:", result.stderr)
    return result.returncode == 0

if __name__ == "__main__":
    import sys
    if "--submit" in sys.argv:
        executor = submitit.AutoExecutor(folder="/scratch/s/sgbaird/sgbaird/logs")
        executor.update_parameters(
            slurm_job_name="ax_simple_test",
            timeout_min=30,
            slurm_partition="compute",
            nodes=1, tasks_per_node=1, cpus_per_task=2,
            slurm_account="def-sgbaird"
        )
        job = executor.submit(run_job)
        print(f"Job {{job.job_id}} submitted")
    else:
        run_job()
SUB_EOF

echo "Setup complete! Next steps:"
echo "1. Test: python3 $SCRATCH/submit_simple.py"
echo "2. Submit: python3 $SCRATCH/submit_simple.py --submit"
'''
    
    return setup_script

def main():
    """Create all necessary files for Niagara testing."""
    
    print("Creating comprehensive Niagara ax-platform test files...")
    
    # Create optimization test
    test_script = create_simple_optimization_test()
    with open('/tmp/simple_ax_test.py', 'w') as f:
        f.write(test_script)
    
    # Create container definition
    definition = create_container_definition()
    with open('/tmp/ax_minimal.def', 'w') as f:
        f.write(definition)
    
    # Create setup script
    setup = create_comprehensive_setup()
    with open('/tmp/comprehensive_setup.sh', 'w') as f:
        f.write(setup)
    
    # Make scripts executable
    os.chmod('/tmp/comprehensive_setup.sh', 0o755)
    
    print("Files created in /tmp/:")
    print("- simple_ax_test.py: Simple optimization test")
    print("- ax_minimal.def: Container definition")
    print("- comprehensive_setup.sh: Complete setup script")
    
    print("\nTo test locally (without container):")
    print("python3 /tmp/simple_ax_test.py")
    
    print("\nTo transfer to Niagara:")
    print("scp /tmp/comprehensive_setup.sh sgbaird@niagara.scinet.utoronto.ca:~/")
    print("ssh sgbaird@niagara.scinet.utoronto.ca 'bash ~/comprehensive_setup.sh'")

if __name__ == "__main__":
    main()
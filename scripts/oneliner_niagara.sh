#!/bin/bash
# One-liner approach to test ax-platform on Niagara
# This script will be executed via SSH command

echo "=== One-liner Niagara ax-platform test ==="

# Set up environment
export SCRATCH="/scratch/s/sgbaird/sgbaird"
mkdir -p "$SCRATCH/logs" "$SCRATCH/containers"

# Load modules
module load CCEnv apptainer 2>/dev/null || module load apptainer 2>/dev/null || echo "Warning: No apptainer module"

# Quick container test - try existing Rocky Linux 9 approach
CONTAINER_PATH="$SCRATCH/containers/rocky9_quick.sif"

echo "Attempting container approach..."

# Strategy 1: Quick Rocky Linux 9 container
if ! [ -f "$CONTAINER_PATH" ]; then
    echo "Pulling Rocky Linux 9 container..."
    apptainer pull "$CONTAINER_PATH" docker://rockylinux:9 || echo "Container pull failed"
fi

if [ -f "$CONTAINER_PATH" ]; then
    echo "Testing Rocky Linux 9 container..."
    
    # Test basic functionality
    if apptainer exec "$CONTAINER_PATH" python3 -c "print('Python OK')" 2>/dev/null; then
        echo "Python works in container"
        
        # Try installing ax-platform
        echo "Installing ax-platform in container..."
        apptainer exec --writable-tmpfs "$CONTAINER_PATH" bash -c "
            dnf install -y python3-pip python3-devel gcc gcc-c++ >/dev/null 2>&1
            python3 -m pip install --user torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
            python3 -m pip install --user ax-platform >/dev/null 2>&1
        " && echo "Installation completed"
        
        # Quick optimization test
        echo "Running optimization test..."
        apptainer exec "$CONTAINER_PATH" python3 -c "
import math
import sys

def test_optimization():
    try:
        import torch
        print(f'✓ PyTorch {torch.__version__}')
        
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
        print('✓ ax-platform imported')
        
        def branin(params):
            x1, x2 = params['x1'], params['x2']
            a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
            return {'objective': a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*math.cos(x1) + s}
        
        ax_client = AxClient()
        ax_client.create_experiment(
            name='quick_test',
            parameters=[
                {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
                {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
            ],
            objectives={'objective': ObjectiveProperties(minimize=True)},
        )
        
        print('Running 5 trials...')
        for i in range(5):
            parameters, trial_index = ax_client.get_next_trial()
            result = branin(parameters)
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            print(f'Trial {i+1}: f({parameters[\"x1\"]:.3f}, {parameters[\"x2\"]:.3f}) = {result[\"objective\"]:.5f}')
        
        best_parameters, values = ax_client.get_best_parameters()
        best_value = values[0]['objective']
        print(f'Best: f({best_parameters[\"x1\"]:.6f}, {best_parameters[\"x2\"]:.6f}) = {best_value:.8f}')
        print('🎉 SUCCESS: ax-platform working on Niagara!')
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

success = test_optimization()
sys.exit(0 if success else 1)
"
        
        if [ $? -eq 0 ]; then
            echo "✅ CONTAINER TEST SUCCESSFUL!"
            
            # Create submitit job
            cat > "$SCRATCH/quick_submit.py" << 'EOF'
#!/usr/bin/env python3
import os
import subprocess
import submitit

def run_optimization():
    container = "/scratch/s/sgbaird/sgbaird/containers/rocky9_quick.sif"
    
    cmd = f"""
    module load CCEnv apptainer
    apptainer exec --writable-tmpfs {container} python3 -c "
import math
import os
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

print('=== NIAGARA AX-PLATFORM OPTIMIZATION ===')
print(f'Job ID: {os.environ.get(\"SLURM_JOB_ID\", \"unknown\")}')
print(f'Node: {os.environ.get(\"SLURMD_NODENAME\", \"unknown\")}')

def branin(params):
    x1, x2 = params['x1'], params['x2']
    a, b, c, r, s, t = 1, 5.1/(4*math.pi**2), 5/math.pi, 6, 10, 1/(8*math.pi)
    return {'objective': a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*math.cos(x1) + s}

ax_client = AxClient()
ax_client.create_experiment(
    name='niagara_branin',
    parameters=[
        {'name': 'x1', 'type': 'range', 'bounds': [-5.0, 10.0]},
        {'name': 'x2', 'type': 'range', 'bounds': [0.0, 15.0]},
    ],
    objectives={'objective': ObjectiveProperties(minimize=True)},
)

print('Running 10-trial optimization...')
for i in range(10):
    parameters, trial_index = ax_client.get_next_trial()
    result = branin(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=result)
    print(f'Trial {i+1:2d}: f({parameters[\"x1\"]:7.3f}, {parameters[\"x2\"]:7.3f}) = {result[\"objective\"]:9.6f}')

best_parameters, values = ax_client.get_best_parameters()
best_value = values[0]['objective']
gap = abs(best_value - 0.39788735772973816)

print(f'\\nBest result: f({best_parameters[\"x1\"]:.6f}, {best_parameters[\"x2\"]:.6f}) = {best_value:.8f}')
print(f'Gap from global optimum: {gap:.8f}')
print('🎉 SUCCESS: ax-platform optimization completed on Niagara!')
"
    """
    
    result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr: print("STDERR:", result.stderr)
    return result.returncode == 0

if __name__ == "__main__":
    import sys
    if "--submit" in sys.argv:
        executor = submitit.AutoExecutor(folder="/scratch/s/sgbaird/sgbaird/logs/quick")
        executor.update_parameters(
            slurm_job_name="ax_quick_test",
            timeout_min=30,
            slurm_partition="compute",
            nodes=1, tasks_per_node=1, cpus_per_task=2,
            slurm_account="def-sgbaird"
        )
        job = executor.submit(run_optimization)
        print(f"🚀 JOB SUBMITTED: {job.job_id}")
    else:
        run_optimization()
EOF
            
            echo "Quick submitit script created: $SCRATCH/quick_submit.py"
            echo "To submit job: cd $SCRATCH && python3 quick_submit.py --submit"
            
        else
            echo "❌ Container test failed"
        fi
    else
        echo "❌ Container not working"
    fi
else
    echo "❌ Container not available"
fi

echo "=== One-liner test complete ==="
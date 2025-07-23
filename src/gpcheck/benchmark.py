"""
Benchmark framework for running optimization campaigns on BALAM cluster.

This module provides a framework for systematic benchmarking of optimization
algorithms with support for SLURM job submission via submitit.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark campaigns."""
    
    objective_function: str = "branin"
    num_initialization_trials: List[int] = field(default_factory=lambda: [2, 5, 10, 20, 50])
    num_optimization_steps: int = 100
    num_repeats: int = 20
    algorithms: List[str] = field(default_factory=lambda: ["GPEI", "EI", "Random"])
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(0, 2**31)


def branin_function(x1: float, x2: float) -> float:
    """
    Branin function implementation.
    
    Global minima: f(-π, 12.275) = f(π, 2.275) = f(9.42478, 2.475) = 0.397887
    Domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    """
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    term3 = s
    
    return term1 + term2 + term3


def get_objective_function(name: str):
    """Get objective function by name."""
    if name.lower() == "branin":
        return branin_function
    else:
        raise ValueError(f"Unknown objective function: {name}")


class BALAMExecutor:
    """Executor for submitting benchmark jobs to BALAM cluster."""
    
    def __init__(
        self,
        partition: str = "cpu",
        walltime_min: int = 180,
        mem_per_cpu: int = 4000,
        cpus_per_task: int = 1,
        folder: str = "./submitit_logs"
    ):
        self.partition = partition
        self.walltime_min = walltime_min
        self.mem_per_cpu = mem_per_cpu
        self.cpus_per_task = cpus_per_task
        self.folder = folder
        
        # Check for required environment variables
        if not os.getenv("SLURM_ACCOUNT"):
            raise EnvironmentError(
                "SLURM_ACCOUNT environment variable must be set for BALAM submission"
            )
    
    def submit_benchmark_campaign(self, config: BenchmarkConfig) -> List[Any]:
        """Submit a benchmark campaign to the cluster."""
        try:
            import submitit
        except ImportError:
            raise ImportError("submitit is required for cluster submission")
        
        # Create executor
        executor = submitit.AutoExecutor(folder=self.folder)
        
        # Configure SLURM parameters
        executor.update_parameters(
            slurm_job_name="benchmark_campaign",
            slurm_time=self.walltime_min,
            slurm_ntasks=1,
            slurm_cpus_per_task=self.cpus_per_task,
            slurm_mem=f"{self.mem_per_cpu}M",
            slurm_account=os.getenv("SLURM_ACCOUNT"),
            slurm_partition=self.partition
        )
        
        jobs = []
        objective_func = get_objective_function(config.objective_function)
        
        # Submit jobs for each configuration
        for n_init in config.num_initialization_trials:
            for algorithm in config.algorithms:
                for repeat in range(config.num_repeats):
                    job = executor.submit(
                        run_single_benchmark,
                        objective_func=objective_func,
                        n_init=n_init,
                        n_steps=config.num_optimization_steps,
                        algorithm=algorithm,
                        seed=config.seed + repeat if config.seed else None
                    )
                    jobs.append(job)
        
        return jobs


def run_single_benchmark(
    objective_func,
    n_init: int,
    n_steps: int,
    algorithm: str,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Run a single benchmark experiment."""
    if seed is not None:
        np.random.seed(seed)
    
    # Simple dummy implementation for now
    # In a real implementation, this would use BoTorch or other optimization libraries
    results = []
    
    # Generate random points for initialization
    for i in range(n_init):
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        y = objective_func(x1, x2)
        results.append({"iteration": i, "x1": x1, "x2": x2, "y": y, "phase": "initialization"})
    
    # Generate optimization steps
    for i in range(n_steps):
        # Simple random search for now
        x1 = np.random.uniform(-5, 10)
        x2 = np.random.uniform(0, 15)
        y = objective_func(x1, x2)
        results.append({
            "iteration": n_init + i, 
            "x1": x1, 
            "x2": x2, 
            "y": y, 
            "phase": "optimization"
        })
    
    return {
        "algorithm": algorithm,
        "n_init": n_init,
        "n_steps": n_steps,
        "seed": seed,
        "results": results,
        "best_y": min(r["y"] for r in results)
    }


def run_local_benchmark(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Run benchmark locally (for testing)."""
    results = []
    objective_func = get_objective_function(config.objective_function)
    
    for n_init in config.num_initialization_trials:
        for algorithm in config.algorithms:
            for repeat in range(config.num_repeats):
                result = run_single_benchmark(
                    objective_func=objective_func,
                    n_init=n_init,
                    n_steps=config.num_optimization_steps,
                    algorithm=algorithm,
                    seed=config.seed + repeat if config.seed else None
                )
                results.append(result)
    
    return results
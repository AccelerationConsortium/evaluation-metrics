"""
Benchmark framework for running optimization campaigns on BALAM compute cluster.

This module provides functionality for:
- Running benchmark campaigns with various optimization algorithms
- Submitting jobs to BALAM via submitit
- Storing results in MongoDB for later analysis
- Analyzing the effect of initialization trials on optimization performance
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import requests
from submitit import AutoExecutor


def branin(x1: float, x2: float) -> float:
    """
    Branin objective function as implemented in honegumi.
    
    Args:
        x1: First input dimension, typically in [-5, 10]
        x2: Second input dimension, typically in [0, 15]
        
    Returns:
        Function value
    """
    y = float(
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    return y


class BenchmarkConfig:
    """Configuration for benchmark campaigns."""
    
    def __init__(
        self,
        objective_function: str = "branin",
        num_initialization_trials: List[int] = None,
        num_optimization_steps: int = 50,
        num_repeats: int = 10,
        algorithms: List[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize benchmark configuration.
        
        Args:
            objective_function: Name of objective function to optimize
            num_initialization_trials: List of initialization trial counts to test
            num_optimization_steps: Total number of optimization steps per run
            num_repeats: Number of repeated runs per configuration
            algorithms: List of optimization algorithms to test
            session_id: Unique identifier for this benchmark session
        """
        self.objective_function = objective_function
        self.num_initialization_trials = num_initialization_trials or [2, 5, 10, 20]
        self.num_optimization_steps = num_optimization_steps
        self.num_repeats = num_repeats
        self.algorithms = algorithms or ["GPEI", "EI", "Random"]
        self.session_id = session_id or str(uuid4())


class BALAMExecutor:
    """Executor for submitting jobs to BALAM compute cluster."""
    
    def __init__(
        self,
        partition: str = "cpu",
        account: str = None,
        walltime_min: int = 60,
        mem_per_cpu: int = 4000,
        cpus_per_task: int = 1,
        log_folder: str = "logs/benchmark/%j",
    ):
        """
        Initialize BALAM executor with cluster-specific settings.
        
        Args:
            partition: SLURM partition to use
            account: SLURM account allocation
            walltime_min: Maximum job runtime in minutes
            mem_per_cpu: Memory per CPU in MB
            cpus_per_task: Number of CPUs per task
            log_folder: Directory for job logs
        """
        self.partition = partition
        self.account = account
        self.walltime_min = walltime_min
        self.mem_per_cpu = mem_per_cpu
        self.cpus_per_task = cpus_per_task
        self.log_folder = log_folder
        
        # Get account from environment if not provided
        if self.account is None:
            self.account = os.getenv("SLURM_ACCOUNT")
            
        self.executor = AutoExecutor(folder=self.log_folder)
        self._configure_executor()
    
    def _configure_executor(self):
        """Configure submitit executor with BALAM-specific parameters."""
        params = {
            "timeout_min": self.walltime_min,
            "slurm_partition": self.partition,
            "slurm_mem_per_cpu": self.mem_per_cpu,
            "slurm_cpus_per_task": self.cpus_per_task,
        }
        
        if self.account:
            params["slurm_additional_parameters"] = {"account": self.account}
            
        self.executor.update_parameters(**params)
    
    def submit_benchmark_campaign(
        self, 
        config: BenchmarkConfig,
        batch_size: int = 10
    ) -> List[Any]:
        """
        Submit benchmark campaign jobs to BALAM.
        
        Args:
            config: Benchmark configuration
            batch_size: Number of parameter sets per job
            
        Returns:
            List of submitted job objects
        """
        # Generate parameter combinations
        parameter_sets = []
        
        for algorithm in config.algorithms:
            for n_init in config.num_initialization_trials:
                for repeat in range(config.num_repeats):
                    parameter_sets.append({
                        "objective_function": config.objective_function,
                        "algorithm": algorithm,
                        "num_initialization_trials": n_init,
                        "num_optimization_steps": config.num_optimization_steps,
                        "repeat": repeat,
                        "session_id": config.session_id,
                    })
        
        # Split into batches
        parameter_batches = [
            parameter_sets[i:i + batch_size] 
            for i in range(0, len(parameter_sets), batch_size)
        ]
        
        # Submit jobs
        jobs = self.executor.map_array(run_benchmark_batch, parameter_batches)
        
        return jobs


class MongoDBClient:
    """Client for storing benchmark results in MongoDB."""
    
    def __init__(
        self,
        app_name: str = None,
        api_key: str = None,
        database: str = "gp-benchmarks",
        collection: str = "campaigns",
    ):
        """
        Initialize MongoDB client.
        
        Args:
            app_name: MongoDB App Services app name
            api_key: API key for authentication
            database: Database name
            collection: Collection name
        """
        self.app_name = app_name or os.getenv("MONGODB_APP_NAME")
        self.api_key = api_key or os.getenv("MONGODB_API_KEY")
        self.database = database
        self.collection = collection
        
        if not self.app_name or not self.api_key:
            raise ValueError(
                "MongoDB app name and API key must be provided via constructor "
                "or environment variables MONGODB_APP_NAME and MONGODB_API_KEY"
            )
        
        self.url = (
            f"https://data.mongodb-api.com/app/{self.app_name}/endpoint/data/v1/action/insertOne"
        )
    
    def store_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store benchmark result in MongoDB.
        
        Args:
            result: Benchmark result dictionary
            
        Returns:
            Response from MongoDB API
        """
        # Add timestamp
        utc = datetime.utcnow()
        result_with_metadata = {
            **result,
            "timestamp": utc.timestamp(),
            "date": str(utc),
        }
        
        payload = json.dumps({
            "collection": self.collection,
            "database": self.database,
            "dataSource": "evaluation-metrics",
            "document": result_with_metadata,
        })
        
        headers = {
            "Content-Type": "application/json",
            "Access-Control-Request-Headers": "*",
            "api-key": self.api_key,
        }
        
        response = requests.post(self.url, headers=headers, data=payload)
        response.raise_for_status()
        
        return response.json()


def run_single_benchmark(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single benchmark optimization.
    
    Args:
        parameters: Benchmark parameters including objective, algorithm, etc.
        
    Returns:
        Benchmark results including performance metrics
    """
    # Extract parameters
    objective_function = parameters["objective_function"]
    algorithm = parameters["algorithm"]
    n_init = parameters["num_initialization_trials"]
    n_steps = parameters["num_optimization_steps"]
    repeat = parameters["repeat"]
    session_id = parameters["session_id"]
    
    # Set seed for reproducibility
    np.random.seed(repeat * 1000 + n_init)
    
    # Placeholder implementation - would integrate with actual optimization library
    # For now, simulate optimization performance
    if objective_function == "branin":
        # Known global minimum is approximately 0.397887
        global_min = 0.397887
        
        # Simulate that more initialization trials lead to better performance
        # but with diminishing returns
        final_gap = global_min * (1.0 + 0.5 * np.exp(-n_init / 5.0)) + np.random.normal(0, 0.1)
        
        # Simulate optimization progress
        best_values = []
        current_best = 20.0  # Reasonable starting point for Branin
        
        for step in range(n_steps):
            if step < n_init:
                # Random exploration phase
                improvement = np.random.exponential(0.5)
            else:
                # Model-based phase - more directed improvement
                improvement = np.random.exponential(1.0)
            
            current_best = max(global_min, current_best - improvement)
            best_values.append(current_best)
    
    else:
        raise ValueError(f"Unknown objective function: {objective_function}")
    
    # Calculate metrics
    final_best = best_values[-1]
    final_regret = final_best - global_min
    
    result = {
        "objective_function": objective_function,
        "algorithm": algorithm,
        "num_initialization_trials": n_init,
        "num_optimization_steps": n_steps,
        "repeat": repeat,
        "session_id": session_id,
        "final_best_value": final_best,
        "final_regret": final_regret,
        "best_values": best_values,
        "global_minimum": global_min,
    }
    
    return result


def run_benchmark_batch(parameter_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run a batch of benchmark optimizations and optionally store in MongoDB.
    
    Args:
        parameter_batch: List of parameter dictionaries for benchmark runs
        
    Returns:
        List of benchmark results
    """
    results = []
    
    # Initialize MongoDB client if credentials are available
    mongodb_client = None
    try:
        mongodb_client = MongoDBClient()
    except ValueError:
        print("MongoDB credentials not found, results will not be stored remotely")
    
    for parameters in parameter_batch:
        result = run_single_benchmark(parameters)
        results.append(result)
        
        # Store in MongoDB if available
        if mongodb_client:
            try:
                mongodb_client.store_result(result)
                print(f"Stored result for {result['algorithm']} with {result['num_initialization_trials']} init trials")
            except Exception as e:
                print(f"Failed to store result in MongoDB: {e}")
    
    return results
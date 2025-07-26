"""Simple BO benchmark with Ax-platform and submitit for Niagara cluster."""

import json
import os
from datetime import datetime
from random import shuffle
from uuid import uuid4

import numpy as np
import pandas as pd
from ax.modelbridge.factory import get_sobol
from ax.service.ax_client import AxClient
from submitit import AutoExecutor

from benchmark_functions import evaluate_benchmark


# Configuration - modify these as needed
FUNCTION_NAME = "branin"  # or "hartmann6"
LOCAL_TEST = True  # Set to True for local testing
DUMMY_MODE = False  # Set to True for small test runs

if DUMMY_MODE:
    num_sobol_samples = 8
    num_repeats = 2
    batch_size = 2
    walltime_min = 15
else:
    num_sobol_samples = 15
    num_repeats = 5
    batch_size = 50
    walltime_min = 120

SOBOL_SEED = 42
session_id = str(uuid4())

# Benchmark function configurations
FUNCTION_CONFIGS = {
    "branin": {
        "parameters": [
            {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
        ],
        "global_min": 0.39788735772973816
    },
    "hartmann6": {
        "parameters": [
            {"name": f"x{i+1}", "type": "range", "bounds": [0.0, 1.0]}
            for i in range(6)
        ],
        "global_min": -3.3223699212379664
    }
}


def generate_parameter_sets(function_name):
    """Generate parameter sets using Ax Sobol sampling."""
    config = FUNCTION_CONFIGS[function_name]
    parameters = config["parameters"] + [
        {"name": "function", "type": "fixed", "value": function_name}
    ]
    
    from ax.service.utils.instantiation import ObjectiveProperties
    
    ax_client = AxClient()
    ax_client.create_experiment(
        name=f"{function_name}_sobol",
        parameters=parameters,
        objectives={"value": ObjectiveProperties(minimize=True)},
    )
    
    search_space = ax_client.experiment.search_space
    m = get_sobol(search_space, fallback_to_sample_polytope=True, seed=SOBOL_SEED)
    gr = m.gen(n=num_sobol_samples)
    param_df = pd.DataFrame([arm.parameters for arm in gr.arms])
    
    # Create repeats with different random seeds
    param_dfs = []
    for repeat in range(num_repeats):
        tmp_df = param_df.copy()
        tmp_df["repeat"] = repeat
        tmp_df["session_id"] = session_id
        param_dfs.append(tmp_df)
    
    all_params = pd.concat(param_dfs, ignore_index=True)
    parameter_sets = all_params.to_dict(orient="records")
    shuffle(parameter_sets)
    
    return parameter_sets


def mongodb_evaluate(parameters, verbose=False):
    """Evaluate and store to MongoDB. Required environment variables:
    - MONGODB_URI: MongoDB connection string
    """
    utc = datetime.utcnow()
    
    result = {
        **parameters,
        "value": evaluate_benchmark(parameters),
        "timestamp": utc.timestamp(),
        "date": str(utc),
        "sobol_seed": SOBOL_SEED,
        "num_sobol_samples": num_sobol_samples,
        "num_repeats": num_repeats,
    }
    
    # MongoDB storage is required
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    import pymongo
    client = pymongo.MongoClient(mongodb_uri)
    client["bo_benchmarks"]["benchmark_results"].insert_one(result.copy())
    client.close()
    
    return result


def mongodb_evaluate_batch(parameter_sets, verbose=False):
    """Evaluate a batch of parameter sets."""
    return [mongodb_evaluate(p, verbose=verbose) for p in parameter_sets]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def submit_niagara_benchmark(function_names=None):
    """Submit benchmark jobs to Niagara cluster for all functions."""
    if function_names is None:
        function_names = list(FUNCTION_CONFIGS.keys())
    
    all_jobs = []
    
    for function_name in function_names:
        # Generate parameter sets for this function
        parameter_sets = generate_parameter_sets(function_name)
        if DUMMY_MODE:
            parameter_sets = parameter_sets[:10]
        
        parameter_batch_sets = list(chunks(parameter_sets, batch_size))
        
        # Configure submitit for Niagara
        log_folder = "/scratch/s/sgbaird/sgbaird/submitit_logs/bo_benchmarks/%j"
        
        executor = AutoExecutor(folder=log_folder)
        executor.update_parameters(
            timeout_min=walltime_min,
            cpus_per_task=1,  # 1 CPU per task
            slurm_partition="compute",  # Niagara compute partition
            slurm_account=os.getenv("SLURM_ACCOUNT", "def-sgbaird"),
            slurm_job_name=f"bo_benchmark_{function_name}"
        )
        
        print(f"Submitting {len(parameter_batch_sets)} batch jobs for {function_name}")
        jobs = executor.map_array(mongodb_evaluate_batch, parameter_batch_sets)
        print(f"Submitted jobs: {[job.job_id for job in jobs]}")
        all_jobs.extend(jobs)
    
    return all_jobs


# Run the benchmark
if LOCAL_TEST:
    # Run locally for testing
    parameter_sets = generate_parameter_sets(FUNCTION_NAME)[:5]
    results = [mongodb_evaluate(p, verbose=True) for p in parameter_sets]
    print(f"Local test completed. Sample results: {results[:2]}")
else:
    # Submit to cluster for all functions
    jobs = submit_niagara_benchmark()
    print(f"Submitted {len(jobs)} total jobs to Niagara cluster")
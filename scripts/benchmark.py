"""Minimal benchmark for studying initialization trial effects on optimization performance."""

import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood


def branin(x1, x2):
    """Branin function as implemented in honegumi."""
    y = float(
        (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5.0 / np.pi * x1 - 6.0) ** 2
        + 10 * (1 - 1.0 / (8 * np.pi)) * np.cos(x1)
        + 10
    )
    return y


def run_single_benchmark(num_init_trials: int = 5, num_opt_steps: int = 20, seed: int = 42):
    """
    Run a single optimization benchmark studying initialization trial effects.
    
    Args:
        num_init_trials: Number of initialization trials before switching to GP
        num_opt_steps: Number of optimization steps after initialization
        seed: Random seed for reproducibility
        
    Returns:
        dict: Results including trajectory, metadata, and GP predictions for all data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Branin domain bounds
    bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
    
    # Initialize with random points
    X_init = torch.rand(num_init_trials, 2)
    X_init = unnormalize(X_init, bounds)
    y_init = torch.tensor([branin(x[0].item(), x[1].item()) for x in X_init])
    
    X = X_init.clone()
    y = y_init.clone()
    
    trajectory = []
    metadata = []
    
    # Record initial best
    best_y = y.min().item()
    trajectory.append(best_y)
    
    # Run GP-based optimization
    for step in range(num_opt_steps):
        # Normalize for GP
        X_norm = normalize(X, bounds)
        
        # Fit GP
        gp = SingleTaskGP(X_norm, y.unsqueeze(-1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # Get predictions for all current data points
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(X_norm)
            means = posterior.mean.squeeze(-1)
            variances = posterior.variance.squeeze(-1)
            uncertainties = variances.sqrt()
        
        # Store metadata for this step
        step_metadata = {
            "step": step,
            "X": X.numpy().tolist(),
            "y": y.numpy().tolist(),
            "predicted_means": means.numpy().tolist(),
            "predicted_uncertainties": uncertainties.numpy().tolist(),
            "num_init_trials": num_init_trials,
            "current_best": y.min().item()
        }
        metadata.append(step_metadata)
        
        # Optimize acquisition function
        EI = ExpectedImprovement(gp, best_f=y.max())
        candidate, _ = optimize_acqf(
            EI, bounds=torch.stack([torch.zeros(2), torch.ones(2)]), q=1, num_restarts=10, raw_samples=100
        )
        
        # Evaluate candidate
        candidate_unnorm = unnormalize(candidate, bounds)
        new_y = branin(candidate_unnorm[0, 0].item(), candidate_unnorm[0, 1].item())
        
        # Update data
        X = torch.cat([X, candidate_unnorm])
        y = torch.cat([y, torch.tensor([new_y])])
        
        # Update best and record
        best_y = min(best_y, new_y)
        trajectory.append(best_y)
    
    return {
        "trajectory": trajectory,
        "final_best": best_y,
        "num_init_trials": num_init_trials,
        "total_evaluations": len(y),
        "metadata": metadata,
        "final_X": X.numpy().tolist(),
        "final_y": y.numpy().tolist(),
        "seed": seed
    }
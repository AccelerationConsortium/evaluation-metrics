from typing import Tuple

from botorch.test_functions.synthetic import Hartmann
import numpy as np
from scipy.stats.qmc import LatinHypercube
import torch


def create_hartmann_dataset(
    seed: int = 0,
    dim: int = 6,
    n_train: int = 80,
    n_test: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Hartmann]:
    """
    Build Hartmann dataset for testing GP models with configurable parameters.

    Args:
        seed: Random seed for reproducibility
        dim: Dimensionality of the problem
        n_train: Number of training points
        n_test: Number of test points

    Returns:
        Tuple containing:
        - train_X: Training inputs
        - train_Y: Training outputs
        - test_X: Test inputs
        - test_Y: Test outputs
        - f: The objective function
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    sampler = LatinHypercube(d=dim, rng=np.random.default_rng(seed))
    X_all_np = sampler.random(n=n_train + n_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_all = torch.tensor(X_all_np, device=device).double()

    train_X, test_X = X_all[:n_train], X_all[n_train:]
    f = Hartmann(dim=dim, negate=False).to(device)

    train_Y = f(train_X).unsqueeze(-1).double()
    test_Y = f(test_X).unsqueeze(-1).double()

    return train_X, train_Y, test_X, test_Y, f

from typing import Tuple

from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch

from .config import GPConfig


class GPModel:
    """
    A wrapper for a GP model providing standardized fitting, prediction,
    and methods to extract common diagnostic metrics like lengthscales.
    """

    def __init__(self, config: GPConfig):
        self.config = config
        self._setup_reproducibility()

    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        torch.manual_seed(self.config.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_default_dtype(torch.double)

    def fit(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        """Fit the GP model to the training data."""

        train_X = train_X.double()
        train_Y = train_Y.double()

        ard_dims = (
            self.config.ard_num_dims if self.config.ard_num_dims is not None else train_X.size(-1)
        )
        covar = ScaleKernel(RBFKernel(ard_num_dims=ard_dims))
        self.model = self.config.model_class(
            train_X,
            train_Y,
            covar_module=covar,
            outcome_transform=Standardize(m=1) if self.config.use_standardize else None,
        ).to(self.config.device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        # prepare for making predictions rather than further training
        self.model.eval()
        self.model.likelihood.eval()

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty estimates."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        post = self.model.posterior(X)

        return (post.mean.squeeze(-1), post.variance.sqrt().squeeze(-1))

    def get_lengthscales(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lengthscales and their importance."""
        if self.model is None:
            raise ValueError("Model must be fitted before getting lengthscales")
        ls = self.model.covar_module.base_kernel.lengthscale.squeeze().detach().cpu().numpy()
        imp = (1 / ls) / (1 / ls).sum()

        return (ls, imp)

"""
This script contains the code for evaluating the performance of a GP model.
It contains the following metrics:
- R²
- Lengthscales / Feature importance
- Rank correlation
- Leave-One-Out Pseudo-Likelihood
More details can be found in the README.md file.

Code can still be orgtanized far better, obviously. For it's mainly to reach alignment on
- which metrics to track and how to use them
- how to calculate the metrics in the best way

__author__ = "Willi Gottstein"
__status__ = "Development"
"""

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Tuple, Optional

from botorch.test_functions.synthetic import Hartmann
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.settings import fast_pred_var

@dataclass
class GPConfig:
    """Configuration for GP model. I guess that should be updated further, to e.g. also include
    the model etc. but for now it's not critical, I guess."""
    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ard_num_dims: Optional[int] = None
    use_standardize: bool = True

class GPModel:
    """
    A wrapper for a SingleTaskGP model providing standardized fitting, prediction,
    and methods to extract common diagnostic metrics like lengthscales.
    """
    
    def __init__(self, config: GPConfig):
        self.config = config
        self.model: Optional[SingleTaskGP] = None
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
        
        ard_dims = self.config.ard_num_dims if self.config.ard_num_dims is not None else train_X.size(-1)
        covar = ScaleKernel(RBFKernel(ard_num_dims=ard_dims))
        self.model = SingleTaskGP(
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

def build_dataset(seed: int = 0, dim: int = 6, n_train: int = 80, n_test: int = 20):
    """
    Build dataset for testing GP models with configurable parameters.
    
    Args:
        seed: Random seed for reproducibility
        dim: Dimensionality of the problem
        n_train: Number of training points
        n_test: Number of test points
        
    Returns:
        Training and test data, along with the objective function
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sampler = LatinHypercube(d=dim, seed=seed)
    X_all_np = sampler.random(n=n_train + n_test)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_all = torch.tensor(X_all_np, device=device).double()
    
    train_X, test_X = X_all[:n_train], X_all[n_train:]
    f = Hartmann(dim=dim, negate=False).to(device)
    
    return (train_X, f(train_X).unsqueeze(-1).double(), test_X, f(test_X).unsqueeze(-1).double(), f)

def loo_pseudo_likelihood(model, train_X, train_Y) -> float:
    """
    Negative mean leave‑one‑out log‑likelihood (lower is better).
    
    Note: The GPInputWarning about matching training data is expected here
    as we're calculating LOO-PL on the training data. Not sure how to deal with the warning...
    """
    loo_mll = LeaveOneOutPseudoLikelihood(model.likelihood, model)

    # GP output at the training inputs
    with torch.no_grad(), fast_pred_var():
        f_dist = model(train_X)
        target = train_Y.squeeze(-1)

    # average over points
    return (-loo_mll(f_dist, target).mean()).item()


def importance_concentration(imp: np.ndarray) -> dict:
    """
    Calculate cumulative importance concentration across dimensions.
    Returns a dictionary where keys are the number of features (1, 2, 3, ...)
    and values are the cumulative importance of the respective features.
    """
    sorted_imp = imp[np.argsort(imp)[::-1]]
    cum_sum = np.cumsum(sorted_imp)
    
    return dict(zip(range(1, len(cum_sum)+1), cum_sum))


if __name__ == "__main__":
    
    config = GPConfig()
    
    # one can play with the number of points that heavily impact the metrics
    dim = 6
    n_train = 80
    n_test = 20
    
    train_X, train_Y, test_X, test_Y, f_true = build_dataset(
        seed=config.seed, 
        dim=dim, 
        n_train=n_train, 
        n_test=n_test
    )
    
    print(
        f"Created dataset with {dim} dimensions, "
        f"{train_X.shape[0]} training points and "
        f" {test_X.shape[0]} test points"
    )
    
    gp_model = GPModel(config)
    gp_model.fit(train_X, train_Y)

    mean_t, std_t = gp_model.predict(test_X)

    r2 = r2_score(
        test_Y.detach().cpu().numpy(),
        mean_t.detach().cpu().numpy()
    )
    
    ls, imp = gp_model.get_lengthscales()
    imp_dict = importance_concentration(imp)
        
    loo = loo_pseudo_likelihood(gp_model.model, train_X, train_Y)
    hold_nll = -gp_model.model.likelihood(gp_model.model(test_X)).log_prob(test_Y).mean().item()
    
    rank_tau = kendalltau(
        test_Y.squeeze().detach().cpu().numpy(),
        mean_t.detach().cpu().numpy()
    ).statistic

    metrics = dict(
        r2=r2,
        imp_cumsum=imp_dict,
        loo_nll=loo,
        holdout_nll=hold_nll,
        rank_tau=rank_tau,
    )

    test_np = test_Y.squeeze().detach().cpu().numpy()
    mean_np = mean_t.detach().cpu().numpy()
    std_np = std_t.detach().cpu().numpy()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"prediction vs true (R²={r2:.4f})",
            "feature importance",
            "lengthscales",
            f"rank correlation (τ={rank_tau:.4f})"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_np,
            y=mean_np,
            mode='markers',
            error_y=dict(
                array=std_np,
                visible=True
            ),
            name="predictions"
        ),
        row=1, col=1
    )
    min_val = min(test_np.min(), mean_np.min())
    max_val = max(test_np.max(), mean_np.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash'),
            name="perfect prediction"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=[f"x{i+1}" for i in range(len(ls))],
            y=imp,
            name="importance"
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Bar(
            x=[f"x{i+1}" for i in range(len(ls))],
            y=ls,
            name="lengthscales"
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.argsort(test_np).argsort(),
            y=np.argsort(mean_np).argsort(),
            mode='markers',
            name="rank correlation"
        ),
        row=2,
        col=2
    )

    fig.add_trace(
        go.Scatter(
            x=[0, len(test_np)-1],
            y=[0, len(test_np)-1],
            mode='lines',
            line=dict(dash='dash'),
            name="perfect rank"
        ),
        row=2,
        col=2
    )

    fig.update_layout(
        height=800,
        width=1200,
        showlegend=False,
        title_text="GP model evaluation metrics",
        title_x=0.5
    )

    fig.update_xaxes(title_text="true value", row=1, col=1)
    fig.update_yaxes(title_text="predicted value", row=1, col=1)
    fig.update_xaxes(title_text="feature", row=1, col=2)
    fig.update_yaxes(title_text="normalized importance (1/ℓ)", row=1, col=2)
    fig.update_xaxes(title_text="feature", row=2, col=1)
    fig.update_yaxes(title_text="lengthscale (ℓ)", row=2, col=1)
    fig.update_xaxes(title_text="true value rank", row=2, col=2)
    fig.update_yaxes(title_text="predicted value rank", row=2, col=2)

    fig.show()

    print("\n=== Summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
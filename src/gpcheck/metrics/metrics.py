from gpytorch.mlls import LeaveOneOutPseudoLikelihood
from gpytorch.settings import fast_pred_var
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
import torch


def loo_pseudo_likelihood(model, train_X: torch.Tensor, train_Y: torch.Tensor) -> float:
    """
    Negative mean leave‑one‑out log‑likelihood (lower is better).

    Note: The GPInputWarning about matching training data is expected here
    as we're calculating LOO-PL on the training data.
    """
    loo_mll = LeaveOneOutPseudoLikelihood(model.likelihood, model)

    # GP output at the training inputs
    with torch.no_grad(), fast_pred_var():
        f_dist = model(train_X)
        target = train_Y.squeeze(-1)

    # average over points
    return (-torch.as_tensor(loo_mll(f_dist, target)).mean()).item()


def importance_concentration(imp: np.ndarray) -> dict:
    """
    Calculate cumulative importance concentration across dimensions.
    Returns a dictionary where keys are the number of features (1, 2, 3, ...)
    and values are the cumulative importance of the respective features.
    """
    sorted_imp = imp[np.argsort(imp)[::-1]]
    cum_sum = np.cumsum(sorted_imp)

    return dict(zip(range(1, len(cum_sum) + 1), cum_sum))


def calculate_metrics(
    gp_model,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    test_X: torch.Tensor,
    test_Y: torch.Tensor,
) -> dict:
    """
    Calculate comprehensive evaluation metrics for a GP model.

    This replicates the original functionality from evaluation_metrics.py
    """
    mean_t, std_t = gp_model.predict(test_X)

    # R^2
    r2 = r2_score(test_Y.detach().cpu().numpy(), mean_t.detach().cpu().numpy())

    # feature importance
    ls, imp = gp_model.get_lengthscales()
    imp_dict = importance_concentration(imp)

    # leave-one-out pseudo-likelihood
    loo = loo_pseudo_likelihood(gp_model.model, train_X, train_Y)

    # rank correlation
    rank_tau = kendalltau(test_Y.squeeze().detach().cpu().numpy(), mean_t.detach().cpu().numpy())[0]

    return {
        "r2": r2,
        "lengthscales": ls,
        "importance": imp,
        "imp_cumsum": imp_dict,
        "loo_nll": loo,
        "rank_tau": rank_tau,
        "predictions": {
            "mean": mean_t.detach().cpu().numpy(),
            "std": std_t.detach().cpu().numpy(),
            "true": test_Y.squeeze().detach().cpu().numpy(),
        },
    }

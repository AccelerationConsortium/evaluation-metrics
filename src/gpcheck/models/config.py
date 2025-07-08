from dataclasses import dataclass
from typing import Optional

from botorch.models import SingleTaskGP
import torch


@dataclass
class GPConfig:
    """Configuration for GP model."""

    seed: int = 0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ard_num_dims: Optional[int] = None
    use_standardize: bool = True
    model_class: type = SingleTaskGP

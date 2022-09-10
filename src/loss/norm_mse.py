import torch
import torch.nn as nn
from torch.nn import functional as F

class NormalizedMSE(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compuetes Normalized MSE

        Args:
            x (torch.Tensor): tensor
            y (torch.Tensor): tensor

        Returns:
            torch.Tensor: normalized MSE
        """
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1)


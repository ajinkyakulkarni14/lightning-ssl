import torch
import torch.nn as nn
from src.loss.norm_mse import NormalizedMSE

class BYOLLoss(nn.Module):
    
    def __init__(self, base: str = "norm_mse") -> None:
        super().__init__()
        assert base in ["norm_mse", "xent"], "base must be norm_mse or xent"
        if base == "norm_mse": self.criterion = NormalizedMSE()
        if base == "xent": self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        """mdoel outputs (logits1, logits2, target1, target2)

        Args:
            outputs (torch.Tensor): (logits1, logits2, target1, target2)

        Returns:
            torch.Tensor: BYOL loss
        """
        logits1, logits2, target1, target2 = outputs
        return torch.mean(
            self.criterion(logits1, target2) + self.criterion(logits2, target1)
        )
    
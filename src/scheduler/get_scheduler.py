from typing import Dict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, _LRScheduler

def get_scheduler(
    optimizer: Optimizer, 
    name: str,
    **kwargs
) -> _LRScheduler:
    """returns lr scheduler

    Args:
        optimizer (Optimizer): optimizer
        scheduler (str): which scheduler

    Returns:
        _LRScheduler: scheduler
    """
    support = ["linear_cosine", "cosine"]
    assert name in support, f"Only {support} schedulers are supported."
    
    if name == "linear_cosine":
        return LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            **kwargs
        )
    if name == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            **kwargs
        )

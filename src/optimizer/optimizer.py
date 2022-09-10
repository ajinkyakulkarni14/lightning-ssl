import torch.nn as nn
from pl_bolts.optimizers.lars import LARS
from torch.optim import Optimizer, Adam, SGD, AdamW

def optimizer(
    model: nn.Module, 
    algo: str = "adam", 
    lr: float = 1e-4, 
    weight_decay: float = 1e-6, 
    momentum: float = .9
) -> Optimizer:
    """returns the optimizer to train the model

    Args:
        model (nn.Module): model
        algo (str, optional): optimizer algorithm. Defaults to "Adam".
        lr (float, optional): learning rate. Defaults to 1e-4.
        weight_decay (float, optional): weighe decay. Defaults to 1e-6.
        momentum (float, optional): sgd momentum. Defaults to .9.

    Returns:
        Optimizer: optimizer
    """
    support = ["lars", "adamw", "adam", "sgd"]
    assert algo in support, f"Only {support} optimizers are supported."
    
    if algo == "lars":
        return LARS(
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    if algo == "adamw":
        return AdamW(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    if algo == "adam":
        return Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    if algo == "sgd":
        return SGD(
            params=model.parameters(),
            lr=lr,
            momentum=momentum
        )
from typing import Union
from src.loss.byol import BYOLLoss
from src.loss.dino import DINOLoss

def ssl_loss(
    model: str,
    **kwargs
) -> Union[DINOLoss, BYOLLoss]:
    """returns self supervised loss

    Args:
        model (str): model name

    Returns:
        Union[DINOLoss, BYOLLoss]: self supervised loss
    """
    
    if model == "byol":
        return BYOLLoss(**kwargs)

    if model == "dino":
        return DINOLoss(**kwargs)

    print(f"{model} not supported.")
    quit()
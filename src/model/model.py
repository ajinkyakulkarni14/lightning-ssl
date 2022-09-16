import torch.nn as nn
from typing import Union
from src.model.byol import BYOL
from src.model.dino import DINO

def ssl_model(
    model: str,
    **kwargs
) -> Union[BYOL, DINO]:
    
    if model == "byol":
        return BYOL(**kwargs)
    if model == "dino":
        return DINO(**kwargs)
    
    print(f"{model} not supported.")
    quit()
     
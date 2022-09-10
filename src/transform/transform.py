from typing import Callable, Union
from src.transform.base import SSLTransform, DINOTransform

def train_transform(
    model: str, 
    **kwargs
) -> Union[SSLTransform, DINOTransform]:
    """retunrs train image transformations class

    Args:
        model (str): model name (DINO/BYOL)

    Returns:
        Union[SSLTransform, DINOTransform]: train transformation
    """
    
    if model == "dino":
        return DINOTransform(**kwargs)
    
    if model == "byol":
        return SSLTransform(train=True, **kwargs)    
    
    print(f"{model} not supported.")
    quit()

def val_transform(
    model: str, 
    **kwargs
) -> Union[SSLTransform, DINOTransform]:
    """retunrs val image transformations class

    Args:
        model (str): model name (DINO/BYOL)

    Returns:
        Union[SSLTransform, DINOTransform]: validation transformation
    """
    
    if model == "dino":
        return DINOTransform(**kwargs)
    
    if model == "byol":
        return SSLTransform(train=False, **kwargs)    
    
    print(f"{model} not supported.")
    quit()

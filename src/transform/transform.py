from typing import Callable, Tuple
from src.transform.base import SSLTransform, DINOTransform

def get_transform(model: str, **kwargs) -> Tuple[Callable, Callable]:
    """returns train and val transformations based on ssl model

    Args:
        model (str): SSL model

    Returns:
        Tuple[Callable, Callable]: train and val transformations
    """
    if model == "dino": return DINOTransform(**kwargs), DINOTransform(**kwargs)
    if model == "byol": return SSLTransform(train=True, **kwargs), SSLTransform(train=False, **kwargs)

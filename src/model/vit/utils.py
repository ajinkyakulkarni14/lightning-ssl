import timm
import torch.nn as nn
from src.model.vit.vit import VisionTransformer

def load_state_dict(model: VisionTransformer, model_name: str) -> VisionTransformer:
    """loads state dict for custom VisionTransformer

    Args:
        model (VisionTransformer): custom ViT model
        model_name (str): timm base name

    Returns:
        VisionTransformer: custom ViT with loaded pretrained weights
    """
    print(f"Loading pretrained weights from timm's {model_name}")
    pretrained_model = timm.create_model(model_name, pretrained=True)    
    pretrained_state_dict = pretrained_model.state_dict()
    for layer_name in model.state_dict():
        pre_w = pretrained_state_dict[layer_name]
        model.state_dict()[layer_name].copy_(pre_w)
    return model
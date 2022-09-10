import timm
import torch.nn as nn
from functools import partial
from src.model.vit.vit import VisionTransformer
from src.model.vit.utils import load_state_dict

def custom_vit(
    vit_base: str,
    model_size: str, 
    pretrained: bool,
    patch_size: int, 
    img_size: int,
) -> nn.Module:
    """returns a custom ViT backbone

    Args:
        vit_base (str): vit/deit
        model_size (str): model size (tiny, small, base)
        patch_size (int): patch size
        img_size (int): image size
        
    Returns:
        nn.Module: Custom ViT
    """
    
    assert model_size in ["tiny", "small", "base"], "Only tiny, small and base are supported for custom ViT"
    
    # getting embed_dim + num_heads for custom ViT
    if model_size == "tiny": 
        embed_dim=192 
        num_heads=3
    if model_size == "small": 
        embed_dim=384 
        num_heads=6
    if model_size == "base": 
        embed_dim=768 
        num_heads=12
    vit = VisionTransformer(
        img_size=[img_size],
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    if pretrained:
        vit = load_state_dict(
            model=vit,
            model_name=f"{vit_base}_{model_size}_patch{patch_size}_{img_size}"
        )
    return vit

def create_backbone(
    backbone: str, 
    pretrained: bool = True
) -> nn.Module:
    """creates model's backbone

    Args:
        backbone (str): backbone name
        pretrained (bool, optional): pretrained. Defaults to True.

    Returns:
        nn.Module: backbone model
    """
    
    if backbone.startswith("custom_"):
        model_info=backbone.split("_")
        return custom_vit(
            vit_base=model_info[1],
            model_size=model_info[2],
            pretrained=pretrained,
            patch_size=int(model_info[3].replace("patch", "")),
            img_size=int(model_info[-1]),
        )
    else:
        return timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=0
        )

def get_out_features(
    model_name: str
) -> int:
    """returns the out features dim of a

    Args:
        model_name (str): model name for timm

    Returns:
        int: model out features dim
    """
    if model_name.startswith("custom_"):
        model_info = model_name.split("_")
        if model_info[2] == "tiny": return 192
        if model_info[2] == "small": return 384
        if model_info[2] == "base": return 768
    else:
        model = timm.create_model(
            model_name=model_name, 
            pretrained=False
        )
        layers = list(model.children())
        return layers[-1].in_features

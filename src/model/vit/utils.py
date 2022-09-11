import timm
import torch.nn as nn
from functools import partial
from src.model.vit.vit import VisionTransformer

def create_vit(
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
        if img_size not in [224, 384]:
            print(f"[ERROR] ViT input image size is {img_size}. No pretrained weights are available.")
        else:
            vit = load_state_dict(
                model=vit,
                model_name=f"{vit_base}_{model_size}_patch{patch_size}_{img_size}"
            )
    return vit

def load_state_dict(
    model: VisionTransformer, 
    model_name: str
) -> VisionTransformer:
    """loads state dict for custom VisionTransformer

    Args:
        model (VisionTransformer): custom ViT model
        model_name (str): timm base name

    Returns:
        VisionTransformer: custom ViT with loaded pretrained weights
    """
    print(f"> Loading pretrained weights from timm's {model_name}")
    pretrained_model = timm.create_model(model_name, pretrained=True)    
    pretrained_state_dict = pretrained_model.state_dict()
    for layer_name in model.state_dict():
        pre_w = pretrained_state_dict[layer_name]
        model.state_dict()[layer_name].copy_(pre_w)
    return model
import timm
import torch.nn as nn
from src.model.vit import create_vit


def create_backbone(
    backbone: str, 
    pretrained: bool = True,
    img_size: int = None
) -> nn.Module:
    """creates model's backbone

    Args:
        backbone (str): backbone name
        pretrained (bool, optional): pretrained. Defaults to True.
        img_size (int, optional): input image size. Defaults to 224.

    Returns:
        nn.Module: backbone model
    """
    
    if backbone.startswith("custom_"):
        model_info=backbone.split("_")
        img_size = int(model_info[-1]) if img_size is None else img_size
        return create_vit(
            vit_base=model_info[1],
            model_size=model_info[2],
            pretrained=pretrained,
            patch_size=int(model_info[3].replace("patch", "")),
            img_size=img_size,
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

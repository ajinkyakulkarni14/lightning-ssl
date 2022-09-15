import timm
import torch
import torch.nn as nn
from typing import Dict, Tuple
from src.model.vit import create_vit
from src.model.classifier import LinearClassifier

def create_linear_model(
    backbone: str,
    ckpt: str,
    img_size: int,
    num_classes: int,
    n_last_blocks: int = 4,
    avgpool: bool = False,
    freeze: bool = True
) -> Tuple[nn.Module, nn.Module]:
    """creates ssl_backbone and clf_head for the linear evaluation

    Args:
        backbone (str): ssl backbone name
        ckpt (str): path to checkpoint file
        img_size (int): input image size
        num_classes (int): number of classes
        n_last_blocks (int, optional): number of last attention blocks to consider (only for ViT). Defaults to 4.
        avgpool (bool, optional): if avgpool attentions outputs (only for ViT). Defaults to False.
        freeze (bool, optional): if True, the backbone weights will be frozen. Defaults to True.

    Returns:
        Tuple[nn.Module, nn.Module]: ssl_backbone, clf_head
    """
    
    ssl_backbone = create_model(backbone=backbone, pretrained=False, img_size=img_size)
    ssl_state_dict = torch.load(ckpt, map_location=torch.device('cpu'))["state_dict"]
    ssl_backbone = load_state_dict_ssl(
        model=ssl_backbone,
        ssl_state_dict=ssl_state_dict
    )
    embed_dim = get_out_features(model_name=backbone)
    if "vit" in backbone:
        embed_dim = embed_dim * (n_last_blocks + int(avgpool))
        
    model = LinearClassifier(
        backbone=ssl_backbone,
        is_vit=True if "vit" in backbone else False,
        in_feat=embed_dim,
        num_classes=num_classes,
        n_last_blocks=n_last_blocks,
        avgpool=avgpool,
    )
    
    return model
    
def create_model(
    backbone: str, 
    pretrained: bool = True,
    img_size: int = None,
    num_classes: int = 0
) -> nn.Module:
    """creates model's backbone

    Args:
        backbone (str): backbone name
        pretrained (bool, optional): pretrained. Defaults to True.
        img_size (int, optional): input image size. Defaults to 224.
        num_classes (int, optional): number of output classes. Defaults to 0.

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
            num_classes=num_classes
        )
    else:
        return timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes
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

def load_state_dict_ssl(
    model: nn.Module, 
    ssl_state_dict: Dict, 
    initials: str = "model.student.backbone."
) -> nn.Module:
    """loads weights from self-supervised model into model based on initials params (e.g. "model.student.backbone." are the layers to consider for TeacherStudentSSL models)

    Args:
        model (nn.Module): model
        ssl_state_dict (Dict): self supervised model state dict
        initials (str, optional): layers' initial to consider for loading weights to model. Defaults to "model.student.backbone.".

    Returns:
        nn.Module: model with loaded weights
    """
    count = 0
    for k, v in ssl_state_dict.items():
        if k.startswith(initials):
            _k = k.replace(initials, "")
            if _k in model.state_dict().keys():
                model.state_dict()[_k].copy_(v)
                count += 1
    print(f"> Loaded weights into model for {count}/{len(list(model.state_dict().keys()))} layers.")
    if count == 0:
        print("[WARNING] Unable to load weights properly.")
    return model
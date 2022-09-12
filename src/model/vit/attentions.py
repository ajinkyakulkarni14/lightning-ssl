import torch
import numpy as np
import torch.nn as nn
from src.model.vit.vit import VisionTransformer

def compute_attentions(
    model: VisionTransformer,
    x: torch.Tensor,
    patch_size: int = 16
) -> np.array:
    
    if not hasattr(model, "get_last_selfattention"):
        print("Method get_lastselfattention is required.")
        return None
    
    # praparing model
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    # w, h feature map size
    w_featmap = x.shape[-2] // patch_size
    h_featmap = x.shape[-1] // patch_size
    
    # get last self attention
    attentions = model.get_last_selfattention(x)
    # number of heads
    nh = attentions.shape[1]
    
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), 
        scale_factor=patch_size, 
        mode="nearest"
    )[0].cpu().numpy()    
    return attentions
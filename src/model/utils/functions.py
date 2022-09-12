import math
import timm
import torch

def fills_val_trunc_normal_(
    tensor: torch.Tensor, 
    mean: float = 0., 
    std: float = 1., 
    a: float = -2, 
    b: float = 2
) -> torch.Tensor:
    """Fills the input torch.Tensor with values drawn from a truncated normal distribution. If values fall over [2, -2], they are drawn again until within that range.

    Args:
        tensor (torch.Tensor): Input tensor
        mean (float, optional): mean of the normal distribution. Defaults to 0..
        std (float, optional): standard deviation of the normal distribution. Defaults to 1..
        a (float, optional): the minimum cutoff value. Defaults to -2.
        b (float, optional): the maximum cutoff value. Defaults to 2.
    """

    def normal_cdf(x: float):
        # The erf() function can be used to compute traditional statistical functions such as the cumulative standard normal distribution
        return (1. + math.erf(x / math.sqrt(2.)))/2.
    
    if (mean < a - 2*std) or (mean > b + 2*std):
        print("mean is more than 2 std from [a, b]. The distribution of values may be incorrect.")

    # used in inference
    with torch.no_grad():

        # Values are generated by using a truncated uniform distribution and the using the inverse CDF for the normal distribution
        low = normal_cdf((a-mean)/std)
        upper = normal_cdf((b-mean)/std)

        # Uniformely fill tensor with values from [low, upper], then translate to [2*low-1, 2*upper-1]
        tensor.uniform_(2*low-1, 2*upper-1)

        # Use inverse cdf transform for normal distribution to get truncated standard normal
        tensor.erfinv_()

        # Transform tensor to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure tensor is in the proper range
        tensor.clamp_(min=a, max=b)
        
        return tensor
    
def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def cancel_gradients_last_layer(epoch, model, frozen_epochs):
    if epoch >= frozen_epochs:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return fills_val_trunc_normal_(tensor, mean, std, a, b)
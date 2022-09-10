import torch
import torch.nn as nn
from src.model.utils.functions import fills_val_trunc_normal_

class MLP(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        hidden_dim: int = 4096, 
        proj_dim: int = 256,
        num_layers: int = 2,
        use_bn: bool = True,
        use_gelu: bool = False,
        drop_p: float = 0.,
        init_weights: bool = True
    ) -> None:
        """MLP implementation

        Args:
            in_features (int): input features size
            hidden_dim (int, optional): hidden layer features size. Defaults to 4096.
            proj_dim (int, optional): output features size (projection). Defaults to 256.
            num_layers (int, optional): number of layers in the MLP. Defaults to 3.
            use_bn (bool, optional): whether to apply BN. Defaults to True.
            use_gelu (bool, optional): whether to use GELU (True) or ReLU (False). Defaults to True.
            drop_p (float, optional): dropout prob. Defaults to 0.
            init_weights (bool, optional): if True initialize weights. Defaults to True.
        """
        super().__init__()
        
        num_layers = max(1, num_layers)
        if num_layers == 1:
            self.model = nn.Linear(in_features=in_features, out_features=proj_dim)
        else:
            # adding first layer 
            layers = [nn.Linear(in_features=in_features, out_features=hidden_dim)]
            if use_bn: layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
            
            # adding all the other layers
            for _ in range(num_layers-2):
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                if use_bn: layers.append(nn.BatchNorm1d(num_features=hidden_dim))
                layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
                
            layers.append(nn.Linear(in_features=hidden_dim, out_features=proj_dim))
            layers.append(nn.Dropout(drop_p))
            
            self.model = nn.Sequential(*layers)
            
            if init_weights: self.apply(self._init_weights)
                
    def _init_weights(self, m):
        """init weights fn

        Args:
            m (nn.Module): torch nn Module
        """
        if isinstance(m, nn.Linear):
            fills_val_trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on input tensor x

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        
        return self.model(x)

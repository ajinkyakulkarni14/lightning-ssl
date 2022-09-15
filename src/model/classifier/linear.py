import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(
        self,
        backbone: nn.Module,
        is_vit: bool,
        in_feat: int, 
        num_classes: int = 10,
        n_last_blocks: int = 4,
        avgpool: bool = True
    ):
        """Linear Classifier built on top of frozen features

        Args:
            backbone (nn.Module): backbone that extract frozen features
            is_vit (bool): if the model is ViT
            in_feat (int): input features
            num_classes (int, optional): number of output classes. Defaults to 10.
            n_last_blocks (int, optional): (only for ViT) number of last attenion blocks to consider. Defaults to 4.
            avgpool (bool, optional): (only for ViT) if run avgpool in ViT output. Defaults to True.
        """
        super(LinearClassifier, self).__init__()
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone
        self.is_vit = is_vit
        self.num_classes = num_classes
        self.linear = nn.Linear(in_feat, num_classes)
        self.n_last_blocks = n_last_blocks
        self.avgpool = avgpool
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()


    def forward(self, x):
        with torch.no_grad():
            if self.is_vit:
                intermediate_out = self.backbone.get_intermediate_layers(x, self.n_last_blocks)
                out = torch.cat([x[:, 0] for x in intermediate_out], dim=-1)
                if self.avgpool:
                    out = torch.cat((out.unsqueeze(-1), torch.mean(intermediate_out[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    out = out.reshape(out.shape[0], -1)
            else:
                out = self.backbone(x)

        # flatten
        out = out.view(out.size(0), -1)
        # linear layer
        return self.linear(out)

import torch
import torch.nn as nn
from typing import List, Tuple
from src.model.layers import Encoder

class DINO(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        img_size: int,
        pretrained: bool = True,
        hidden_dim: int = 4096,
        proj_dim: int = 256,
        out_dim: int = 65568,
        num_layers: int = 3,
        use_bn: bool = False,
        use_gelu: bool = False,
        drop_p: float = 0.,
        init_weights: bool = True,
        norm_last_layer: bool = True,
        beta: float = 0.996,
    ) -> None:
        """DINO Model initialization.

        Args:
            backbone (str): backbone architecture
            img_size (int): input image size
            pretrained (bool, optional): load pretrained weights. Defaults to True.
            hidden_dim (int, optional): encoder hidden dim. Defaults to 4096.
            proj_dim (int, optional): encoder projector output dim. Defaults to 256.
            out_dim (int, optional): dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.. Defaults to 65568.
            num_layers (int, optional): encoder num of linear layers. Defaults to 3.
            use_bn (bool, optional): use batch norm in encoder. Defaults to False.
            use_gelu (bool, optional): use gelu in encoder. Defaults to False.
            drop_p (float, optional): dropout in encoder. Defaults to 0..
            init_weights (bool, optional): init weights in encoder. Defaults to True.
            norm_last_layer (bool, optional): normalize last layer in encoder. Defaults to True.
            beta (float, optional): EMA update weight. Defaults to 0.996.
        """
        super().__init__()
        
        self.student = Encoder(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights,
            dino=True,
            dino_out_dim=out_dim,
            norm_last_layer=norm_last_layer
        )
        
        self.teacher = Encoder(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights,
            dino=True,
            dino_out_dim=out_dim,
            norm_last_layer=norm_last_layer
        )
        
        self.beta = beta
        
        self._init_teacher_weights()
        
    def _init_teacher_weights(self):
        """inits f() weights with g() weights. Also sets the gradient of f() to False.
        """
        for params_student, params_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            params_teacher.data.copy_(params_student.data) # copying g params
            params_teacher.requires_grad = False # no gradient updates
            
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of the target network
        """
        # TODO: implementare lo scheduler di beta come da paper (va da 0.996 a 1)
        for params_student, params_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            params_teacher.data = self.beta * params_teacher.data + (1 - self.beta) * params_student.data
            
    def embeds(self, x: torch.Tensor) -> torch.Tensor:
        return self.student.backbone(x)
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x_global = x[:2]
        x_local = x[2:]
        
        # Teacher Output - global crops
        teacher_crops = len(x_global)
        x_teacher = torch.cat(x_global, dim=0) # (batch_size * 2, 3, size, size)
        teacher_logits = self.teacher(x_teacher) # (batch_size * 2, proj_dim)
        
        # Student Output - local + global crops
        # global + local
        student_crops = len(x)
        x_global_student = torch.cat(x_global, dim=0)
        x_local_student = torch.cat(x_local, dim=0)
        student_global_logits = self.student(x_global_student)
        student_local_logits = self.student(x_local_student)
        student_logits = torch.cat((student_global_logits, student_local_logits), dim=0) # (batch_size * n_crops, out_dim) -- n_crops is 2+n_local_crops (2 is global)
        
        return student_logits.chunk(student_crops), teacher_logits.chunk(teacher_crops)
        
        
        
        
        

    
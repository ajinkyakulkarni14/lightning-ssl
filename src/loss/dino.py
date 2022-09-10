import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F

class DINOLoss(nn.Module):
    
    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9
    ) -> None:
        """DINO loss initialization.

        Args:
            out_dim (int): dimensionality of the final layer (we computed the softmax over)
            teacher_temp (float, optional): softmax temperature of the teacher resp. student. Defaults to 0.04.
            student_temp (float, optional): softmax temperature of the student. Defaults to 0.1.
            center_momentum (float, optional): hyperparameter for the exponential moving average that determines the center logits. 
                The higher the more the running average metters. Defaults to 0.9.
        """
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        
    def forward(self, output: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """DINO loss computation

        Args:
            output (Tuple[torch.Tensor, torch.Tensor]): student_logits, teacher_logits

        Returns:
            torch.Tensor: DINO loss value
        """
        student_logits, teacher_logits = output # output already chunked in n_crops (global+local and global)
        student_out = [s / self.student_temp for s in student_logits]
        # teacher centering and sharpening
        teacher_out = [(t - self.center) / self.teacher_temp for t in teacher_logits]
        
        student_sm = [F.log_softmax(s, dim=-1) for s in student_out]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_out]
        
        total_loss = 0
        n_loss_terms = 0
        
        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue
                loss = torch.sum(-t*s, dim=-1) # (n_samples, )
                total_loss += loss.mean() # scalar
                n_loss_terms += 1
        total_loss /= n_loss_terms
        # update center param for teacher output
        self.update_center(teacher_logits)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output

        Args:
            teacher_output (List[torch.Tensor]): _description_
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True) # (1, out_dim)
        self.center = self.center * self.center_momentum + (1 - self.center_momentum) * batch_center
                
        
        
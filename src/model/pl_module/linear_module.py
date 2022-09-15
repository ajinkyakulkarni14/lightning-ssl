import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import _LRScheduler

class LinearClassifierModule(pl.LightningModule):
    
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        n_last_blocks: int = 4,
        avgpool: bool = False
    ) -> None:
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_last_blocks = n_last_blocks # just for ViT models
        self.avgpool = avgpool
        self.acc = Accuracy()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        x, target = batch
        logits = self(x)
        loss = self.criterion(logits, target)
        
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        
        x, target = batch
        logits = self(x)
        loss = self.criterion(logits, target)
        
        self.acc.update(
            preds=logits,
            target=target
        )
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        acc = self.acc.compute()
        self.acc.reset()
        # just for lightning compatibility
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("loss/val", avg_loss, sync_dist=True, prog_bar=True)
        self.log("acc/val", acc, sync_dist=True, prog_bar=True)
        
    def training_epoch_end(self, outputs):
        pass
        
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
        
        
        
        
from grpc import Call
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from typing import Callable, Tuple
from torch.optim.lr_scheduler import _LRScheduler
from src.evaluation.evaluation import KNNEvaluator
from src.model.utils.functions import cancel_gradients_last_layer

class TeacherStudentSSLModule(pl.LightningModule):
    
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        last_layer_frozen: int = 2
    ) -> None:
        
        super().__init__()        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.last_layer_frozen = last_layer_frozen
        self.knn_evaluator = KNNEvaluator()
        
        
    def forward(self, views):
        return self.model(views)
        
    def training_step(self, batch, batch_idx):
        
        x, views, targets = batch
        outputs = self(views)
        loss = self.criterion(outputs)
        
        cancel_gradients_last_layer(
            epoch=self.current_epoch,
            model=self.model,
            frozen_epochs=self.last_layer_frozen
        )
        # EMA update        
        self.model.update_teacher()
        
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        
        self.knn_evaluator.update(
            split="train",
            embeds=self.model.embeds(x),
            targets=targets
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, views, targets = batch
        outputs = self(views)
        loss = self.criterion(outputs)
        
        self.knn_evaluator.update(
            split="val",
            embeds=self.model.embeds(x),
            targets=targets
        )
        
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = self.knn_evaluator.compute()
        self.knn_evaluator.reset()
        self.log("loss/val", avg_loss, sync_dist=True, prog_bar=True)
        self.log("acc/val", acc, sync_dist=True, prog_bar=True)
        
    def training_epoch_end(self, outputs):
        pass
        
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
import os
from typing import List
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase

def get_callbacks(output_dir: str) -> List[pytorch_lightning.Callback]:
    """Returns list of pl.Callbacks

    Args:
        output_dir (str): output dir

    Returns:
        List[pytorch_lightning.Callback]: list of pl.Callbacks
    """
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir, 
            filename="epoch={epoch}-step={step}-val_loss={loss/val:.3f}-val_acc={acc/val:.3f}",
            monitor="loss/val",
            verbose=True,
            mode="min",
            save_top_k=20,
            auto_insert_metric_name=False
        )
    )
    
    callbacks.append(
         LearningRateMonitor(
             logging_interval="epoch",
             log_momentum=True
        )
    )
    
    callbacks.append(ModelSummary(max_depth=1))
    
    callbacks.append(
        EarlyStopping(
            monitor="loss/val",
            min_delta=0.0,
            patience=20,
            verbose=False,
            mode="min",
            check_finite=True,
            stopping_threshold=None,
            divergence_threshold=None,
            check_on_train_epoch_end=None
        )
    )
    
    return callbacks

def get_logger(output_dir: str) -> LightningLoggerBase:
    """returns a logger

    Args:
        output_dir (str): output dir
        name (str): name
        version (str): versione

    Returns:
        pytorch_lightning.loggers.LightningLoggerBase: logger
    """
    return TensorBoardLogger(
        save_dir=os.path.join(output_dir, "tensorboard")
    )
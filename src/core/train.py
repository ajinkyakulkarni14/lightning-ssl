import os
import torch
import pytorch_lightning as pl
from src.io.io import load_config
from src.loss.byol import BYOLLoss
from src.loss.dino import DINOLoss
from src.model.byol.byol import BYOL
from src.model.dino.dino import DINO
from src.datamodule import SSLDataModule
from src.transform.transform import get_transform
from src.optimizer.get_optimizer import get_optimizer
from src.scheduler.get_scheduler import get_scheduler
from src.utils.trainer import get_callbacks, get_logger
from src.model.ssl_module import TeacherStudentSSLModule

def train(args):
    
    pl.seed_everything(42)
    
    # TODO implement hydra
    device = "gpu" if torch.cuda.is_available() else "cpu"
    config_path = os.path.join("config", f"{args.model}_{device}.yml")
    config = load_config(config_path)

    train_transform, val_transform = get_transform(
        model=args.model, 
        **config["transform"]
    )
    
    datamodule = SSLDataModule(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        **config["data_module"]
    )
    
    # setting up model, loss, optimizer, lr_scheduler
    model = DINO(**config["model"]) if args.model == "dino" else BYOL(**config["model"])
    criterion = DINOLoss(**config["loss"]) if args.model == "dino" else BYOLLoss(**config["loss"])
    # adapt lr to rule (base_lr * batch_size / 256)
    config["optimizer"]["lr"] *= config["data_module"]["batch_size"] / 256.
    optimizer = get_optimizer(model=model, **config["optimizer"])
    lr_scheduler = get_scheduler(optimizer=optimizer, name=config["lr_scheduler"]["name"], **config["lr_scheduler"]["params"])
    
    ssl_model = TeacherStudentSSLModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    logger = get_logger(output_dir=args.checkpoint_dir)
    
    callbacks = get_callbacks(output_dir=args.checkpoint_dir)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.checkpoint_dir,
        **config["trainer"]
    )
    
    trainer.fit(model=ssl_model, datamodule=datamodule)
    
    
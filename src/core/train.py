import os
from src.loss import SSLLOSS
import pytorch_lightning as pl
from src.io.io import load_config
from src.optimizer import Optimizer
from src.scheduler import LRScheduler
from src.datamodule import SSLDataModule
from src.transform import TrainTransform, ValTransform
from src.utils.trainer import get_callbacks, get_logger
from src.model import SSLModel, TeacherStudentSSLModule

def train(args):
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    # Train + Validation transform
    train_transform = TrainTransform(
        model=args.model,
        **config["transform"]
    ) 
    val_transform = ValTransform(
        model=args.model,
        **config["transform"]
    )
    
    # Data Module
    datamodule = SSLDataModule(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        **config["datamodule"]
    )
    
    # setting up model, loss, optimizer, lr_scheduler
    model = SSLModel(
        model=args.model,
        img_size=config["transform"]["img_size"],
        **config["model"]
    )
    
    criterion = SSLLOSS(
        model=args.model,
        **config["loss"]
    )
    
    # adapt lr to rule (base_lr * batch_size / 256)
    config["optimizer"]["lr"] *= config["datamodule"]["batch_size"] / 256.
    optimizer = Optimizer(
        model=model, 
        **config["optimizer"]
    )
    lr_scheduler = LRScheduler(
        optimizer=optimizer, 
        **config["lr_scheduler"]
    )
    
    ssl_model = TeacherStudentSSLModule(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    logger = get_logger(output_dir=args.checkpoint_dir)
    callbacks = get_callbacks(output_dir=args.checkpoint_dir)
    
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from,
        **config["trainer"]
    )
    
    trainer.fit(model=ssl_model, datamodule=datamodule)
    
    
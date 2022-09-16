import torch.nn as nn
from torch.optim import SGD
import pytorch_lightning as pl
from src.io.io import load_config
from src.model import LinearClassifierModule
from src.transform import ClassifierTransform
from src.datamodule import ClassifierDataModule
from src.model.utils import create_linear_model
from src.utils.trainer import get_callbacks, get_logger
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def linear_eval(args):
    
    pl.seed_everything(args.seed)

    ssl_config = load_config(args.ssl_config)
    clf_config = load_config(args.clf_config)
    
    # Train + Validation transform
    train_transform = ClassifierTransform(
        train=True,
        img_size=ssl_config["transform"]["img_size"],
    )
    val_transform = ClassifierTransform(
        train=False,
        img_size=ssl_config["transform"]["img_size"]
    )
    
    # Data Module
    datamodule = ClassifierDataModule(
        data_dir=args.data_dir,
        train_transform=train_transform,
        val_transform=val_transform,
        **clf_config["datamodule"]
    )
    
    # Setting up model, loss, optimizer, lr_scheduler
    linear_model = create_linear_model(
        backbone=ssl_config["model"]["backbone"],
        ckpt=args.ssl_ckpt,
        img_size=ssl_config["transform"]["img_size"],
        **clf_config["model"]
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(filter(lambda p: p.requires_grad, linear_model.parameters()), lr=0.001, momentum=.9)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=10,
        T_mult=2,
        eta_min=0.0000001,
        last_epoch=-1
    )
    
    model = LinearClassifierModule(
        model=linear_model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_last_blocks=clf_config["model"]["n_last_blocks"],
        avgpool=clf_config["model"]["avgpool"]
    )
    
    logger = get_logger(output_dir=args.checkpoint_dir)
    callbacks = get_callbacks(
        output_dir=args.checkpoint_dir,
        **clf_config["callbacks"]
    )
    
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from,
        **clf_config["trainer"]   
    )
    
    trainer.fit(model=model, datamodule=datamodule)
    
    
    
    
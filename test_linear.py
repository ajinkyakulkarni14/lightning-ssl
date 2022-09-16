import os
import torch
import torch.nn as nn
from torch.optim import SGD
from src.dataset import STL10
from src.io.io import load_config
from torch.utils.data import DataLoader
from src.transform import ClassifierTransform
from src.model.utils import create_linear_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def train_epoch(
    clf_backbone,
    clf_head,
    data_loader,
    criterion,
    optimizer,
    lr_scheduler: CosineAnnealingWarmRestarts,
    n: int = 4,
    avgpool=False,
    log_every: float = .2,
):
    
    clf_head.train()
    
    log_every = int(log_every * len(data_loader))
    
    for batch_idx, (x, target) in enumerate(data_loader):
        optimizer.zero_grad()
        
        x = x.to("mps")
        target = target.to("mps")
        
        with torch.no_grad():
            intermediate_out = clf_backbone.get_intermediate_layers(x, n)
            out = torch.cat([x[:, 0] for x in intermediate_out], dim=-1)
            if avgpool:
                out = torch.cat((out.unsqueeze(-1), torch.mean(intermediate_out[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                out = out.reshape(out.shape[0], -1)
            
        logits = clf_head(out)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        if (batch_idx) % log_every == 0:
            print(f"\t > step [{batch_idx}/{len(data_loader)}] - train/loss={loss.item():.4f} - lr={lr_scheduler.get_last_lr()[0]}")
    lr_scheduler.step()

@torch.no_grad()
def val_epoch(
    clf_backbone,
    clf_head,
    data_loader,
    criterion,
    n: int = 4,
    avgpool=False,
    log_every: float = .2,
):
    val_loss = 0
    correct = 0
    total = 0
    clf_head.eval()
    log_every = int(log_every * len(data_loader))
    for batch_idx, (x, target) in enumerate(data_loader):
        
        x = x.to("mps")
        target = target.to("mps")
        
        with torch.no_grad():
            intermediate_out = clf_backbone.get_intermediate_layers(x, n)
            out = torch.cat([x[:, 0] for x in intermediate_out], dim=-1)
            if avgpool:
                out = torch.cat((out.unsqueeze(-1), torch.mean(intermediate_out[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                out = out.reshape(out.shape[0], -1)
            
        logits = clf_head(out)
        loss = criterion(logits, target)
        val_loss += loss.item()
        preds = torch.max(logits, dim=1).indices
        correct += torch.sum(preds==target).item()
        total += target.size(0)
        
        if (batch_idx) % log_every == 0:    
            print(f"\t> validation step [{batch_idx}/{len(data_loader)}] - val/loss={loss.item():.4f} - val/acc={correct/total:.4f}")
 
    val_acc = correct/total
    val_loss = val_loss/len(data_loader)
    print(f"> Validation Stats val/loss={val_loss:.4f} - val/acc={val_acc:.4f}")

    return val_loss, val_acc
    

if __name__ == "__main__":
    
    config = load_config("/Users/riccardomusmeci/Developer/experiments/lightning-ssl/dino-vit-tiny-sagemaker-stl10-96/dino.yml")
    output_dir ="/Users/riccardomusmeci/Developer/experiments/lightning-ssl/dino-vit-tiny-sagemaker-stl10-96/classifier"
    n_last_blocks = 4
    avgpool_patchtokens=False
    
    
    os.makedirs(output_dir, exist_ok=True)
    train_dataset = STL10(
        root="/Users/riccardomusmeci/Developer/data/stl10",
        train=True,
        transform=ClassifierTransform(
            train=True,
            img_size=config["transform"]["img_size"],
        )
    )
    val_dataset = STL10(
        root="/Users/riccardomusmeci/Developer/data/stl10",
        train=False,
        transform=ClassifierTransform(
            train=False,
            img_size=config["transform"]["img_size"]
        )
    )
    num_classes = len(STL10.CLASSES)
    
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=5,
        drop_last=True
    )
    
    val_dl = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=5,
        drop_last=True
    )
    
    # model
    model_name = config["model"]["backbone"]
    clf_backbone = create_model(
        backbone=model_name,
        pretrained=False,
        img_size=config["transform"]["img_size"]
    )
    
    # loading state dict from DINO
    dino_state_dict_path = "/Users/riccardomusmeci/Developer/experiments/lightning-ssl/dino-vit-tiny-sagemaker-stl10-96/backbone/epoch=144-step=226490-val_loss=3.853.ckpt"
    
    # sending model to mps
    clf_backbone.load_state_dict(model_state_dict)
    clf_backbone.to("mps")
    
    
    embed_dim = clf_backbone.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
    
    clf_head = LinearClassifier(
        in_feat=embed_dim,
        num_classes=num_classes
    )
    
    clf_head.to("mps")

    clf_backbone.eval()
        
    epochs = 200
    
    # criterion, optimizer, lr_scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(clf_head.parameters(), lr=0.001, momentum=.9)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=10,
        T_mult=2,
        eta_min=0.0000001,
        last_epoch=-1
    )
    
    for epoch in range(epochs):
        print(f"Training epoch: {epoch+1}/{epochs}")
        train_epoch(
            clf_backbone=clf_backbone,
            clf_head=clf_head,
            n=n_last_blocks,
            avgpool=avgpool_patchtokens,
            data_loader=train_dl,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        print(f"Validation epoch: {epoch+1}/{epochs}")
        val_loss, val_acc = val_epoch(
            clf_backbone=clf_backbone,
            clf_head=clf_head,
            n=n_last_blocks,
            avgpool=avgpool_patchtokens,
            data_loader=val_dl,
            criterion=criterion
        )
        
        # save model
        output_fname = f"{model_name}-epoch={epoch+1}-val_loss={val_loss:5f}-val_acc={val_acc:5f}.pth"
        output_path = f"{output_dir}/{output_fname}"
        print(f"> Saving model pth at {output_path}")
        torch.save(clf_head.state_dict(), output_path)
    
    
    
    




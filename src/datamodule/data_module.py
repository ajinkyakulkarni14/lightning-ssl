import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataset.utils import collate_fn
from src.dataset.imagenette import ImagenetteDataset
from typing import List, Dict, Union, Callable, Optional


class SSLDataModule(pl.LightningDataModule):
    
    def __init__(
        self, 
        data_dir: str,
        batch_size: int,
        train_transform: Callable = None, 
        val_transform: Callable = None, 
        shuffle: bool = True,
        num_workers: int = 5,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last 
        self.persistent_workers=persistent_workers
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == "fit" or stage is None:
            
            self.train_dataset = ImagenetteDataset(
                root=self.data_dir,
                train=True,
                transform=self.train_transform
            )
            
            self.val_dataset = ImagenetteDataset(
                root=self.data_dir,
                train=False,
                transform=self.val_transform
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ImagenetteDataset(
                root=self.data_dir,
                train=False,
                transform=self.val_transform
            )
            
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
        
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
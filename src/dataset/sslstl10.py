
import os
from tkinter.messagebox import NO
from typing import Callable, Tuple
from torch.utils.data import Dataset

from src.io.io import read_rgb

class SSLSTL10(Dataset):
    
    def __init__(
        self,
        root: str, 
        train: bool,
        transform: Callable = None
    ) -> None:
        """Self Supervised STL10 Dataset support

        Args:
            root (str): data dir
            train (bool): if True unlabeled data is provided, else test data.
            transform (Callable, optional): self-sup transform function. Defaults to None.
        """
        if not os.path.exists(root):
            print(f"{root} does not exists. Quitting.")
            quit()
            
        self.data_dir = os.path.join(root, "unlabeled" if train else "test")
        self.img_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if not f.startswith(".")]
        if not train:
            self.classes = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(self.data_dir) if not f.startswith(".")]
        else:
            self.classes = None
        self.transform = transform
        
    def __getitem__(self, index) -> Tuple:
        
        img_path = self.img_paths[index]
        if self.classes is not None:
            label = self.classes[index]
        else:
            label = None
            
        img = read_rgb(img_path)
        views = None
        if self.transform:
            img, views = self.transform(img)
        
        return img, views, label
    
    def __len__(self) -> int:
        return len(self.img_paths)
        
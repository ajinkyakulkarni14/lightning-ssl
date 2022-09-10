import os
import numpy as np
from src.io.io import read_rgb
from src.utils.type import to_tensor
from torch.utils.data import Dataset
from typing import Callable, List, Tuple

class ImagenetteDataset(Dataset):
    
    def __init__(self,
        root: str,
        train: bool,
        transform: Callable = None
    ) -> None:
        
        if not os.path.exists(root):
            print(f"{root} does not exists. Quitting.")
            quit()
            
        self.data_dir = os.path.join(root, "train" if train else "val")
        
        self.classes = [c for c in os.listdir(self.data_dir) if not c.startswith(".DS")]
        
        self.img_paths = []
        self.labels = []
        for i, c in enumerate(self.classes):
            _paths = [os.path.join(self.data_dir, c, f) for f in os.listdir(os.path.join(self.data_dir, c)) if f.lower().endswith(".jpeg")]
            self.img_paths.extend(_paths)
            self.labels.extend([i for _ in _paths])
        
        self.transform = transform
        
    def __getitem__(self, index) ->Tuple[np.array, List[np.array], int]:
        
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = read_rgb(img_path)
        if self.transform:
            img, views = self.transform(img)

        return img, views, label
    
    def __len__(self) -> int:
        return len(self.img_paths)
import os
from src.io.io import read_rgb
from typing import Callable, Tuple
from torch.utils.data import Dataset

class STL10(Dataset):
    
    CLASSES = {
        1: "airplane",
        2: "bird",
        3: "car",
        4: "cat",
        5: "deer",
        6: "dog",
        7: "horse",
        8: "monkey",
        9: "ship",
        10: "truck"
    }
    
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
        
        self.data_dir = os.path.join(root, "train" if train else "test")
        classes = [c for c in os.listdir(self.data_dir) if not c.startswith(".")]
        self.targets = []
        self.img_paths = []
        self.labels = []
        for c_i, c in enumerate(classes):
            c_dir = os.path.join(self.data_dir, c)
            for f in os.listdir(c_dir):
                if not f.startswith("."):
                    self.img_paths.append(os.path.join(self.data_dir, c, f))
                    self.targets.append(c_i)
                    self.labels.append(self.CLASSES[int(c)])
        
        self.transform=transform
        
    def __getitem__(self, index) -> Tuple:
        
        img_path = self.img_paths[index]
        label = self.targets[index]
            
        img = read_rgb(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self) -> int:
        return len(self.img_paths)
        
# **How to extend the repository capabilities**

## **New Dataset**
If you want to use a different dataset, e.g. *MyNewDataset*, here the steps to follow.

1) Define the class under *src/dataset/*
```python

from src.io.io import read_rgb # custom utils to read RGB images
from torch.utils.data import Dataset

class MyNewDataset(Dataset):
    
    def __init__(
        self,
        # your params
    ):
        # your init code

    def __getitem__(self, index) -> Tuple:
        
        img_path = # get image path based on index
        label = # get image label 
        img = read_rgb(img_path)
        views = None
        if self.transform:
            img, views = self.transform(img)
        
        # views are necessary for Teacher-Student Self-Supervised Techniques
        return img, views, label
```








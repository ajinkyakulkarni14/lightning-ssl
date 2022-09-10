import PIL
import numpy as np
from PIL import Image
import albumentations as A
from src.utils.type import to_tensor
from typing import Union, List, Tuple

class SSLTransform:
    
    def __init__(
        self,
        train: bool,
        img_size: Union[int, list, tuple],
        mean: list = [0.485, 0.456, 0.406], 
        std: list = [0.229, 0.224, 0.225],
        brightness=0.8, 
        contrast=0.8, 
        saturation=0.8, 
        hue=0.2,
        color_jitter_p=.5,
        grayscale_p=.2,
        h_flip_p=.5,
        kernel=(3, 3),
        sigma=(.1, 2),
        gaussian_blur_p=.1,
    ) -> None:
        """Self-Supervised augmentations

        Args:
            train (bool): train/val mode.
            img_size (Union[int, list, tuple]): image size. 
            mean (list, optional): normalization mean. Defaults to [0.485, 0.456, 0.406].
            std (list, optional): normalization std. Defaults to [0.229, 0.224, 0.225].
            brightness (float, optional): color jitter brightness val. Defaults to 0.4.
            contrast (float, optional): color jitter contrast val. Defaults to 0.4.
            saturation (float, optional): color jitter saturation val. Defaults to 0.2.
            hue (float, optional): color jitter hue val. Defaults to 0.1.
            color_jitter_p (float, optional): color jitter prob. Defaults to 0.8.
            grayscale_p (float, optional): grayscale prob. Defaults to 0.1.
            h_flip_p (float, optional): horizontal flip prob. Defaults to 0.5.
            kernel (tuple, optional): gaussian blur kernel. Defaults to (3, 3).
            sigma (tupla, optional): gaussian blur std. Defaults to (.1, 2).
            gaussian_blur_p (float, optional): gaussian blur prob. Defaults to 0.1.
        """
        
        if isinstance(img_size, tuple) or isinstance(img_size, list):
            height = img_size[0]
            width = img_size[1]
        else:
            height = img_size
            width = img_size
            
        if train:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    p=color_jitter_p
                ),
                A.ToGray(p=grayscale_p),
                A.HorizontalFlip(p=h_flip_p),
                A.GaussianBlur(
                    blur_limit=kernel, 
                    sigma_limit=sigma, 
                    p=gaussian_blur_p
                ),
                A.RandomResizedCrop(height=height, width=width),
                A.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=mean, std=std),
            ])
        
        self.vanilla_transform = A.Compose([
                A.Resize(height=height, width=width),
                A.Normalize(mean=mean, std=std),
            ])
            
    def __call__(self, img: Union[np.array, PIL.Image.Image]) -> Tuple[np.array, np.array, np.array]:
        """Apply augmentations

        Args:
            img (Union[np.array, PIL.Image.Image]): input image

        Returns:
            Tuple[np.array, np.array, np.array]: vanilla img (resize + normalize), view 1, view 2
        """
        
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        view_1 = self.transform(image=img)['image']
        view_2 = self.transform(image=img)['image']
        img = self.vanilla_transform(image=img)['image']
        
        img = to_tensor(img)
        view_1 = to_tensor(view_1)
        view_2 = to_tensor(view_2)
        
        return img, (view_1, view_2)

class DINOTransform:
    def __init__(
        self,  
        img_size: Union[int, list, tuple], 
        local_crop_size: Union[int, list, tuple] = 96,
        global_crops_scale: tuple = (0.4, 1), 
        local_crops_scale: tuple =(0.05, .4), 
        n_local_crops: int = 8,
        mean: list = [0.485, 0.456, 0.406], 
        std: list = [0.229, 0.224, 0.225],
        crop_resize_p: float = 0.5,
        brightness: float = 0.4, 
        contrast: float = 0.4, 
        saturation: float = 0.2, 
        hue: float = 0.1,
        color_jitter_p: float = .5,
        grayscale_p: float = 0.2,
        h_flip_p: float = .5,
        kernel: tuple = (5, 5),
        sigma: tuple = (.1, 2),
        solarization_p: float = 0.2,
        solarize_t: int = 170,
    ):
        """DINO Transform

        Args:
            img_size (Union[int, list, tuple]): image size. 
            local_crop_size (Union[int, list, tuple], optional): local crop size. Defaults to 96.
            global_crops_scale (tuple, optional): global crop scale range. Defaults to (0.4, 1).
            local_crops_scale (tuple, optional): local crop scale range. Defaults to (0.05, .4).
            n_local_crops (int, optional): number of local crops. Total of crops will be 2+n_local_crops. Defaults to 8.
            mean (list, optional): normalization mean. Defaults to [0.485, 0.456, 0.406].
            std (list, optional): normalization std. Defaults to [0.229, 0.224, 0.225].
            crop_resize_p (float, optional): crop and resize prob. Defaults to 0.5.
            brightness (float, optional): color jitter brightness val. Defaults to 0.4.
            contrast (float, optional): color jitter contrast val. Defaults to 0.4.
            saturation (float, optional): color jitter saturation val. Defaults to 0.2.
            hue (float, optional): color jitter hue val. Defaults to 0.1.
            color_jitter_p (float, optional): color jitter prob. Defaults to 0.8.
            grayscale_p (float, optional): grayscale prob. Defaults to 0.1.
            h_flip_p (float, optional): horizontal flip prob. Defaults to 0.5.
            kernel (tuple, optional): gaussian blur kernel. Defaults to (3, 3).
            sigma (tupla, optional): gaussian blur std. Defaults to (.1, 2).
            gaussian_blur_p (float, optional): gaussian blur prob. Defaults to 0.1.
            solarization_p (float, optional): solarization prob. Defaults to 0.2.
            solarize_t (int, optional): solarization threshold. Defaults to 170.
        """

        if isinstance(img_size, tuple) or isinstance(img_size, list):
            height = img_size[0]
            width = img_size[1]
        else:
            height = img_size
            width = img_size
            
        if isinstance(local_crop_size, tuple) or isinstance(local_crop_size, list):
            crop_height = local_crop_size[0]
            crop_width = local_crop_size[1]
        else:
            crop_height = local_crop_size
            crop_width = local_crop_size
        
        # Global Augmentation 1
        
        self.global_transform_1 = A.Compose([
            # A.Resize(height=height, width=width),
            A.RandomResizedCrop(height=height, width=width, scale=global_crops_scale, p=crop_resize_p, interpolation=Image.BICUBIC),
            # Flip and ColorJitter
            A.HorizontalFlip(p=h_flip_p),
            A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=color_jitter_p),
            A.ToGray(p=grayscale_p),
            # Gaussion Blur
            A.GaussianBlur(blur_limit=kernel, sigma_limit=sigma, p=1.0), # always applied for global_transform_1
            # Normalization
            A.Normalize(mean=mean, std=std)            
        ])
        
        self.global_transform_2 = A.Compose([
            # A.Resize(height=height, width=width),
            A.RandomResizedCrop(height=height, width=width, scale=global_crops_scale, p=crop_resize_p, interpolation=Image.BICUBIC),
            # Flip and ColorJitter
            A.HorizontalFlip(p=h_flip_p),
            A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=color_jitter_p),
            A.ToGray(p=grayscale_p),
            # Gaussion Blur + Solarization
            A.GaussianBlur(blur_limit=kernel, sigma_limit=sigma, p=0.1), 
            A.Solarize(threshold=solarize_t, p=solarization_p),
            # Normalization
            A.Normalize(mean=mean, std=std)            
        ])
        
        # transformation for the local small crops
        self.n_local_crops = n_local_crops
        self.local_transform = A.Compose([
            #A.Resize(height=height, width=width),
            A.RandomResizedCrop(height=crop_height, width=crop_width, scale=local_crops_scale, p=crop_resize_p, interpolation=Image.BICUBIC),
            # Flip and ColorJitter
            A.HorizontalFlip(p=h_flip_p),
            A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=color_jitter_p),
            A.ToGray(p=grayscale_p),
            # Gaussion Blur
            A.GaussianBlur(blur_limit=kernel, sigma_limit=sigma, p=0.5),
            # Normalization
            A.Normalize(mean=mean, std=std)             
        ])
        
        self.vanilla_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std),
        ])
        
    def __call__(self, img) -> Tuple[np.array, List[np.array]]:
        """Apply augmentations

        Args:
            img (Union[np.array, PIL.Image.Image]): input image

        Returns:
            Tuple[np.array, List[np.array]]: vanilla image (resize+normalize), list of np.array crops (first 2 globals, remaining local)
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        crops = []
        
        crops.append(to_tensor(self.global_transform_1(image=img)["image"].astype(np.float32)))
        crops.append(to_tensor(self.global_transform_2(image=img)["image"].astype(np.float32)))
        for _ in range(self.n_local_crops):
            crops.append(to_tensor(self.local_transform(image=img)["image"].astype(np.float32)))
        img = to_tensor(self.vanilla_transform(image=img)['image'])
        
        return img, crops

## **Self-Supervised Learning (BYOL+DINO) with PyTorch Lightning**
Pytorch-Lightning implementation of two of the most important self-supervised learning techniques: 

* **BYOL** ([`paper`](https://arxiv.org/pdf/2006.07733.pdf), [`repository`](https://github.com/deepmind/deepmind-research/tree/master/byol))
<p align="center">
    <img width="70%" src="static/byol_diagram.png" alt>
</p>

* **DINO** ([`arXiv`](https://arxiv.org/pdf/2104.14294.pdf), [`repository`](https://github.com/facebookresearch/dino))
<p align="center">
    <img width="70%%" src="static/dino.gif" alt>
</p>

## **Dataset**
Models are trained on the [STL10 dataset](https://ai.stanford.edu/~acoates/stl10/). The dataset was downloaded and then converted to *.png* images split into *train*, *test*, and *unlabelled* folders.

Train and test folders must be divided into folders, every one representing a class.

## **Train**
The repository supports [timm](https://github.com/rwightman/pytorch-image-models) models as backbones for both BYOL and DINO. 

Both BYOL and DINO come with a YAML configuration file in *config/* folder. Play with it to change some training parameters such us backbones, augmentations, schedulers, etc.

To train the model, please run:
```
python train.py --config config/YOUR_CONFIG.yml --model dino/byol --data-dir PATH/TO/STL10 --checkpoints-dir PATH/TO/DIR/TO/SAVE/PTH
```

## **Custom ViT**
Custom implementation of ViT is provided to be flexible on the image size. These the models supported:
* custom_vit_tiny_patch16_224
* custom_vit_small_patch16_224
* custom_vit_base_patch16_224

Here the *224* is left for timm's compatibility. Image size will always be the one specified in the configuration file under the *transform* section.

## **MPS Support**
The repository has *mps* support to train on M1 GPUs.

## **Warnings**
:warning: As of today ResNet and ViT backbones work fine. Other models might fail.

## **TO-DOs**
[ ] Add models performances on STL10

[ ] Add notebook to train a classifier on DINO/BYOL weigths

[ ] Add notebook for feature visualization 

[ ] Add notebook to show GradCAM










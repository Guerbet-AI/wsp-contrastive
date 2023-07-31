# Weakly-supervised positional contrastive learning: application to cirrhosis classification

<p align="center">
<img width="700" alt="png_miccai" src="https://github.com/Guerbet-AI/wsp-contrastive/assets/55430451/a4cebc73-d4dc-4db1-8728-b842aa2a1812">
</p>

Official implementation of "Weakly-supervised positional contrastive learning: application to cirrhosis classification", accepted paper at MICCAI 2023.  

**Authors**: *Emma Sarfati* $^{*[1,2]}$, *Alexandre Bône* $^{[1]}$, *Marc-Michel Rohé* $^{[1]}$, *Pietro Gori* $^{[2]}$, *Isabelle Bloch* $^{[2,3]}$.  

[1] Guerbet Research, Villepinte, France  
[2] LTCI, Télécom Paris, Institut Polytechnique de Paris, France  
[3] Sorbonne Université, CNRS, LIP6, Paris, France  
$*$ Corresponding author


This paper introduces a new contrastive learning method based on a generic kernel-loss function that allows to leverage discrete and continuous meta-labels for medical imaging.  

# Method

Let $x_t$ be a random image in our dataset $\mathcal{X}$, called an anchor, and $(y_t,d_t)$ be a pair of respectively discrete and continuous random variables associated to $x_t$. Let $x_j^-$ and $x_i^+$ be two semantically different and similar images respectively, w.r.t $x_t$.  
In contrastive learning (CL), one wants to find a parametric function $f_{\theta}:\mathcal{X}\rightarrow \mathrm{S}^d$ such that:
$$s_{tj}^- - s_{ti}^+ \leq 0 \quad \forall t,j,i$$
where $s_{tj}^-=sim(f_\theta(x_t),f_\theta(x_j^-))$ and $s_{ti}^+=sim(f_\theta(x_t),f_\theta(x_i^+))$, with $sim$ a similarity function defined here as $sim(a,b)=\frac{a^Tb}{\tau}$ with $\tau>0$.  
In practice, one does not know the definition of negative and positive. This is the main difference between each CL method. In SimCLR [1], positives are two random augmentations of the anchor $x_t$ and negatives are the other images. In SupCon [2], positives are all the images with the same discrete label $y$. In [3], all samples are considered positives but have a continuous degree of positiveness according to the associated continuous variables $d$ provided by a Gaussian kernel. In this work, we propose to leverage at the same time a discrete variable $y$ as well as a continuous one $y$ by introducing a composite kernel such that:
$$w_\delta(y_t,y_i) \cdot w_\sigma(d_t,d_i) (s_{tj}-s_{ti}) \leq 0 \quad \forall t,i,j\neq i \quad (1)$$

where the indices $t,i,j$ traverse all $N$ images in the batch since there are no ``hard'' positive or negative samples, as in SimCLR or SupCon, but all images are considered as positive and negative at the same time. After simplification, our final loss function leads to:  
$$\mathcal{L_{WSP}}=-\sum_{t=1}^{N} \sum_{i\in P(t)} w_\sigma(d_t,d_i) \log \left( \frac{\exp(s_{ti})}{ \sum_{j\neq i} \exp(s_{tj})} \right)$$

References:  
[1] [SimCLR](https://arxiv.org/abs/2002.05709)  
[2] [SupCon](https://arxiv.org/abs/2004.11362)  
[3] [y-Aware](https://arxiv.org/abs/2106.08808)  

# Codes

This repo contains the official codes for WSP Contrastive Learning. The codes are implemented using PyTorch-Lightning. 

## Requirements

### Image data

All the images must be stored in the `path_to_data` path and must contain two folders inside:
- `/train`: training images in Nifty format.
- `/validation`: validation images in Nifty format.

### DataFrame 

To run properly the codes, you will have to provide a Pandas DataFrame with the following index and columns:
- Index: name of the subjects.
- Column `class`: radiological class or histological class depending on the type of task (pretraining or classification).
- Column `label`: histological class (if available).
We provide the dataframe for the public LIHC dataset that we used in our paper for edvaluation , in the `dataframe_lihc.csv` file.  
The CT-scans are also available for downloading here: *put path here*.

## Launching the codes 

The file `main.py`can be launched in two different modes: pretraining or finetuning. Many other arguments follow, that you will have to indicate by following this convention:

```
python main.py --mode <put mode here> --rep_dim <put number here> --num_classes <put number here>
```

And so on. All the arguments are available in the file `config.py` and are provided below.

```
mode: str = 'finetuning',  
rep_dim: int = 512,  
hidden_dim: int = 256,  
output_dim: int = 128,  
num_classes: int = 4,  
encoder: str = 'tiny',  
n_layer: int = 18,  
lr: float = 1e-5,  
weight_decay: float = 1e-5,  
label_name: str = 'label',  
n_fold: int = 4,  
cross_val: bool = False,  
pretrained_path: str = None,  
sigma: float = 0.85,  
temperature: float = 0.1,  
kernel: str = 'rbf',  
max_epochs: int = 40,  
batch_size: int = 64,  
pretrained: bool = False,  
path_to_data: str = "path_to_data",  
lght_dir: str = "path_to_models"  
```

For either pretraining or finetuning mode, the data that are fed to the models are, in this order: `data, label, subject_id, z`.
- `data`: 2D image of shape (1,512,512).
- `label`: discrete label corresponding to the variable $y$ in Eq. (1).
- `subject_id`: name of the subject, for convenience.
- `z`: continuous label corresponding to the variable $d$ in Eq. (1).
Please note that the normalized positional coordinate $d\in [0,1]$ (named `z` in the code) is computed automatically given each volume at the beginning of the `dataset.py` file. Hence you will only have to provide the discrete label in the dataframe. If you wish to use other labels related to the patients you can do it by providing other columns in the original dataframe. You will need to change the implementation of the loss accordingly by adding discrete/continuous kernels.

### Pretraining

For pretraining, launch the following line of code.
```
main.py --mode pretraining
```

### Finetuning

For finetuning, launch the following line of code.
```
main.py --mode finetuning
```

For adding an argument, you can follow the protocol described above.

### Tensorboard

We use TensorBoard for following metrics. To access it, launch in your terminal:

```
tensorboard --logdir=<path of where your codes are>
```

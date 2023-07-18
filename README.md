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

**Context**

Let $x_t$ be a random image in our dataset $\mathcal{X}$, called an anchor, and $(y_t,d_t)$ be a pair of respectively discrete and continuous random variables associated to $x_t$. Let $x_j^-$ and $x_i^+$ be two semantically different and similar images respectively, w.r.t $x_t$.  
In contrastive learning (CL), one wants to find a parametric function $f_{\theta}:\mathcal{X}\rightarrow \mathrm{S}^d$ such that:
$$s_{tj}^- - s_{ti}^+ \leq 0 \quad \forall t,j,i$$
where $s_{tj}^-=sim(f_\theta(x_t),f_\theta(x_j^-))$ and $s_{ti}^+=sim(f_\theta(x_t),f_\theta(x_i^+))$, with $sim$ a similarity function defined here as $sim(a,b)=\frac{a^Tb}{\tau}$ with $\tau>0$.  
In practice, one does not know the definition of negative and positive. This is the main difference between each CL method. In SimCLR [1], positives are two random augmentations of the anchor $x_t$ and negatives are the other images. In SupCon [2], positives are all the images with the same discrete label $y$. In [3], all samples are considered positives but have a continuous degree of positiveness according to the associated continuous variables $d$ provided by a Gaussian kernel. In this work, we propose to leverage at the same time a discrete variable $y$ as well as a continuous one $y$ by introducing a composite kernel such that:
$$w_\delta(y_t,y_i) \cdot w_\sigma(d_t,d_i) (s_{tj}-s_{ti}) \leq 0 \quad \forall t,i,j\neq i      (1)$$

where the indices $t,i,j$ traverse all $N$ images in the batch since there are no ``hard'' positive or negative samples, as in SimCLR or SupCon, but all images are considered as positive and negative at the same time. After simplification, our final loss function leads to:  
$$\mathcal{L_{WSP}}=-\sum_{t=1}^{N} \sum_{i\in P(t)} w_\sigma(d_t,d_i) \log \left( \frac{\exp(s_{ti})}{ \sum_{j\neq i} \exp(s_{tj})} \right)$$

References:  
[1] [SimCLR](https://arxiv.org/abs/2002.05709)  
[2] [SupCon](https://arxiv.org/abs/2004.11362)  
[3] [y-Aware](https://arxiv.org/abs/2106.08808)  

**Codes**

This repo contains the official codes for WSP Contrastive Learning. The codes are implemented using PyTorch-Lightning. In this code you have to provide a dataframe with:
- Index: name of the subjects.
- Column `class`: radiological class or histological class depending on the type of task (pretraining or classification).
- Column `label`: histological class (if available).

The data that are fed to the models are, in this order: `data, label, subject_id, z`.
- `data`: 2D image of shape (1,512,512).
- `label`: discrete label corresponding to the variable $y$ in Eq. (1).
- `subject_id`: name of the subject, for convenience.
- `z`: continuous label corresponding to the $d$ in Eq. (1).
- $d$: continuous positional variable which is computed beforehand in the `dataset.py` file.

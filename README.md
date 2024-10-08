# Localization of Synthetic Manipulations in Western Blot Images

Official repository of the [homonymous paper](https://arxiv.org/pdf/2408.13786), accepted at IEEE WIFS 2024.  
Authors: Anmol Manjunath, Viola Negroni, Sara Mandelli, Daniel Moreira, Paolo Bestagini.

In this repository, we present two new datasets of manipulated Western blot images: 
1. Automatically manipulated Western blots.
2. Realistically manipulated Western blots.
  
Additionally, we release the scripts to reproduce the experiments presented in the paper.

## Datasets

### Automatically manipulated Western blots
A dataset of automatically tampered Western blots. Synthetic images in this dataset have been created by inserting squared patches of 64x64 at a random location into genuine host images. The amount of synthetic patches used is equally balanced between CycleGAN, Pix2pix, StyleGAN2-ADA, and DDPM.    
Synthetic images from this dataset are available, along with their ground truth masks, at:   

```bash
data/automatically_tampered
```

### Realistically manipulated Western blots
A dataset of manually tampered Western blots. To emulate real-world forgery, synthetic images in this dataset have been created by manipulating pristine images by hand. 
We employed three different tampering tools: DALL·E 2, Cleanup, and GIMP.  
Synthetic images, divided by tampering method, and ground truth masks are available at:  

```bash
data/tampered_blots
```

Both these datasets have been built on top of a subset of the dataset Western blots dataset released by _Mandelli et al._ in [this paper](https://ieeexplore.ieee.org/abstract/document/9785655).

## Prerequisites

Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate detection_pbp
```

## Run the Synthetic Image Detector

TODO

## How to cite

```bibtex
@article{manjunath2024localization,
  title={Localization of Synthetic Manipulations in Western Blot Images},
  author={Manjunath, Anmol and Negroni, Viola and Mandelli, Sara and Moreira, Daniel and Bestagini, Paolo},
  journal={arXiv preprint arXiv:2408.13786},
  year={2024}
}

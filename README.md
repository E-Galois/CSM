# PyTorch implementation of "Beyond One-Hot Labels: Semantic Mixing for Model Calibration"

The code will be made available soon!

This repo is the official pytorch implementation of "Beyond One-Hot Labels: Semantic Mixing for Model Calibration".

## Dependencies
* Python >= 3.8
* PyTorch >= 1.8.1

## Run
Augmented samples generation:
> sh gen_edm-mix/run_mix.sh

Annotation:
> python annotate.py

Training:
> python train_net.py

## Acknowledgements
Our code is mainly based on the code of [`RankMixup`]. This is our example code for training on the CIFAR-100 dataset.

## Citation

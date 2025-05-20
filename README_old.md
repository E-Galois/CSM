# PyTorch implementation of "Beyond One-Hot Labels: Semantic Mixing for Model Calibration"

This is the implementation of our paper "Beyond One-Hot Labels: Semantic Mixing for Model Calibration (ICLR 2025)".

For details, check out our paper at this [PDF page](https://arxiv.org/pdf/2504.13548).

## Dependencies
* Python >= 3.8
* PyTorch >= 1.8.1

Please: 1. create a virtual environment, 2. install environments in `gen_edm-mix/environment.yml` for sample augmentation and in `requirement.txt` for training & testing.

## Augmented Samples
1. Generate augmented samples:
```shell

cd gen_edm_mix
sh run_mix.sh
```

2. Post-process to get augmented image files in `gen_edm-mix/out_{dset}_imgs_mix`:
```shell

python convert_pths.py
```

## Sample Reannotation

Annotations can be done by running (*e.g.* CIFAR100):
```shell

cd ..
python annotate.py \
--dset_name c100 [c10 | tiny] \
--clip_path [CLIP download path] \
--train_path [training images path] \
--mix_path ./gen_edm-mix/out_c10_imgs_mix
```

Remember to fill in the paths of your own inputs here. The annotations would be saved at `annotations/clip_lam_anno_edm_{args.dset_name}.pth`.

## Train & Test:
1. Modify `configs/defaults.yaml` to determine the dataset and model architecture.
2. (CIFAR100 as an example) Modify `configs/cifar100.yaml` to specify your own data paths.
3. Start training with calibration tests by running:
```shell

python train_net.py
```

## Bibtex
```
@article{luo2025beyond,
  title={Beyond One-Hot Labels: Semantic Mixing for Model Calibration},
  author={Luo, Haoyang and Tao, Linwei and Dong, Minjing and Xu, Chang},
  journal={arXiv preprint arXiv:2504.13548},
  year={2025}
}
```

## References
Our code is based on the official implementations of [RankMixup](https://github.com/cvlab-yonsei/RankMixup) [1] and [EDM](https://github.com/NVlabs/edm) [2]. Thanks a lot for their efforts!

[1] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35, 26565-26577.

[2] Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., & Yan, S. (2023, July). Better diffusion models further improve adversarial training. In International Conference on Machine Learning (pp. 36246-36263). PMLR.

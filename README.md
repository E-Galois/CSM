
# Beyond One-Hot Labels: Semantic Mixing for Model Calibration (PyTorch)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)

Official PyTorch implementation of our ICLR 2025 paper **"Beyond One-Hot Labels: Semantic Mixing for Model Calibration"**.

📄 [Read the full paper on arXiv](https://arxiv.org/pdf/2504.13548)

## Table of Contents
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
  - [Augmented Samples](#augmented-samples)
  - [Sample Reannotation](#sample-reannotation)
  - [Training & Testing](#training--testing)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.8.1

**Setup Instructions:**
1. Create a virtual environment
2. Install dependencies:
   ```bash
   # For sample augmentation
   cd gen_edm_mix
   conda env create -f environment.yml
   
   # For training & testing
   pip install -r requirements.txt
   ```

## Getting Started

### Augmented Samples

1. **Generate augmented samples:**
   ```bash
   cd gen_edm_mix
   sh run_mix.sh
   ```

2. **Post-process to get image files:**
   ```bash
   python convert_pths.py
   ```
   Output will be saved in `gen_edm-mix/out_{dset}_imgs_mix`

### Sample Reannotation

Run annotation for your dataset (e.g., CIFAR100):
```bash
python annotate.py \
  --dset_name c100 [c10 | tiny] \
  --clip_path [CLIP_DOWNLOAD_PATH] \
  --train_path [TRAINING_IMAGES_PATH] \
  --mix_path ./gen_edm-mix/out_c10_imgs_mix
```

Annotations will be saved at `annotations/clip_lam_anno_edm_{args.dset_name}.pth`

### Training & Testing

1. Configure your experiment:
   - Modify `configs/defaults.yaml` for dataset and model selection
   - Update paths in `configs/cifar100.yaml` (or corresponding dataset config)

2. Start training with calibration tests:
   ```bash
   python train_net.py
   ```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{luo2025beyond,
  title={Beyond One-Hot Labels: Semantic Mixing for Model Calibration},
  author={Luo, Haoyang and Tao, Linwei and Dong, Minjing and Xu, Chang},
  journal={arXiv preprint arXiv:2504.13548},
  year={2025}
}
```

## Acknowledgements

We are inspired by these excellent works:
1. [RankMixup](https://github.com/cvlab-yonsei/RankMixup) [1]
2. [EDM](https://github.com/NVlabs/edm) [2]
3. [DM-Improves-AT](https://github.com/wzekai99/DM-Improves-AT) [3]

We thank the authors for sharing their code.

[1] Noh, J., Park, H., Lee, J., & Ham, B. (2023). Rankmixup: Ranking-based mixup training for network calibration. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1358-1368).

[2] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35, 26565-26577.

[3] Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., & Yan, S. (2023, July). Better diffusion models further improve adversarial training. In International Conference on Machine Learning (pp. 36246-36263). PMLR.

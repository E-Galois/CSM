# Data Generation

The generation codes are modifed based on [the official implementation of EDM ](https://github.com/NVlabs/edm) [1].

To generate the semantically-mixed images for cifar10 or other datasets, run or modify
> sh runmix.sh

The checkpoints are borrowed from this work [DM-Improves-AT](https://github.com/wzekai99/DM-Improves-AT/tree/main/edm) [2]

# References

[1] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35, 26565-26577.

[2] Wang, Z., Pang, T., Du, C., Lin, M., Liu, W., & Yan, S. (2023, July). Better diffusion models further improve adversarial training. In International Conference on Machine Learning (pp. 36246-36263). PMLR.
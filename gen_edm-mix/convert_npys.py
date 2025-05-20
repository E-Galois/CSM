import os

import numpy as np
from PIL import Image

n_cls = 100
for i in range(n_cls):
    cls_images_1 = np.load(f'./out_cifar100/{i}.npy')
    cls_images_2 = np.load(f'./out_cifar100-2/{i}.npy')
    cls_images = np.concatenate([cls_images_1, cls_images_2], axis=0)
    print(cls_images.shape)
    #os.mkdir(f'./out_cifar100_imgs/{i}')
    for j, img in enumerate(cls_images):
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img.save(f'./out_cifar100_imgs/{i}-{j}.png')

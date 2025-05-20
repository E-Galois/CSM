import os
import numpy as np
from PIL import Image
import torch

dsets = ['c10', 'c100', 'tiny']
for dset in dsets:
    pth_dir = f'./out_{dset}_mix'
    out_imgs_dir = f'./out_{dset}_imgs_mix'
    pth_i = 0
    while os.path.exists(os.path.join(pth_dir, f'{pth_i}.pth')):
        mdict = torch.load(os.path.join(pth_dir, f'{pth_i}.pth'))
        for fn, img in mdict.items():
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img.save(os.path.join(out_imgs_dir, fn))
        print(f'saved images in {pth_i}.pth!')
        pth_i += 1
    print(f'{dset} done!')


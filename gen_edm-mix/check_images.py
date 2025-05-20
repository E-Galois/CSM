import os
import numpy as np
from PIL import Image
import torch


def keys_to_num(k):
    seed, c1, c2 = k.split('_')
    #seed = int(seed)
    #c1 = int(c1)
    #c2 = int(c2)
    return int(seed)


def keys_to_nums(k):
    seed, c1, c2 = k.split('_')
    seed = int(seed)
    c1 = int(c1)
    c2 = int(c2)
    return (seed, c1, c2)


out_imgs_dir = './out_c100_imgs_mix'
pngs = os.listdir(out_imgs_dir)
all_keys_cnt = {}

for name in pngs:
    if name[-4:] != '.png':
        continue
    info = name[:-4]
    seed, c1, c2, lam = info.split('_')
    k = f'{seed}_{c1}_{c2}'
    if all_keys_cnt.get(k) is None:
        all_keys_cnt[k] = 0
    all_keys_cnt[k] += 1
    assert all_keys_cnt[k] <= 8

keys = sorted(list(all_keys_cnt.keys()), key=keys_to_num)
nums = set(keys_to_nums(k) for k in keys)

seed = 0
c1 = 0
c2 = 1
epoch = 0
while epoch < 4:
    if (seed, c1, c2) not in nums:
        print((seed, c1, c2))
    seed += 1
    c2 += 1
    if c2 == 100:
        c1 += 1
        c2 = c1 + 1
        if c1 == 100 - 1:
            # there is now no bigger class values for `other_idx`
            c1 = 0
            c2 = 1
            epoch += 1
 

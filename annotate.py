import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import argparse


@torch.no_grad()
def calc_class_protos(img_dir):
    all_images = os.listdir(img_dir)
    all_images = [fn for fn in all_images if fn.endswith('.png')]
    cls_protos = {}
    for fn in tqdm(all_images):
        if not fn.endswith('.png'):
            continue
        info = fn[:-4]
        idx, cls = info.split('-')
        idx, cls = int(idx), int(cls)
        img = preprocess(Image.open(os.path.join(img_dir, fn))).unsqueeze(0).to(device)
        f_img = model.encode_image(img)
        if cls_protos.get(cls) is None:
            cls_protos[cls] = []
        cls_protos[cls].append(f_img)

    cls_protos_list = [None] * len(cls_protos)
    for cls in cls_protos:
        cls_protos_list[cls] = torch.mean(torch.cat(cls_protos[cls], dim=0), keepdim=True).to(device)
    cat_cls_protos = torch.cat(cls_protos_list, dim=0)
    print('Class prototypes: \n', cat_cls_protos)
    return cat_cls_protos


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='CSM')
    parser.add_argument('--dset_name', type=str, default='c10', help='Dataset label')  # c10/c100/tiny
    parser.add_argument('--clip_path', type=str, default='./your/path/to/CLIP', help='CLIP download path')
    parser.add_argument('--train_path', type=str, default='./your/path/to/train/set', help='Path to the training images')
    parser.add_argument('--mix_path', type=str, default='./your/path/to/generated/samples', help='Path to the generated samples')
    args = parser.parse_args()

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=args.clip_path)
    dset_root = args.train_path
    cls_protos = calc_class_protos(dset_root)

    root = args.mix_path
    all_images_in_root = os.listdir(root)
    all_keys = []
    for fn in tqdm(all_images_in_root):
        if not fn.endswith('.png'):
            continue
        info = fn[:-4]
        gen_seed, c1, c2, gen_lam = info.split('_')
        all_keys.append(f'{gen_seed}_{c1}_{c2}')

    lams_all = []
    lam_dict_to_save = {}

    for fn_pref in tqdm(all_keys):
        all_fns = [fn for fn in all_images_in_root if fn.startswith(fn_pref)]
        group_images = []

        for fn in all_fns:
            info = fn[:-4]
            gen_seed, c1, c2, gen_lam = info.split('_')
            gen_seed, c1, c2, gen_lam = int(gen_seed), int(c1), int(c2), float(gen_lam)
            img = preprocess(Image.open(os.path.join(root, fn))).unsqueeze(0).to(device)
            group_images.append(img)
        group_images = torch.cat(group_images, dim=0)
        in_img_features = model.encode_image(group_images)
        src_feature = cls_protos[c1]
        tgt_feature = cls_protos[c2]
        delta_x = src_feature - tgt_feature
        delta_x_2 = delta_x @ delta_x.t()
        lams = []
        for in_img_feature in in_img_features:
            lam_by_x = ((in_img_feature - tgt_feature) @ delta_x.t() / delta_x_2).item()
            lams.append(lam_by_x)
            lams_all.append(lam_by_x)
            lam_dict_to_save[fn] = lam_by_x
    torch.save(lam_dict_to_save, f'./annotations/clip_lam_anno_edm_{args.dset_name}.pth')

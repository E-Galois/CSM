"""
Create train, valid, test iterators for CIFAR-100.
Train set size: 45000
Val set size: 5000
Test set size: 10000
"""

import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import os
from PIL import Image


class ImageCalibDatasetMultiAug(Dataset):
    def __init__(self, img_root, calib_root=None, n_cls=10, img_split='-', calib_split='_', transform=None, in_memory=False,
                 n_aug=3, target_re_file=None, valid_only=False, std_set_size=8, s_value=None):
        imgs = os.listdir(img_root)
        calibs = os.listdir(calib_root)
        self.img_root = img_root
        self.calib_root = calib_root
        self.n_cls = n_cls
        self.img_dict = {}
        self.calib_dict = {}
        self.transform = transform
        self.in_memory = in_memory
        self.n_aug = n_aug
        self.valid_only = valid_only
        self.std_set_size = std_set_size
        self.target_re_data = torch.load(target_re_file) if target_re_file is not None else None
        assert n_aug <= std_set_size - 2

        for name in imgs:
            info = name[:-4]
            img_id, img_cls = info.split(img_split)
            img_id = int(img_id)
            img_cls = int(img_cls)
            img_read = os.path.join(img_root, name)
            if in_memory:
                img_read = self.read_image(img_read)
            self.img_dict[img_id] = ((img_read, img_cls))
        if not valid_only:
            print(f's_value is {s_value}')
            for name in calibs:
                info = name[:-4]
                gen_seed, c1, c2, gen_lam = info.split(calib_split)
                gen_seed, c1, c2, gen_lam = int(gen_seed), int(c1), int(c2), float(gen_lam)
                calib_id = f'{gen_seed}_{c1}_{c2}'
                if self.calib_dict.get(calib_id) is None:
                    self.calib_dict[calib_id] = []
                calib_read = os.path.join(calib_root, name)
                if in_memory:
                    calib_read = self.read_image(calib_read)
                lam = float(self.target_re_data[name])
                lam = 1 / (1 + np.exp(-(s_value * (lam - 0.5))))
                if lam > 1.0:
                    lam = 1.0
                elif lam < 0.0:
                    lam = 0.0
                self.calib_dict[calib_id].append((calib_read, gen_seed, c1, c2, lam))
            self.calib_dict_keys = sorted(list(self.calib_dict.keys()))
            for calib_id in self.calib_dict:
                self.calib_dict[calib_id] = sorted(self.calib_dict[calib_id], key=lambda item: item[-1], reverse=True)

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, index):
        data = {}
        # load from original first
        img_read, img_cls = self.img_dict[index]
        if not self.in_memory:
            img_read = self.read_image(img_read)
        data['inputs'] = img_read
        data['labels'] = np.array(img_cls)
        if self.valid_only:
            return data
        i_group = np.random.randint(len(self.calib_dict_keys))
        k_group = self.calib_dict_keys[i_group]
        c1 = int(k_group.split('_')[1])
        c2 = int(k_group.split('_')[2])
        calib_set_size = len(self.calib_dict[k_group])
        while (c1 != img_cls and c2 != img_cls) or calib_set_size != 8:
            i_group = np.random.randint(len(self.calib_dict_keys))
            k_group = self.calib_dict_keys[i_group]
            c1 = int(k_group.split('_')[1])
            c2 = int(k_group.split('_')[2])
            calib_set_size = len(self.calib_dict[k_group])
        index_calib = np.random.permutation(calib_set_size - 2)[:self.n_aug] + 1
        index_calib = np.concatenate([np.array([0]) if c1 == img_cls else np.array([calib_set_size - 1]), index_calib])
        calib_read_all = []
        label_all = []
        for i_calib in index_calib:
            calib_read, gen_seed, c1, c2, lam = self.calib_dict[k_group][i_calib]
            if not self.in_memory:
                calib_read = self.read_image(calib_read)
            calib_read_all.append(calib_read)
            lab_tar_one_hot = torch.zeros((self.n_cls,), requires_grad=False, dtype=torch.float32)
            lab_tar_one_hot[c1] = lam
            lab_tar_one_hot[c2] = 1 - lam
            label_all.append(lab_tar_one_hot)
        
        calib_read_all = torch.stack(calib_read_all)
        label_all = torch.stack(label_all)
        data['mix_inputs'] = calib_read_all
        data['target_re'] = label_all
        return data

    def read_image(self, path):
        img = Image.open(path)
        if (img.mode == 'L'):
            img = img.convert('RGB')
        assert self.transform is not None
        return self.transform(img)


def get_train_valid_loader(batch_size,
                           augment,
                           random_seed,
                           s_value=None,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           get_val_temp=0,
                           data_dir=None,
                           calib_dir=None,
                           target_re_file=None):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - get_val_temp: set to 1 if temperature is to be set on a separate
                    val set other than normal val set.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    data_dir = os.path.expanduser(data_dir)
    calib_dir = os.path.expanduser(calib_dir)
    target_re_file = os.path.expanduser(target_re_file)
    n_cls = 10
    n_aug = 1
    train_dataset = ImageCalibDatasetMultiAug(data_dir, n_cls=n_cls, calib_root=calib_dir, transform=train_transform, n_aug=n_aug, target_re_file=target_re_file, s_value=s_value)
    valid_dataset = ImageCalibDatasetMultiAug(data_dir, n_cls=n_cls, transform=valid_transform, n_aug=n_aug, valid_only=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    if get_val_temp > 0:
        valid_temp_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=False, transform=valid_transform,
        )
        split = int(np.floor(get_val_temp * split))
        valid_idx, valid_temp_idx = valid_idx[split:], valid_idx[:split]
        valid_temp_sampler = SubsetRandomSampler(valid_temp_idx)
        valid_temp_loader = torch.utils.data.DataLoader(
            valid_temp_dataset, batch_size=batch_size, sampler=valid_temp_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
        # worker_init_fn=worker_init_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # worker_init_fn=worker_init_fn,
    )
    if get_val_temp > 0:
        return (train_loader, valid_loader, valid_temp_loader)
    else:
        return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    data_dir=None):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    data_dir = os.path.expanduser(data_dir)
    n_cls = 10
    n_aug = 1
    dataset = ImageCalibDatasetMultiAug(data_dir, n_cls=n_cls, transform=transform, n_aug=n_aug, valid_only=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


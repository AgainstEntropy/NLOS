# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 14:47
# @Author  : WangYihao
# @File    : data.py

import os
import shutil

import torch
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.io import read_image
from tqdm import tqdm

from data.loader import load_frames
from data.preprocess import sub_mean, reduce


class MyDataset(Dataset):
    def __init__(self, ROOT, reduce_mode='W', transform=None):
        assert reduce_mode in 'HW'
        self.reduce_mode = reduce_mode
        self.data = []
        cls_dirs = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
        print(cls_dirs)
        for idx, DIR in enumerate(cls_dirs):
            class_path = os.path.join(ROOT, DIR)
            class_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            self.data.extend(list(zip(class_files, [idx] * len(class_files))))
        self.transform = transform

    def __getitem__(self, index):
        mat_path, label = self.data[index]
        mat = loadmat(mat_path)[f'reduce_{self.reduce_mode}']  # (W, T, C)

        if self.transform is not None:
            mat = self.transform(mat)

        return mat, label

    def __len__(self):
        return len(self.data)


def make_dataset(stat_file='/mnt/cfs/wangyh/blender/blank_wall/datasets/statistic.txt',
                 png_root='/mnt/cfs/wangyh/blender/blank_wall/output_png',
                 dataset_dir='/mnt/cfs/wangyh/blender/blank_wall/datasets',
                 device=None):
    with open(stat_file, mode='r') as f:
        png_dirs = [line.strip() for line in f.readlines()]

    classes = set([line.split('/')[0] for line in png_dirs])
    print(f'{len(classes)} classes in total.')
    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    for png_dir in tqdm(png_dirs):
        mat_path = os.path.join(dataset_dir, f'{png_dir}.mat')
        if os.path.exists(mat_path):
            continue

        png_abs_dir = os.path.join(png_root, png_dir)
        frames = load_frames(png_abs_dir, device)
        frames_sub_mean = sub_mean(frames)
        reduce_H, reduce_W = reduce(frames_sub_mean)  # (W or H, T, RGB)
        save_dict = {"reduce_H": reduce_H.cpu().numpy(),
                     "reduce_W": reduce_W.cpu().numpy()}
        savemat(mat_path, save_dict)


def create_loaders(full_dataset, ratio, kwargs=None):
    assert sum(ratio) == 1
    if kwargs is None:
        kwargs = {
            'batch_size': 16,  # default:1
            'shuffle': True,  # default:False
            'num_workers': 2,  # default:0
            'pin_memory': True,  # default:False
            'drop_last': True,  # default:False
            'prefetch_factor': 4,  # default:2
            'persistent_workers': False  # default:False
        }

    train_size = int(len(full_dataset) * ratio[0])
    validate_size = int(len(full_dataset) * ratio[1])
    test_size = len(full_dataset) - validate_size - train_size
    train_dataset, validate_dataset, test_dataset = \
        random_split(full_dataset, [train_size, validate_size, test_size])

    train_loader = DataLoader(train_dataset, **kwargs)
    val_loader = DataLoader(validate_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)

    return train_loader, val_loader, test_loader

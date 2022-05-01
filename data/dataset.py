# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 14:47
# @Author  : WangYihao
# @File    : data.py

import os
import shutil

import torch
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

from data.loader import load_frames
from data.preprocess import sub_mean, reduce


class MyDataset(Dataset):
    def __init__(self, ROOT, reduce_mode='W', transform=None):
        assert reduce_mode in 'HW'
        self.reduce_mode = reduce_mode
        self.data = []
        cls_dirs = sorted([d for d in os.listdir(ROOT) if os.path.isdir(d)])
        for idx, DIR in enumerate(cls_dirs):
            class_path = os.path.join(ROOT, DIR)
            class_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            self.data.extend(list(zip(class_files, [idx] * len(class_files))))
        self.transform = transform

    def __getitem__(self, index):
        img_dir, label = self.data[index]
        img = loadmat(img_dir)[self.reduce_mode].transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

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
        reduce_H, reduce_W = reduce(frames_sub_mean)
        save_dict = {"reduce_H": reduce_H.cpu().numpy(),
                     "reduce_W": reduce_W.cpu().numpy()}
        savemat(mat_path, save_dict)

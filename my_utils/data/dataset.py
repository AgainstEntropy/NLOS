# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 14:47
# @Author  : WangYihao
# @File    : data.py

import os

import torch
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from my_utils.data.loader import load_frames
from my_utils.data.preprocess import sub_mean, reduce


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


def build_dataset(png_root='/mnt/cfs/wangyh/blender/blank_wall/output_random',
                  stat_file='/mnt/cfs/wangyh/blender/blank_wall/datasets/random/state.txt',
                  dataset_dir='/mnt/cfs/wangyh/blender/blank_wall/datasets/random',
                  noise_factor: float = None,
                  resize=None,
                  device=None):
    with open(stat_file, mode='r') as f:
        png_dirs = [line.strip() for line in f.readlines()]

    classes = set([line.split('/')[0] for line in png_dirs])
    print(f'{len(classes)} classes in total: {classes}')
    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

    for png_dir in tqdm(png_dirs):
        mat_path = os.path.join(dataset_dir, f'{png_dir}.mat')
        if os.path.exists(mat_path):
            continue

        png_abs_dir = os.path.join(png_root, png_dir)
        frames = load_frames(png_abs_dir, output_size=resize, device=device)[:, :, :, :3]  # RGBA -> RGB
        if noise_factor is not None:
            frames = frames + 255 * noise_factor * torch.randn_like(frames)
        frames_sub_mean = sub_mean(frames)
        reduce_H, reduce_W = reduce(frames_sub_mean)  # (W or H, T, RGB)
        save_dict = {"reduce_H": reduce_H.cpu().numpy(),
                     "reduce_W": reduce_W.cpu().numpy()}
        savemat(mat_path, save_dict)


def make_dataset(
        dataset_dir: str,
        phase: str = 'train',
        ratio: float = 0.8,
        reduced_mode='W',
        transform=None
):
    full_dataset = MyDataset(ROOT=dataset_dir, reduce_mode=reduced_mode, transform=transform)
    if phase == 'train':
        train_size = int(len(full_dataset) * ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == 'test':
        return full_dataset


def check_dataset(check_root='/mnt/cfs/wangyh/blender/blank_wall/output_random',
                  output_dir='/mnt/cfs/wangyh/blender/blank_wall/datasets/random',
                  ):
    actions = ['Clapping', 'Flair', 'Hip Hop Dancing', 'Jump', 'Jumping Jacks',
               'Mma Kick', 'Rumba Dancing', 'Standing', 'Sweep Fall', 'Waving']
    actions_num = dict.fromkeys(actions)
    completed_dirs = []
    data_dict = {
        'floor': {
            'white_tiling': 0,
            'grey_tiling': 0,
            'wooden_floor_1': 0,
            'wooden_floor_2': 0
        },
        'rot_num': [0, ] * 4,
        'pos_num': [0, ] * 5,
        'character': {
            'Bryce': 0,
            'Jody': 0,
            'Kate': 0,
            'Sophie': 0
        }
    }
    for action in tqdm(actions):
        action_dir = os.path.join(check_root, action)
        png_dirs = os.listdir(action_dir)
        actions_num[action] = len(png_dirs)
        for png_dir in png_dirs:
            png_num = len([f for f in os.listdir(os.path.join(action_dir, png_dir)) if f.endswith('.png')])
            if png_num == 64:
                completed_dirs.append(action + '/' + png_dir + '\n')
                # with open(png_dir + 'config.yaml', 'r') as stream:
                #     config = yaml.load(stream, Loader=yaml.FullLoader)
                floor, rot_num, pos_num, character = analysis_dir_name(png_dir)
                data_dict['floor'][floor] += 1
                if rot_num is not None:
                    data_dict['rot_num'][rot_num] += 1
                if pos_num is not None:
                    data_dict['pos_num'][pos_num] += 1
                data_dict['character'][character] += 1
            # png_dir = os.path.join(output_root, f'{action}/{floor}_r{rot_num}_p{pos_num}_{character}_{tic}/')

    for k, v in actions_num.items():
        print(f'{k}: {v}')
    for k, v in data_dict.items():
        print(f'{k}\t {v}')

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, 'state.txt')
    with open(result_file, mode='a+') as f:
        f.truncate(0)
        f.writelines(completed_dirs)


def analysis_dir_name(png_dir: str):
    floor, rot_num, pos_num, character = (None,) * 4
    for f in ['white_tiling', 'grey_tiling', 'wooden_floor_1', 'wooden_floor_2']:
        if f in png_dir:
            floor = f
            break

    for ch in ['Bryce', 'Jody', 'Kate', 'Sophie']:
        if ch in png_dir:
            character = ch
            break

    for r in [0, 1, 2, 3]:
        if f'r{r}' in png_dir:
            rot_num = r
            break

    for p in [0, 1, 2, 3, 4]:
        if f'p{p}' in png_dir:
            pos_num = p
            break

    return floor, rot_num, pos_num, character

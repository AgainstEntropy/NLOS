# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 14:47
# @Author  : WangYihao
# @File    : data.py

import os

import torch
import yaml
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from my_utils.data.loader import load_frames, load_video
from my_utils.data.preprocess import sub_mean, reduce
from my_utils.utils import get_cls_from_position


class MyDataset_base(Dataset):
    def __init__(self, dataset_root, raw_data_root=None, reduce_mode='W', transform=None):
        assert reduce_mode in 'HW'
        self.reduce_mode = reduce_mode
        self.dataset_root = dataset_root
        self.raw_data_root = raw_data_root
        self.transform = transform

        self.data = []
        self.get_all_data()

    def get_all_data(self):
        raise NotImplementedError

    def __getitem__(self, index):
        mat_path, label = self.data[index]
        mat = loadmat(mat_path)[f'reduce_{self.reduce_mode}']  # (W, T, C)

        if self.transform is not None:
            mat = self.transform(mat)

        return mat, label

    def __len__(self):
        return len(self.data)


class MyDataset_actions(MyDataset_base):
    def __init__(self, dataset_root, reduce_mode='W', transform=None):
        super().__init__(dataset_root=dataset_root, reduce_mode=reduce_mode, transform=transform)

    def get_all_data_old(self):
        cls_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)]
        cls_dirs = sorted([d for d in cls_dirs if os.path.isdir(d)])
        print([os.path.split(d)[-1] for d in cls_dirs])
        for idx, DIR in enumerate(cls_dirs):
            class_path = os.path.join(self.dataset_root, DIR)
            class_files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            self.data.extend(list(zip(class_files, [idx] * len(class_files))))

    def get_all_data(self):
        classes = ['Clap', 'Crouch to Stand', 'Dance', 'Idle', 'Jump']
        cls_to_idx = dict(zip(classes, range(len(classes))))

        mats = sorted([f for f in os.listdir(self.dataset_root) if f.endswith('.mat')])
        cls_list = [mat.split('_')[0] for mat in mats]
        labels = [cls_to_idx[cls] for cls in cls_list]
        mat_abs_paths = [os.path.join(self.dataset_root, mat) for mat in mats]

        self.data.extend(list(zip(mat_abs_paths, labels)))


class MyDataset_positions(MyDataset_base):
    def __init__(self, dataset_root, raw_data_root, reduce_mode='W', transform=None):
        super().__init__(dataset_root=dataset_root, raw_data_root=raw_data_root,
                         reduce_mode=reduce_mode, transform=transform)

    def get_all_data(self):
        cls_dirs = [os.path.join(self.dataset_root, d) for d in os.listdir(self.dataset_root)]
        cls_dirs = sorted([d for d in cls_dirs if os.path.isdir(d)])
        for cls in cls_dirs:
            class_abs_dir = os.path.join(self.dataset_root, cls)
            for mat in os.listdir(class_abs_dir):
                mat_abs_path = os.path.join(class_abs_dir, mat)
                # /mnt/cfs/wangyh/blender/blank_wall/datasets/random/Clapping/Bryce_grey_tiling_05-18_02:22.mat
                raw_png_dir = os.path.join(self.raw_data_root, cls, mat.replace('.mat', ''))
                # /mnt/cfs/wangyh/blender/blank_wall/output_random/Clapping/Bryce_grey_tiling_05-18_02:22
                pos_cls = get_cls_from_position(os.path.join(raw_png_dir, 'configs.yaml'))
                self.data.append((mat_abs_path, pos_cls))


def build_dataset(png_root='/mnt/cfs/wangyh/blender/blank_wall/output_random',
                  stat_file='/mnt/cfs/wangyh/blender/blank_wall/datasets/random/state.txt',
                  dataset_dir='/mnt/cfs/wangyh/blender/blank_wall/datasets/random',
                  noise_factor: float = None,
                  resize=None):
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
        frames = load_frames(png_abs_dir, output_size=resize)[:, :, :, :3]  # RGBA -> RGB
        if noise_factor is not None:
            frames = frames + 255 * noise_factor * torch.randn_like(frames)
        frames_sub_mean = sub_mean(frames)
        reduce_H, reduce_W = reduce(frames_sub_mean)  # (W or H, T, RGB)
        save_dict = {"reduce_H": reduce_H.cpu().numpy(),
                     "reduce_W": reduce_W.cpu().numpy()}
        savemat(mat_path, save_dict)


def make_dataset(
        dataset_root: str,
        raw_data_root: str,
        cls_type: str = 'action',
        phase: str = 'train',
        ratio: float = 0.8,
        reduced_mode='W',
        transform=None
):
    DS_class = {'action': MyDataset_actions, 'position': MyDataset_positions}[cls_type]
    full_dataset = DS_class(dataset_root=dataset_root, raw_data_root=raw_data_root,
                            reduce_mode=reduced_mode, transform=transform)
    if phase == 'train':
        train_size = int(len(full_dataset) * ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == 'test':
        return full_dataset


def check_dataset(check_root='/mnt/cfs/wangyh/blender/blank_wall/output_variety',
                  output_dir='/mnt/cfs/wangyh/blender/blank_wall/datasets/variety',
                  mode='train'
                  ):
    if mode == 'unseen':
        check_root += '_unseen'
        output_dir += '_unseen'
    actions = ['Clap', 'Crouch to Stand', 'Dance', 'Idle', 'Jump']
    floors = {
        'train': ['white_tiling', 'grey_tiling', 'wooden_floor_1', 'wooden_floor_2'],
        'unseen': ['yellow_tiling']
    }

    data_dict = {
        'floor': dict.fromkeys(floors[mode], 0),
        'rot_num': [0, ] * 4,
        'pos_num': [0, ] * (5 if mode == 'train' else 4)
    }

    actions_num = dict.fromkeys(actions, 0)
    completed_dirs = []
    png_dirs = os.listdir(check_root)
    for png_dir in tqdm(png_dirs):
        png_dir = os.path.join(check_root, png_dir)
        render_state, action, rot_num, pos_num, floor = get_configs(png_dir)
        if render_state == 'finish':
            png_num = len([f for f in os.listdir(os.path.join(check_root, png_dir)) if f.endswith('.png')])
            if png_num == 64:
                completed_dirs.append(png_dir + '\n')
                actions_num[action] += 1
                data_dict['floor'][floor] += 1
                data_dict['rot_num'][rot_num] += 1
                data_dict['pos_num'][pos_num] += 1

    print(actions_num)
    for k, v in data_dict.items():
        print(f'{k}\t {v}')

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, 'complete.txt')
    with open(result_file, mode='a+') as f:
        f.truncate(0)
        f.writelines(completed_dirs)


def analysis_dir_name(png_dir: str):
    floor, rot_num, pos_num, character = (None,) * 4
    for f in ['white_tiling', 'grey_tiling', 'wooden_floor_1', 'wooden_floor_2', 'yellow_tiling']:
        if f in png_dir:
            floor = f
            break

    for ch in ['Bryce', 'Jody', 'Kate', 'Sophie', 'Suzie']:
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


def get_configs(dir_path: str):
    with open(os.path.join(dir_path, 'configs.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config['render_state'], config['action'], config['rot_num'], config['pos_num'], config['floor']


class raw_png_processor(object):
    def __init__(self,
                 project_root: str = '/mnt/cfs/wangyh/blender/blank_wall/',
                 dataset_name: str = 'variety'):

        self.result_file = None
        self.raw_png_root = os.path.join(project_root, 'output_' + dataset_name)
        self.output_dir = os.path.join(project_root, 'datasets/' + dataset_name)

        self.actions = ['Clap', 'Crouch to Stand', 'Dance', 'Idle', 'Jump']

    def check_data(self, mode='train'):
        if mode == 'unseen':
            check_root = self.raw_png_root + '_unseen'
            output_dir = self.output_dir + '_unseen'
        else:
            check_root = self.raw_png_root
            output_dir = self.output_dir
        floors = {
            'train': ['white_tiling', 'grey_tiling', 'wooden_floor_1', 'wooden_floor_2'],
            'unseen': ['yellow_tiling']
        }

        data_dict = {
            'floor': dict.fromkeys(floors[mode], 0),
            'rot_num': [0, ] * 4,
            'pos_num': [0, ] * (5 if mode == 'train' else 4)
        }

        actions_num = dict.fromkeys(self.actions, 0)
        completed_dirs = []
        png_dirs = os.listdir(check_root)
        for png_dir in tqdm(png_dirs):
            abs_dir = os.path.join(check_root, png_dir)
            render_state, action, rot_num, pos_num, floor = get_configs(abs_dir)
            if render_state == 'finish':
                png_num = len([f for f in os.listdir(os.path.join(check_root, abs_dir)) if f.endswith('.png')])
                if png_num == 64:
                    completed_dirs.append(png_dir + '\n')
                    actions_num[action] += 1
                    data_dict['floor'][floor] += 1
                    data_dict['rot_num'][rot_num] += 1
                    data_dict['pos_num'][pos_num] += 1

        print(actions_num)
        for k, v in data_dict.items():
            print(f'{k}\t {v}')

        os.makedirs(output_dir, exist_ok=True)
        self.result_file = os.path.join(output_dir, 'complete.txt')
        with open(self.result_file, mode='a+') as f:
            f.truncate(0)
            f.writelines(completed_dirs)

    def build_dataset(self,
                      noise_factor: float = None,
                      resize=None):
        with open(self.result_file, mode='r') as f:
            png_dirs = [line.strip() for line in f.readlines()]

        print(f'{len(self.actions)} classes in total: {self.actions}')

        for png_dir in tqdm(png_dirs):
            mat_path = os.path.join(self.output_dir, f'{png_dir}.mat')
            if os.path.exists(mat_path):
                continue

            png_abs_dir = os.path.join(self.raw_png_root, png_dir)
            frames = load_frames(png_abs_dir, output_size=resize)
            if noise_factor is not None:
                frames = frames + 255 * noise_factor * torch.randn_like(frames)
            frames_sub_mean = sub_mean(frames)
            reduce_H, reduce_W = reduce(frames_sub_mean)  # (W or H, T, RGB)
            save_dict = {"reduce_H": reduce_H.cpu().numpy(),
                         "reduce_W": reduce_W.cpu().numpy()}
            savemat(mat_path, save_dict)


class raw_video_processor(object):
    def __init__(self,
                 project_root: str = '/mnt/cfs/wangyh/blender/blank_wall/',
                 dataset_name: str = 'video_test'):
        self.result_file = None
        self.raw_video_root = os.path.join(project_root, 'videos/' + dataset_name)
        self.output_dir = os.path.join(project_root, 'datasets/' + dataset_name)

        self.actions = ['Clapping', 'Flair', 'Hip Hop Dancing', 'Jump', 'Jumping Jacks',
                        'Mma Kick', 'Rumba Dancing', 'Standing', 'Sweep Fall', 'Waving']

    def build_dataset(self, resize=(256, 256)):
        for action in tqdm(self.actions):
            action_output_dir = os.path.join(self.output_dir, action)
            os.makedirs(action_output_dir, exist_ok=True)
            action_video_dir = os.path.join(self.raw_video_root, action)
            for video in os.listdir(action_video_dir):
                video_path = os.path.join(action_video_dir, video)
                vframes = load_video(video_path, output_size=resize)
                frames_sub_mean = sub_mean(vframes)
                reduce_H, reduce_W = reduce(frames_sub_mean)  # (W or H, T, RGB)
                save_dict = {"reduce_H": reduce_H.cpu().numpy(),
                             "reduce_W": reduce_W.cpu().numpy()}
                mat_path = os.path.join(action_output_dir, f'{action}_{video}')
                savemat(mat_path, save_dict)


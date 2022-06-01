# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 14:47
# @Author  : WangYihao
# @File    : data.py

import os

import torch
import yaml
from scipy.io import savemat, loadmat
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from my_utils.data.loader import load_frames, load_video
from my_utils.data.preprocess import sub_mean, reduce
from my_utils.utils import get_cls_from_position


class MyDataset(Dataset):
    def __init__(self, dataset_root, reduce_mode='W', transform=None,
                 mat_name='N0', cls_mode='action'):
        assert reduce_mode in 'HW'
        self.reduce_mode = reduce_mode
        self.dataset_root = dataset_root
        self.transform = transform
        self.mat_name = 'reduced_data_' + mat_name + '.mat'
        self.cls_mode = cls_mode

        self.data = []
        self.get_all_data()

    def get_all_data(self):
        print('Loading dataset:')
        mat_paths = []
        labels = []

        for png_dir in tqdm(os.listdir(self.dataset_root)):
            png_abs_dir = os.path.join(self.dataset_root, png_dir)
            if not os.path.isdir(png_abs_dir):
                continue
            mat_abs_path = os.path.join(png_abs_dir, self.mat_name)
            if os.path.exists(mat_abs_path):
                mat_paths.append(mat_abs_path)
                if self.cls_mode == 'action':
                    classes = ['Clap', 'Crouch to Stand', 'Dance', 'Idle', 'Jump', 'Kick', 'Punch', 'Sit']
                    cls_to_idx = dict(zip(classes, range(len(classes))))
                    cls_idx = cls_to_idx[get_configs(png_abs_dir)['action']]
                elif self.cls_mode == 'position':
                    cls_idx = get_configs(png_abs_dir)['pos_num']
                labels.append(cls_idx)

        self.data.extend(list(zip(mat_paths, labels)))

    def __getitem__(self, index):
        mat_path, label = self.data[index]
        mat = loadmat(mat_path)[f'reduce_{self.reduce_mode}']  # (W, T, C)

        if self.transform is not None:
            mat = self.transform(mat)

        return mat, label

    def __len__(self):
        return len(self.data)


def split_dataset(
        dataset_root: str,
        cls_mode: str = 'action',
        phase: str = 'train',
        ratio: float = 0.8,
        reduced_mode='W',
        mat_name='N0',
        transform=None
):
    full_dataset = MyDataset(dataset_root=dataset_root, reduce_mode=reduced_mode,
                             transform=transform, mat_name=mat_name, cls_mode=cls_mode)
    if phase == 'train':
        train_size = int(len(full_dataset) * ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == 'test':
        return full_dataset


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
    return config


class raw_png_processor(object):
    def __init__(self,
                 project_root: str = '/mnt/cfs/wangyh/blender/blank_wall/',
                 dataset_name: str = 'variety',
                 mode: str = 'train'):

        self.raw_png_root = os.path.join(project_root, 'raw_pngs/' + dataset_name)
        self.result_file = None
        self.mode = mode
        if mode == 'unseen':
            self.raw_png_root = self.raw_png_root + '_unseen'
        assert os.path.exists(self.raw_png_root)

        self.actions = ['Clap', 'Crouch to Stand', 'Dance', 'Idle', 'Jump', 'Kick', 'Punch', 'Sit']

    def check_data(self):
        floors = {
            'train': ['white_tiling', 'grey_tiling', 'wooden_floor_1', 'wooden_floor_2'],
            'unseen': ['yellow_tiling']
        }

        data_dict = {
            'floor': dict.fromkeys(floors[self.mode], 0),
            'rot_num': [0, ] * 4,
            'pos_num': [0, ] * (5 if self.mode == 'train' else 4)
        }

        actions_num = dict.fromkeys(self.actions, 0)
        completed_dirs = []
        png_dirs = os.listdir(self.raw_png_root)
        for png_dir in tqdm(png_dirs):
            abs_dir = os.path.join(self.raw_png_root, png_dir)
            if not os.path.isdir(abs_dir):
                continue
            config = get_configs(abs_dir)
            if config['render_state'] == 'finish':
                png_num = len([f for f in os.listdir(os.path.join(self.raw_png_root, abs_dir)) if f.endswith('.png')])
                if png_num == 64:
                    completed_dirs.append(png_dir + '\n')
                    actions_num[config['action']] += 1
                    data_dict['floor'][config['floor']] += 1
                    data_dict['rot_num'][config['rot_num']] += 1
                    data_dict['pos_num'][config['pos_num']] += 1

        print(actions_num)
        for k, v in data_dict.items():
            print(f'{k}\t {v}')

        self.result_file = os.path.join(self.raw_png_root, 'render_state.txt')
        with open(self.result_file, mode='a+') as f:
            f.truncate(0)
            f.writelines(completed_dirs)

    def build_dataset(self,
                      noise_factor: float,
                      resize=None):
        with open(self.result_file, mode='r') as f:
            png_dirs = [line.strip() for line in f.readlines()]

        print(f'{len(self.actions)} classes in total: {self.actions}')

        for png_dir in tqdm(png_dirs):
            png_abs_dir = os.path.join(self.raw_png_root, png_dir)
            mat_path = os.path.join(png_abs_dir, f'reduced_data_N{noise_factor}.mat')
            if os.path.exists(mat_path):
                continue

            frames = load_frames(png_abs_dir, output_size=resize)
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


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


# from my_utils.utils import get_cls_from_position


class MyDataset(Dataset):
    def __init__(self,
                 dataset_root: str,
                 modal='image',
                 mat_name='N0',  # for video, mat_name is 128_N0
                 reduce_mode='W',
                 transform=None,
                 cls_mode='action'):
        assert reduce_mode in 'HW'
        self.dataset_root = dataset_root
        self.transform = transform
        self.cls_mode = cls_mode
        if cls_mode == 'action':
            classes = ['Being hit', 'Clap', 'Crouch to Stand', 'Dance', 'Hanging',
                       'Idle', 'Jump', 'Kick', 'Lying down', 'Punch',
                       'Sit', 'Spin', 'Squat', 'Stand to crouch', 'Stand to kneel',
                       'Strafing', 'Throw', 'Turn around', 'Waving hands', 'Yelling']
            self.cls_to_idx = dict(zip(classes, range(len(classes))))

        self.modal = modal
        mat_modal = 'reduced_data' if modal == 'image' else 'video'
        self.mat_name = f'{mat_modal}_' + mat_name + '.mat'
        self.mat_paths = []
        self.labels = []
        self.get_all_data()

        if self.modal == 'video':
            self.mat_key = 'video'  # (T, H, W, C)
        elif self.modal == 'image':
            self.mat_key = f'reduce_{reduce_mode}'  # (W, T, C)

    def get_all_data(self):
        for png_dir in tqdm(os.listdir(self.dataset_root)):
            png_abs_dir = os.path.join(self.dataset_root, png_dir)
            if not os.path.isdir(png_abs_dir):
                continue
            mat_abs_path = os.path.join(png_abs_dir, self.mat_name)
            if os.path.exists(mat_abs_path):
                self.mat_paths.append(mat_abs_path)
                if self.cls_mode == 'action':
                    action = get_configs(png_abs_dir)['action']
                    cls_idx = self.cls_to_idx[action['class'] if type(action) == dict else action]
                elif self.cls_mode == 'position':
                    cls_idx = get_configs(png_abs_dir)['pos_num']
                self.labels.append(cls_idx)

    def __getitem__(self, index):
        mat_path = self.mat_paths[index]
        mat = loadmat(mat_path)[self.mat_key]  # (T, H, W, C) or (W, T, C)

        if self.modal == 'image' and self.transform is not None:
            mat = self.transform(mat)  # (C, H, W)
        elif self.modal == 'video':
            mat = torch.from_numpy(mat)  # (C, T, H, W)

        return mat, self.labels[index]

    def __len__(self):
        return len(self.mat_paths)


def split_dataset(
        dataset_root: str,
        modal: str = ' video',
        cls_mode: str = 'action',
        phase: str = 'train',
        ratio: float = 0.8,
        reduced_mode='W',
        mat_name='N0',
        transform=None
):
    full_dataset = MyDataset(dataset_root=dataset_root, modal=modal, reduce_mode=reduced_mode,
                             transform=transform, mat_name=mat_name, cls_mode=cls_mode)
    if phase == 'train':
        train_size = int(len(full_dataset) * ratio)
        val_size = len(full_dataset) - train_size
        return random_split(full_dataset, [train_size, val_size])
    elif phase == 'test':
        return full_dataset


def get_configs(dir_path: str):
    with open(os.path.join(dir_path, 'configs.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


class raw_png_processor(object):
    def __init__(self,
                 project_root: str = '/mnt/cfs/wangyh/blender/blank_wall/',
                 dataset_root: str = '/mnt/lustre/wangyihao/nlos_raw_pngs',
                 mode: str = 'train'):

        self.actions_num = None
        self.data_dict = None
        self.actions = None
        self.raw_png_root = os.path.join(dataset_root, mode)
        assert os.path.exists(self.raw_png_root)
        self.result_file = os.path.join(self.raw_png_root, 'render_state.txt')
        self.mode = mode

    def check_data(self, resume=False):
        if resume:
            with open(self.result_file, mode='r') as f:
                completed_dirs = [line.strip() for line in f.readlines()]
                # print(completed_dirs[:5])
        else:
            floors = {
                'train': ['white_tiling', 'grey_tiling', 'wooden_floor_1', 'wooden_floor_2'],
                'unseen': ['yellow_tiling']
            }
            self.actions = ['Being hit', 'Clap', 'Crouch to Stand', 'Dance', 'Hanging',
                            'Idle', 'Jump', 'Kick', 'Lying down', 'Punch',
                            'Sit', 'Spin', 'Squat', 'Stand to crouch', 'Stand to kneel',
                            'Strafing', 'Throw', 'Turn around', 'Waving hands', 'Yelling']
            self.data_dict = {
                'floor': dict.fromkeys(floors[self.mode], 0),
                'rot_num': [0, ] * 4,
                'pos_num': [0, ] * (5 if self.mode == 'train' else 4)
            }
            self.actions_num = dict.fromkeys(self.actions, 0)
            completed_dirs = []

        png_dirs = os.listdir(self.raw_png_root)
        for png_dir in tqdm(png_dirs):
            if png_dir in completed_dirs:
                continue
            abs_dir = os.path.join(self.raw_png_root, png_dir)
            if not os.path.isdir(abs_dir):
                continue
            config = get_configs(abs_dir)
            if config['render_state'] == 'finish':
                png_num = len([f for f in os.listdir(os.path.join(self.raw_png_root, abs_dir)) if f.endswith('.png')])
                if png_num == 64:
                    completed_dirs.append(png_dir)
                    action = config['action']
                    self.actions_num[action['class'] if type(action) == dict else action] += 1
                    self.data_dict['floor'][config['floor']] += 1
                    self.data_dict['rot_num'][config['rot_num']] += 1
                    self.data_dict['pos_num'][config['pos_num']] += 1

        print(self.actions_num)
        for k, v in self.data_dict.items():
            print(f'{k}\t {v}')

        with open(self.result_file, mode='a+') as f:
            f.truncate(0)
            f.writelines([line + '\n' for line in completed_dirs])

    def build_dataset(self,
                      modal: str = 'image',
                      noise_factor: float = 0,
                      resize=None):

        assert modal in ['image', 'video']
        if modal == 'video':
            assert resize[0] == resize[1]
        mat_name = 'reduced_data' if modal == 'image' else f'video_{resize[0]}'

        with open(self.result_file, mode='r') as f:
            png_dirs = [line.strip() for line in f.readlines()]

        # print(f'{len(self.actions)} classes in total: {self.actions}')

        for png_dir in tqdm(png_dirs):
            png_abs_dir = os.path.join(self.raw_png_root, png_dir)
            mat_path = os.path.join(png_abs_dir, f'{mat_name}_N{noise_factor}.mat')
            if os.path.exists(mat_path):
                continue

            frames = load_frames(png_abs_dir, output_size=resize)
            frames = frames + 255 * noise_factor * torch.randn_like(frames)
            frames_sub_mean = sub_mean(frames)  # (T, H, W, RGB)
            if modal == 'video':
                save_dict = {'video': frames_sub_mean.numpy().permute(3, 0, 1, 2)}  # (RGB, T, H, W)
            elif modal == 'image':
                reduce_H, reduce_W = reduce(frames_sub_mean)  # (W or H, T, RGB)
                save_dict = {"reduce_H": reduce_H.numpy(),
                             "reduce_W": reduce_W.numpy()}
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


def check_dataset(check_root='/mnt/lustre/wangyihao/nlos_raw_pngs',
                  output_dir='/mnt/lustre/wangyihao/nlos_raw_pngs',
                  mode='train'
                  ):
    check_root = os.path.join(check_root, mode)
    output_dir = os.path.join(output_dir, mode)
    actions = ['Being hit', 'Clap', 'Crouch to Stand', 'Dance', 'Hanging',
               'Idle', 'Jump', 'Kick', 'Lying down', 'Punch',
               'Sit', 'Spin', 'Squat', 'Stand to crouch', 'Stand to kneel',
               'Strafing', 'Throw', 'Turn around', 'Waving hands', 'Yelling']
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

        config = get_configs(png_dir)
        if config['render_state'] == 'finish':
            png_num = len([f for f in os.listdir(os.path.join(check_root, png_dir)) if f.endswith('.png')])
            if png_num == 64:
                completed_dirs.append(png_dir + '\n')
                action = config['action']
                actions_num[action['class'] if type(action) == dict else action] += 1
                data_dict['floor'][config['floor']] += 1
                data_dict['rot_num'][config['rot_num']] += 1
                data_dict['pos_num'][config['pos_num']] += 1

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

# -*- coding: utf-8 -*-
# @Date    : 2022/1/23 14:05
# @Author  : WangYihao
# @File    : vis.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from .data.preprocess import sub_mean
from .utils import get_device


def grid_vis(loader, row_num, multiply=20, model=None, norm=True):
    imgs, labels = next(iter(loader))
    if model is not None:
        device = get_device(model)
        scores = model(imgs.to(device))
        preds = scores.argmax(axis=1)

    if imgs.size(-1) == 3:
        vis = imgs.permute(0, 2, 1, 3)
    else:
        vis = imgs.permute(0, 3, 2, 1)
    batch_size = vis.size(0)
    if row_num ** 2 < batch_size:
        subfig_num = row_num ** 2
        col_num = row_num
    else:
        subfig_num = batch_size
        col_num = batch_size // row_num
    fig = plt.figure(figsize=(2 * col_num, 0.8 * row_num))
    for i in range(subfig_num):
        plt.subplot(row_num, col_num, i + 1)
        if model is not None:
            plt.title(f"GT:{labels[i]}  Pre:{preds[i]}")
        else:
            plt.title(f"{labels[i]}")
        if norm:
            vis[i] = (vis[i] - vis[i].min()) / (vis[i].max() - vis[i].min())
        else:
            vis[i] *= multiply
        plt.imshow(vis[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def vis_act(act, label, row_num=6):
    act = act.permute(1, 2, 3, 0)  # (1, C, H, W) -> (C, H, W, 1)
    chans_num = act.size(0)
    if row_num ** 2 < chans_num:
        subfig_num = row_num ** 2
        col_num = row_num
    else:
        subfig_num = chans_num
        row_num = int(np.sqrt(chans_num))
        col_num = chans_num // row_num + 1
    fig = plt.figure()
    for i in range(subfig_num):
        plt.subplot(row_num, col_num, i + 1)
        plt.title(f"GT:{label}")
        plt.imshow(act[i], 'gray')
        plt.axis('off')
    plt.tight_layout()


def frame2video(frames: torch.Tensor, save_path: str, fps: int = 30):
    N, H, W, C = frames.shape

    frames_sub_mean = sub_mean(frames)
    frames_sub_mean_norm = (frames_sub_mean - frames_sub_mean.min()) / (frames_sub_mean.max() - frames_sub_mean.min())
    frames_write = np.array(255 * frames_sub_mean_norm, dtype=np.uint8)
    bs = frames_write[:, :, :, 2]
    gs = frames_write[:, :, :, 1]
    rs = frames_write[:, :, :, 0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, (H, W))
    for i in tqdm(range(N)):
        img = cv2.merge([bs[i], gs[i], rs[i]])
        videoWriter.write(img)
    videoWriter.release()

    print(f'Successfully write frames into {save_path}!')

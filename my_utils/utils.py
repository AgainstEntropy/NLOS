# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:39
# @Author  : WangYihao
# @File    : functions.py
import os
import time

import numpy as np
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from my_utils import models


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = -1
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_rate(preds: torch.Tensor,
                 labels: torch.Tensor):
    assert len(preds) == len(labels)
    num_correct = (preds == labels).sum()
    return num_correct / len(preds)


def save_model(model, optimizer, scheduler, save_dir, acc=00):
    model_paras = model.checkpoint()
    optim_paras = optimizer.checkpoint()
    scheduler_main_paras = scheduler.checkpoint()

    save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = os.path.join(save_dir, f'{acc:.1f}_{save_time}.pt')
    torch.save({
        "model_paras": model_paras,
        "optim_paras": optim_paras,
        "scheduler_paras": scheduler_main_paras
    }, save_path)

    print(f"\nSuccessfully saved model, optimizer and scheduler to {save_path}")


def get_device(model):
    if next(model.parameters()).device.type == 'cuda':
        index = next(model.parameters()).device.index
        device = torch.device(f'cuda:{index}')
    else:
        device = torch.device('cpu')
    return device


def check_accuracy(model, loader, training=False):
    confusion_matrix = np.zeros((model.num_classes, ) * 2)
    device = get_device(model)
    model.eval()  # set model to evaluation mode
    tic = time.time()
    with torch.no_grad():
        with autocast():
            for batch_idx, (X, Y) in tqdm(enumerate(loader)):
                X = X.to(device)  # move to device, e.g. GPU
                Y = Y.to(device)
                _, preds = model((X, Y))
                for label, pred in zip(Y, preds):
                    confusion_matrix[label, pred] += 1
    num_correct = confusion_matrix.trace()
    test_acc = float(num_correct) / confusion_matrix.sum()
    if training:
        return test_acc
    else:
        print(f"Test accuracy is : {100. * test_acc:.2f}%\tInfer time: {time.time() - tic}")
        return confusion_matrix


def get_cls_from_position(path: str):
    candidate_pos = [
        np.array([0, 0]),
        np.array([0.75, 0.8]),
        np.array([-0.75, 0.8]),
        np.array([0.75, -0.8]),
        np.array([-0.75, -0.8]),
    ]

    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    position = np.array(list(config['armature']['properties']['position'].values()))
    distances = [np.linalg.norm(position - pos) for pos in candidate_pos]
    return np.argmin(distances)


def load_model(run_name: str,
               log_dir: str = '/home/wangyh/01-Projects/01-NLOS/test_runs/',
               ckpt_name: str = 'best') -> torch.nn.Module:
    run_dir = os.path.join(log_dir, run_name)
    checkpoint = torch.load(os.path.join(run_dir, f'checkpoints/{ckpt_name}.pth'))
    with open(os.path.join(run_dir, 'configs.yaml'), 'r') as stream:
        run_config = yaml.load(stream, Loader=yaml.FullLoader)
    num_classes = {'action': 8, 'position': 5}[run_config['train_configs']['class_type']]

    print('best val acc is:', checkpoint['best_val_acc'].item())
    model = models.NLOS_Conv(num_classes=num_classes, **run_config['model_configs'])
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(new_state_dict)
    print('Successfully load model parameters!')

    return model

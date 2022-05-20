# -*- coding: utf-8 -*-
# @Date    : 2021/12/18 19:39
# @Author  : WangYihao
# @File    : functions.py
import os

import torch
from torch import nn
import torch.nn.functional as F

import time


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
    model_paras = model.state_dict()
    optim_paras = optimizer.state_dict()
    scheduler_main_paras = scheduler.state_dict()

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


def check_accuracy(test_model, loader, training=False):
    num_correct = 0
    num_samples = 0
    device = get_device(test_model)
    test_model.eval()  # set model to evaluation mode
    tic = time.time()
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(loader):
            X = X.to(device, dtype=torch.float32)  # move to device, e.g. GPU
            Y = Y.to(device, dtype=torch.int)
            scores = test_model(X)
            num_correct += (scores.argmax(axis=1) == Y).sum()
            num_samples += len(scores)
    test_acc = float(num_correct) / num_samples
    if training:
        return test_acc
    else:
        print(f"Test accuracy is : {100. * test_acc:.2f}%\tInfer time: {time.time() - tic}")


# def train(model, optimizer, scheduler, loss_fn, train_loader,
#           check_fn, check_loaders, batch_step, epochs=2, log_every=10, writer=None):
#     """
#
#     Args:
#         batch_step (int):
#         epochs (int):
#         log_every (int):
#         writer :
#
#     Returns:
#         batch_step (int):
#     """
#     device = get_device(model)
#     batch_size = train_loader.batch_size
#     check_loader_train = check_loaders['train']
#     check_loader_val = check_loaders['val']
#     iters = len(train_loader)
#     for epoch in range(1, epochs + 1):
#         tic = time.time()
#         for batch_idx, (X, Y) in enumerate(train_loader):
#             batch_step += 1
#             model.train()
#             X = X.to(device, dtype=torch.float32)
#             Y = Y.to(device, dtype=torch.int64)
#             scores = model(X)
#             loss = loss_fn(scores, Y)
#             if writer is not None:
#                 writer.add_scalar('loss', loss.item(), batch_step)
#                 writer.add_scalar('lr', optimizer.param_groups[0]['lr'], batch_step)
#
#             # back propagate
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             scheduler.step(batch_step / iters)
#
#             # check accuracy
#             if batch_idx % log_every == 0:
#                 model.eval()
#                 train_acc = check_fn(model, check_loader_train, training=True)
#                 val_acc = check_fn(model, check_loader_val, training=True)
#                 if writer is not None:
#                     writer.add_scalars('acc', {'train': train_acc, 'val': val_acc}, batch_step)
#                 print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\tVal acc: {:.1f}%'.format(
#                     epoch, batch_idx * batch_size, len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader),
#                     loss, 100. * val_acc))
#
#         print('====> Epoch: {}\tTime: {}s'.format(epoch, time.time() - tic))
#
#     return batch_step

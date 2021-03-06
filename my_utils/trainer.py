# -*- coding: utf-8 -*-
# @Date    : 2022/4/30 15:52
# @Author  : WangYihao
# @File    : trainer.py
import os
import platform
import random
import time
from decimal import Decimal

import numpy as np
import wandb
import yaml
from loguru import logger
from prettytable import PrettyTable
from tqdm import tqdm
from fairscale.optim.oss import OSS
import torch
from torch import optim, nn, distributed
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

from my_utils import models
from my_utils.data.dataset import split_dataset
from my_utils.utils import AverageMeter, correct_rate


# cudnn.benchmark = True


def seed_worker(worker_id):
    # print(torch.initial_seed())
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _set_seed(seed, deterministic=False):
    """
    seed manually to make runs reproducible
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option
        for CUDNN backend
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


class Trainer(object):
    def __init__(self, cfg):
        tic = time.time()
        self.dist_cfgs = cfg['distributed_configs']
        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Loading configurations...")
        self.cfg = cfg
        self.model_cfgs = cfg['model_configs']
        self.train_cfgs = cfg['train_configs']
        self.dataset_cfgs = cfg['dataset_configs']
        self.loader_kwargs = cfg['loader_kwargs']
        self.optim_kwargs = cfg['optim_kwargs']
        self.schedule_cfgs = cfg['schedule_configs']
        self.log_cfgs = cfg['log_configs']

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Initializing trainer...")
        if self.dist_cfgs['distributed']:
            distributed.init_process_group(backend='nccl',
                                           init_method='tcp://127.0.0.1:' + self.dist_cfgs['port'],
                                           world_size=self.dist_cfgs['world_size'],
                                           rank=self.dist_cfgs['local_rank'])
        _set_seed(self.train_cfgs['seed'] + self.dist_cfgs['local_rank'], deterministic=True)
        if torch.cuda.is_available():
            self.device = f'cuda:{self.dist_cfgs["local_rank"]}'
        else:
            self.device = "cpu"
        self.dist_cfgs['device'] = self.device

        save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        run_dir = os.path.join(os.getcwd(), self.log_cfgs['log_dir'])
        self.log_dir = os.path.join(run_dir, save_time)
        self.ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.dist_cfgs['local_rank'] == 0:
            with open(os.path.join(self.log_dir, 'configs.yaml'), 'w', encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, default_flow_style=False, allow_unicode=True)

        self.start_epoch = 0
        self.steps = 0
        self.epoch = 0
        self.min_loss = float('inf')
        self.val_best_acc_total = 0.0
        self.val_metrics = {'current_acc': 0.0, 'best_acc': 0.0,
                            'best_epoch': 0}

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Loading dataset...")
        if self.train_cfgs['mode'] == 'train':
            (self.train_loader, self.train_sampler), (self.val_loader, self.val_sampler) \
                = self._load_dataset()
        if self.train_cfgs['mode'] == 'test':
            self.test_loader, self.test_sampler = self._load_dataset()

        if self.dist_cfgs['local_rank'] == 0:
            logger.info("Building model...")
        self._build_model()
        if self.dist_cfgs['distributed']:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.dist_cfgs['local_rank']],
                             output_device=self.dist_cfgs['local_rank'],
                             find_unused_parameters=False)

        self._load_optimizer()

        if self.dist_cfgs['local_rank'] == 0:
            self._init_recorder(project_name='NLOS', run_name=save_time)

        if self.train_cfgs['resume']:
            checkpoint_path = self.train_cfgs['resume_path']
            assert os.path.exists(checkpoint_path)
            self.load_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            print(f"{time.time() - tic:.2f} sec are used to initialize a Trainer.")

    def _init_recorder(self, project_name, run_name):
        train_config = {
            "dataset": self.dataset_cfgs['name'],
            # "class_type": self.train_cfgs['class_type'],
            "kernel_size": self.model_cfgs['kernel_size'],
            "depths": self.model_cfgs['depths'],
            "dims": self.model_cfgs['dims'],
            "modal": self.train_cfgs['modal'],
            "batch_size": self.train_cfgs['batch_size'],
            "lr_backbone": self.optim_kwargs['lr'],
            "optimizer": self.optim_kwargs['optimizer'],
            "weight_decay": self.optim_kwargs['weight_decay'],
            "epochs": self.schedule_cfgs['max_epoch'],
            "fig_size": self.dataset_cfgs['fig_resize'],
            "reduced_mode": self.dataset_cfgs['reduced_mode']
        }
        if self.train_cfgs['recorder'] == 'wandb':
            wandb.init(project=project_name, entity="against-entropy", name=run_name, dir=self.log_dir,
                       config=train_config)
            wandb.watch(self.model)
        elif self.train_cfgs['recorder'] == 'tensorboard':
            self.writer = SummaryWriter(log_dir=self.log_dir)

        config_table = PrettyTable()
        config_table.add_column('Phase', list(train_config))
        config_table.add_column('Value', list(train_config.values()))
        logger.info('\n' + config_table.get_string())

    def _load_dataset(self):
        dataset_dir = os.path.join(self.dataset_cfgs['dataset_root'], self.dataset_cfgs['name'])
        if self.train_cfgs['modal'] == 'image':
            trans = T.Compose([
                T.ToTensor(),
                T.Resize(self.dataset_cfgs['fig_resize'])
            ])
            if self.dataset_cfgs['normalize']:
                trans = T.Compose([trans, T.Normalize(self.dataset_cfgs['mean'], self.dataset_cfgs['std'])])
        elif self.train_cfgs['modal'] == 'video':
            trans = None

        if self.train_cfgs['mode'] == 'train':
            train_dataset, val_dataset = split_dataset(dataset_root=dataset_dir,
                                                       modal=self.train_cfgs['modal'],
                                                       phase='train',
                                                       ratio=self.dataset_cfgs['train_ratio'],
                                                       reduced_mode=self.dataset_cfgs['reduced_mode'],
                                                       transform=trans)

            if self.dist_cfgs['distributed']:
                train_sampler = DistributedSampler(train_dataset, shuffle=True)
                val_sampler = DistributedSampler(val_dataset, shuffle=True)
            else:
                train_sampler = None
                val_sampler = None

            train_loader = DataLoader(train_dataset, **self.loader_kwargs, drop_last=True)
            val_loader = DataLoader(val_dataset, **self.loader_kwargs, drop_last=False)
            return (train_loader, train_sampler), (val_loader, val_sampler)

        elif self.train_cfgs['mode'] == 'test':
            test_dataset = split_dataset(dataset_root=dataset_dir,
                                         phase='test',
                                         reduced_mode=self.dataset_cfgs['reduced_mode'],
                                         transform=trans)
            test_sampler = DistributedSampler(test_dataset, shuffle=True) if self.dist_cfgs['distributed'] else None
            test_loader = DataLoader(test_dataset, **self.loader_kwargs, drop_last=False)
            return test_loader, test_sampler

    def _build_model(self):
        if self.train_cfgs['class_type'] == 'action':
            self.model_cfgs['num_classes'] = 20
        elif self.train_cfgs['class_type'] == 'position':
            self.model_cfgs['num_classes'] = 5

        model_name = self.model_cfgs.pop('model_name')
        if self.train_cfgs['modal'] == 'video':
            model_builder = {'r21d': models.my_NLOS_r21d,
                             '3d': models.my_NLOS_3d}[model_name]
        elif self.train_cfgs['modal'] == 'image':
            model_builder = models.NLOS_Conv
            self.model_cfgs['groups'] = len(self.dataset_cfgs['reduced_mode'])

        self.model = model_builder(**self.model_cfgs)
        self.model.to(self.device)

    def _load_optimizer(self):
        base_optimizer = None
        optim_type = self.optim_kwargs.pop('optimizer')
        if optim_type == 'SGD':
            base_optimizer = optim.SGD
            self.optim_kwargs['momentum'] = 0.9
        elif optim_type == 'Adam':
            base_optimizer = optim.Adam
            self.optim_kwargs['betas'] = (0.9, 0.999)
        elif optim_type == 'AdamW':
            base_optimizer = optim.AdamW
            self.optim_kwargs['betas'] = (0.9, 0.999)
        else:
            print(f"{optim_type} not support.")
            exit(0)

        if self.dist_cfgs['distributed']:
            # Wrap a base optimizer into OSS
            self.optimizer = OSS(
                params=self.model.parameters(),
                optim=base_optimizer,
                **self.optim_kwargs,
            )
        else:
            self.optimizer = base_optimizer(
                params=self.model.parameters(),
                **self.optim_kwargs,
            )

        if self.schedule_cfgs['schedule_type'] == 'cosine_warm':
            self.schedule_cfgs['max_epoch'] = \
                int((self.schedule_cfgs['cos_mul'] ** self.schedule_cfgs['cos_iters'] - 1) / \
                    (self.schedule_cfgs['cos_mul'] - 1) * self.schedule_cfgs['cos_T'])
            self.scheduler = \
                optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                               T_0=self.schedule_cfgs['cos_T'], T_mult=2)
        elif self.schedule_cfgs['schedule_type'] == 'cosine':
            self.schedule_cfgs['max_epoch'] = self.schedule_cfgs['cos_iters'] * self.schedule_cfgs['cos_T']
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.schedule_cfgs['cos_T'])

        if self.train_cfgs['amp']:
            self.scaler = GradScaler()

        self.optim_kwargs['optimizer'] = optim_type

    def run(self):
        if self.dist_cfgs['local_rank'] == 0:
            logger.info("--- Begin to run! ---")
        for epoch in range(self.start_epoch, self.schedule_cfgs['max_epoch']):

            if self.dist_cfgs['distributed']:
                self.train_sampler.set_epoch(epoch)

            train_loss, train_acc = self.train(epoch)
            self.min_loss = min(self.min_loss, train_loss)

            val_loss, val_acc = self.val(epoch)
            self.epoch += 1

            if self.dist_cfgs['local_rank'] == 0:

                if self.train_cfgs['recorder'] == 'wandb':
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        wandb.log({f"optimizer/lr_group_{i}": param_group['lr']}, step=epoch + 1)
                    wandb.log({
                        'Metric/acc/train': train_acc,
                        'Metric/acc/val': val_acc,
                        'Metric/acc/best_acc': self.val_metrics['best_acc'],
                        'Metric/loss/train': train_loss,
                        'Metric/loss/val': val_loss
                    }, step=epoch + 1)
                elif self.train_cfgs['recorder'] == 'tensorboard':
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        self.writer.add_scalar(tag=f'optimizer/lr_group_{i}',
                                               scalar_value=param_group['lr'],
                                               global_step=epoch)
                        self.writer.add_scalars('Metric', {'acc_train': train_acc,
                                                           'acc_val': val_acc,
                                                           'loss_train': train_loss,
                                                           'loss_val': val_loss}, epoch + 1)

            self.scheduler.step()

            if ((epoch + 1) % self.log_cfgs['save_epoch_interval'] == 0) \
                    or (epoch + 1) == self.schedule_cfgs['max_epoch']:
                checkpoint_path = os.path.join(self.ckpt_dir, f"epoch_{(epoch + 1)}.pth")
                self.save_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            if self.train_cfgs['recorder'] == 'wandb':
                wandb.finish()
            elif self.train_cfgs['recorder'] == 'tensorboard':
                self.writer.close()

        if self.dist_cfgs['distributed']:
            distributed.destroy_process_group()

    def train(self, epoch):
        self.model.train()
        len_loader = len(self.train_loader)
        iter_loader = iter(self.train_loader)

        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()

        pbar = None
        if self.dist_cfgs['local_rank'] == 0:
            pbar = tqdm(total=len_loader,
                        dynamic_ncols=True,
                        ascii=(platform.version() == 'Windows'))

        for step in range(len_loader):
            try:
                inputs, labels = next(iter_loader)
            except Exception as e:
                logger.critical(e)
                continue

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            if self.train_cfgs['amp']:
                with autocast():
                    loss, preds = self.model((inputs, labels))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, preds = self.model((inputs, labels))
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            self.steps += 1
            loss = loss.detach().clone()
            acc_recorder.update(correct_rate(preds, labels), batch_size)

            if self.dist_cfgs['distributed']:
                distributed.reduce(loss, 0)
                loss /= self.dist_cfgs['world_size']
            loss_recorder.update(loss.item(), batch_size)

            if self.dist_cfgs['local_rank'] == 0:
                last_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
                last_lr_string = "lr " + ' '.join(f"{Decimal(lr):.1E}" for lr in last_lr)

                pbar.set_description(
                    f"train epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                    f"Iter {self.steps}/{len_loader * self.schedule_cfgs['max_epoch']}  "
                    f"{last_lr_string}  "
                    f"----  "
                    f"loss {loss_recorder.avg:.4f}  "
                    f"top1_acc {acc_recorder.avg:.2%}")
                pbar.update()

                if self.steps % self.log_cfgs['snapshot_interval'] == 0:
                    checkpoint_path = os.path.join(self.ckpt_dir, "latest.pth")
                    self.save_checkpoint(checkpoint_path)

        if self.dist_cfgs['local_rank'] == 0:
            pbar.close()

            # logger.info(
            #     f"train epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
            #     f"Iter {self.steps}/{len_loader * self.schedule_cfgs['max_epoch']}  "
            #     f"----  "
            #     f"loss {loss_recorder.avg:.4f} "
            #     f"top1_acc {acc_recorder.avg:.2%}")

        return loss_recorder.avg, acc_recorder.avg

    def val(self, epoch):
        self.model.eval()
        len_loader = len(self.val_loader)
        iter_loader = iter(self.val_loader)

        loss_recorder = AverageMeter()
        acc_recorder = AverageMeter()

        pbar = None
        if self.dist_cfgs['local_rank'] == 0:
            pbar = tqdm(total=len_loader,
                        dynamic_ncols=True,
                        ascii=(platform.version() == 'Windows'))

        for step in range(len_loader):
            try:
                inputs, labels = next(iter_loader)
            except Exception as e:
                logger.critical(e)
                continue

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            with torch.no_grad():
                if self.train_cfgs['amp']:
                    with autocast():
                        loss, preds = self.model((inputs, labels))
                else:
                    loss, preds = self.model((inputs, labels))

            loss = loss.detach().clone()
            acc_recorder.update(correct_rate(preds, labels), batch_size)

            if self.dist_cfgs['distributed']:
                distributed.reduce(loss, 0)
                loss /= self.dist_cfgs['world_size']
            loss_recorder.update(loss.item(), batch_size)

            if self.dist_cfgs['local_rank'] == 0:
                pbar.set_description(
                    f"val epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
                    f"Step {step}/{len_loader}  "
                    f"------  "
                    f"loss {loss_recorder.avg:.4f}  "
                    f"top1_acc {acc_recorder.avg:.2%}")
                pbar.update()

        if self.dist_cfgs['local_rank'] == 0:
            pbar.close()

            # logger.info(
            #     f"val epoch {epoch + 1}/{self.schedule_cfgs['max_epoch']}  "
            #     f"------  "
            #     f"loss {loss_recorder.avg:.4f}  "
            #     f"top1_acc {acc_recorder.avg:.2%}")

            self.val_metrics['current_acc'] = acc_recorder.avg
            if acc_recorder.avg > self.val_metrics['best_acc']:
                self.val_metrics['best_acc'] = acc_recorder.avg
                self.val_metrics['best_epoch'] = epoch + 1

                checkpoint_path = os.path.join(self.ckpt_dir, "best.pth")
                self.save_checkpoint(checkpoint_path)

            res_table = PrettyTable()
            res_table.add_column('Phase', ['Current Acc', 'Best Acc', 'Best Epoch'])
            res_table.add_column('Val', [f"{self.val_metrics['current_acc']:.2%}",
                                         f"{self.val_metrics['best_acc']:.2%}",
                                         self.val_metrics['best_epoch']])
            logger.info(f'Performance on validation set at epoch: {epoch + 1}\n' + res_table.get_string())

        return loss_recorder.avg, acc_recorder.avg

    def save_checkpoint(self, path):
        # self.optimizer.consolidate_state_dict()
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        if self.dist_cfgs['local_rank'] == 0:
            save_dict = {
                'model': self.model.state_dict(),
                # 'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'iteration': self.steps,
                'best_val_acc': self.val_metrics['best_acc'],
                'best_epoch': self.val_metrics['best_epoch'],
                'val_best_acc_total': self.val_best_acc_total,
            }
            torch.save(save_dict, path)

    def load_checkpoint(self, path):
        ckpt = None
        if self.dist_cfgs['local_rank'] == 0:
            ckpt = torch.load(path, map_location={'cuda:0': f'cuda:{self.dist_cfgs["local_rank"]}'})
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch']
        self.steps = ckpt['iteration']
        self.val_metrics['best_epoch'] = ckpt['best_epoch']
        self.val_metrics['best_acc'] = ckpt['best_val_acc']
        self.val_best_acc_total = ckpt['val_best_acc_total']

# -*- coding: utf-8 -*-
# @Date    : 2022/4/30 15:52
# @Author  : WangYihao
# @File    : train.py

import argparse
import os

import torch
import torch.multiprocessing as mp
import yaml


def main(cfg):
    loader_cfgs = cfg['loader_kwargs']
    train_cfgs = cfg['train_configs']
    dist_cfgs = cfg['distributed_configs']
    log_cfgs = cfg['log_configs']

    os.makedirs(log_cfgs['log_dir'], exist_ok=True)
    # os.environ["CUDA_VISIBLE_DEVICES"] = dist_cfgs['device_ids']
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    world_size = len(dist_cfgs['device_ids'].split(','))
    dist_cfgs['distributed'] = True if world_size > 1 else False
    dist_cfgs['world_size'] = world_size
    loader_cfgs['batch_size'] = train_cfgs['batch_size'] // world_size

    if dist_cfgs['distributed']:
        # print(f"Using devices: {dist_cfgs['device_ids']}")
        mp.spawn(worker, nprocs=world_size, args=(cfg,))
    else:
        worker(0, cfg)


def worker(rank, cfg):
    torch.cuda.set_device(rank)
    cfg['distributed_configs']['local_rank'] = rank
    from my_utils.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training a classifier convnet")
    parser.add_argument('-cfg', '--config', type=str, default='configs/S video.yaml')
    # parser.add_argument('-ks', '--kernel_size', type=int, default=7)
    parser.add_argument('--depths', type=lambda x: tuple(int(i) for i in x.split(',')), default='4,1')
    parser.add_argument('--dims', type=lambda x: tuple(int(i) for i in x.split(',')), default='16,32')
    # parser.add_argument('-act', '--activation', type=str, choices=['relu', 'gelu'], default='relu')
    # parser.add_argument('-norm', '--normalization', type=str, choices=['BN', 'LN'], default='BN')

    parser.add_argument('-dn', '--dataset_name', type=str, default='train')
    # parser.add_argument('--reduced_mode', type=str, choices=['H', 'W', ''], default='H')
    # parser.add_argument('--file_name', type=str, choices=['N0', 'N0.05', '128_N0'], default='128_N0')

    parser.add_argument('-b', '--batch_size', type=int, default=64)
    # parser.add_argument('-r', '--resume', action='store_true', help='load previously saved checkpoint')
    # parser.add_argument('-ct', '--class_type', type=str, choices=['action', 'position'], required=True)

    # parser.add_argument('-op', '--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW')
    parser.add_argument('-lr_b', '--lr_backbone', type=float, default=3e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5.0e-3)

    parser.add_argument('-T', '--cos_T', type=int, default=25)
    parser.add_argument('--cos_iters', type=int, default=2)

    parser.add_argument('-g', '--gpu_ids', required=True, type=lambda x: x.replace(" ", ""), default='0,1',
                        help='available gpu ids')
    parser.add_argument('--port', type=str, default='4250', help='port number of distributed init')

    # parser.add_argument('-log', '--log_dir', type=str, default='test_runs', help='where to log train results')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    # config['model_configs']['kernel_size'] = args.kernel_size
    config['model_configs']['depths'] = args.depths
    config['model_configs']['dims'] = args.dims
    # config['model_configs']['act'] = args.activation
    # config['model_configs']['norm'] = args.normalization

    config['dataset_configs']['name'] = args.dataset_name
    # config['dataset_configs']['reduced_mode'] = args.reduced_mode
    # config['dataset_configs']['file_name'] = args.file_name

    config['train_configs']['batch_size'] = args.batch_size
    # config['train_configs']['resume'] = args.resume
    # config['train_configs']['class_type'] = args.class_type

    # config['optim_kwargs']['optimizer'] = args.optimizer
    config['optim_kwargs']['lr'] = args.lr_backbone
    config['optim_kwargs']['weight_decay'] = args.weight_decay

    config['schedule_configs']['cos_T'] = args.cos_T
    config['schedule_configs']['cos_iters'] = args.cos_iters

    config['distributed_configs']['device_ids'] = args.gpu_ids
    config['distributed_configs']['port'] = args.port

    # config['log_configs']['log_dir'] = args.log_dir

    main(config)

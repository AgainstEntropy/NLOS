import os
from typing import Tuple, Any

import torch
from tqdm import tqdm


def sub_mean(frames: torch.Tensor) -> torch.Tensor:
    mean_frame = frames.mean(axis=0, keepdim=True)
    frames_sub_mean = frames - mean_frame

    if frames.shape[-1] == 4:
        frames_sub_mean = frames_sub_mean[:, :, :, :3]  # (T, H, W, RGBA) -> (T, H, W, RGB)

    return frames_sub_mean


def reduce(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_H = frames.mean(axis=1).permute(1, 0, 2)  # (T, H, W, RGB) -> (W, T, RGB)
    reduce_W = frames.mean(axis=2).permute(1, 0, 2)  # (T, H, W, RGB) -> (H, T, RGB)

    return reduce_H, reduce_W


def check_data(check_dict=None,
               check_root='/mnt/cfs/wangyh/blender/blank_wall/output_png',
               output_dir='/mnt/cfs/wangyh/blender/blank_wall/datasets',
               stat_file='stat-7'):
    if check_dict is None:
        check_dict = {
            'actions': ['Clapping', 'Flair', 'Jump', 'Jumping Jacks', 'Rumba Dancing'],
            'characters': ['Bryce', 'Jody', 'Kate', 'Sophie', 'Suzie'],
            'light_types': ['AREA', 'POINT'],
            'light_energy_modes': ['1', '2'],
            'light_position_modes': ['1', '2'],
            'scenes': ['Wall7', 'Wall8', 'Wall12', 'concrete_31', 'concrete_new', 'white_concrete'],
            'rotate': ['r0', 'r180'],
        }

    completed_dirs = []
    action_samples = dict.fromkeys(check_dict['actions'], 0)

    for scene in tqdm(check_dict['scenes']):
        # for r in check_dict['rotate']:
        for action in check_dict['actions']:
            for character in check_dict['characters']:
                for lt in check_dict['light_types']:
                    for le in check_dict['light_energy_modes']:
                        for lp in check_dict['light_position_modes']:

                            dir_name = f'{action}/{action}_{character}_lt{lt}_le{le}_lp{lp}_{scene}'
                            dir_path = os.path.join(check_root, dir_name)
                            if os.path.exists(dir_path):
                                png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
                                png_num = len(png_files)
                                if png_num == 64:
                                    completed_dirs.append(dir_name + '\n')
                                    action_samples[action] += 1

    for k, v in action_samples.items():
        print(f'{k}\t {v}')

    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, stat_file)
    with open(result_file, mode='a+') as f:
        f.truncate(0)
        f.writelines(completed_dirs)

    return action_samples


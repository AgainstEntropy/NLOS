import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchvision.io import read_video, read_image
from torchvision.transforms.functional import resize


def load_video(
        path: str,
        start_sec: Optional[Union[int, float]] = 0,
        end_sec: Optional[Union[int, float]] = None,
        frame_range: tuple = (0, 64),
        output_size: tuple = None
) -> torch.Tensor:

    vframes, _, _ = read_video(path, start_sec, end_sec, pts_unit='sec')
    end_frame = min(frame_range[1], len(vframes))
    vframes = vframes[frame_range[0]: end_frame].permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    if output_size is not None:
        vframes = resize(vframes, size=output_size)
    vframes = vframes.type(torch.float32) / 255.

    return vframes.permute(0, 2, 3, 1)  # (T, H, W, C)


def load_frames(
        root: str,
        frame_range: Union[None, Tuple[int, int]] = None,
        output_size: Union[None, Tuple[int, int]] = None,
        rgb_only=True
) -> torch.Tensor:
    # TODO:
    frame_list = sorted([f for f in os.listdir(root) if f.endswith('.png')])
    if frame_range is not None:
        frame_list = frame_list[frame_range[0]: frame_range[1]]
    frame_paths = [os.path.join(root, f) for f in frame_list]

    C, H, W = read_image(frame_paths[0]).shape
    frame_num = len(frame_list)
    if output_size is None:
        output_size = (H, W)
    if C == 4 and rgb_only:
        frames = torch.zeros((frame_num, *output_size, 3))
    else:
        frames = torch.zeros((frame_num, *output_size, C))
    for i in range(frame_num):
        frame = read_image(frame_paths[i])
        if C == 4 and rgb_only:
            frame = frame[:3]
        if output_size is not None:
            frame = resize(frame, size=output_size)
        frames[i] = frame.permute(1, 2, 0)
    return frames  # (N, H, W, C)

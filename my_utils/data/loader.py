import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchvision.io import read_video, read_image
from torchvision.transforms.functional import resize


def load_video(
        path: str,
        pts_unit: str = "sec",
        start_pts: Optional[Union[int, float]] = 0,
        end_pts: Optional[Union[int, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """

    :param path:
    :param pts_unit:
    :param start_pts:
    :param end_pts:
    :return:
    """
    # TODO:
    vframes, aframes, info = read_video(path, start_pts, end_pts, pts_unit)
    vframes = vframes.type(torch.float16) / 255.

    return vframes, aframes, info


def load_frames(
        root: str,
        frame_range: Union[None, Tuple[int, int]] = None,
        output_size: Union[None, Tuple[int, int]] = None,
        device=None,
        rgb_only=True
) -> Tensor:
    """

    :param root:
    :param frame_range: (start frame, end_frame), not include endpoint.
    :param output_size:
    :param device:
    :return:
    """
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
    if device is not None:
        frames = frames.to(device)
    for i in range(frame_num):
        frame = read_image(frame_paths[i])
        if C == 4 and rgb_only:
            frame = frame[:3]
        if output_size is not None:
            frame = resize(frame, size=output_size)
        frames[i] = frame.permute(1, 2, 0)
    return frames  # (N, H, W, C)

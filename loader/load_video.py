import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torchvision.io import read_video


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


def preprocess_video(video: torch.Tensor) -> torch.Tensor:
    # TODO:
    pass



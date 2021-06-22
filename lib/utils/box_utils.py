from typing import Union

import numpy as np
import torch


__all__ = ['xywh_to_xyxy', 'xyxy_to_xywh', 'compute_delta', 'apply_delta']


def xywh_to_xyxy(
    boxes: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    if not inplace:
        boxes = boxes.clone()
    boxes[..., 2:4] = boxes[..., 2:4] - boxes[..., :2]
    boxes[..., :2] = boxes[..., :2] + boxes[..., 2:4] / 2
    return boxes


def xyxy_to_xywh(
    boxes: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    if not inplace:
        boxes = boxes.clone()
    boxes[..., :2] = boxes[..., :2] - boxes[..., 2:4] / 2
    boxes[..., 2:4] = boxes[..., :2] + boxes[..., 2:4]
    return boxes


_DEFAULT_WEIGHTS = torch.tensor([10., 10., 5., 5.])


def compute_delta(
    src_boxes: torch.Tensor,
    dst_boxes: torch.Tensor,
    weights: torch.Tensor = _DEFAULT_WEIGHTS,
) -> torch.Tensor:
    """
    Compute transformation deltas from the source boxes
    to the target boxes
    """

    src_boxes = xyxy_to_xywh(src_boxes)
    dst_boxes = xyxy_to_xywh(dst_boxes)

    delta = torch.cat([
        (dst_boxes[..., :2] - src_boxes[..., :2]) / src_boxes[..., 2:4],
        (dst_boxes[..., 2:4] / src_boxes[..., 2:4]).log()
    ], dim=-1) * weights.to(src_boxes.device)

    return delta


def apply_delta(
    src_boxes: torch.Tensor,
    delta: torch.Tensor,
    weights: torch.Tensor = _DEFAULT_WEIGHTS,
) -> torch.Tensor:
    """
    Apply transformation deltas to the source boxes
    """

    src_boxes = xyxy_to_xywh(src_boxes)
    delta = delta / weights.to(src_boxes.device)

    dst_boxes = torch.cat([
        delta[..., :2] * src_boxes[..., 2:4] + src_boxes[..., :2],
        delta[..., 2:4].exp() * src_boxes[..., 2:4]
    ], dim=1)

    return xywh_to_xyxy(dst_boxes)
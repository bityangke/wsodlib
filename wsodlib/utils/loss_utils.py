from typing import Dict

import torch


__all__ = ['reduce_loss_dict']


def reduce_loss_dict(
    loss_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    return sum(v for k, v in loss_dict.items() if 'loss' in k)  # type: ignore

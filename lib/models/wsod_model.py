import abc
from typing import Dict, List, Optional

import torch
from torch import nn


__all__ = ['WsodModel']


class WsodModel(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        proposals: List[torch.Tensor],
        objectness: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        pass
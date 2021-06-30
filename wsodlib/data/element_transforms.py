from wsodlib.data.structures import WsodElement
from typing import Any, Optional, Sequence, Tuple

from numpy import random
from torch import nn
from torchvision import transforms as T

from wsodlib.data.structures import WsodElement, WsodElementLabels


__all__ = ['Compose', 'DropSmallProposals', 'RandomHorizontalFlip']


class Compose(nn.Module):
    """
    Chains multiple transformations together
    """
    def __init__(
        self,
        *modules: nn.Module
    ):
        super().__init__()
        self._modules = modules

    def forward(
        self,
        *args: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        for module in self._modules:
            args = module(*args)
        return args


class DropSmallProposals(nn.Module):
    """
    Removes region proposals below a given size threshold
    """
    def __init__(
        self,
        min_size: float = 20.,
    ):
        super().__init__()
        self.min_size = min_size

    def forward(
        self,
        element: WsodElement,
        labels: WsodElementLabels,
    ) -> Tuple[WsodElement, WsodElementLabels]:
        if element.proposals is not None:
            wh = element.proposals[..., 2:4] - element.proposals[..., 0:2]
            mask = (wh >= self.min_size).all(-1)
            element.proposals = element.proposals[mask]
            element.objectness = element.objectness[mask]  # type: ignore
        return element, labels


class RandomHorizontalFlip(nn.Module):
    """
    Randomly flips the image and proposals horizontally
    """
    def __init__(
        self,
        p: float = 0.5,
    ):
        super().__init__()
        self.p = p

    def forward(
        self,
        element: WsodElement,
        labels: WsodElementLabels,
    ) -> Tuple[WsodElement, WsodElementLabels]:
        if random.random() < self.p:
            element.image = T.functional.hflip(element.image)
            if element.proposals is not None:
                element.proposals[..., [0, 2]] = element.image_size[0] - element.proposals[..., [2, 0]] - 1
        return element, labels

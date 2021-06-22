from lib.data.structures import WsodElement
from typing import Any, Optional, Sequence, Tuple

from numpy import random
from torch import nn
from torchvision import transforms as T

from lib.data.structures import WsodElement


__all__ = ['Compose', 'DropSmallProposals', 'RandomHorizontalFlip', 'RandomResizeSmallestEdge', 'ResizeLargestEdge', 
           'ResizeSmallestEdge']


class Compose(nn.Module):
    """
    Chains multiple transformations together
    """
    def __init__(
        self,
        *modules: nn.Module
    ):
        self().__init__()
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
    ) -> WsodElement:
        if element.proposals is not None:
            wh = element.proposals[..., 2:4] - element.proposals[..., 0:2]
            mask = (wh >= self.min_size).all(-1)
            element.proposals = element.proposals[mask]
        return element


class RandomHorizontalFlip(nn.Module):
    """
    Randomly flips the image and proposals horizontally
    """
    def __init__(
        self,
        p: float = 0.5,
        rng_seed: Optional[int] = None,
    ):
        super().__init__()
        self.p = p
        self._rng = random.default_rng(rng_seed)

    def forward(
        self,
        element: WsodElement
    ) -> WsodElement:
        if self._rng.random() < self.p:
            element.image = T.functional.hflip(element.image)
            if element.proposals is not None:
                element.proposals[..., [0, 2]] = element.image_size[0] - element.proposals[..., [2, 0]] - 1
        return element


class ResizeSmallestEdge(nn.Module):
    """
    Resize the smallest edge of an image to a fixed size, not exceeding a maximum
    """
    def __init__(
        self,
        size: int = 480,
        max_size: int = 2000,
    ):
        super().__init__()
        assert size <= max_size, "Size must not exceed max_size"
        self.size = size
        self.max_size = max_size

    def forward(
        self,
        element: WsodElement
    ) -> WsodElement:
        min_ratio = self.size / min(element.image_size)
        max_ratio = self.max_size / max(element.image_size)
        ratio = min(min_ratio, max_ratio)
        new_size = (element.image_size * ratio).round().astype(element.image_size.dtype)
        element.image = T.functional.resize(element.image, new_size)
        if element.proposals is not None:
            element.proposals = element.proposals * ratio
        return element


class RandomResizeSmallestEdge(ResizeSmallestEdge):
    """
    Resize the smallest to one of a list of scales, selected randomly
    """
    def __init__(
        self,
        sizes: Sequence[int],
        max_size: int = 2000,
        rng_seed: int = None,
    ):
        super().__init__(sizes[0], max_size)
        self.sizes = sizes
        self._rng = random.default_rng(rng_seed)

    def forward(
        self, 
        element: WsodElement
    ) -> WsodElement:
        self.size = self._rng.choice(self.sizes)
        return super().forward(element)


class ResizeLargestEdge(nn.Module):
    """
    Resize the largest edge of an image to a fixed size
    """
    def __init__(
        self,
        size: int = 480,
    ):
        super().__init__()
        self.size = size

    def forward(
        self,
        element: WsodElement
    ) -> WsodElement:
        ratio = self.size / max(element.image_size)
        new_size = (element.image_size * ratio).round().astype(element.image_size.dtype)
        element.image = T.functional.resize(element.image, new_size)
        if element.proposals is not None:
            element.proposals = element.proposals * ratio
        return element

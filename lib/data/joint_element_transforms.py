from typing import Sequence, Tuple, Optional

from numpy import random
from PIL import Image
from torch import nn

from lib.data.structures import WsodElement, WsodElementLabels


class RandomResizeSmallestEdge(nn.Module):
    def __init__(
        self,
        sizes: Sequence[int] = [480, 576, 688, 864, 1200],
        max_size: int = 2000,
        rng_seed: Optional[int] = None
    ):
        super().__init__()
        assert all([size <= max_size for size in sizes]), "sizes must not exceed max_size"
        self.sizes = sizes
        self.max_size = max_size
        self._rng = random.default_rng(rng_seed)

    def forward(
        self,
        *elements_and_labels: Tuple[WsodElement, WsodElementLabels]
    ) -> Sequence[Tuple[WsodElement, WsodElementLabels]]:
        size = self._rng.choice(self.sizes)
        resized_elements_and_labels = []
        for (element, label) in elements_and_labels:
            ratio = min(size / element.image_size.min(),
                        self.max_size / element.image_size.max())
            new_size = (ratio * size).round().astype(int)
            
            element.image = element.image.resize(new_size, resample=Image.BILINEAR)
            if element.proposals is not None:
                element.proposals = element.proposals * ratio
            
            resized_elements_and_labels.append((element, label))
        return resized_elements_and_labels


class RandomResizeLargestEdge(nn.Module):
    def __init__(
        self,
        sizes: Sequence[int] = [480, 576, 688, 864, 1200],
        rng_seed: Optional[int] = None
    ):
        super().__init__()
        self.sizes = sizes
        self._rng = random.default_rng(rng_seed)

    def forward(
        self,
        *elements_and_labels: Tuple[WsodElement, WsodElementLabels]
    ) -> Sequence[Tuple[WsodElement, WsodElementLabels]]:
        size = self._rng.choice(self.sizes)
        resized_elements_and_labels = []
        for (element, label) in elements_and_labels:
            ratio = size / element.image_size.max()
            new_size = (ratio * size).round().astype(int)
            
            element.image = element.image.resize(new_size, resample=Image.BILINEAR)
            if element.proposals is not None:
                element.proposals = element.proposals * ratio
            
            resized_elements_and_labels.append((element, label))
        return resized_elements_and_labels
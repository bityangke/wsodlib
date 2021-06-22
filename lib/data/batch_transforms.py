from torch import nn
from torchvision import transforms as T

from lib.data.structures import WsodBatch


__all__ = ['BatchResizeLargestEdge', 'BatchResizeSmallestEdge']


class BatchResizeSmallestEdge(nn.Module):
    """
    Resize the smallest edge of a batch of images
    """
    def __init__(
        self,
        size: int = 480,
        max_size: int = 2000,
    ):
        super().__init__()
        self.size = size
        self.max_size = max_size

    def forward(
        self,
        batch: WsodBatch,
    ) -> WsodBatch:
        min_ratio = self.size / min(batch.images.shape[-2:])
        max_ratio = self.max_size / max(batch.images.shape[-2:])
        ratio = min(min_ratio, max_ratio)
        new_size = (batch.images.shape[[-1, -2]] * ratio).round().astype(batch.image_sizes.dtype)
        batch.images = T.functional.resize(batch.images, new_size)
        batch.image_sizes = (batch.image_sizes * ratio).round().astype(batch.image_sizes.dtype)
        for i in range(len(batch.proposals)):
            if batch.proposals[i] is not None:
                batch.proposals[i] = batch.proposals[i] * ratio
        return batch


class BatchResizeLargestEdge(nn.Module):
    """
    Resize the largest edge of a batch of images
    """
    def __init__(
        self,
        size: int = 480,
    ):
        super().__init__()
        self.size = size

    def forward(
        self,
        batch: WsodBatch,
    ) -> WsodBatch:
        ratio = self.size / max(batch.images.shape[-2:])
        new_size = (batch.images.shape[[-1, -2]] * ratio).round().astype(batch.image_sizes.dtype)
        batch.images = T.functional.resize(batch.images, new_size)
        batch.image_sizes = (batch.image_sizes * ratio).round().astype(batch.image_sizes.dtype)
        for i in range(len(batch.proposals)):
            if batch.proposals[i] is not None:
                batch.proposals[i] = batch.proposals[i] * ratio
        return batch

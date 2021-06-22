from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


__all__ = ['WsodBatch', 'WsodElement']


@dataclass
class WsodElement:
    """
    A single (i.e. unbatched) WSOD training element
    """
    image: Image.Image
    filename: str
    img_id: str
    image_labels: np.ndarray
    original_size: np.ndarray
    image_size: np.ndarray
    proposals: Optional[np.ndarray] = None
    objectness: Optional[np.ndarray] = None


@dataclass
class WsodBatch:
    """
    A batch of WSOD training elements
    """
    images: torch.Tensor
    filenames: Tuple[str, ...]
    img_ids: Tuple[str, ...]
    image_labels: torch.Tensor
    original_sizes: torch.Tensor
    image_sizes: torch.Tensor
    proposals: Tuple[Optional[torch.Tensor], ...]
    objectness: Tuple[Optional[torch.Tensor], ...]

    def pin_memory(
        self,
    ) -> WsodBatch:  # type: ignore
        self.images = self.images.pin_memory()
        self.image_labels = self.images.pin_memory()
        self.proposals = tuple(p.pin_memory() if p is not None else None for p in self.proposals)
        self.objectness = tuple(o.pin_memory() if o is not None else None for o in self.objectness)
        return self

    def to(
        self,
        *args,
        **kwargs,
    ) -> WsodBatch:  # type: ignore
        self.images = self.images.to(*args, **kwargs)
        self.image_labels = self.image_labels.pin_memory().to(*args, **kwargs)
        self.proposals = tuple(p.to(*args, **kwargs) if p is not None else None for p in self.proposals)
        self.objectness = tuple(o.to(*args, **kwargs) if o is not None else None for o in self.objectness)
        return self

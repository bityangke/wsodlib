from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
    image_size: np.ndarray
    proposals: Optional[np.ndarray] = None
    objectness: Optional[np.ndarray] = None


@dataclass
class WsodElementLabels:
    filename: str
    img_id: str
    image_labels: np.ndarray
    original_size: np.ndarray


@dataclass
class WsodBatch:
    """
    A batch of WSOD training elements
    """
    images: torch.Tensor
    image_sizes: torch.Tensor
    proposals: Sequence[Optional[torch.Tensor]]
    objectness: Sequence[Optional[torch.Tensor]]

    def pin_memory(
        self,
    ) -> WsodBatch:  # type: ignore
        self.images = self.images.pin_memory()
        self.proposals = tuple(p.pin_memory() if p is not None else None for p in self.proposals)
        self.objectness = tuple(o.pin_memory() if o is not None else None for o in self.objectness)
        return self

    def to(
        self,
        *args,
        **kwargs,
    ) -> WsodBatch:  # type: ignore
        self.images = self.images.to(*args, **kwargs)
        self.proposals = tuple(p.to(*args, **kwargs) if p is not None else None for p in self.proposals)
        self.objectness = tuple(o.to(*args, **kwargs) if o is not None else None for o in self.objectness)
        return self


@dataclass
class WsodBatchLabels:
    filenames: Sequence[str]
    img_ids: Sequence[str]
    image_labels: torch.Tensor
    original_sizes: torch.Tensor
    
    def pin_memory(
        self,
    ) -> WsodBatchLabels:  # type: ignore
        self.image_labels = self.image_labels.pin_memory()
        return self

    def to(
        self,
        *args,
        **kwargs,
    ) -> WsodBatchLabels:  # type: ignore
        self.image_labels = self.image_labels.to(*args, **kwargs)
        return self
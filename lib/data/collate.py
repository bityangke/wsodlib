from typing import Callable, Optional, Sequence, Tuple

import torch
import torchvision.transforms as T
from torch import nn

from lib.data.structures import WsodBatch, WsodElement


__all__ = ['pad_norm_and_collate', 'PadAndCollate']


def pad_norm_and_collate(
    elements: Sequence[Tuple[WsodElement, None]],
    norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
) -> WsodBatch:
    all_images, filenames, img_ids, image_labels = [], [], [], []
    original_sizes, image_sizes, proposals, objectness = [], [], [], []
    max_size = torch.zeros(4, dtype=torch.int32)

    for element, _ in elements:
        all_images.append(norm(T.functional.to_tensor(element.image)))
        filenames.append(element.filename)
        img_ids.append(element.img_id)
        image_labels.append(torch.as_tensor(element.image_labels))
        original_sizes.append(torch.as_tensor(element.original_size))
        image_sizes.append(torch.as_tensor(element.image_size))
        proposals.append(torch.as_tensor(element.proposals))
        objectness.append(torch.as_tensor(element.objectness))
        max_size = torch.maximum(max_size, all_images[-1])
    tensor_image_labels = torch.stack(image_labels)
    tensor_original_sizes = torch.stack(original_sizes)
    tensor_image_sizes = torch.stack(image_sizes)

    tensor_images = torch.zeros([len(all_images)] + max_size.tolist(), dtype=all_images[-1].dtype)
    for i, image in enumerate(all_images):
        tensor_images[i, :image.size(0), :image.size(1), :image.size(2)].copy_(image)

    return WsodBatch(
        tensor_images,
        tuple(filenames),
        tuple(img_ids),
        tensor_image_labels,
        tensor_original_sizes,
        tensor_image_sizes,
        tuple(proposals),
        tuple(objectness),
    )


class PadAndCollate(nn.Module):
    def __init__(
        self,
        norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        batch_transforms: Optional[Callable[[WsodBatch], WsodBatch]] = None,
    ):
        super().__init__()
        self.norm = norm
        self.batch_transforms = batch_transforms if batch_transforms is not None else lambda x: x

    def forward(
        self,
        elements: Sequence[Tuple[WsodElement, None]],
    ) -> WsodBatch:
        batch = pad_norm_and_collate(elements, self.norm)
        return self.batch_transforms(batch)

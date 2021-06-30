from typing import Callable, Optional, Sequence, Tuple

import torch
import torchvision.transforms as T
from torch import nn

from wsodlib.data.structures import WsodBatch, WsodBatchLabels, WsodElement, WsodElementLabels


__all__ = ['pad_norm_and_collate', 'PadAndCollate']


def pad_norm_and_collate(
    elements_and_labels: Sequence[Tuple[WsodElement, WsodElementLabels]],
    norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
) -> Tuple[WsodBatch, WsodBatchLabels]:
    all_images, filenames, img_ids, image_labels = [], [], [], []
    original_sizes, image_sizes, proposals, objectness = [], [], [], []
    max_size = torch.zeros(3, dtype=torch.int32)

    for element, element_labels in elements_and_labels:
        all_images.append(norm(T.functional.to_tensor(element.image)))
        image_sizes.append(torch.as_tensor(element.image_size))
        proposals.append(torch.as_tensor(element.proposals))
        objectness.append(torch.as_tensor(element.objectness))
        max_size = torch.maximum(max_size, torch.as_tensor(all_images[-1].size()))

        filenames.append(element_labels.filename)
        img_ids.append(element_labels.img_id)
        image_labels.append(torch.as_tensor(element_labels.image_labels))
        original_sizes.append(torch.as_tensor(element_labels.original_size))

    tensor_image_labels = torch.stack(image_labels)
    tensor_original_sizes = torch.stack(original_sizes)
    tensor_image_sizes = torch.stack(image_sizes)

    tensor_images = torch.zeros([len(all_images)] + max_size.tolist(), dtype=all_images[-1].dtype)
    for i, image in enumerate(all_images):
        tensor_images[i, :image.size(0), :image.size(1), :image.size(2)].copy_(image)

    return (
        WsodBatch(
            tensor_images,
            tensor_image_sizes,
            tuple(proposals),
            tuple(objectness),
        ),
        WsodBatchLabels(
            tuple(filenames),
            tuple(img_ids),
            tensor_image_labels,
            tensor_original_sizes,
        )
    )


class PadAndCollate(nn.Module):
    def __init__(
        self,
        norm: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        joint_element_transforms: Optional[Callable[[Sequence[Tuple[WsodElement, WsodElementLabels]]], 
                                                    Sequence[Tuple[WsodElement, WsodElementLabels]]]] = None,
        batch_transforms: Optional[Callable[[WsodBatch], WsodBatch]] = None,
    ):
        super().__init__()
        self.norm = norm
        self.joint_element_transforms = (joint_element_transforms 
                                         if joint_element_transforms is not None else lambda *x: x)
        self.batch_transforms = batch_transforms if batch_transforms is not None else lambda x: x

    def forward(
        self,
        elements_and_labels: Sequence[Tuple[WsodElement, WsodElementLabels]],
    ) -> Tuple[WsodBatch, WsodBatchLabels]:
        elements_and_labels = self.joint_element_transforms(*elements_and_labels)  # type: ignore
        batch, batch_labels = pad_norm_and_collate(elements_and_labels, self.norm)
        return self.batch_transforms(batch), batch_labels

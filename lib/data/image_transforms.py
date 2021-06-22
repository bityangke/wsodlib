from typing import Union

from torch import nn
from torchvision import transforms as T

from lib.data.structures import WsodBatch, WsodElement


__all__ = ['AutoAugment']


class AutoAugment(nn.Module):
    """
    Applies the autoaugment augmentation policy, with any spatial transforms removed
    """
    def __init__(
        self,
        policy: T.AutoAugmentPolicy = T.AutoAugmentPolicy.IMAGENET,
    ):
        super().__init__()
        autoaugment = T.AutoAugment(policy=policy)
        for i, transforms in enumerate(autoaugment.transforms):
            transforms = tuple(tfm if not tfm[0] in 'Rotate ShearX' else None for tfm in transforms)
            autoaugment.transforms[i] = transforms
        self.autoaugment = autoaugment

    def forward(
        self,
        input: Union[WsodBatch, WsodElement]
    ) -> Union[WsodBatch, WsodElement]:
        if isinstance(input, WsodBatch):
            input.images = self.autoaugment(input.images)
        else:  # isinstance(input, WsodElement)
            input.image = self.autoaugment(input.image)
        return input

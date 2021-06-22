from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image

from lib.data.annotation_transform import DatasetAnnotationParser
from lib.data.structures import WsodElement


__all__ = ['VOCWsodAnnotationParser']


_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
_LABEL_TO_ID = dict(zip(_LABELS, range(len(_LABELS))))


class VOCWsodAnnotationParser(DatasetAnnotationParser):
    """
    Parses the VOC XML Entity Tree dict into WSOD annotations
    """
    def __init__(
        self,
        use_difficult: bool = False,
    ):
        super().__init__()
        self.use_difficult = use_difficult

    def forward(
        self,
        image: Image.Image,
        annotation: Dict[str, Any],
    ) -> WsodElement:
        annotation = annotation['annotation']

        # Construct the image labels
        label_ids = set()
        for obj in annotation['object']:
            if self.use_difficult or obj['difficult'] == '0':
                label_ids.add(int(_LABEL_TO_ID[obj['name']]))
        img_labels = np.zeros((20,), dtype=np.float32)
        img_labels[list(label_ids)] = 1.

        element = WsodElement(
            image,
            annotation['filename'],
            annotation['filename'][:-4],
            img_labels,
            np.array([int(annotation['size']['width']), int(annotation['size']['height'])]),
            np.array([int(annotation['size']['width']), int(annotation['size']['height'])]),
        )

        return element

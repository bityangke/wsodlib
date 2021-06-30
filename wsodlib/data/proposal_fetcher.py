from wsodlib.data.structures import WsodElement
import pickle
from typing import Any, Dict, Tuple

import numpy as np
from torch import nn

from wsodlib.data.structures import WsodElement, WsodElementLabels


__all__ = ['ProposalFetcher']


def _load_proposals(
    file: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """ Loads precomputed proposals in Detectron2 format """
    with open(file, 'rb') as f:
        d: Dict[str, np.ndarray] = pickle.load(f)

    proposal_dict = dict(zip(d['ids'], d['boxes']))
    objectness_dict = dict(zip(d['ids'], d['objectness_logits']))
    return proposal_dict, objectness_dict


class ProposalFetcher(nn.Module):
    """
    Can be inserted into a `torchvision.transforms` pipeline to load
    precomputed proposals into the annotation dictionary
    """
    def __init__(
        self,
        proposal_file: str,
    ):
        super().__init__()
        self.proposal_dict, self.objectness_dict = _load_proposals(proposal_file)

    def forward(
        self,
        element: WsodElement,
        labels: WsodElementLabels,
    ) -> Tuple[WsodElement, WsodElementLabels]:
        element.proposals = self.proposal_dict[labels.img_id].copy()
        element.objectness = self.objectness_dict[labels.img_id].copy()
        return element, labels

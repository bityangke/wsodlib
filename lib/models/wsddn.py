from lib.data.structures import WsodBatch, WsodBatchLabels
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import ops

from lib.layers.midn import MidnLayer
from lib.models.wsod_model import WsodModel


__all__ = ['Wsddn', 'simple_wsddn_loss', 'spatial_wsddn_loss']


class Wsddn(WsodModel):
    """
    Unverified implementation of WSDDN by Bilen and Vedaldi
    """
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module,
        neck: nn.Module,
        head: MidnLayer,
        use_objectness: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler
        self.neck = neck
        self.head = head
        self.use_objectness = use_objectness

    def forward(
        self,
        images: torch.Tensor,
        proposals: List[torch.Tensor],
        objectness: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        features = self.backbone(images)
        spp_features = self.pooler(features, proposals)
        if self.use_objectness:
            spp_features = spp_features * torch.cat(objectness, 0).view(-1, 1, 1, 1)
        neck_features = self.neck(spp_features)
        proposal_counts = [len(p) for p in proposals]
        return [
            {
                'latent': nf,
                'midn': self.head(nf),
                'proposals': p,
            }
            for nf, p in zip(neck_features.split(proposal_counts, 0), proposals)
        ]


_EPS = 1e-8

def simple_wsddn_loss(
    predictions: List[Dict[str, torch.Tensor]],
    labels: WsodBatchLabels,
) -> Dict[str, torch.Tensor]:
    image_level_predictions = torch.stack([p['midn'].sum(0) for p in predictions], 0).clamp(_EPS, 1-_EPS)
    return {
        'midn_loss': F.binary_cross_entropy(image_level_predictions, labels.image_labels, 
                                            reduction='sum').div(image_level_predictions.size(0))
    }


def spatial_wsddn_loss(
    predictions: List[Dict[str, torch.Tensor]],
    labels: WsodBatchLabels,
    tau: float = 0.6,
) -> Dict[str, torch.Tensor]:
    loss_dict = simple_wsddn_loss(predictions, labels)
    for i, prediction in enumerate(predictions):
        klasses = labels.image_labels[i].nonzero()[:, 0]
        
        top_scores, top_idxs = prediction['midn'][:, klasses].max(0)
        top_boxes = prediction['proposals'][top_idxs]
        top_features = prediction['latent'][top_idxs]
        max_overlaps, gt_assignment = ops.box_iou(prediction['proposals'], top_boxes).max(1)
        pos_mask = max_overlaps > tau
        assignment_weights = top_scores.gather(0, gt_assignment) ** 2
        gt_features = top_features[gt_assignment, :]
        loss_dict['sp_reg_loss'] = (loss_dict.get('sp_reg_loss', 0.) + 
                                    (assignment_weights[pos_mask] * (gt_features[pos_mask] - prediction['latent'][pos_mask])).sum() 
                                    / pos_mask.sum())
    return loss_dict


def postprocess(
    prediction: Dict[str, torch.Tensor],
    rescale_factor: float = 1.,
    nms_thresh: float = 0.4,
    pre_nms_max: int = 1000,
    post_nms_max: int = 100,
) -> Dict[str, torch.Tensor]:
    scores = prediction['midn']
    boxes = (prediction['proposals'] / rescale_factor).repeat_interleave(scores.size(1), 0)
    klasses = torch.arange(scores.size(1)).repeat(scores.size(0)).to(scores.device)
    scores = scores.flatten()

    sorted_idxs = scores.argsort(0, descending=True)[:pre_nms_max]
    scores = scores[sorted_idxs]
    boxes = boxes[sorted_idxs]
    klasses = klasses[sorted_idxs]

    keep_idxs = ops.batched_nms(boxes, scores, klasses, nms_thresh)[:post_nms_max]
    return {
        'scores': scores[keep_idxs],
        'boxes': boxes[keep_idxs],
        'labels': klasses[keep_idxs],
    }
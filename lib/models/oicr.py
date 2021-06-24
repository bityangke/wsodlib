from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import ops

from lib.data.structures import WsodBatch, WsodBatchLabels
from lib.layers.midn import MidnLayer
from lib.models.wsddn import simple_wsddn_loss
from lib.utils.box_utils import compute_delta


class Oicr(nn.Module):
    """
    Unverified implementation of OICR model by Tang et. al.
    """
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module,
        neck: nn.Module,
        num_classes: int = 20,
        k_refinements: int = 3,
        use_regression: bool = False,
        neck_output_features: int = 4096,
        use_objectness: bool = False,
    ):
        super().__init__()
        self.use_objectness = use_objectness
        self.use_regression = use_regression
        self.k_refinements = k_refinements
        self.num_classes = num_classes

        self.backbone = backbone
        self.pooler = pooler
        self.neck = neck
        self.midn_head = MidnLayer(neck_output_features, num_classes, 1.)
        self.refinement_heads = nn.Linear(neck_output_features, (num_classes+1) * k_refinements)
        self.regression_head = nn.Linear(neck_output_features, 4* num_classes) if use_regression else None

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
                'features': nf,
                'midn': self.midn_head(nf),
                'refinements': self.refinement_heads(nf).reshape(-1, self.k_refinements, self.num_classes+1),
                'regression': self.regression_head(nf).reshape(-1, self.num_classes, 4)
                if self.regression_head is not None else None,
                'proposals': p,
            }
            for nf, p in zip(neck_features.split(proposal_counts, 0), proposals)
        ]


def max_pseudo_label(
    prediction: Dict[str, torch.Tensor],
    proposals: torch.Tensor,
    klasses: torch.Tensor,
    midn_scale_factor: float = 3.,
    bg_thresh: float = 0.5,
    ig_thresh: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # determine the pseudo ground-truths
    probs = torch.cat([midn_scale_factor * prediction['midn'].unsqueeze(1),
                       prediction['refinements'].softmax(1)[:, :-1, 1:]], dim=1)
    gt_weights, gt_idxs = probs[..., klasses].max(0)
    gt_boxes = proposals[gt_idxs]
    gt_klasses = klasses[:, None].repeat(1, gt_idxs.size(0))

    # Compute assignment
    overlaps = ops.box_iou(proposals, gt_boxes.view(-1, 4)).view(proposals.size(0), gt_boxes.size(0), gt_boxes.size(1))
    max_overlaps, gt_assignment = overlaps.max(-1)

    # Assign labels and weights
    gt_labels = (gt_klasses + 1).gather(0, gt_assignment)
    gt_labels[max_overlaps < bg_thresh] = 0
    label_weights = gt_weights.gather(0, gt_assignment)
    label_weights[max_overlaps < ig_thresh] = 0.

    reg_box_targets = gt_boxes[-1].gather(0, gt_assignment[-1, ..., None].repeat(1, 4))
    reg_delta_targets = compute_delta(proposals, reg_box_targets)

    return gt_labels, label_weights, reg_delta_targets


def oicr_loss(
    predictions: List[Dict[str, torch.Tensor]],
    labels: WsodBatchLabels,
) -> Dict[str, torch.Tensor]:
    loss_dict = simple_wsddn_loss(predictions, labels)
    for i, prediction in enumerate(predictions):
        klasses = labels.image_labels[i].nonzero()[:, 0]
        
        gt_labels, label_weights, reg_delta_targets = max_pseudo_label(prediction, prediction['proposals'], klasses)
        loss_dict['refinement_loss'] = (loss_dict.get('refinement_loss', 0.) + 
                                        (label_weights.flatten() 
                                         * F.cross_entropy(prediction['refinements'].flatten(end_dim=1), 
                                                           gt_labels.flatten(),
                                                           reduction='none')).mean())
        if prediction['regression'] is not None:
            pos_mask = gt_labels[..., -1] > 0
            loss_dict['regression_loss'] = (loss_dict.get('regression_loss', 0.) +
                                            (label_weights[pos_mask, -1, None]
                                             * F.smooth_l1_loss(prediction['regression'], 
                                                                reg_delta_targets, 
                                                                reduction='none')).mean())
    return loss_dict
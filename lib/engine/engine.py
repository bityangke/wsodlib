from typing import Any, Callable, Dict, Optional, Union

from fastprogress.fastprogress import progress_bar
import torch

from lib.data.structures import WsodBatchLabels
from lib.models.wsod_model import WsodModel
from lib.utils.loss_utils import reduce_loss_dict


class _NoopScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self): pass
    def step(self, *args, **kwargs): pass


def train_one_epoch(
    model: WsodModel,
    loss_fn: Callable[[Dict[str, torch.Tensor], WsodBatchLabels], Dict[str, torch.Tensor]],
    loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    step_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Union[str, torch.device] = 'cuda:0',
    ema_init: float = 1.
):
    if step_scheduler is None:
        step_scheduler = _NoopScheduler()

    model.train()
    pbar = progress_bar(loader)
    ema = ema_init
    for batch, batch_labels in pbar:
        # forward pass
        batch = batch.to(device)
        batch_labels = batch_labels.to(device)
        predictions = model(batch.images, batch.proposals, batch.objectness)
        loss_dict = loss_fn(predictions, batch_labels)
        loss = reduce_loss_dict(loss_dict)

        # backward pass
        loss.backward()
        optim.step()
        optim.zero_grad(True)
        step_scheduler.step()
        
        float_loss = loss.item()
        ema = ema * 0.95 + float_loss * 0.05
        pbar.comment = f'Loss: {float_loss:7.04f}, EMA: {ema:7.04f}'


@torch.no_grad()
def val_one_epoch(
    model: WsodModel,
    loader: torch.utils.data.DataLoader,
    eval_fn: Callable[..., Dict[str, Any]],
    postprocess: Callable = lambda x: x,
    device: Union[str, torch.device] = 'cuda:0'
) -> Dict[str, Any]:
    model.eval()
    pbar = progress_bar(loader)
    id_to_prediction = {}
    for batch, labels in pbar:
        batch = batch.to(device)
        labels = labels.to(device)
        predictions = model(batch.images, batch.proposals, batch.objectness)
        postprocessed = [postprocess(prediction, ns[0] / os[0]) 
                         for prediction, os, ns in zip(predictions, labels.original_sizes, batch.image_sizes)]
        for img_id, pp in zip(labels.img_ids, postprocessed):
            id_to_prediction[img_id] = pp
    return eval_fn(id_to_prediction)

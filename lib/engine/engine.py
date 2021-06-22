from typing import Any, Callable, Dict, Optional, Union

from tqdm import tqdm
import torch

from lib.data.structures import WsodBatch
from lib.models.wsod_model import WsodModel
from lib.utils.loss_utils import reduce_loss_dict


class _NoopScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self): pass
    def step(self, *args, **kwargs): pass


def train_one_epoch(
    model: WsodModel,
    loss_fn: Callable[[Dict[str, torch.Tensor], WsodBatch], Dict[str, torch.Tensor]],
    loader: torch.utils.data.DataLoader,
    optim: torch.optim.Optimizer,
    step_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Union[str, torch.device] = 'cuda:0',
):
    if step_scheduler is None:
        step_scheduler = _NoopScheduler()

    model.train()
    pbar = tqdm(loader)
    for batch in pbar:
        # forward pass
        batch = batch.to(device)
        predictions = model(batch.images, batch.proposals, batch.objectness)
        loss_dict = loss_fn(predictions, batch)
        loss = reduce_loss_dict(loss_dict)

        # backward pass
        loss.backward()
        optim.step()
        optim.zero_grad(True)
        step_scheduler.step()


@torch.no_grad()
def val_one_epoch(
    model: WsodModel,
    loader: torch.utils.data.DataLoader,
    eval_fn: Callable[..., Dict[str, Any]],
    postprocess: Callable = lambda x: x,
    device: Union[str, torch.device] = 'cuda:0'
) -> Dict[str, Any]:
    model.eval()
    pbar = tqdm(loader)
    id_to_prediction = {}
    for batch in pbar:
        batch = batch.to(device)
        predictions = model(batch.images, batch.proposals, batch.objectness)
        postprocessed = postprocess(predictions)
        id_to_prediction[batch.img_ids] = postprocessed
    return eval_fn(id_to_prediction)

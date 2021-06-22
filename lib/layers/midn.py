import torch
from torch import nn


__all__ = ['MidnLayer']


class MidnLayer(nn.Module):
    def __init__(
        self,
        in_features: int = 4096,
        out_features: int = 20,
        temperature: float = 1.,
    ):
        super().__init__()
        self.midn_c = nn.Linear(in_features, out_features)
        self.midn_d = nn.Linear(in_features, out_features, bias=False)
        self.temperature = 1. / temperature

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        c = self.midn_c(x).softmax(-1)
        d = self.midn_d(x).mul(self.temperature).softmax(-2)
        return c * d

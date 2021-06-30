from typing import Optional

import torch
from torch import nn


__all__ = ['MultiHeadAttention', 'MultiHeadSelfAttention', 'Residual']


class MultiHeadAttention(nn.Module):
    """
    Based heavily on the implementation at https://github.com/rwightman/pytorch-image-models
    """
    def __init__(
        self,
        dimension: int,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or input_dimension or dimension
        self.num_heads = num_heads
        self.head_dimension = self.dimension // self.num_heads
        self.scale = self.head_dimension ** -0.5

        self.q_project = nn.Linear(self.input_dimension, dimension, bias=False)
        self.kv_project = nn.Linear(self.input_dimension, dimension * 2, bias=False)
        self.output_project = nn.Linear(self.dimension, self.output_dimension, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        B, Nx, _ = x.shape
        Ny = y.shape[1]
        q = self.q_project(x).reshape(B, Nx, self.num_heads, self.head_dimension)
        k, v = self.kv_project(y).reshape(B, Ny, 2, self.num_heads, self.head_dimension).unbind(2)

        attn = (q @ k.transpose(-2, -1)).mul(self.scale).softmax(-1)
        r = (attn @ v).transpose(1, 2).reshape(B, Nx, self.dimension)
        r = self.output_project(r)
        return r


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(  # type: ignore
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return super().forward(x, x)


class Residual(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):
        super().__init__()
        self.module = module

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x + self.module(x, *args, **kwargs)

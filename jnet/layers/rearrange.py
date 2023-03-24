from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange


class Rearrange(nn.Module):
    "A module that rearranges the input tensor based on the given pattern using the einops.rearrange function."

    def __init__(self, pattern: str, **axes_length: int) -> None:
        super().__init__()

        self.pattern = pattern
        self.axes_length = axes_length

    def forward(self, x: Tensor, **axes_length: int) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return rearrange(x, self.pattern, **self.axes_length, **axes_length)  # type: ignore

    def extra_repr(self) -> str:
        return f'pattern="{self.pattern}"'

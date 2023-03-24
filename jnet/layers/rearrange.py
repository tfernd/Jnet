from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange


class Rearrange(nn.Module):
    "A module that rearranges the input tensor based on the given pattern using the einops.rearrange function."

    def __init__(self, pattern: str, **kwargs: int) -> None:
        super().__init__()

        self.kwargs = kwargs
        self.method = partial(rearrange, pattern=pattern, **kwargs)

    def forward(self, x: Tensor, **kwargs: int) -> Tensor:
        with torch.set_grad_enabled(self.training):
            return self.method(x, **kwargs)  # type: ignore

    def extra_repr(self) -> str:
        return f"pattern={self.pattern}"

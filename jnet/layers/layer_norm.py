from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        x = rearrange(x, "B C H W -> B (H W) C")
        x = self.layernorm(x)
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)

        return x

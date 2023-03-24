from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ColorMix(nn.Module):
    """
    A module to convert an (RGB) image into an arbitrary learnable color space (Y′CBCR).
    """

    scale: float = 255 / 2

    W: nn.Parameter
    W_inv: nn.Parameter
    bias: nn.Parameter
    bias_inv: nn.Parameter

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()

        self.num_channels = num_channels

        # RGB to Y′CBCR
        if num_channels == 3:
            W = torch.tensor(
                [
                    [0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312],
                ]
            )
            bias = torch.tensor([-1, 0, 0]).float()
        else:
            W = torch.eye(num_channels)
            bias = torch.zeros(num_channels)
        Winv = W.inverse()

        self.enc = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.dec = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        self.enc.weight.data = W[..., None, None]
        self.dec.weight.data = Winv[..., None, None]

        assert self.enc.bias is not None
        assert self.dec.bias is not None

        self.enc.bias.data = bias
        self.dec.bias.data = -bias @ Winv.T

    def encode(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.uint8

        with torch.set_grad_enabled(self.training):
            out = self.enc(x / self.scale)

            # clamp but keep gradients
            clamped = out.detach().clamp(-1, 1)
            if not self.training:
                return clamped

            return clamped + (out - out.detach())  # [-1, 1]

    def decode(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            out = self.dec(x) * self.scale

            # clamp but keep gradients
            discrete = out.detach().clamp(0, 255).round().byte()
            if not self.training:
                return discrete

            return discrete + (out - out.detach())  # [0, 255]

    def extra_repr(self) -> str:
        return f"num_channels={self.num_channels}"

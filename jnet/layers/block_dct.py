from __future__ import annotations
from typing import Optional

from functools import partial

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rearrange import Rearrange


class BlockDCT(nn.Module):
    """
    A module to perform the Discrete Cosine Transform
    (DCT) on an input tensor in patches.
    """

    W: nn.Parameter
    W_inv: nn.Parameter

    def __init__(
        self,
        blocks: int = 8,
        num_channels: int = 3,
    ) -> None:
        super().__init__()

        assert blocks >= 1
        assert num_channels >= 1

        self.blocks = blocks
        self.num_channels = num_channels

        self.emb_size = emb_size = num_channels * blocks**2

        # Construct the DCT matrix W
        i = torch.arange(blocks).view(1, -1)

        W = torch.cos((2 * i.T + 1) * i * torch.pi / (2 * blocks))
        W *= math.sqrt(2 / blocks)
        W[:, 0].fill_(math.sqrt(1 / blocks))

        self.W = nn.Parameter(W)
        self.W_inv = nn.Parameter(W.inverse())  # W.T also works :)

        # patch-wise matrix multiplication
        self.pw_mm = partial(torch.einsum, "bhwcHW,Hi,Wj->bhwcij")
        # channel-wise matrix multiplication
        self.cw_mm = partial(torch.einsum, "bchw,ck->bkhw")

        # Define functions for rearranging input tensor into patches and vice-versa
        self.to_patches = Rearrange("b c (h ph) (w pw) -> b h w c ph pw", ph=blocks, pw=blocks)
        self.from_patches = Rearrange("b h w c ph pw -> b c (h ph) (w pw)")

        # Define functions for merging and splitting channels of blocked patches
        self.merge_block_channels = Rearrange("b h w c ph pw -> b (c ph pw) h w")
        self.split_block_channels = Rearrange("b (c ph pw) h w -> b h w c ph pw", ph=blocks, pw=blocks)

        # entropy encoding. reorder FFT-channels by lowest frequency
        r = i**2 + i.T**2
        r = r.expand(num_channels, -1, -1)
        if num_channels == 3:
            # gives less weight to the Y-channel in Y′CBCR so it retains its frequency
            b = blocks**2
            r = r * torch.tensor([1, b, b]).view(-1, 1, 1)
        r = r.flatten()

        idx = r.argsort()
        W = torch.zeros(emb_size, emb_size)
        for i, j in enumerate(idx):
            W[i, j] = 1

        self.freq_shuffle = nn.Conv2d(emb_size, emb_size, kernel_size=1)
        self.freq_unshuffle = nn.Conv2d(emb_size, emb_size, kernel_size=1)

        self.freq_shuffle.weight.data = W[..., None, None]
        self.freq_unshuffle.weight.data = W.T[..., None, None]

        assert self.freq_shuffle.bias is not None
        assert self.freq_unshuffle.bias is not None

        self.freq_shuffle.bias.data.fill_(0)
        self.freq_unshuffle.bias.data.fill_(0)

    def dct(self, x: Tensor) -> Tensor:
        "Performs the Discrete Cosine Transform (DCT) on the input tensor."

        x = self.pw_mm([x, self.W, self.W])
        x = x / self.blocks

        return x

    def idct(self, x: Tensor) -> Tensor:
        "Performs the Inverse Discrete Cosine Transform (IDCT) on the input tensor."

        x = x * self.blocks
        x = self.pw_mm([x, self.W_inv, self.W_inv])

        return x

    def pad(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape

        # pad data if height and width are not multiples of block
        b = self.blocks
        pad_h = (b - height % b) % b
        pad_w = (b - width % b) % b

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return x

    def crop(self, x: Tensor, size: Optional[tuple[int, int]]) -> Tensor:
        if size is not None:
            height, width = size

            x = x[..., :height, :width]

        return x

    def encode(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            x = self.pad(x)
            x = self.to_patches(x)
            x = self.dct(x)
            x = self.merge_block_channels(x)
            x = self.freq_shuffle(x)

            return x

    def decode(
        self,
        x: Tensor,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        B, C, H, W = x.shape

        with torch.set_grad_enabled(self.training):
            x = self.freq_unshuffle(x)
            x = self.split_block_channels(x)
            x = self.idct(x)
            x = self.from_patches(x)
            x = self.crop(x, size)

            return x

    def extra_repr(self) -> str:
        return f"block={self.blocks}, num_channels={self.num_channels}"

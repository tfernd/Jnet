from __future__ import annotations
from typing import Optional
from typing_extensions import Literal

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lightning.pytorch as pl

from .layers import ColorMix, BlockDCT


BlockKind = Literal["gated", "normal"]


def make_block(kind: BlockKind):
    def normal_block(channels: int, kernel_size: int) -> nn.Sequential:
        conv = partial(nn.Conv2d, kernel_size=kernel_size, padding="same")

        return nn.Sequential(
            conv(channels, channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            conv(channels, channels),
            nn.BatchNorm2d(channels),
        )

    def gated_block(channels: int, kernel_size: int) -> nn.Sequential:
        conv = partial(nn.Conv2d, kernel_size=kernel_size, padding="same")

        return nn.Sequential(
            conv(channels, 2 * channels),
            nn.BatchNorm2d(2 * channels),
            nn.GLU(dim=1),
            conv(channels, channels),
            nn.BatchNorm2d(channels),
        )

    if kind == "normal":
        return normal_block
    return gated_block


class JNet(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        blocks: int = 8,
        latent_size: int = 4,
        kernel_size: int = 3,
        num_layers: int | tuple[int, int] = 1,
        kind: BlockKind = "normal",
    ) -> None:
        super().__init__()

        num_layers = (num_layers, num_layers) if isinstance(num_layers, int) else num_layers

        self.num_channels = num_channels
        self.blocks = blocks
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.kind = kind

        self.emb_size = emb_size = num_channels * blocks**2
        assert 1 <= latent_size <= emb_size

        self.compression = 2 * latent_size / emb_size  # uint8 -> half

        self.receptive_field = (kernel_size - 1) * 2 * sum(num_layers) * blocks + 1

        self.color_mix = ColorMix(num_channels)
        self.block_dct = BlockDCT(blocks, num_channels)

        block = make_block(kind)
        self.enc_blocks = nn.ModuleList([block(emb_size, kernel_size) for _ in range(num_layers[0])])
        self.dec_blocks = nn.ModuleList([block(emb_size, kernel_size) for _ in range(num_layers[1])])

        self.enc_scale = nn.Parameter(torch.zeros(num_layers[0]).fill_(0.001))
        self.dec_scale = nn.Parameter(torch.zeros(num_layers[1]).fill_(0.001))

        # bottleneck # TODO transform into conv2d with kernel-size=1
        W = torch.eye(emb_size)[:, :latent_size]
        self.channel_crop = nn.Parameter(W)
        self.channel_uncrop = nn.Parameter(W.T)

        # channel-wise matrix multiplication
        self.cw_mm = partial(torch.einsum, "bchw,ck->bkhw")

        self.eval()

    def encode(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            x = self.color_mix.encode(x)
            x = self.block_dct.encode(x)

            for scale, block in zip(self.enc_scale, self.enc_blocks):
                x = x + scale * block(x)

            x = self.cw_mm([x, self.channel_crop])

            return x

    def decode(
        self,
        x: Tensor,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        with torch.set_grad_enabled(self.training):
            x = self.cw_mm([x, self.channel_uncrop])

            for scale, block in zip(self.dec_scale, self.dec_blocks):
                x = x + scale * block(x)

            x = self.block_dct.decode(x, size)
            x = self.color_mix.decode(x)

            return x


class JNetModel(pl.LightningModule):
    lr: float = 1e-3

    def __init__(self, model: JNet) -> None:
        super().__init__()

        self.model = model

    def training_step(self, x: Tensor, batch_idx: int) -> Tensor:
        B, C, H, W = x.shape

        z = self.model.encode(x)
        xhat = self.model.decode(z, size=(H, W))

        loss = F.l1_loss(xhat, x.float())

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer

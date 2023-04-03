from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lightning.pytorch as pl

from .layers import ColorMix, BlockDCT, LayerNorm


def block(channels: int, kernel_size: int, ratio: float) -> nn.Sequential:
    mid_channels = round(channels * ratio)

    return nn.Sequential(
        LayerNorm(channels),
        nn.Conv2d(channels, mid_channels, kernel_size, padding="same"),
        nn.GELU(),
        nn.Conv2d(mid_channels, channels, kernel_size, padding="same"),
    )


class JNet(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        blocks: int = 8,
        latent_size: int = 4,
        kernel_size: int = 3,
        ratio: float = 4,
        num_layers: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()

        num_layers = (num_layers, num_layers) if isinstance(num_layers, int) else num_layers

        self.num_channels = num_channels
        self.blocks = blocks
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.emb_size = emb_size = num_channels * blocks**2
        assert 1 <= latent_size <= emb_size

        self.compression = 2 * latent_size / emb_size  # uint8 -> half

        self.receptive_field = (kernel_size - 1) * 2 * sum(num_layers) * blocks + 1

        self.color_mix = ColorMix(num_channels)
        self.block_dct = BlockDCT(blocks, num_channels)

        self.enc_blocks = nn.ModuleList([block(emb_size, kernel_size, ratio) for _ in range(num_layers[0])])
        self.dec_blocks = nn.ModuleList([block(emb_size, kernel_size, ratio) for _ in range(num_layers[1])])

        self.enc_scale = nn.Parameter(torch.zeros(num_layers[0]).fill_(0.001))
        self.dec_scale = nn.Parameter(torch.zeros(num_layers[1]).fill_(0.001))

        # bottleneck
        W = torch.eye(emb_size)[:, :latent_size]

        self.channel_crop = nn.Conv2d(emb_size, latent_size, kernel_size=1, bias=False)
        self.channel_uncrop = nn.Conv2d(latent_size, emb_size, kernel_size=1, bias=False)

        self.channel_crop.weight.data = W.T[..., None, None]
        self.channel_uncrop.weight.data = W[..., None, None]

    def encode(self, x: Tensor) -> Tensor:
        with torch.set_grad_enabled(self.training):
            x = self.color_mix.encode(x)
            x = self.block_dct.encode(x)

            for scale, block in zip(self.enc_scale, self.enc_blocks):
                x = x + scale * block(x)

            x = self.channel_crop(x)

            return x

    def decode(
        self,
        x: Tensor,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        with torch.set_grad_enabled(self.training):
            x = self.channel_uncrop(x)

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

        x = x.to(self.dtype)
        loss = F.l1_loss(xhat, x)

        dx_hat = x[..., 1:, :] - x[..., :-1, :]
        dy_hat = x[..., :, 1:] - x[..., :, :-1]

        dxhat_hat = xhat[..., 1:, :] - xhat[..., :-1, :]
        dyhat_hat = xhat[..., :, 1:] - xhat[..., :, :-1]

        grad_loss = F.l1_loss(dx_hat, dxhat_hat) + F.l1_loss(dy_hat, dyhat_hat)

        self.log("loss", loss)
        self.log("grad_loss", grad_loss)

        return loss + grad_loss * 0.1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer

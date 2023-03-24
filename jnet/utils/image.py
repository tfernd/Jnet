from __future__ import annotations

from pathlib import Path
from PIL import Image, ExifTags

import numpy as np

import torch
from torch import Tensor

from einops import rearrange


def open_image(path: str | Path) -> Image.Image:
    """
    Opens an image file located at the given path
    with PIL, and rotates it according to its
    orientation metadata if necessary.
    """

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Get orientation flag from image metadata
    exif_data = getattr(img, "_getexif", lambda: None)()
    if exif_data:
        for tag, value in exif_data.items():
            if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == "Orientation":
                orientation = value
                # Apply rotation if necessary
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
                break

    return img


def image2tensor(img: Image.Image) -> Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.asarray(img)
    arr = rearrange(arr, "h w c -> c h w")

    return torch.from_numpy(arr)


@torch.no_grad()
def tensor2images(x: Tensor) -> list[Image.Image]:
    assert x.dtype == torch.uint8

    if x.ndim == 3:
        x = x[None]

    x = x.cpu()

    x = rearrange(x, "b c h w -> b h w c")
    xs: list[Tensor] = x.unbind(dim=0)

    return [Image.fromarray(x.numpy()) for x in xs]

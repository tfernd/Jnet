from __future__ import annotations

from pathlib import Path

from PIL import Image

import random
import math

import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from .utils import image2tensor


class ImageDataset(Dataset):
    def __init__(self, root: str | Path, height: int, width: int, suffix: str = ".jpg") -> None:
        root = Path(root)

        self.height = height
        self.width = width

        paths = list(root.rglob(f"*{suffix}"))

        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]

        img = Image.open(path)
        width, height = img.size

        # random resize
        if width > self.width and height > self.height:
            scale = min(width / self.width, height / self.height)
            scale = random.uniform(1, scale)

            w, h = math.ceil(width / scale), math.ceil(height / scale)
            img = img.resize((w, h), resample=Image.LANCZOS)
            width, height = img.size

        # random crop
        if width > self.width or height > self.height:
            pw = random.randint(0, width - self.width)
            ph = random.randint(0, height - self.height)

            img = img.crop((pw, ph, pw + self.width, ph + self.height))

        return image2tensor(img)

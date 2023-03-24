from __future__ import annotations

import torch.nn as nn


def num_parameters(layer: nn.Module) -> int:
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

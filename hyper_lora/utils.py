from __future__ import annotations
from typing import Literal, Union

import torch

Device = Union[torch.device, Literal["cuda", "cpu", "auto"]]


def auto(device: Device, /) -> tuple[torch.device, torch.dtype]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    dtype = torch.float32 if device.type == "cpu" else torch.float16

    return device, dtype

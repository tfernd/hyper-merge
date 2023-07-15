from __future__ import annotations
from typing import Literal, Union

from pathlib import Path

import torch
from torch import Tensor
from safetensors.torch import save_file

Device = Union[torch.device, Literal["cuda", "cpu", "auto"]]


def auto(device: Device, /) -> tuple[torch.device, torch.dtype]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    dtype = torch.float32 if device.type == "cpu" else torch.float16

    return device, dtype


def save_ckpt(obj: dict[str, Tensor], path: Path, /) -> None:
    if path.exists():
        path.unlink()

    assert path.suffix == ".safetensors"

    path.parent.mkdir(exist_ok=True, parents=True)
    save_file(obj, path)

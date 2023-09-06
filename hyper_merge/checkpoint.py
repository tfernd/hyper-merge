from __future__ import annotations
from typing import Optional

import logging
from pathlib import Path

import torch
from torch import Tensor
from safetensors.torch import load_file, save_file


def load_ckpt(
    path: str | Path,
    /,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        ckpt = load_file(path, device="cpu")
    else:
        logging.warning("Please use .safetensors!")

        ckpt = torch.load(path, map_location="cpu")
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    return {key: weight.to(dtype=dtype, device=device) for (key, weight) in ckpt.items()}


def save_ckpt(
    model: dict[str, Tensor],
    path: Path,
    /,
    *,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    if path.exists():
        logging.warning(f"Deleting {path}!")
        path.unlink()

    assert path.suffix == ".safetensors"

    path.parent.mkdir(exist_ok=True, parents=True)
    save_file(model, path, metadata=metadata)

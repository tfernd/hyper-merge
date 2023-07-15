from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm

import re
import json
from pathlib import Path

import torch
from torch import Tensor

from .utils import save_ckpt, auto, Device
from .svd import svd, to_weight, SVDBasis
from .civitai import load_ckpt

# Read LoRA mapping for SD 1.5
with open("sd_1.5-lora_mapping.json", "r") as f:
    SD15MAP: dict[str, str] = json.load(f)
    SD15MAP_INV = {value: key for key, value in SD15MAP.items()}
    LORA_KEYS = set(SD15MAP.values())


def make_lora_keys(lora_key: str, /) -> tuple[str, str, str]:
    return lora_key + ".lora_up.weight", lora_key + ".lora_down.weight", lora_key + ".alpha"


def add_lora_value_(lora: dict[str, Tensor], lora_key: str, U: Tensor, V: Tensor, alpha: Tensor, /) -> None:
    up_key, down_key, alpha_key = make_lora_keys(lora_key)

    lora[up_key] = U.to(device="cpu", dtype=torch.float16, non_blocking=True)
    lora[down_key] = V.to(device="cpu", dtype=torch.float16, non_blocking=True)
    lora[alpha_key] = alpha.to(device="cpu", dtype=torch.float16, non_blocking=True)


def get_lora_value(lora: dict[str, Tensor], lora_key: str, /, device: Device, dtype: torch.dtype) -> SVDBasis:
    up_key, down_key, alpha_key = make_lora_keys(lora_key)

    U = lora[up_key].to(device=device, dtype=dtype, non_blocking=True)
    V = lora[down_key].to(device=device, dtype=dtype, non_blocking=True)
    alpha = lora[alpha_key].to(device=device, dtype=dtype, non_blocking=True)

    return SVDBasis(U, V, alpha)


@torch.no_grad()
def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    /,
    *,
    save_path: str | Path,
    rank: int | float,
    conv_rank: Optional[int | float] = None,  # LoCon if != 0
    max_rank: int = 128,
    device: Device = "auto",
) -> None:
    conv_rank = rank if conv_rank is None else conv_rank

    assert rank > 0
    assert conv_rank > 0

    device, dtype = auto(device)

    base = load_ckpt(base_path)
    tuned = load_ckpt(tuned_path)

    out: dict[str, Tensor] = {}
    for sd_key, lora_key in tqdm(SD15MAP.items(), "Converting to LORA"):
        base_tensor = base[sd_key].to(device=device, dtype=dtype, non_blocking=True)
        tuned_tensor = tuned[sd_key].to(device=device, dtype=dtype, non_blocking=True)

        diff = tuned_tensor - base_tensor
        del base_tensor, tuned_tensor

        is_conv = diff.ndim == 4 and tuple(diff.shape[2:]) == (3, 3)

        # No convolution layers
        if is_conv and conv_rank == 0:
            continue

        general_rank = rank if not is_conv else conv_rank
        U, V, alpha = svd(diff, general_rank, max_rank)
        del diff

        add_lora_value_(out, lora_key, U, V, alpha)
        del U, V, alpha

    save_path = Path(save_path)
    if save_path.suffix != ".safetensors":
        save_path = save_path / (Path(tuned_path).name + "_from_" + Path(base_path).name + ".safetensors")

    save_ckpt(out, save_path)


@torch.no_grad()
def merge_lora(
    base_path: str | Path,
    lora_path: str | Path,
    /,
    *,
    save_path: str | Path,
    multiplier: float = 1,
    device: Device = "auto",
) -> None:
    device, dtype = auto(device)

    base = load_ckpt(base_path)
    lora = load_ckpt(lora_path)

    lora_base_keys = set(re.sub(r".(alpha|lora_(down|up).weight)", "", k) for k in lora.keys())

    assert len(lora_base_keys - LORA_KEYS) == 0

    for lora_key in tqdm(lora_base_keys, "Merging to LoRA"):
        sd_key = SD15MAP_INV[lora_key]

        U, V, alpha = get_lora_value(lora, lora_key, device, dtype)
        dW = to_weight(U, V, alpha, multiplier)
        del U, V, alpha

        W0 = base[sd_key].to(device=device, dtype=dtype, non_blocking=True)

        base[sd_key] = W0.add(dW).contiguous().to(device="cpu", dtype=torch.float16, non_blocking=True)
        del W0, dW

    save_path = Path(save_path)
    if save_path.is_dir():
        save_path = save_path / (Path(base_path).stem + "_add_" + Path(lora_path).name + f"({multiplier})")

    save_ckpt(base, save_path)


@torch.no_grad()
def resize_lora(
    lora_path: str | Path,
    /,
    save_path: str | Path,
    rank: int | float,
    conv_rank: Optional[int | float] = None,  # LoCon if != 0
    max_rank: int = 128,
    device: Device = "auto",
):
    conv_rank = rank if conv_rank is None else conv_rank

    assert rank > 0
    assert conv_rank > 0

    device, dtype = auto(device)

    lora = load_ckpt(lora_path)

    for lora_key in tqdm(LORA_KEYS, "Resizing LoRA"):
        U, V, alpha = get_lora_value(lora, lora_key, device, dtype)

        dW = to_weight(U, V, alpha)
        del U, V, alpha

        is_conv = dW.ndim == 4 and tuple(dW.shape[2:]) == (3, 3)

        general_rank = rank if not is_conv else conv_rank
        U, V, alpha = svd(dW, general_rank, max_rank)
        del dW

        add_lora_value_(lora, lora_key, U, V, alpha)
        del U, V, alpha

    # Save the updated LoRA model
    save_path = Path(save_path)
    if save_path.suffix != ".safetensors":
        save_path = save_path / (Path(lora_path).stem + f"_{rank}_{conv_rank}.safetensors")

    save_ckpt(lora, save_path)

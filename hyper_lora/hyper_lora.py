from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm, trange

import re
import json
from pathlib import Path

import torch
from torch import Tensor
from safetensors.torch import save_file

from .utils import save_ckpt, auto, Device
from .svd import svd, to_weight, SVDBasis
from .civitai import load_ckpt
from .lora import get_lora_value, add_lora_value_, LORA_KEYS


def stack_layer_weights(
    loras: list[dict[str, Tensor]],
    lora_key: str,
    /,
    *,
    device: Device = "auto",
) -> tuple[Tensor, int]:
    device, dtype = auto(device)

    max_rank = 1
    dWs = []
    for lora in loras:
        # TODO may fail if lora does not have a key. In this case, give a mask and fill it with zeros dW

        U, V, alpha = get_lora_value(lora, lora_key, device, dtype)
        max_rank = max(max_rank, U.size(1))

        dW = to_weight(U, V, alpha)
        dWs.append(dW)

        del dW, U, V, alpha

    dW = torch.stack(dWs).contiguous()
    del dWs

    return dW, max_rank


@torch.no_grad()
def hyper_lora(
    lora_paths: list[str | Path] | list[str] | list[Path],
    /,
    *,
    save_path: str | Path,
    device: Device = "auto",
    steps: int = 1,
) -> Tensor:
    device, dtype = auto(device)

    loras = [load_ckpt(path) for path in tqdm(lora_paths, desc="Loading LoRAs")]

    # initialize super-lora with mean of loras and unit scale
    hyper_lora: dict[str, Tensor] = {}
    for lora_key in tqdm(LORA_KEYS, desc="Initializing hyper-LoRA"):
        dWs, rank = stack_layer_weights(loras, lora_key, device=device)
        dW = dWs.mean(0)
        del dWs

        U, V, alpha = svd(dW, rank)
        del dW

        add_lora_value_(hyper_lora, lora_key, U, V, alpha)
        del U, V, alpha
    scales = torch.ones(len(loras), device=device, dtype=dtype)  # not used, but for completness

    for step in trange(steps, desc="Iteractive part"):
        # Compute new scales
        num, den = torch.zeros(len(loras), device=device), torch.zeros(1, device=device)
        for lora_key in tqdm(LORA_KEYS, desc="Computing new scale", leave=step == steps - 1):
            W, rank = stack_layer_weights([hyper_lora], lora_key, device=device)
            dWs, rank = stack_layer_weights(loras, lora_key, device=device)

            num += dWs.mul(W).flatten(1).sum(1).float()
            den += W.square().flatten(1).sum(1).float()
            del W, dWs
        scales = num.div(den).to(dtype=dtype)
        del num, den

        # compute new weights
        for lora_key in tqdm(LORA_KEYS, desc="Computing new weights", leave=step == steps - 1):
            dWs, rank = stack_layer_weights(loras, lora_key, device=device)

            dim = dWs.ndim - 1
            s = scales.view(-1, *([1] * dim))

            W = dWs.mul(s).sum(0) / scales.square().sum()
            del dWs

            U, V, alpha = svd(W, rank)
            del W

            add_lora_value_(hyper_lora, lora_key, U, V, alpha)
            del U, V, alpha

    save_path = Path(save_path)
    if save_path.suffix != ".safetensors":
        save_path = save_path / "hyper-lora.safetensors"

    save_ckpt(hyper_lora, save_path)

    return scales.cpu()

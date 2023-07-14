from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm, trange

import re
import json
from pathlib import Path

import torch
from torch import Tensor
from safetensors.torch import save_file

from .utils import auto, Device
from .svd import svd, to_weight
from .civitai import load_ckpt

# Read LoRA mapping for SD 1.5
with open("sd_1.5-lora_mapping.json", "r") as f:
    SD15MAP: dict[str, str] = json.load(f)
    SD15MAP_INV = {value: key for key, value in SD15MAP.items()}
    LORA_KEYS = set(SD15MAP.values())


def make_lora_keys(lora_key: str, /) -> tuple[str, str, str]:
    return lora_key + ".lora_up.weight", lora_key + ".lora_down.weight", lora_key + ".alpha"


def _save(obj: dict[str, Tensor], path: Path, /) -> None:
    if path.exists():
        path.unlink()

    assert path.suffix == ".safetensors"

    path.parent.mkdir(exist_ok=True, parents=True)
    save_file(obj, path)


@torch.no_grad()
def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    /,
    *,
    save_path: str | Path,
    rank: int | float,
    conv_rank: Optional[int | float] = None,
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

        up_key, down_key, alpha_key = make_lora_keys(lora_key)

        out[up_key] = U.to(device="cpu", dtype=torch.float16, non_blocking=True)
        out[down_key] = V.to(device="cpu", dtype=torch.float16, non_blocking=True)
        out[alpha_key] = alpha.to(device="cpu", dtype=torch.float16, non_blocking=True)

        del U, V, alpha

    save_path = Path(save_path)
    if save_path.suffix != ".safetensors":
        save_path = save_path / (Path(tuned_path).name + "_from_" + Path(base_path).stem)

    _save(out, save_path)


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

        W0 = base[sd_key].to(device=device, dtype=dtype, non_blocking=True)

        up_key, down_key, alpha_key = make_lora_keys(lora_key)

        U = lora[up_key].to(device=device, dtype=dtype, non_blocking=True)
        V = lora[down_key].to(device=device, dtype=dtype, non_blocking=True)
        alpha = lora[alpha_key].to(device=device, dtype=dtype, non_blocking=True)

        dW = to_weight(U, V, alpha, multiplier)

        base[sd_key] = W0.add(dW).contiguous().to(device="cpu", dtype=torch.float16, non_blocking=True)

    save_path = Path(save_path)
    if save_path.is_dir():
        save_path = save_path / (Path(base_path).stem + "_add_" + Path(lora_path).name + f"({multiplier})")

    _save(base, save_path)


def stack_layer_weights(
    loras: list[dict[str, Tensor]],
    lora_key: str,
    /,
    *,
    device: Device = "auto",
) -> tuple[Tensor, int]:
    device, dtype = auto(device)

    up_key, down_key, alpha_key = make_lora_keys(lora_key)

    max_rank = 1
    dWs = []
    for lora in loras:
        U = lora[up_key].to(device=device, dtype=dtype, non_blocking=True)
        V = lora[down_key].to(device=device, dtype=dtype, non_blocking=True)
        alpha = lora[alpha_key].to(device=device, dtype=dtype, non_blocking=True)

        max_rank = max(max_rank, U.size(1))

        dW = to_weight(U, V, alpha)
        dWs.append(dW)

        del dW, U, V, alpha

    dW = torch.stack(dWs).contiguous()
    del dWs

    return dW, max_rank


def main_hyper_lora(
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
        up_key, down_key, alpha_key = make_lora_keys(lora_key)

        dWs, rank = stack_layer_weights(loras, lora_key, device=device)
        dW = dWs.mean(0)
        del dWs

        U, V, alpha = svd(dW, rank)
        del dW

        hyper_lora[up_key] = U.to(device="cpu", dtype=torch.float16, non_blocking=True)
        hyper_lora[down_key] = V.to(device="cpu", dtype=torch.float16, non_blocking=True)
        hyper_lora[alpha_key] = alpha.to(device="cpu", dtype=torch.float16, non_blocking=True)

        del U, V, alpha
    scales = torch.ones(len(loras), device=device, dtype=dtype)  # not used, but for completness

    for step in trange(steps, desc="Iteractive part"):
        # Compute new scales
        num, den = torch.zeros(len(loras), device=device), torch.zeros(1, device=device)
        for lora_key in tqdm(LORA_KEYS, desc="Computing new scale", leave=step == steps - 1):
            up_key, down_key, alpha_key = make_lora_keys(lora_key)

            W, rank = stack_layer_weights([hyper_lora], lora_key, device=device)
            dWs, rank = stack_layer_weights(loras, lora_key, device=device)

            num += dWs.mul(W).flatten(1).sum(1).float()
            den += W.square().flatten(1).sum(1).float()
        scales = num.div(den).to(dtype=dtype)

        # compute new weights
        for lora_key in tqdm(LORA_KEYS, desc="Computing new weights", leave=step == steps - 1):
            up_key, down_key, alpha_key = make_lora_keys(lora_key)

            dWs, rank = stack_layer_weights(loras, lora_key, device=device)

            dim = dWs.ndim - 1
            s = scales.view(-1, *([1] * dim))

            W = dWs.mul(s).sum(0) / scales.square().sum()
            del dWs

            U, V, alpha = svd(W, rank)
            del W

            hyper_lora[up_key] = U.to(device="cpu", dtype=torch.float16, non_blocking=True)
            hyper_lora[down_key] = V.to(device="cpu", dtype=torch.float16, non_blocking=True)
            hyper_lora[alpha_key] = alpha.to(device="cpu", dtype=torch.float16, non_blocking=True)

            del U, V, alpha

    save_path = Path(save_path)
    if save_path.suffix != ".safetensors":
        save_path = save_path / "hyper-lora.safetensors"

    _save(hyper_lora, save_path)

    return scales.cpu()
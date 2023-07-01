from __future__ import annotations
from typing import Literal, Optional

from functools import cache
import logging
from tqdm.auto import tqdm

import json
from pathlib import Path
import requests

import math
import torch
from torch import Tensor

from safetensors.torch import load_file, save_file

@cache
def get_civitai_model_url(modelId: int | str) -> tuple[str, str]:
    url = f"https://civitai.com/api/v1/models/{modelId}"

    response = requests.get(url)
    if response.status_code == 200:
        obj = response.json()
        obj = obj["modelVersions"][0]["files"][0]

        return obj["name"], obj["downloadUrl"]  # type: ignore

    raise ValueError 


def download_ckpt(url: str, path: str | Path, /) -> None:
    path = Path(path)
    assert path.suffix == ".safetensors"

    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {url}") as pbar:
            for chunk in response.iter_content(chunk_size=8_192):
                pbar.update(len(chunk))
                if chunk:
                    f.write(chunk)


def load_ckpt(path: str | Path, /) -> dict[str, Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_file(path)

    logging.warning("Please use .safetensors!")

    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def clamp_rank(rank: int | float, N: int, /, min_dim: int, max_dim: int) -> int:
    if isinstance(rank, float):
        rank = math.ceil(rank * N)

    rank = min(max(rank, min_dim), max_dim)

    return min(max(1, rank), N)


def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    /,
    *,
    dim: int | float = 0.04,
    min_dim: int = 4,
    max_dim: int = 1_024,
    clamp_quantile: Optional[float] = 0.99,
    save_path: Optional[str | Path] = None,
    device: Literal["cuda", "cpu", "auto"] = "auto",
) -> dict[str, Tensor]:
    """
    Extract LoRA (Low-Rank Adaptation) parameters from provided base and tuned models.
    """

    with open("sd_1.5-lora_mapping.json", "r") as f:
        mapping = json.load(f)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = Path(save_path) if save_path else None

    base = load_ckpt(base_path)
    tuned = load_ckpt(tuned_path)

    out: dict[str, Tensor] = {}
    for key, lora_key in tqdm(mapping.items(), "Converting to LORA"):
        base_tensor = base[key].to(device=device, dtype=torch.float32, non_blocking=True)
        tuned_tensor = tuned[key].to(device=device, dtype=torch.float32, non_blocking=True)

        diff = tuned_tensor - base_tensor

        shape = tuple(diff.shape)
        is_conv = diff.ndim == 4
        out_dim, in_dim, *kernel_size = shape

        diff = diff.flatten(1)
        N = diff.size(1)

        rank = clamp_rank(dim, N, min_dim, max_dim)
        rank = min(rank, out_dim, N)

        U, S, Vh = torch.svd_lowrank(diff, q=rank)
        U = U @ torch.diag(S)

        if clamp_quantile:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            value = torch.quantile(dist, clamp_quantile)

            U = U.clamp(-value, value)
            Vh = Vh.clamp(-value, value)

        U = U.half().cpu().contiguous()
        Vh = Vh.T.half().cpu().contiguous()

        if is_conv:
            U = U.view(out_dim, rank, 1, 1).contiguous()
            Vh = Vh.view(rank, in_dim, *kernel_size).contiguous()

        out[lora_key + ".lora_up.weight"] = U
        out[lora_key + ".lora_down.weight"] = Vh
        out[lora_key + ".alpha"] = torch.tensor(Vh.size(0)).half()

    if save_path is not None:
        if save_path.is_dir():
            save_path = save_path / Path(tuned_path).name

        if save_path.exists():
            save_path.unlink()

        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_file(out, save_path)

    return out

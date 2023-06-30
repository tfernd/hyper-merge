from typing import Optional

import logging

from tqdm.auto import tqdm

import json
from pathlib import Path
import requests
import hashlib

import math
import torch
from torch import Tensor

from safetensors.torch import load_file, save_file


def load_ckpt(
    path: str | Path,
    /,
    *,
    cache_dir: Optional[str | Path] = None,
) -> dict[str, Tensor]:
    # Download url
    if isinstance(path, str) and path.startswith("http"):
        url = path

        hash_object = hashlib.md5(path.encode())
        hash_name = hash_object.hexdigest()

        cache_dir = Path(cache_dir) if cache_dir is not None else Path.home() / "sd-models-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        assert cache_dir.is_dir()
        path = cache_dir / f"{hash_name}.safetensors"

        if path.exists():
            # If partially downloaded of failed.
            try:
                return load_ckpt(path)
            except:
                pass

        response = requests.get(url, stream=True)

        with open(path, "wb") as f:
            total_size = int(response.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {url}") as pbar:
                for chunk in response.iter_content(chunk_size=8_192):
                    pbar.update(len(chunk))
                    if chunk:
                        f.write(chunk)
        return load_ckpt(path)

    path = Path(path)
    if path.suffix == ".safetensors":
        return load_file(path)

    logging.warning("Please use .safetensors!")

    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def clamp_rank(rank: int | float, /) -> int | float:
    if isinstance(rank, float):
        return max(0, min(1, rank))

    return max(1, rank)


def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    /,
    *,
    dim: int | float,
    conv_dim: Optional[int | float] = None,
    max_dim: Optional[int] = None,
    clamp_quantile: Optional[float] = 0.99,
    device: str = "cpu",
    save_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
) -> dict[str, Tensor]:
    """
    Extract LoRA (Low-Rank Adaptation) parameters from provided base and tuned models.
    """

    save_path = Path(save_path) if save_path else None

    dim = clamp_rank(dim)
    conv_dim = clamp_rank(conv_dim or dim)

    with open("sd_1.5-lora_mapping.json", "r") as f:
        mapping = json.load(f)

    base = load_ckpt(base_path, cache_dir=cache_dir)
    tuned = load_ckpt(tuned_path, cache_dir=cache_dir)

    out: dict[str, Tensor] = {}
    for key, lora_key in tqdm(mapping.items(), "Converting to LORA"):
        base_tensor = base[key].to(device=device, dtype=torch.float32, non_blocking=True)
        tuned_tensor = tuned[key].to(device=device, dtype=torch.float32, non_blocking=True)

        diff = tuned_tensor - base_tensor

        shape = tuple(diff.shape)
        ndim = diff.ndim
        is_conv = ndim == 4
        out_dim, in_dim, *kernel_size = shape

        diff = diff.flatten(1)
        N = diff.size(1)

        if is_conv and tuple(kernel_size) != (1, 1):
            rank = min(conv_dim, N) if isinstance(conv_dim, int) else math.ceil(conv_dim * N)
            rank = min(rank, in_dim, out_dim)
        else:
            rank = min(dim, N) if isinstance(dim, int) else math.ceil(dim * N)

        if max_dim is not None:
            rank = min(rank, max_dim)

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
            save_path = save_path / tuned_path.name

        if save_path.exists():
            save_path.unlink()

        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_file(out, save_path)

    return out

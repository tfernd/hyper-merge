from __future__ import annotations
from typing import Literal, Optional

from functools import cache
import logging
from tqdm.auto import tqdm

import re
import json
from pathlib import Path
import requests

import math
import torch
from torch import Tensor

from safetensors.torch import load_file, save_file

with open("sd_1.5-lora_mapping.json", "r") as f:
    SD15MAP: dict[str, str] = json.load(f)
    SD15MAP_INV = {value: key for key, value in SD15MAP.items()}


@cache
def get_civitai_model_url(modelId: int | str) -> tuple[str, str]:
    """
    Get the download URL for a CivitAI model.

    Args:
        modelId (int | str): The ID of the model.

    Returns:
        tuple[str, str]: A tuple containing the name and download URL of the model.

    Raises:
        ValueError: If the request to the CivitAI API fails.
    """

    url = f"https://civitai.com/api/v1/models/{modelId}"

    response = requests.get(url)
    if response.status_code == 200:
        obj = response.json()

        files = obj['modelVersions'][0]["files"]
        safe_files = list(filter(lambda o: o['metadata']['format'] == 'SafeTensor', files))
        file = safe_files[0] if len(safe_files) >= 1 else files[0]

        return file["name"], file["downloadUrl"]  # type: ignore

    raise ValueError  # TODO better message


def download_ckpt(url: str, path: str | Path, /) -> Path:
    """
    Download a checkpoint file from a given URL and save it to the specified path.

    Args:
        url (str): The URL of the checkpoint file.
        path (str | Path): The path to save the downloaded file.

    """

    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {url}") as pbar:
            for chunk in response.iter_content(chunk_size=8_192):
                pbar.update(len(chunk))
                if chunk:
                    f.write(chunk)
    return path


def load_ckpt(path: str | Path, /) -> dict[str, Tensor]:
    """
    Load a checkpoint file.

    Args:
        path (str | Path): The path to the checkpoint file.

    Returns:
        dict[str, Tensor]: The loaded checkpoint data.

    Warnings:
        If the file extension is not '.safetensors', a warning is logged suggesting the use of '.safetensors' files.
    """

    path = Path(path)
    if path.suffix == ".safetensors":
        return load_file(path)

    logging.warning("Please use .safetensors!")

    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def clamp_rank(rank: int | float, N: int, /, min_dim: int, max_dim: int) -> int:
    """
    Clamp the rank value within the specified bounds.

    Args:
        rank (int | float): The rank value. It can be either an integer or a percentage of the total number of elements.
        N (int): The total number of elements.
        min_dim (int): The minimum allowed rank value.
        max_dim (int): The maximum allowed rank value.

    Returns:
        int: The clamped rank value.
    """

    if isinstance(rank, float):
        rank = math.ceil(rank * N)

    rank = min(max(rank, min_dim), max_dim)

    return min(max(1, rank), N)


def auto_device(device: Literal["cuda", "cpu", "auto"], /) -> Literal["cuda", "cpu"]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    /,
    *,
    dim: int | float = 0.04,
    min_dim: int = 4,
    max_dim: int = 1_024,
    clamp_quantile: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    device: Literal["cuda", "cpu", "auto"] = "auto",
) -> dict[str, Tensor]:
    """
    Extract LoRA (Low-Rank Adaptation) parameters from the provided base and tuned models.

    Args:
        base_path (str | Path): The path to the base model checkpoint file.
        tuned_path (str | Path): The path to the tuned model checkpoint file.
        dim (int | float, optional): The desired rank. It can be either an integer or a percentage of the total number of elements.
        min_dim (int, optional): The minimum allowed rank value.
        max_dim (int, optional): The maximum allowed rank value.
        clamp_quantile (float, optional): The quantile value used for clamping.
        save_path (str | Path, optional): The path to save the extracted parameters.
        device (Literal["cuda", "cpu", "auto"], optional): The device to use for computation.

    Returns:
        dict[str, Tensor]: A dictionary containing the extracted LoRA parameters.

    Warnings:
        If the specified device is "auto" and CUDA is available, the function will use CUDA. If CUDA is not available,
        the function will fall back to CPU. If you want to explicitly specify the device, use "cuda" or "cpu" instead.
    """

    device = auto_device(device)

    save_path = Path(save_path) if save_path else None

    base = load_ckpt(base_path)
    tuned = load_ckpt(tuned_path)

    out: dict[str, Tensor] = {}
    for key, lora_key in tqdm(SD15MAP.items(), "Converting to LORA"):
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

        U = U.half().contiguous().cpu()
        Vh = Vh.T.half().contiguous().cpu()

        if is_conv:
            U = U.unflatten(1, (rank, 1, 1)).contiguous()
            Vh = Vh.unflatten(1, (in_dim, *kernel_size)).contiguous()

        out[lora_key + ".lora_up.weight"] = U
        out[lora_key + ".lora_down.weight"] = Vh
        out[lora_key + ".alpha"] = torch.tensor(rank).half()

    if save_path is not None:
        if not lora.is_file():
            save_path = save_path / (Path(base_path).stem + "_to_" + Path(tuned_path).name)

        if save_path.exists():
            save_path.unlink()

        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_file(out, save_path)

    return out


def merge_lora(
    base_path: str | Path,
    lora_path: str | Path,
    /,
    *,
    multiplier: float = 1,
    save_path: Optional[str | Path] = None,
    device: Literal["cuda", "cpu", "auto"] = "auto",
) -> dict[str, Tensor]:
    device = auto_device(device)

    save_path = Path(save_path) if save_path else None

    base = load_ckpt(base_path)
    lora = load_ckpt(lora_path)

    lora_base_keys = set(re.sub(r".(alpha|lora_(down|up).weight)", "", k) for k in lora.keys())

    assert len(lora_base_keys - set(SD15MAP.values())) == 0

    for key in list(base.keys()):
        if base[key].dtype == torch.float32:
            base[key] = base[key].half().contiguous()

    for lora_key in tqdm(lora_base_keys, "Mergin to LORA"):
        key = SD15MAP_INV[lora_key]

        out_dim, in_dim, *kernel_size = base[key].shape

        U = lora[lora_key + ".lora_up.weight"].to(device=device, dtype=torch.float32, non_blocking=True)
        Vh = lora[lora_key + ".lora_down.weight"].to(device=device, dtype=torch.float32, non_blocking=True)
        alpha = lora[lora_key + ".alpha"].to(device=device, dtype=torch.float32, non_blocking=True)

        rank = U.size(1)

        dW = multiplier * alpha / rank * (U.flatten(1) @ Vh.flatten(1))
        dW = dW.unflatten(1, (in_dim, *kernel_size))

        base[key] += dW.half().contiguous().cpu()

    if save_path is not None:
        if save_path.is_dir():
            save_path = save_path / (Path(base_path).stem + "_to_" + Path(lora_path).name)

        if save_path.exists():
            save_path.unlink()

        save_path.parent.mkdir(exist_ok=True, parents=True)
        save_file(base, save_path)

    return base

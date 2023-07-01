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
import torch.optim as optim
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

        files = obj["modelVersions"][0]["files"]
        safe_files = list(filter(lambda o: o["metadata"]["format"] == "SafeTensor", files))
        file = safe_files[0] if len(safe_files) >= 1 else files[0]

        return file["name"], file["downloadUrl"]  # type: ignore

    # TODO remove cache if error
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

    # TODO if any error, remove the file

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
    """
    Automatically determine the device (CPU or CUDA) to use for computation.

    Args:
        device (Literal["cuda", "cpu", "auto"]): The device argument.

    Returns:
        Literal["cuda", "cpu"]: The determined device.

    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def svd(
    dW: Tensor,
    rank: int,
    /,
    shape: tuple[int, int] | tuple[int, int, int, int],
) -> tuple[Tensor, Tensor]:
    """
    Perform singular value decomposition (SVD) on a matrix using low-rank approximation.

    Args:
        dW (Tensor): The input matrix for SVD.
        rank (int): The desired rank of the approximation.
        shape (tuple[int, int] | tuple[int, int, int, int]): The shape of the original matrix.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the U and Vh matrices obtained from the SVD.

    Notes:
        The function assumes that the input matrix `dW` is a 2-dimensional tensor.
        The SVD is performed using low-rank approximation, which reduces the matrix to the desired rank.
        The function returns a tuple (U, Vh) representing the matrices obtained from the SVD.
        The matrix `dW` can be approximately reconstructed using the formula dW = U @ Vh.
    """

    assert dW.ndim == 2

    U, S, Vh = torch.svd_lowrank(dW.float(), q=rank)
    U = U @ torch.diag(S)

    out_dim, in_dim = shape[:2]
    kernel_size = shape[2:]
    pre_kernel = (1, 1) if len(kernel_size) == 2 else ()

    U = U.unflatten(1, (rank, *pre_kernel)).half().contiguous()
    Vh = Vh.T.unflatten(1, (in_dim, *kernel_size)).half().contiguous()

    return U, Vh


def extract_lora(
    base_path: str | Path,
    tuned_path: str | Path,
    /,
    *,
    dim: int | float = 0.08,
    min_dim: int = 8,
    max_dim: int = 96,
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

    if isinstance(dim, int):
        min_dim = max_dim = dim

    device = auto_device(device)

    save_path = Path(save_path) if save_path else None

    base = load_ckpt(base_path)
    tuned = load_ckpt(tuned_path)

    out: dict[str, Tensor] = {}
    for key, lora_key in tqdm(SD15MAP.items(), "Converting to LORA"):
        base_tensor = base[key].to(device=device, dtype=torch.float32, non_blocking=True)
        tuned_tensor = tuned[key].to(device=device, dtype=torch.float32, non_blocking=True)

        diff = (tuned_tensor - base_tensor).flatten(1)

        shape = tuple(tuned_tensor.shape)
        out_dim, in_dim, *kernel_size = shape
        N = diff.size(1)
        rank = clamp_rank(dim, N, min_dim, min(max_dim, out_dim, N))

        U, Vh = svd(diff, rank, shape)

        out[lora_key + ".lora_up.weight"] = U.cpu()
        out[lora_key + ".lora_down.weight"] = Vh.cpu()
        out[lora_key + ".alpha"] = torch.tensor(rank).half()

    if save_path is not None:
        # Not a file
        if save_path.suffix != ".safetensors":
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
    """
    Merge LoRA (Low-Rank Adaptation) parameters into a base model.

    Args:
        base_path (str | Path): The path to the base model checkpoint file.
        lora_path (str | Path): The path to the LoRA (Low-Rank Adaptation) model checkpoint file.
        multiplier (float, optional): The multiplier applied to the LoRA parameters before merging. Defaults to 1.
        save_path (str | Path, optional): The path to save the merged model. If None, the merged model is not saved.
        device (Literal["cuda", "cpu", "auto"], optional): The device to use for computation. Defaults to "auto".

    Returns:
        dict[str, Tensor]: A dictionary containing the merged model parameters.

    Raises:
        AssertionError: If the LoRA model contains parameters that do not match the expected LoRA keys.

    Notes:
        - The function loads the base model and LoRA model checkpoints.
        - The LoRA model parameters are merged into the base model by adding the LoRA parameters to the corresponding base model parameters.
        - The multiplier can be used to control the magnitude of the LoRA parameters during the merge.
        - The resulting merged model parameters are returned as a dictionary.
        - If a save path is provided, the merged model is saved at the specified location.

    Warnings:
        - The function assumes that the base model and LoRA model checkpoints are in the same format (e.g., both .safetensors files).
        - If the specified device is "auto" and CUDA is available, the function will use CUDA. If CUDA is not available, it will fall back to CPU. To explicitly specify the device, use "cuda" or "cpu" instead.
    """

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


def directional(dWs: Tensor, /, steps: int, lr: float = 5e-3) -> tuple[Tensor, Tensor, float]:
    """
    Perform directional optimization to find the principal direction of a set of low-rank matrices.

    Args:
        dWs (Tensor): The input low-rank matrices.
        steps (int): The number of optimization steps to perform.
        lr (float, optional): The learning rate for the optimizer.

    Returns:
        tuple[Tensor, Tensor, float]: A tuple containing the principal direction λ, the corresponding vector v, and the final loss value.

    Notes:
        - The function assumes that the input dWs is a 3-dimensional tensor.
        - The principal direction is found by optimizing λ and v to minimize the mean squared error between dWs and v * λ.
        - The optimization is performed using the AdamW optimizer with the specified learning rate.
        - The function returns a tuple (λ, v, loss), where λ is a tensor of shape (n, 1, 1), v is a tensor of shape (1, m, n), and loss is a scalar representing the final loss value.
    """

    assert dWs.ndim == 3

    n = dWs.size(0)

    λ = dWs.new_full((n, 1, 1), 1 / n).requires_grad_()
    v = dWs.mean(0, keepdim=True).requires_grad_()

    optimizer = optim.AdamW([λ, v], lr=lr)

    loss_init = (dWs - v).square().mean(0).sum().item()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = (dWs - v * λ).square().mean(0).sum() / loss_init
        loss.backward()
        optimizer.step()

    v.detach_()
    λ.detach_()

    # re-scale
    v *= λ.sum()
    λ /= λ.sum()

    return λ, v, loss.item()  # type: ignore


@torch.no_grad()
def directional2(dWs: Tensor, /, steps: int = 2) -> tuple[Tensor, Tensor, float]:
    """
    Perform simplified directional optimization to find the principal direction of a set of low-rank matrices.

    Args:
        dWs (Tensor): The input low-rank matrices.
        steps (int, optional): The number of optimization steps to perform. Defaults to 2.

    Returns:
        tuple[Tensor, Tensor, float]: A tuple containing the principal direction λ, the corresponding vector v, and the final loss value.

    Notes:
        - The function assumes that the input dWs is a 3-dimensional tensor.
        - The principal direction is found by iteratively updating λ and v to minimize the mean squared error between dWs and v * λ.
        - The optimization is performed for the specified number of steps.
        - The function returns a tuple (λ, v, loss), where λ is a tensor of shape (n, 1, 1), v is a tensor of shape (1, m, n), and loss is a scalar representing the final loss value.
    """

    assert dWs.ndim == 3

    n = dWs.size(0)

    λ = dWs.new_full((n, 1, 1), 1)
    v = dWs.mean(0, keepdim=True)

    loss_init = (dWs - v).square().mean(0).sum().item()

    for _ in range(steps):
        λ = v.mul(dWs).sum(dim=(1, 2), keepdim=True) / v.square().sum(dim=(1, 2), keepdim=True)
        v = λ.mul(dWs).sum(dim=(0), keepdim=True) / λ.square().sum(dim=(0), keepdim=True)

        loss = (dWs - v * λ).square().mean(0).sum().item() / loss_init

    return λ, v, loss  # type: ignore


def ortho_lora(
    loras_path: list[str | Path],
    /,
    *,
    save_path: Optional[str | Path] = None,
    device: Literal["cuda", "cpu", "auto"] = "auto",
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Perform orthogonalization of LoRA (Low-Rank Adaptation) parameters.

    Args:
        loras_path (list[str | Path]): The paths to the LoRA model checkpoint files.
        save_path (str | Path, optional): The path to save the orthogonalized LoRA parameters. If None, the parameters are not saved. Defaults to None.
        device (Literal["cuda", "cpu", "auto"], optional): The device to use for computation. Defaults to "auto".

    Returns:
        tuple[dict[str, Tensor], dict[str, Tensor]]: A tuple containing two dictionaries representing the orthogonalized LoRA parameters.
            - The first dictionary contains the parameters for the principal direction.
            - The second dictionary contains the parameters for the secondary direction.

    Notes:
        - The function performs orthogonalization of the LoRA parameters by finding the principal and secondary directions for each parameter.
        - The orthogonalization is performed by iterating over the LoRA model checkpoint files.
        - The resulting orthogonalized parameters are returned as two dictionaries, one for the principal direction and one for the secondary direction.
        - If a save path is provided, the orthogonalized parameters are saved at the specified location.

    Warnings:
        - The function assumes that the LoRA model checkpoint files are in the same format (e.g., .safetensors files).
        - If the specified device is "auto" and CUDA is available, the function will use CUDA. If CUDA is not available, it will fall back to CPU. To explicitly specify the device, use "cuda" or "cpu" instead.
    """

    device = auto_device(device)
    save_path = Path(save_path) if save_path else None

    loras = [load_file(path) for path in tqdm(loras_path, desc="Loading LoRAs weights.")]

    lora1: dict[str, Tensor] = {}
    lora2: dict[str, Tensor] = {}
    with tqdm(SD15MAP.items(), desc="Orthogonalizing") as pbar:
        for _, lora_key in pbar:
            Us, Vhs, alphas = [], [], []
            for lora in loras:
                Us.append(lora[lora_key + ".lora_up.weight"].to(device=device, dtype=torch.float32, non_blocking=True))
                Vhs.append(lora[lora_key + ".lora_down.weight"].to(device=device, dtype=torch.float32, non_blocking=True))

                # TODO re-scale using the alpha

                alphas.append(lora[lora_key + ".alpha"])

            Us = torch.stack(Us)
            Vhs = torch.stack(Vhs)
            alphas = torch.stack(alphas)

            assert alphas.std() == 0

            shape = (Us.size(1), Vhs.size(2), *Vhs.shape[3:])
            rank = Us.size(2)

            dWs = Us.flatten(2) @ Vhs.flatten(2)

            # Find principal and secondary directions
            λ1, v1, loss1 = directional2(dWs)
            λ2, v2, loss2 = directional2(dWs - λ1 * v1)

            pbar.set_postfix(loss1=loss1, loss2=loss2)

            U1, Vh1 = svd(v1.squeeze(0), rank, shape)
            U2, Vh2 = svd(v2.squeeze(0), rank, shape)

            # TODO recalculate new better alpha

            lora1[lora_key + ".lora_up.weight"] = U1.cpu()
            lora1[lora_key + ".lora_down.weight"] = Vh1.cpu()
            lora1[lora_key + ".alpha"] = alphas.mean()

            lora2[lora_key + ".lora_up.weight"] = U2.cpu()
            lora2[lora_key + ".lora_down.weight"] = Vh2.cpu()
            lora2[lora_key + ".alpha"] = alphas.mean()

            del Us, Vhs, alphas, dWs, U1, Vh1, U2, Vh2, v1, v2, λ1, λ2

    if save_path is not None:
        # Not a file
        if save_path.suffix != ".safetensors":
            lora1_path = save_path / "lora_principal.safetensors"
            lora2_path = save_path / "lora_secondary.safetensors"
        else:
            lora1_path = save_path.parent / f"{save_path.stem}_principal.safetensors"
            lora2_path = save_path.parent / f"{save_path.stem}_secondary.safetensors"

        save_path.parent.mkdir(exist_ok=True, parents=True)

        save_file(lora1, lora1_path)
        save_file(lora2, lora2_path)

    return lora1, lora2

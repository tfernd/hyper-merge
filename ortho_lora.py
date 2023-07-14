from __future__ import annotations
from typing import Literal, NamedTuple, Optional, Sequence

import logging
from tqdm.auto import tqdm

import re
import json
from pathlib import Path
import requests

import math
import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
from safetensors.torch import load_file, save_file


# def directional(dWs: Tensor, /, steps: int, lr: float = 5e-3) -> tuple[Tensor, Tensor, float]:
#     """
#     Perform directional optimization to find the principal direction of a set of low-rank matrices.

#     Args:
#         dWs (Tensor): The input low-rank matrices.
#         steps (int): The number of optimization steps to perform.
#         lr (float, optional): The learning rate for the optimizer.

#     Returns:
#         tuple[Tensor, Tensor, float]: A tuple containing the principal direction λ, the corresponding vector v, and the final loss value.

#     Notes:
#         - The function assumes that the input dWs is a 3-dimensional tensor.
#         - The principal direction is found by optimizing λ and v to minimize the mean squared error between dWs and v * λ.
#         - The optimization is performed using the AdamW optimizer with the specified learning rate.
#         - The function returns a tuple (λ, v, loss), where λ is a tensor of shape (n, 1, 1), v is a tensor of shape (1, m, n), and loss is a scalar representing the final loss value.
#     """

#     assert dWs.ndim == 3

#     n = dWs.size(0)

#     λ = dWs.new_full((n, 1, 1), 1 / n).requires_grad_()
#     v = dWs.mean(0, keepdim=True).requires_grad_()

#     optimizer = optim.AdamW([λ, v], lr=lr)

#     loss_init = (dWs - v).square().mean(0).sum().item()

#     for _ in range(steps):
#         optimizer.zero_grad()
#         loss = (dWs - v * λ).square().mean(0).sum() / loss_init
#         loss.backward()
#         optimizer.step()

#     v.detach_()
#     λ.detach_()

#     # re-scale
#     v *= λ.sum()
#     λ /= λ.sum()

#     return λ, v, loss.item()  # type: ignore


# @torch.no_grad()
# def directional_approx(dWs: Tensor, /, steps: int = 2) -> tuple[Tensor, Tensor, float]:
#     """
#     Perform simplified directional optimization to find the principal direction of a set of low-rank matrices.

#     Args:
#         dWs (Tensor): The input low-rank matrices.
#         steps (int, optional): The number of optimization steps to perform. Defaults to 2.

#     Returns:
#         tuple[Tensor, Tensor, float]: A tuple containing the principal direction λ, the corresponding vector v, and the final loss value.

#     Notes:
#         - The function assumes that the input dWs is a 3-dimensional tensor.
#         - The principal direction is found by iteratively updating λ and v to minimize the mean squared error between dWs and v * λ.
#         - The optimization is performed for the specified number of steps.
#         - The function returns a tuple (λ, v, loss), where λ is a tensor of shape (n, 1, 1), v is a tensor of shape (1, m, n), and loss is a scalar representing the final loss value.
#     """

#     assert dWs.ndim == 3

#     n = dWs.size(0)

#     λ = dWs.new_full((n, 1, 1), 1)
#     v = dWs.mean(0, keepdim=True)

#     loss_init = (dWs - v).square().mean(0).sum().item()

#     for _ in range(steps):
#         λ = v.mul(dWs).sum(dim=(1, 2), keepdim=True) / v.square().sum(dim=(1, 2), keepdim=True)
#         v = λ.mul(dWs).sum(dim=(0), keepdim=True) / λ.square().sum(dim=(0), keepdim=True)

#         loss = (dWs - v * λ).square().mean(0).sum().item() / loss_init

#     return λ, v, loss  # type: ignore


# def ortho_lora(
#     loras_path: list[str | Path],
#     /,
#     *,
#     dim: int | float = 0.08,
#     min_dim: int = 8,
#     max_dim: int = 96,
#     save_path: Optional[str | Path] = None,
#     device: Literal["cuda", "cpu", "auto"] = "auto",
# ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
#     """
#     Perform orthogonalization of LoRA (Low-Rank Adaptation) parameters.

#     Args:
#         loras_path (list[str | Path]): The paths to the LoRA model checkpoint files.
#         save_path (str | Path, optional): The path to save the orthogonalized LoRA parameters. If None, the parameters are not saved. Defaults to None.
#         device (Literal["cuda", "cpu", "auto"], optional): The device to use for computation. Defaults to "auto".

#     Returns:
#         tuple[dict[str, Tensor], dict[str, Tensor]]: A tuple containing two dictionaries representing the orthogonalized LoRA parameters.
#             - The first dictionary contains the parameters for the principal direction.
#             - The second dictionary contains the parameters for the secondary direction.

#     Notes:
#         - The function performs orthogonalization of the LoRA parameters by finding the principal and secondary directions for each parameter.
#         - The orthogonalization is performed by iterating over the LoRA model checkpoint files.
#         - The resulting orthogonalized parameters are returned as two dictionaries, one for the principal direction and one for the secondary direction.
#         - If a save path is provided, the orthogonalized parameters are saved at the specified location.

#     Warnings:
#         - The function assumes that the LoRA model checkpoint files are in the same format (e.g., .safetensors files).
#         - If the specified device is "auto" and CUDA is available, the function will use CUDA. If CUDA is not available, it will fall back to CPU. To explicitly specify the device, use "cuda" or "cpu" instead.
#     """

#     if isinstance(dim, int):
#         min_dim = max_dim = dim

#     device = auto_device(device)
#     save_path = Path(save_path) if save_path else None

#     loras = [load_file(path) for path in tqdm(loras_path, desc="Loading LoRAs weights.")]

#     lora1: dict[str, Tensor] = {}
#     lora2: dict[str, Tensor] = {}
#     with tqdm(SD15MAP.items(), desc="Orthogonalizing") as pbar:
#         for _, lora_key in pbar:
#             dWs = []
#             for lora in loras:
#                 up_key, down_key, alpha_key = lora_key + ".lora_up.weight", lora_key + ".lora_down.weight", lora_key + ".alpha"

#                 if up_key not in lora or down_key not in lora or alpha_key not in lora:
#                     continue

#                 U = lora[up_key].to(device=device, dtype=torch.float32, non_blocking=True)
#                 Vh = lora[down_key].to(device=device, dtype=torch.float32, non_blocking=True)
#                 alpha = lora[alpha_key]

#                 rank = U.size(1)
#                 shape = (U.size(0), Vh.size(1), *Vh.shape[2:])
#                 dW = U.flatten(1) @ Vh.flatten(1)
#                 dW *= alpha / rank  # re-scale

#                 dW = dW.view(*shape)
#                 dWs.append(dW)

#             if len(dWs) == 0:
#                 continue

#             dWs = torch.stack(dWs)
#             n = dWs.size(0)
#             shape = dWs.shape[1:]
#             out_dim, in_dim, *kernel_size = shape

#             if n == 1:
#                 lora1[lora_key + ".lora_up.weight"] = U.cpu()  # type: ignore
#                 lora1[lora_key + ".lora_down.weight"] = Vh.cpu()  # type: ignore
#                 lora1[lora_key + ".alpha"] = alpha  # type: ignore
#                 continue

#             # Find principal and secondary directions
#             dWs = dWs.flatten(2)
#             λ1, v1, loss1 = directional_approx(dWs)
#             λ2, v2, loss2 = directional_approx(dWs - λ1 * v1)

#             pbar.set_postfix(loss1=loss1, loss2=loss2)

#             N = dWs.size(2)
#             rank = clamp_rank(dim, N, min_dim, min(max_dim, out_dim, N))

#             U1, Vh1 = svd(v1.squeeze(0), rank, shape)
#             U2, Vh2 = svd(v2.squeeze(0), rank, shape)
#             alpha = torch.tensor(rank).half()

#             lora1[lora_key + ".lora_up.weight"] = U1.cpu()
#             lora1[lora_key + ".lora_down.weight"] = Vh1.cpu()
#             lora1[lora_key + ".alpha"] = alpha

#             lora2[lora_key + ".lora_up.weight"] = U2.cpu()
#             lora2[lora_key + ".lora_down.weight"] = Vh2.cpu()
#             lora2[lora_key + ".alpha"] = alpha

#     if save_path is not None:
#         # Not a file
#         if save_path.suffix != ".safetensors":
#             lora1_path = save_path / "lora_principal.safetensors"
#             lora2_path = save_path / "lora_secondary.safetensors"
#         else:
#             lora1_path = save_path.parent / f"{save_path.stem}_principal.safetensors"
#             lora2_path = save_path.parent / f"{save_path.stem}_secondary.safetensors"

#         save_path.parent.mkdir(exist_ok=True, parents=True)

#         save_file(lora1, lora1_path)
#         save_file(lora2, lora2_path)

#     return lora1, lora2

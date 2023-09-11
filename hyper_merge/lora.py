from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm, trange

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import Checkpoint, Checkpoints, PathLike, PathsLike, SVDCheckpoint
from .constants import LORA_KEYS, LORA_MAPPING, CPU
from .utils import free_cuda
from .checkpoint import load_checkpoint_, load_checkpoints, create_average_checkpoint_
from .svd import svd_compress, svd_reconstruct


def extract_lora_(
    checkpoint: PathLike | Checkpoint,
    base_checkpoint: PathLike | Checkpoint,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    rank: int = 64,
) -> tuple[Checkpoint, float]:
    free_cuda()

    checkpoint = load_checkpoint_(checkpoint, dtype, device, keys=LORA_KEYS)
    base_checkpoint = load_checkpoint_(base_checkpoint, dtype, device, keys=LORA_KEYS)

    K = len(LORA_KEYS)

    loss = 0
    svd_checkpoint: SVDCheckpoint = {}
    for key in tqdm(LORA_KEYS, desc="Extracing LoRA"):
        lora_key = LORA_MAPPING[key]

        dW = checkpoint[key] - base_checkpoint[key]
        svd_checkpoint[lora_key] = svd_compress(dW, rank)

        dWr = svd_reconstruct(svd_checkpoint[lora_key])
        loss += (dW - dWr).square().float().sum().div(K).item()
        del dW, dWr
    free_cuda()

    return make_lora_checkpoint(svd_checkpoint), loss


def extract_hyper_lora_(
    checkpoints: PathsLike | Checkpoints,
    avg_checkpoint: PathLike | Checkpoint,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    rank: int = 64,
    iterations: int = 6,
) -> tuple[Checkpoint, Tensor, Tensor]:
    M = len(checkpoints)

    avg_checkpoint = load_checkpoint_(avg_checkpoint, dtype, device, keys=LORA_KEYS)
    checkpoints = load_checkpoints(checkpoints, dtype, CPU, keys=LORA_KEYS)

    # initialize scales and empty checkpoints
    λ = torch.ones(M, device=device, dtype=dtype)
    svd_checkpoint: SVDCheckpoint = {}  # GPU
    loss = torch.zeros(M, device=device)

    with trange(iterations, desc="Optimizing hyper-checkpoint") as pbar:
        for i in pbar:
            # Update differential weights
            free_cuda()
            λ2 = λ.square().mean()  # scalar
            for key in tqdm(LORA_KEYS, desc="Updating diff weights", leave=False):
                Wavg = avg_checkpoint[key][..., None]
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)

                dW = (λ / λ2).mul(Ws - Wavg).float().mean(-1).to(dtype)
                svd_checkpoint[key] = svd_compress(dW, rank)

                del Wavg, Ws, dW
            del λ2

            # ! This make it worst!
            # Update average weights
            # free_cuda()
            # for key in tqdm(LORA_KEYS, desc="Updating diff weights", leave=False):
            #     Ws = torch.stack([checkpoint[key] for checkpoint in checkpoints], dim=-1)
            #     dW = svd_reconstruct(svd_checkpoint[key])[..., None]

            #     Wavg = (Ws - λ * dW).div(M).float().mean(-1).to(dtype)

            #     avg_checkpoint[key] *= 0.9
            #     avg_checkpoint[key] += 0.1 * Wavg

            #     del Wavg, Ws, dW

            # Update scales
            free_cuda()
            num = torch.zeros(M, device=device)
            den = torch.zeros(1, device=device)
            for key in tqdm(LORA_KEYS, desc="Updating scales", leave=False):
                Wavg = avg_checkpoint[key][..., None]
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                dW = svd_reconstruct(svd_checkpoint[key])[..., None]

                num += dW.mul(Ws - Wavg).flatten(0, -2).float().mean(0)
                den += dW.square().float().mean()

                del Ws, Wavg, dW
            λ = num.div(den).to(dtype=dtype)
            del num, den
            if i < iterations - 1 and M > 1:
                λ /= λ.abs().amax()

            # compute loss
            free_cuda()
            loss = loss.zero_()
            for key in tqdm(LORA_KEYS, desc="Computing loss", leave=False):
                Wavg = avg_checkpoint[key][..., None]
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                dW = svd_reconstruct(svd_checkpoint[key])[..., None]

                loss += (Ws - (Wavg + λ * dW)).square().float().flatten(0, -2).sum(0)
            loss /= len(LORA_KEYS)
            pbar.set_postfix(loss=loss.tolist())

    return make_lora_checkpoint(svd_checkpoint), λ, loss


def make_lora_checkpoint(lora_uv: SVDCheckpoint, /) -> Checkpoint:
    """
    Create a LoRA-checkpoint based on the provided checkpoint and specified rank.

    This function takes an existing checkpoint, filters it to only include keys relevant to LoRA, and then performs SVD-based rank decomposition for each tensor. The rank-decomposed weights are then organized into a new checkpoint, following a specific naming convention for LoRA-compatible keys.

    The rank decomposition is specifically designed to match the architecture of the model to which the checkpoint belongs. The function reshapes weights accordingly, so that they can be used in both linear and convolutional layers.
    """

    lora: Checkpoint = {}
    for lora_key, ((U, V), shape, rank) in tqdm(lora_uv.items(), desc="Making LoRA"):
        up_key = lora_key + ".lora_up.weight"
        down_key = lora_key + ".lora_down.weight"
        alpha_key = lora_key + ".alpha"

        out_dim, in_dim = shape[:2]
        kernel_size = shape[2:]
        pre_kernel = (1, 1) if len(kernel_size) == 2 else ()

        # Reshape for linear or conv-layer
        U = U.unflatten(1, (rank, *pre_kernel)).contiguous().to(device="cpu", non_blocking=True)
        V = V.unflatten(1, (in_dim, *kernel_size)).contiguous().to(device="cpu", non_blocking=True)

        alpha = torch.tensor(rank, dtype=U.dtype)

        lora[up_key] = U
        lora[down_key] = V
        lora[alpha_key] = alpha

    return lora

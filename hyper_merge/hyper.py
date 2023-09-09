from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm, trange

import torch
from torch import Tensor

from .utils import free_cuda
from .checkpoint import Checkpoint, transfer_checkpoint_, transfer_checkpoints_, filter_checkpoint_, filter_checkpoints_
from .constants import LORA_KEYS, CPU
from .lora import svd, reconstruct_weights
from .types import SVDCheckpoint


def create_hyper_checkpoint(
    checkpoints: list[Checkpoint],
    average_checkpoint: Checkpoint,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    rank: int = 128,
    iterations: int = 8,
) -> tuple[SVDCheckpoint, Tensor]:
    """
    Optimize a 'hyper-checkpoint' using an iterative algorithm based on provided checkpoints and an average-checkpoint.

    The hyper-checkpoint serves as a representative model that minimizes the average squared difference with all the given checkpoints.
    Scales for each checkpoint and differential weights for each tensor key are computed iteratively to converge to an optimal hyper-checkpoint.

    The function performs two types of updates in each iteration:
    1. Updating differential weights between the average checkpoint and individual checkpoints.
    2. Updating scales for individual checkpoints.

    The process involves multiple tensor operations, SVD-based tensor compression.
    Loss is computed at the end of each iteration to gauge the quality of the hyper-checkpoint.
    """

    free_cuda()

    M = len(checkpoints)

    checkpoints = filter_checkpoints_(checkpoints, keys=LORA_KEYS)
    checkpoints = transfer_checkpoints_(checkpoints, dtype)

    average_checkpoint = filter_checkpoint_(average_checkpoint, keys=LORA_KEYS)
    average_checkpoint = transfer_checkpoint_(average_checkpoint, dtype, device)  # GPU

    # initialize scales and empty checkpoints
    λ = torch.ones(M, device=device, dtype=dtype)
    hyper_diff_uv: SVDCheckpoint = {}  # GPU

    with trange(iterations, desc="Optimizing hyper-checkpoint") as pbar:
        prev_loss = 1e4
        for i in pbar:
            # Update differential weights
            free_cuda()
            λ2 = λ.square().mean()  # scalar
            for key in tqdm(LORA_KEYS, desc="Updating diff weights", leave=False):
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                Wavg = average_checkpoint[key][..., None]

                dW = (λ / λ2).mul(Ws - Wavg).float().mean(-1).to(dtype)

                # Compress/decompress
                hyper_diff_uv[key] = svd(dW, rank)

                del Ws, Wavg, dW
            del λ2

            # Update scales
            free_cuda()
            num = torch.zeros(M, device=device)
            den = torch.zeros(1, device=device)
            for key in tqdm(LORA_KEYS, desc="Updating scales", leave=False):
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                Wavg = average_checkpoint[key][..., None]
                dW = reconstruct_weights(hyper_diff_uv[key])[..., None]

                num += dW.mul(Ws - Wavg).flatten(0, -2).float().mean(0)
                den += dW.square().float().mean()

                del Ws, Wavg, dW
            λ = num.div(den).to(dtype=dtype)
            del num, den
            if i < iterations - 1:
                λ /= λ.abs().amax()

            # compute loss
            free_cuda()
            loss = torch.zeros(1, device=device)
            for key in tqdm(LORA_KEYS, desc="Computing loss", leave=False):
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                Wavg = average_checkpoint[key][..., None]
                dW = reconstruct_weights(hyper_diff_uv[key])[..., None]

                loss += (Ws - (Wavg + λ * dW)).square().float().mean()
            pbar.set_postfix(loss=loss.item())

            # < 0.1% change # TODO add as a parameter
            if abs(loss - prev_loss) / prev_loss < 0.1 / 100:
                break
            prev_loss = loss

    return hyper_diff_uv, λ


# TODO This is slow!
def remove_direction(
    checkpoints: list[Checkpoint],
    diff_uv: SVDCheckpoint,
    λ: Tensor,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,  # ? not used?
) -> list[Checkpoint]:
    free_cuda()

    checkpoints = transfer_checkpoints_(checkpoints, dtype, CPU)

    for checkpoint, scale in tqdm(list(zip(checkpoints, λ)), desc="Removing directions"):
        free_cuda()

        for key, weights in checkpoint.items():
            weights.data -= reconstruct_weights(diff_uv[key]).mul(scale).cpu()
    free_cuda()

    return checkpoints

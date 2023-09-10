from __future__ import annotations

from tqdm.auto import tqdm

import torch
from torch import Tensor

from .checkpoint import Checkpoint
from .constants import LORA_MAPPING
from .types import SVDOutput, SVDCheckpoint


def svd(
    dW: Tensor,
    /,
    rank: int,
) -> SVDOutput:
    """
    Perform low-rank approximation on the given tensor using Singular Value Decomposition (SVD).

    The function flattens the tensor and then conducts a low-rank SVD to obtain the left singular vectors (U) and right singular vectors (V). The singular vectors are truncated based on the specified rank to generate a low-rank approximation of the original tensor. These are then reshaped back into the original dimensions of the tensor.
    """

    dtype = dW.dtype
    shape = dW.shape

    dW = dW.flatten(1)
    N, M = dW.shape

    rank = max(1, min(rank, N, M))
    # ! this makes the number of elements in the LoRA smaller, otherwise don't use LoRA...
    # ! To be checked a better upper bound
    # rank = min(rank, round(N*M/(N+M)))

    U, S, Vh = torch.svd_lowrank(dW.float(), q=rank)
    U = U @ torch.diag(S)

    U = U.contiguous().to(dtype)
    V = Vh.T.contiguous().to(dtype)

    return (U, V), tuple(shape), rank


def reconstruct_weights(uv: SVDOutput, /) -> Tensor:
    """
    Reconstruct the original tensor weights from a low-rank SVD decomposition.
    """

    (U, V), shape, rank = uv

    return (U @ V).view(*shape)


def make_lora_checkpoint(
    lora_uv: SVDCheckpoint,
    /
) -> Checkpoint:
    """
    Create a LoRA-checkpoint based on the provided checkpoint and specified rank.

    This function takes an existing checkpoint, filters it to only include keys relevant to LoRA, and then performs SVD-based rank decomposition for each tensor. The rank-decomposed weights are then organized into a new checkpoint, following a specific naming convention for LoRA-compatible keys.

    The rank decomposition is specifically designed to match the architecture of the model to which the checkpoint belongs. The function reshapes weights accordingly, so that they can be used in both linear and convolutional layers.
    """

    lora: Checkpoint = {}
    for key, ((U, V), shape, rank) in tqdm(lora_uv.items(), desc="Making LoRA"):
        lora_key = LORA_MAPPING[key] if key in LORA_MAPPING else key # TODO remove
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

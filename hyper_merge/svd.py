from __future__ import annotations

from tqdm.auto import tqdm

import torch
from torch import Tensor

from .checkpoint import Checkpoint
from .constants import LORA_MAPPING
from .types import SVDOutput, SVDCheckpoint


def svd_compress(
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

    U, S, Vh = torch.svd_lowrank(dW.float(), q=rank)
    U = U @ torch.diag(S)

    U = U.contiguous().to(dtype)
    V = Vh.T.contiguous().to(dtype)

    return (U, V), tuple(shape), rank


def svd_reconstruct(uv: SVDOutput, /) -> Tensor:
    """
    Reconstruct the original tensor weights from a low-rank SVD decomposition.
    """

    (U, V), shape, rank = uv

    return (U @ V).view(*shape)

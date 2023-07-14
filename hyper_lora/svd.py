from __future__ import annotations
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor


class SVDBasis(NamedTuple):
    U: Tensor
    V: Tensor
    alpha: Tensor


def svd(
    dW: Tensor,
    /,
    rank: int | float,
    max_rank: int = 2**32 - 1,
) -> SVDBasis:
    device = dW.device
    dtype = dW.dtype
    shape = dW.shape

    M = shape[0]
    N = int(np.prod(shape[1:]))

    if not isinstance(rank, int):
        assert 0 < rank < 1
        rank = round(rank * N)
    rank = max(1, min(rank, N, M, max_rank))

    U, S, Vh = torch.svd_lowrank(dW.flatten(1).float(), q=rank)
    U = U @ torch.diag(S)

    alpha = torch.tensor(rank, dtype=dtype, device=device)

    out_dim, in_dim = shape[:2]
    kernel_size = shape[2:]
    pre_kernel = (1, 1) if len(kernel_size) == 2 else ()

    # Reshape for linear or conv-layer
    U = U.unflatten(1, (rank, *pre_kernel)).contiguous().to(dtype)
    V = Vh.T.unflatten(1, (in_dim, *kernel_size)).contiguous().to(dtype)

    return SVDBasis(U, V, alpha)


def to_weight(U: Tensor, V: Tensor, alpha: Tensor, /, multiplier: float = 1) -> Tensor:
    rank = U.size(1)

    partial_shape = V.shape[1:]

    dW = multiplier * (alpha / rank) * (U.flatten(1) @ V.flatten(1))
    dW = dW.unflatten(1, partial_shape).contiguous()

    return dW

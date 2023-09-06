from __future__ import annotations
from typing import NamedTuple, Optional

from pathlib import Path
import json
from tqdm.auto import tqdm

import numpy as np
import torch
from torch import Tensor

with open(Path(__file__).parent / "sd_1.5-lora_mapping.json") as handle:
    LORA_KEYS: dict[str, str] = json.load(handle)


class SVDBasis(NamedTuple):
    U: Tensor
    V: Tensor
    alpha: Tensor


def svd(dW: Tensor, /, rank: int | float) -> SVDBasis:
    device = dW.device
    dtype = dW.dtype
    shape = dW.shape

    M = shape[0]
    N = int(np.prod(shape[1:]))

    if not isinstance(rank, int):
        assert 0 < rank < 1
        rank = round(rank * N)
    rank = max(1, min(rank, N, M))
    # rank = min(rank, math.ceil(M * N / (M * N)))  # ! rank bigger than this uses more elements!

    U, S, Vh = torch.svd_lowrank(dW.flatten(1).float(), q=rank)  # float for precision
    U = U @ torch.diag(S)

    alpha = torch.tensor(rank, dtype=dtype, device=device)

    out_dim, in_dim = shape[:2]  # TODO out_dim = M
    kernel_size = shape[2:]
    pre_kernel = (1, 1) if len(kernel_size) == 2 else ()

    # Reshape for linear or conv-layer
    U = U.unflatten(1, (rank, *pre_kernel)).to(dtype).contiguous()
    V = Vh.T.unflatten(1, (in_dim, *kernel_size)).to(dtype).contiguous()

    return SVDBasis(U, V, alpha)


def make_lora(
    model: dict[str, Tensor],
    /,
    rank: int | float,
    base_model: Optional[dict[str, Tensor]] = None,
) -> dict[str, Tensor]:
    lora: dict[str, Tensor] = {}
    for key, lora_key in tqdm(LORA_KEYS.items()):
        up_key, down_key, alpha_key = lora_key + ".lora_up.weight", lora_key + ".lora_down.weight", lora_key + ".alpha"

        W = model[key]
        W0 = base_model[key] if base_model is not None else 0
        dW = W - W0

        U, V, alpha = svd(dW, rank)

        lora[up_key] = U.cpu()
        lora[down_key] = V.cpu()
        lora[alpha_key] = alpha.cpu()

        del W, W0, dW, U, V, alpha

    return lora

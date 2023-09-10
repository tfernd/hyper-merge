from __future__ import annotations
from typing import Optional

from tqdm.auto import tqdm, trange
from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from .utils import free_cuda
from .constants import LORA_KEYS, LORA_MAPPING
from .types import Checkpoint, SVDCheckpoint
from .svd import svd, reconstruct_weights, make_lora_checkpoint
from .checkpoint import load_checkpoint, create_average_checkpoint, load_checkpoints

def extract_lora(
    path: str | Path,
    base_path: str | Path,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    rank: int = 64,
) -> tuple[Checkpoint, float]:
    
    checkpoint = load_checkpoint(path, dtype, device, keys=LORA_KEYS)
    base_checkpoint = load_checkpoint(base_path, dtype, device, keys=LORA_KEYS)

    loss = 0
    svd_checkpoint: SVDCheckpoint = {}
    for key in tqdm(LORA_KEYS, desc='Extracing LoRA'):
        lora_key = LORA_MAPPING[key]

        dW = checkpoint[key] - base_checkpoint[key]
        svd_checkpoint[lora_key]  = svd(dW, rank)
        dWr = reconstruct_weights(svd_checkpoint[lora_key])

        loss += F.mse_loss(dW, dWr).item()

        del dW, dWr
    
    return make_lora_checkpoint(svd_checkpoint), loss

def extract_lora_and_average_checkpoint(
    paths: list[str] | list[Path] | list[str|Path],
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    rank: int = 64,
    iterations: int = 6,
    # avg_model # TODO
) -> tuple[Checkpoint, tuple[Checkpoint, Tensor], float]:
    
    M = len(paths)

    avg_model = create_average_checkpoint(paths, dtype, device)
    checkpoints = load_checkpoints(paths, dtype, device, keys=LORA_KEYS)

    # initialize scales and empty checkpoints
    λ = torch.ones(M, device=device, dtype=dtype)
    svd_checkpoint: SVDCheckpoint = {}  # GPU
    loss = torch.zeros(1, device=device)

    with trange(iterations, desc="Optimizing hyper-checkpoint") as pbar:
        for i in pbar:
            # Update differential weights
            free_cuda()
            λ2 = λ.square().mean()  # scalar
            for key in tqdm(LORA_KEYS, desc="Updating diff weights", leave=False):
                Wavg = avg_model[key][..., None]
                Ws = torch.stack([checkpoint[key] for checkpoint in checkpoints], dim=-1)

                dW = (λ / λ2).mul(Ws - Wavg).float().mean(-1).to(dtype)
                svd_checkpoint[key] = svd(dW, rank)

                del Wavg, Ws, dW
            del λ2

            # Update scales
            free_cuda()
            num = torch.zeros(M, device=device)
            den = torch.zeros(1, device=device)
            for key in tqdm(LORA_KEYS, desc="Updating scales", leave=False):
                Wavg = avg_model[key][..., None]
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                dW = reconstruct_weights(svd_checkpoint[key])[..., None]

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
                Wavg = avg_model[key][..., None]
                Ws = torch.stack([checkpoint[key].to(device) for checkpoint in checkpoints], dim=-1)
                dW = reconstruct_weights(svd_checkpoint[key])[..., None]

                loss += (Ws - (Wavg + λ * dW)).square().float().mean()
            pbar.set_postfix(loss=loss.item())

    lora = make_lora_checkpoint(svd_checkpoint)

    return avg_model, (lora, λ), loss.item()
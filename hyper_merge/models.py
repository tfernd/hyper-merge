from __future__ import annotations

from tqdm.auto import tqdm, trange

import torch
from torch import Tensor

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def get_model_keys(models: list[dict[str, Tensor]] | dict[str, Tensor], /) -> list[str]:
    if isinstance(models, dict):
        models = [models]

    keys = set(models[0].keys())
    for model in models[1:]:
        keys &= set(model.keys())
        del model

    return sorted(keys)


def clone_model(
    model: dict[str, Tensor],
    /,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> dict[str, Tensor]:
    return {key: value.clone().to(dtype=dtype, device=device) for (key, value) in model.items()}


def stacked_layers_weights(
    models: list[dict[str, Tensor]],
    key: str,
    /,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> Tensor:
    return torch.stack([model[key].to(device=device, dtype=dtype) for model in models], dim=-1)


def make_model_average(
    models: list[dict[str, Tensor]],
    /,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Tensor]:
    keys = get_model_keys(models)

    avg_model: dict[str, Tensor] = {}
    for key in tqdm(keys, desc="Creating the average model"):
        Ws = stacked_layers_weights(models, key, device, dtype)

        avg_model[key] = Ws.mean(-1).cpu()
        del Ws

    return avg_model


def update_diff_weights(
    models: list[dict[str, Tensor]],
    avg_model: dict[str, Tensor], # TODO needed? can be derived from Ws.mean(-1)
    diff_model: dict[str, Tensor],
    λ: Tensor,
    /,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> dict[str, Tensor]:
    keys = get_model_keys(models)

    λ2 = λ.square().mean()
    for key in tqdm(keys, desc="Updating weights", leave=False):
        Ws = stacked_layers_weights(models, key, device, dtype)
        Wavg = avg_model[key][..., None]

        diff_model[key] = λ.mul(Ws - Wavg).mean(-1) / λ2

        del Ws, Wavg

    return diff_model


def update_λ(
    models: list[dict[str, Tensor]],
    avg_model: dict[str, Tensor],# TODO needed? can be derived from Ws.mean(-1)
    diff_model: dict[str, Tensor],
    /,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    *,
    normalize: bool = False,
) -> Tensor:
    M = len(models)
    keys = get_model_keys(models)

    # Use float for extra-precision
    num = torch.zeros(M, device=device, dtype=torch.float32)
    den = torch.zeros(1, device=device, dtype=torch.float32)

    for key in tqdm(keys, desc="Updating scales", leave=False):
        Ws = stacked_layers_weights(models, key, device, dtype)
        Wavg = avg_model[key][..., None]
        dW = diff_model[key][..., None]

        num += dW.mul(Ws - Wavg).float().flatten(0, -2).mean(0)
        den += dW.square().float().mean()

        del Ws, Wavg, dW

    λ = num.div(den)
    if normalize:
        λ /= λ.abs().amax()
    del num, den

    return λ.to(dtype=dtype)


def compute_loss(
    models: list[dict[str, Tensor]],
    avg_model: dict[str, Tensor],# TODO needed? can be derived from Ws.mean(-1)
    diff_model: dict[str, Tensor],
    λ: Tensor | float, # float = special case
    /,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> float:
    keys = get_model_keys(models)

    loss = torch.zeros(1, device=device, dtype=torch.float32)
    for key in tqdm(keys, desc="Calculating loss", leave=False):
        Ws = stacked_layers_weights(models, key, device, dtype)
        Wavg = avg_model[key][..., None]
        dW = diff_model[key][..., None]

        loss += (Ws - (Wavg + λ * dW)).square().float().mean()
        del Ws

    return loss.item()


def make_diff_model(
    models: list[dict[str, Tensor]],
    avg_model: dict[str, Tensor],
    /,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    *,
    iterations: int = 6,
) -> tuple[dict[str, Tensor], list[float]]:
    keys = get_model_keys(models)

    M = len(models)
    λ = torch.ones(M, device=device, dtype=dtype)

    # null diff
    diff_model: dict[str, Tensor] = {key: torch.zeros(1, device=device, dtype=dtype) for key in keys} 

    best_loss = 2
    initial_loss = compute_loss(models, avg_model, diff_model, 0, device, dtype)
    with trange(iterations) as pbar:
        for i in pbar:
            diff_model = update_diff_weights(models, avg_model, diff_model, λ, device, dtype)
            λ = update_λ(models, avg_model, diff_model, device, dtype, normalize=i < iterations - 1)

            loss = compute_loss(models, avg_model, diff_model, λ, device, dtype)
            loss /= initial_loss
            pbar.set_postfix(loss=f'{loss*100:.3f}%')

            # TODO Early break
            # if abs(loss - best_loss)/best_loss < 0.5/100:
            #     break
            best_loss = loss

    return diff_model, λ.tolist()


def make_hyper_model(
    avg_model: dict[str, Tensor],
    diff_model: dict[str, Tensor],
    /,
    multiplier: float,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
):
    keys = get_model_keys(avg_model)

    hyper_model: dict[str, Tensor] = {}
    for key in tqdm(keys, f"Creating super-model with λ={multiplier}"):
        Wavg = avg_model[key].to(device=device, dtype=dtype)
        dW = diff_model[key]

        hyper_model[key] = Wavg.add(multiplier * dW).cpu()

        del Wavg, dW

    return hyper_model

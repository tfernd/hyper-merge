from __future__ import annotations
from typing import Optional

import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
from safetensors.torch import load_file, save_file

from .types import Checkpoint, PathLike, PathsLike
from .constants import SD_KEYS, FLOAT32
from .utils import free_cuda


# Path-related functions


def load_checkpoint(
    path: PathLike,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    keys: Optional[list[str]] = None,
) -> Checkpoint:
    """
    Load a model checkpoint from a file on disk.
    The checkpoint is transferred to a specified `device` and `dtype`.
    Supports both `.ckpt` and `.safetensors` formats.
    """

    free_cuda()

    path = Path(path)
    if path.suffix == ".safetensors":
        checkpoint = load_file(path, device="cpu")
    else:
        logging.warning("Please use .safetensors!")
        assert path.suffix == ".ckpt", "What are you trying to open?"

        checkpoint = torch.load(path, map_location="cpu")
        checkpoint = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    filter_checkpoint_(checkpoint, keys or SD_KEYS)  # TODO extend to other models
    transfer_checkpoint_(checkpoint, dtype, device)

    return checkpoint


def load_checkpoints(
    paths: PathsLike,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    keys: Optional[list[str]] = None,
) -> list[Checkpoint]:
    """
    Load multiple model checkpoints from a list of file paths.
    Each checkpoint is transferred to a specified `device` and `dtype`.
    """

    return [load_checkpoint(path, dtype, device, keys=keys) for path in tqdm(paths, desc="Loading checkpoints")]


def save_checkpoint_(
    checkpoint: Checkpoint,
    path: str | Path,
    /,
    dtype: Optional[torch.dtype] = None,
    *,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """
    Save the current model checkpoint to disk. Overwrites the file if it already exists.
    The checkpoint is saved in `.safetensors` format and is transferred to the CPU before saving.
    """

    path = Path(path)
    if path.exists():
        logging.warning(f"Deleting {path}!")
        path.unlink()

    assert path.suffix == ".safetensors", "I told you to use safetensors, 😔"
    path.parent.mkdir(exist_ok=True, parents=True)

    transfer_checkpoint_(checkpoint, dtype, device=torch.device("cpu"))
    save_file(checkpoint, path, metadata=metadata)


def create_average_checkpoint(
    paths: PathsLike,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Checkpoint:
    """
    Create an averaged model checkpoint from multiple checkpoint files.
    The averaging is performed element-wise across tensors.
    The averaged checkpoint is then transferred to a specified `device` and `dtype`.
    """

    M = len(paths)
    assert M > 1

    average_checkpoint: Checkpoint = {}
    for path in tqdm(paths, desc="Creating average checkpoint"):
        free_cuda()

        # Use float to avoid overflow
        checkpoint = load_checkpoint(path, FLOAT32, device, keys=SD_KEYS)  # TODO extend keys

        for key, weights in checkpoint.items():
            average_checkpoint[key] = average_checkpoint.get(key, 0) + weights.div(M)

            del checkpoint[key]
        del checkpoint
    free_cuda()

    transfer_checkpoint_(average_checkpoint, dtype, device)

    return average_checkpoint


# checkpoint-related functions


def filter_checkpoint_(
    checkpoint: Checkpoint,
    /,
    keys: list[str],
) -> None:
    """
    Filter out keys from a checkpoint dictionary to keep only the specified keys.
    Useful for selecting specific layers or parameters from a model checkpoint.
    """

    keys_to_remove = [key for key in checkpoint.keys() if key not in keys]
    for key in keys_to_remove:
        del checkpoint[key]


def filter_checkpoints_(
    checkpoints: list[Checkpoint],
    /,
    keys: list[str],  # TODO Sequence? Listable?
) -> None:
    """
    Filter out keys from a list of model checkpoints to keep only the specified keys in each checkpoint.
    Useful for selecting specific layers or parameters from a list of model checkpoints.
    """

    for checkpoint in checkpoints:
        filter_checkpoint_(checkpoint, keys)


def transfer_checkpoint_(
    checkpoint: Checkpoint,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Transfer a model checkpoint to a different `device` and `dtype`.
    """

    for key, weight in checkpoint.items():
        checkpoint[key] = weight.to(dtype=dtype, device=device, non_blocking=True)


def transfer_checkpoints_(
    checkpoints: list[Checkpoint],
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Transfer a list of model checkpoints to a different `device` and `dtype`.
    """

    for checkpoint in checkpoints:
        transfer_checkpoint_(checkpoint, dtype, device)

from __future__ import annotations
from typing import Optional

import logging
from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch import Tensor
from safetensors.torch import load_file, save_file

from .types import Checkpoint, Checkpoints, PathLike, PathsLike
from .constants import SD_KEYS, FLOAT32
from .utils import free_cuda


################### Path-related functions ###################


def load_checkpoint_(
    checkpoint: PathLike | Checkpoint,
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

    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        if path.suffix == ".safetensors":
            checkpoint = load_file(checkpoint, device="cpu")
        else:
            logging.warning("Please use .safetensors!")
            assert path.suffix == ".ckpt", "What are you trying to open?"

            obj = torch.load(path, map_location="cpu")
            checkpoint = obj["state_dict"] if "state_dict" in obj else obj
            assert isinstance(checkpoint, dict)

    filter_checkpoint_(checkpoint, keys or SD_KEYS)  # TODO extend to other models
    transfer_checkpoint_(checkpoint, dtype, device)

    return checkpoint


def load_checkpoints(
    paths: PathsLike | Checkpoints,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    *,
    keys: Optional[list[str]] = None,
) -> Checkpoints:
    """
    Load multiple model checkpoints from a list of file paths.
    Each checkpoint is transferred to a specified `device` and `dtype`.
    """

    return [load_checkpoint_(checkpoint, dtype, device, keys=keys) for checkpoint in tqdm(paths, desc="Loading checkpoints")]


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


def create_average_checkpoint_(
    checkpoints: PathsLike | Checkpoints,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Checkpoint:
    """
    Create an averaged model checkpoint from multiple checkpoint files.
    The averaging is performed element-wise across tensors.
    The averaged checkpoint is then transferred to a specified `device` and `dtype`.

    This consumes the checkpoints!
    """

    M = len(checkpoints)
    assert M > 1

    average_checkpoint: Checkpoint = {}
    for i, checkpoint in enumerate(tqdm(checkpoints, desc="Creating average checkpoint")):
        free_cuda()

        # Use float to avoid overflow
        checkpoint = load_checkpoint_(checkpoint, FLOAT32, device, keys=SD_KEYS)  # TODO extend keys

        for key, weights in checkpoint.items():
            average_checkpoint[key] = average_checkpoint.get(key, 0) + weights.div(M)

        del checkpoints[i]
    free_cuda()

    transfer_checkpoint_(average_checkpoint, dtype, device)

    return average_checkpoint


################### checkpoint-related functions ###################


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
    checkpoints: Checkpoints,
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
    checkpoints: Checkpoints,
    /,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Transfer a list of model checkpoints to a different `device` and `dtype`.
    """

    for checkpoint in checkpoints:
        transfer_checkpoint_(checkpoint, dtype, device)


# ? needed?
def inspect_checkpoint(checkpoint: Checkpoint, /) -> tuple[torch.dtype, torch.device]:
    """
    Get the `dtype` and `device` information of a model checkpoint.

    This function checks that all tensors in the checkpoint have the same dtype and device.
    """

    dtype, device = None, None
    for weights in checkpoint.values():
        if isinstance(weights, Tensor):
            if dtype is None and device is None:
                dtype, device = weights.dtype, weights.device
            else:
                if weights.dtype != dtype:
                    raise ValueError("All tensors in the checkpoint should have the same dtype.")
                if weights.device != device:
                    raise ValueError("All tensors in the checkpoint should be on the same device.")

    if dtype is None or device is None:
        raise ValueError("The checkpoint appears to be empty or does not contain any tensors.")

    return dtype, device

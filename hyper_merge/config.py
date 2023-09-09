from __future__ import annotations
from typing_extensions import Self

from dataclasses import dataclass

from pathlib import Path
import yaml

import torch


@dataclass
class Config:
    models: list[Path]
    device: torch.device
    dtype: torch.dtype
    iterations: int
    ranks: tuple[int, ...]

    @classmethod
    def from_file(cls, path: str | Path, /) -> Self:
        path = Path(path)
        assert path.exists(), f"Expected path '{path}' to exist, but it doesn't."
        assert path.suffix == ".yaml", f"Expected the file to have a '.yaml' extension, but got '{path.suffix}' instead."

        with open(path, "r") as f:
            config_data = yaml.safe_load(f)

        models = [Path(model) for model in config_data["models"]]
        assert models, "models list cannot be empty"

        # Check if the device is cuda or cpu
        device_str = config_data.get("device", "cuda").lower()
        assert device_str in ("cuda", "cpu"), "device must be either 'cuda' or 'cpu'"
        device = torch.device(device_str)

        # Check dtype and convert to the appropriate torch dtype
        dtype_str = config_data.get("dtype", "float16").lower()
        assert dtype_str in ["float16", "float32", "bfloat16"], "dtype must be one of 'float16', 'float32', or 'bfloat16'"
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[dtype_str]

        # This is mandatory! ENFORCE assert!
        models = [Path(model) for model in config_data.get("models", [])]

        # Validate iterations
        iterations = int(config_data.get("iterations", 6))
        assert iterations > 2, "iterations should be greater than 2"

        # Validate and convert ranks
        ranks = tuple(map(int, config_data.get("ranks", (64, 64, 32, 32))))
        assert ranks, "ranks tuple cannot be empty"

        return cls(models, device, dtype, iterations, ranks)

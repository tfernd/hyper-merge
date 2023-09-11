from __future__ import annotations

from pathlib import Path

from torch import Tensor

PathLike = str | Path
PathsLike = list[str] | list[Path] | list[PathLike]

Checkpoint = dict[str, Tensor]
Checkpoints = list[Checkpoint]

SVDOutput = tuple[tuple[Tensor, Tensor], tuple[int, ...], int]  # (U, V), shape, rank
SVDCheckpoint = dict[str, SVDOutput]

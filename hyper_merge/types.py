from __future__ import annotations

from torch import Tensor

Checkpoint = dict[str, Tensor]

SVDOutput = tuple[tuple[Tensor, Tensor], tuple[int, ...], int]
SVDCheckpoint = dict[str, SVDOutput]

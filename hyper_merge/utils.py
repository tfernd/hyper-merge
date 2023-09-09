from __future__ import annotations

import gc
import torch


def free_cuda() -> None:
    """
    Free up unused memory from the CPU and GPU.

    This function runs Python's garbage collection to free up memory from the CPU. Additionally, if CUDA is available, it empties the CUDA cache and collects inter-process cached memory to further free up GPU memory.
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

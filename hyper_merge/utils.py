from __future__ import annotations

import gc
import torch


def free_cuda():
    torch.cuda.empty_cache()
    gc.collect()

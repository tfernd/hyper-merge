from __future__ import annotations

import argparse
from pathlib import Path
from tqdm.auto import tqdm
import logging
import shutil

import torch
from torch import Tensor

from hyper_merge.config import Config
from hyper_merge.utils import free_cuda
from hyper_merge.constants import LORA_KEYS
from hyper_merge.checkpoint import create_average_checkpoint, save_checkpoint_, load_checkpoint_, load_checkpoints, filter_checkpoint_
from hyper_merge.hyper import create_hyper_checkpoint, remove_direction
from hyper_merge.svd import make_lora_checkpoint

# Get terminal width
terminal_width = shutil.get_terminal_size().columns

# Initialize logging with custom formatting
FORMAT = f'\n{"=" * terminal_width}\n\033[91m%(asctime)s\033[0m - %(message)s\n'
logging.basicConfig(level=logging.INFO, format=FORMAT)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Hyper-Merge Script")
    parser.add_argument("--config", type=str, default="config/example.yaml", help="Path to the configuration file. Must be a YAML file.")
    args = parser.parse_args()

    # Load configuration
    logging.info("Loading configuration...")
    config = Config.from_file(args.config)
    dtype, device = config.dtype, config.device
    print(config)

    # Create an average model if not already exists
    average_path = Path(f"models/{config.name}_average.safetensors")
    logging.info("Creating average model checkpoint if not exists...")
    if average_path.exists():
        ...
        # TODO check if metadata is the same, otherwise, remove it
    if not average_path.exists():
        average_checkpoint = create_average_checkpoint(config.models, dtype, device)
        metadata = dict(models=", ".join(config.checkpoint_names))
        save_checkpoint_(average_checkpoint, average_path, dtype, metadata=metadata)
    average_checkpoint = load_checkpoint_(average_path, dtype, device)

    # Filter the average checkpoint to only include LoRA keys
    logging.info("Filtering average model checkpoint with LoRA related weights...")
    filter_checkpoint_(average_checkpoint, LORA_KEYS)

    # Load checkpoints with only LoRA-related weights
    logging.info("Loading checkpoints with LoRA-related weights...")
    trimmed_checkpoints = load_checkpoints(config.models, dtype, keys=LORA_KEYS)

    λs: list[Tensor] = []
    # Main loop for creating hyper-LoRAs
    for step, rank in enumerate(tqdm(config.ranks, desc="Creating hyper-LoRAs")):
        logging.info(f"Initiating hyper-LoRA creation for step={step}, rank={rank}")

        # Free CUDA resources
        free_cuda()

        # Compute better average, differential weights, and scales
        logging.info("Computing hyper-checkpoint...")
        trimmed_diff_uv, λ = create_hyper_checkpoint(
            trimmed_checkpoints, average_checkpoint, dtype, device, rank=rank, iterations=config.iterations
        )
        λs.append(λ)

        # Save the LoRA-checkpoint
        lora_path = Path(f"models/{config.name}_{step}-{rank}.safetensors")
        lora = make_lora_checkpoint(trimmed_diff_uv)
        metadata = {name: str(scale.item()) for (name, scale) in zip(config.checkpoint_names, λ)}
        save_checkpoint_(lora, lora_path, dtype, metadata=metadata)

        # Remove specific directions based on differential weights and scales
        logging.info("Removing specific directions from checkpoints...")
        trimmed_checkpoints = remove_direction(trimmed_checkpoints, trimmed_diff_uv, λ, dtype, device)
    λ = torch.stack(λs)

    # Create and display multipliers
    out = []
    for ckpt_name, scale in zip(config.checkpoint_names, λ.T):
        loras = " ".join([f"<lora:{config.name}_{step}-{rank}:{s.item():.4f}>" for step, (s, rank) in enumerate(zip(scale, config.ranks))])
        out.append(f"{ckpt_name}: {loras}")
    out = "\n".join(out)
    logging.info(out)

    with open(f"models/{config.name}_multipliers.txt", "w") as f:
        logging.info("Saving multipliers'...")
        f.write(out)

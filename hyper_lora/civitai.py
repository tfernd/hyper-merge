from __future__ import annotations
from typing import NamedTuple

import logging
from tqdm.auto import tqdm

from pathlib import Path
import requests

import torch
from torch import Tensor
from safetensors.torch import load_file

# Civitai cached models
CACHE: dict[int, tuple[str, str]] = {}


class ModelDownloadInfo(NamedTuple):
    filename: str
    url: str


def get_civitai_model_url(modelId: int, /) -> ModelDownloadInfo:
    if modelId in CACHE:
        return ModelDownloadInfo(*CACHE[modelId])

    url = f"https://civitai.com/api/v1/models/{modelId}"

    response = requests.get(url)
    if response.status_code == 200:
        obj = response.json()

        files = obj["modelVersions"][0]["files"]
        safe_files = list(filter(lambda o: o["metadata"]["format"] == "SafeTensor", files))
        file = safe_files[0] if len(safe_files) >= 1 else files[0]

        CACHE[modelId] = file["name"], file["downloadUrl"]  # type: ignore

        return ModelDownloadInfo(*CACHE[modelId])

    raise ValueError  # TODO better message


def download_ckpt(url: str, path: str | Path, /) -> Path:
    path = Path(path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {url}") as pbar:
            for chunk in response.iter_content(chunk_size=8_192):
                pbar.update(len(chunk))
                if chunk:
                    f.write(chunk)

            if pbar.n < total_size:
                path.unlink()

                raise ValueError("Download incomplete. File removed.")

    return path


def load_ckpt(path: str | Path, /) -> dict[str, Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_file(path)

    logging.warning("Please use .safetensors!")

    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

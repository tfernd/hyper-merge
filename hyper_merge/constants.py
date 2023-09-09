from __future__ import annotations

from pathlib import Path
import json


HERE = Path(__file__).parent


# TODO extend to other models
with open(HERE / "sd_1.5-keys.json") as handle:
    SD_KEYS: list[str] = json.load(handle)

# TODO extend to other models
with open(HERE / "sd_1.5-lora_mapping.json") as handle:
    LORA_MAPPING: dict[str, str] = json.load(handle)

del HERE

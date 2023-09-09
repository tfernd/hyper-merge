from __future__ import annotations

import argparse

from hyper_merge import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-Merge Script")

    parser.add_argument("--config", type=str, default="config/example.yaml", help="Path to the configuration file. Must be a YAML file.")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    print(config)

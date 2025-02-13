#!/usr/bin/env python3

from pathlib import Path

from datasets.vivit_spatial import ViViTSpatial
from utils.config import get_cli_config
from utils.train import train_vivit_temporal
from datasets.kinetics400 import ROOT_DATA_PATH
import sys


def main(argv):
    config = get_cli_config(
        config_location=Path("configs", "train", "vivit_kinetics400")
    )
    base_name = "spatial_224_25"
    if "spatial_cache_suffix" in config:
        base_name += config["spatial_cache_suffix"]
    train_data = ViViTSpatial(
        Path(ROOT_DATA_PATH, "kinetics400"),
        split="train_40",
        base_name=base_name,
        k=config["k"],
    )
    val_data = ViViTSpatial(
        Path(ROOT_DATA_PATH, "kinetics400"),
        split="val",
        base_name=base_name,
        k=config["k"],
    )
    train_vivit_temporal(config, train_data, val_data)


if __name__ == "__main__":
    main(sys.argv)

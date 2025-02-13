#!/usr/bin/env python3

from pathlib import Path

from datasets.epic_kitchens import EPICKitchens, ROOT_DATA_PATH
from utils.config import get_cli_config
from utils.spatial import compute_vivit_spatial
import sys


def main(argv):
    config = get_cli_config(
        config_location=Path("configs", "spatial", "vivit_epic_kitchens")
    )
    k = config.get("k", 0)
    config_name = config.get("config_uniq_name", k)
    location = Path(ROOT_DATA_PATH, "epic_kitchens")
    for split in "train", "validation":
        print(f"{split.capitalize()}, k={config_name}", flush=True)
        data = EPICKitchens(location, split=split, shuffle=False)
        compute_vivit_spatial(config, location / split / f"spatial_{config_name}", data)


if __name__ == "__main__":
    main(sys.argv)

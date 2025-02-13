#!/usr/bin/env python3

from pathlib import Path

from datasets.kinetics400 import Kinetics400, ROOT_DATA_PATH
from utils.config import get_cli_config
from utils.spatial import compute_vivit_spatial
import sys


def main(argv):
    config = get_cli_config(
        config_location=Path("configs", "spatial", "vivit_kinetics400")
    )
    k = config.get("k", 0)
    token_drop_mode = False
    if 'token_drop' in config['model']['spatial_config'] and config['model']['spatial_config']['token_drop']:
        token_drop_mode = True
        if 'config_uniq_name' in config:
            config_name = config['config_uniq_name']
        else:
            config_name = 'drop_%d' % config['model']['spatial_config']['num_to_drop_in_block'][0]
    location = Path(ROOT_DATA_PATH, "kinetics400")
    for split in "train", "val":
        if not token_drop_mode:
            print(f"{split.capitalize()}, k={k}", flush=True)
        else:
            print(f"{split.capitalize()}, config: {config_name}", flush=True)
        max_tars = config.get("max_tars", None) if (split == "train") else None
        data = Kinetics400(
            location,
            split=split,
            decode_size=224,
            decode_fps=25,
            max_tars=max_tars,
            shuffle=False,
        )
        if max_tars is not None:
            split = f"{split}_{max_tars}"
        if not token_drop_mode:
            target_path = location / split / f"spatial_224_25_{k}"
        else:
            target_path = location / split / f"spatial_224_25_{config_name}"
        compute_vivit_spatial(config, target_path, data)


if __name__ == "__main__":
    main(sys.argv)

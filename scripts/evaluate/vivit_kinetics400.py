#!/usr/bin/env python3

from pathlib import Path

from datasets.kinetics400 import Kinetics400, ROOT_DATA_PATH
from models.vivit import FactorizedViViT
from utils.config import initialize_run
from utils.evaluate import run_evaluations, evaluate_vivit_metrics
import sys


def main(argv):
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_kinetics400")
    )
    data = Kinetics400(
        Path(ROOT_DATA_PATH, "kinetics400"), split="val", decode_size=224, decode_fps=25
    )
    run_evaluations(config, FactorizedViViT, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main(sys.argv)

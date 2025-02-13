#!/usr/bin/env python3

from pathlib import Path

from datasets.epic_kitchens import EPICKitchens, ROOT_DATA_PATH
from models.vivit import FactorizedViViT
from utils.config import initialize_run
from utils.evaluate import evaluate_vivit_metrics, run_evaluations
import sys


def main(argv):
    config = initialize_run(
        config_location=Path("configs", "evaluate", "vivit_epic_kitchens")
    )
    data = EPICKitchens(Path(ROOT_DATA_PATH, "epic_kitchens"), split="validation")
    run_evaluations(config, FactorizedViViT, data, evaluate_vivit_metrics)


if __name__ == "__main__":
    main(sys.argv)

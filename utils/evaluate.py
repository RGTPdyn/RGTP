from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.base import dict_csv_header, dict_csv_line, dict_string
from algorithms.policies import (
    TokenNormThreshold,
    TokenNormTopK,
    TokenNormTopFraction,
)
from utils.misc import (
    TopKAccuracy,
    get_device_description,
    get_pytorch_device,
    set_policies,
    tee_print,
)


def evaluate_vivit_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    top_1 = TopKAccuracy(k=1)
    top_5 = TopKAccuracy(k=5)
    num_workers = config.get("num_workers", 0)
    data = DataLoader(data, batch_size=1, num_workers=num_workers)
    n_items = config.get("n_items", len(data))
    dbg_stop_after = config.get("dbg_stop_after", -1)
    dbg_stop_counter = 0
    loop_obj = tqdm(zip(range(n_items), data), total=n_items, ncols=0)
    for _, (video, label) in loop_obj:
        model.reset()
        with torch.inference_mode():
            output = model(video.to(device))
        label = label.to(device)
        top_1.update(output, label)
        top_5.update(output, label)
        # compute stats for display
        cur_top_1 = top_1.compute()
        loop_obj.set_postfix({'top_1': cur_top_1})
        # debug stop
        if dbg_stop_after > 0:
            dbg_stop_counter += 1
            if dbg_stop_counter == dbg_stop_after:
                break
    metrics = {"top_1": top_1.compute(), "top_5": top_5.compute(), "num_clips": n_items}
    counts = model.total_counts() / n_items
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}


def run_evaluations(config, model_class, data, evaluate_function):
    device = config.get("device", get_pytorch_device())
    if "threads" in config:
        torch.set_num_threads(config["threads"])

    # Load and set up the model.
    model = model_class(**(config["model"]))
    model.load_state_dict(torch.load(config["weights"]))
    model = model.to(device)

    completed = []
    output_dir = Path(config["_output"])

    def do_evaluation(title):
        with open(output_dir / "output.txt", "a") as tee_file:
            # Run the evaluation.
            model.eval()
            results = evaluate_function(device, model, data, config)

            # Print and save results.
            tee_print(title, tee_file)
            tee_print(get_device_description(device), tee_file)
            if isinstance(results, dict):
                save_csv_results(results, output_dir, first_run=(len(completed) == 0))
                for key, val in results.items():
                    tee_print(key.capitalize(), tee_file)
                    tee_print(dict_string(val), tee_file)
            else:
                tee_print(results, tee_file)
            tee_print("", tee_file)
            completed.append(title)

    # Evaluate the model.
    if config.get("vanilla", False):
        do_evaluation("Vanilla")
    for k in config.get("token_top_k", []):
        set_policies(model, TokenNormTopK, k=k)
        do_evaluation(f"Token top k={k}")
    for fraction in config.get("token_top_fraction", []):
        set_policies(model, TokenNormTopFraction, fraction=fraction)
        do_evaluation(f"Token top {fraction * 100:.1f}%")
    for threshold in config.get("token_thresholds", []):
        set_policies(model, TokenNormThreshold, threshold=threshold)
        do_evaluation(f"Token threshold {threshold}")


def save_csv_results(results, output_dir, first_run=False):
    for key, val in results.items():
        with open(output_dir / f"{key}.csv", "a") as csv_file:
            if first_run:
                print(dict_csv_header(val), file=csv_file)
            print(dict_csv_line(val), file=csv_file)

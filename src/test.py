import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger

from src.datasets.loader.cohort_data import load_experiment_cohort, prepare_fold
from src.training.callbacks.rich_progress_bar import RichProgressBar
from src.training.callbacks.simple_logger import SimpleLogger
from src.training.trainer import Trainer
from src.training.utils import DotDict, seed_everything, setup_device
from src.models.model_selector import ModelSelector


@logger.catch(reraise=True)
def main(args):
    config_file = args.experiment_log_dir / "config.yaml"
    cfg = DotDict(yaml.safe_load(config_file.read_text()))

    seed_everything(cfg.training.seed)
    device, _ = setup_device(args.accelerator, args.devices)

    setting_path = args.experiment_log_dir / "cv_setting.json"
    if not setting_path.exists():
        raise ValueError(
            "Missing cv_setting.json: use a new Loader-API experiment; see docs/07_Loader_APIへの移行.md"
        )
    cohort, cv = load_experiment_cohort(cfg.dataset, saved_setting=setting_path)
    try:
        fold_results = evaluate_folds(cfg, args, cohort, cv, device)
    finally:
        cohort.release()

    # Calculate the cross-validation results
    fold_keys = fold_results.keys()
    metric_keys = fold_results[list(fold_keys)[0]].keys()

    average_metrics = {}
    for key in metric_keys:
        values = [fold_results[i][key] for i in fold_keys]
        average = sum(values) / len(values)
        average_metrics[f"average_{key}"] = average

    # Prepare the output file path
    if args.output is None:
        exp_name = args.experiment_log_dir.stem
        output_path = Path(f"results/metrics/{exp_name}.json")
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Save the evaluation results
    contents = {
        "evaluated_at": datetime.now().isoformat(),
        "experiment_log_dir": str(args.experiment_log_dir),
        **average_metrics,
        **fold_results,
    }
    Path(output_path).write_text(json.dumps(contents, ensure_ascii=False, indent=4))


def evaluate_folds(cfg, args, cohort, cv, device):
    fold_results = {}
    for fold_idx in range(cv.n_folds):
        logger.info(f"Start testing fold {fold_idx}")
        stats = None
        if cfg.dataset.get("normalization") == "train-wise":
            stats = torch.load(
                args.experiment_log_dir
                / "weights"
                / f"fold{fold_idx}_normalization.pt",
                map_location="cpu",
            )
        prepared = prepare_fold(
            cohort, cv, cfg.dataset, fold_idx, normalization_stats=stats
        )
        test_loader = DataLoader(
            prepared.test,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model_class = ModelSelector.select(cfg.model.arch)
        model = model_class.load_model(
            args.experiment_log_dir / "weights" / f"fold{fold_idx}_best.pt",
            cfg.model.config,
        ).to(device)
        trainer = Trainer(
            device,
            loss_function=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(model.parameters(), lr=float(cfg.training.lr)),
            max_epochs=cfg.training.max_epochs,
        )
        res = trainer.evaluate(model, test_loader, [SimpleLogger(), RichProgressBar()])
        fold_results[f"fold{fold_idx}"] = res["metrics"]
        del test_loader, prepared, model, trainer
    return fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--experiment-log-dir",
        type=Path,
        required=True,
        help="Path to the experiment log dir",
    )
    parser.add_argument(
        "-a",
        "--accelerator",
        default="auto",
        help="Accelerator type (e.g., 'cpu', 'cuda')",
    )
    parser.add_argument(
        "-d",
        "--devices",
        default="auto",
        help="Device IDs (GPU), must be int, string, or a sequence of ints",
    )
    parser.add_argument(
        "--num-workers",
        default=8,
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to save the evaluation results",
    )
    args = parser.parse_args()

    main(args)

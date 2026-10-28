import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from loguru import logger

from src.losses.loss_selector import LossSelector
from src.training.callbacks.early_stopping import EarlyStopping
from src.training.callbacks.model_checkpoint import ModelCheckpoint
from src.training.callbacks.rich_progress_bar import RichProgressBar
from src.training.callbacks.simple_logger import SimpleLogger
from src.training.trainer import Trainer
from src.training.utils import (
    DotDict,
    save_learning_curve,
    seed_everything,
    setup_device,
)
from src.datasets.loader.cohort_data import (
    load_experiment_cohort,
    prepare_fold,
    save_cv_setting,
)
from src.models.model_selector import ModelSelector


@logger.catch(reraise=True)
def main(cfg, args):
    # Setup result directories
    timestamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    log_dir = Path(f"results/experiments/{cfg.info.name}@{timestamp}")
    log_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    weight_dir = Path(log_dir / "weights")
    weight_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    history_dir = Path(log_dir / "histories")
    history_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Save experiment settings
    shutil.copy(args.config, log_dir / "config.yaml")
    # Setup logger
    logger.remove()
    logger.add(sys.stderr)
    logger.add(log_dir / "training.log")
    logger.info(f"Starting the experiment `{cfg.info.name}`")

    # Init seed & device
    seed_everything(cfg.training.seed)
    device, _ = setup_device(args.accelerator, args.devices)

    cohort, cv = load_experiment_cohort(cfg.dataset)
    try:
        save_cv_setting(log_dir / "cv_setting.json", cohort, cv)
        return train_folds(
            cfg, args, cohort, cv, log_dir, weight_dir, history_dir, device
        )
    finally:
        cohort.release()


def train_folds(cfg, args, cohort, cv, log_dir, weight_dir, history_dir, device):
    fold_results = {}
    for fold_idx in range(cv.n_folds):
        logger.info(f"Start training fold {fold_idx}")
        prepared = prepare_fold(cohort, cv, cfg.dataset, fold_idx)
        if len(cfg.dataset.channel_set) != cfg.model.config.input_channels:
            raise ValueError("dataset.channel_set and model input_channels must match")
        logger.info(
            f"Epochs: train={len(prepared.train)}, val={len(prepared.val)}, "
            f"test={len(prepared.test)}"
        )
        if prepared.normalization_stats is not None:
            torch.save(
                prepared.normalization_stats,
                weight_dir / f"fold{fold_idx}_normalization.pt",
            )
        loader_args = dict(
            batch_size=cfg.training.batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        train_loader = DataLoader(prepared.train, shuffle=True, **loader_args)
        val_loader = DataLoader(prepared.val, shuffle=False, **loader_args)
        test_loader = DataLoader(prepared.test, shuffle=False, **loader_args)

        # Prepare model
        model_class = ModelSelector.select(cfg.model.arch)
        model = model_class(**cfg.model.config)

        # Prepare training settings
        loss_class = LossSelector.select(cfg.training.loss.name)
        training_strategy = {
            "loss_function": loss_class(cfg.training.loss.args),
            "optimizer": optim.Adam(model.parameters(), lr=float(cfg.training.lr)),
            "max_epochs": cfg.training.max_epochs,
        }
        callbacks = [
            RichProgressBar(),
            SimpleLogger(),
            ModelCheckpoint(
                output_dir=log_dir / "weights",
                filename=f"fold{fold_idx}_best.pt",
                **cfg.training.model_checkpoint,
            ),
            EarlyStopping(**cfg.training.early_stopping),
        ]
        trainer = Trainer(device, **training_strategy)
        history = trainer.fit(model, train_loader, val_loader, callbacks)
        trainer.save_model(model, f"{weight_dir}/fold{fold_idx}_last.pt")

        # Save the hisotry csv
        history_path = history_dir / f"fold{fold_idx}_history.csv"
        history_df = pd.DataFrame(history)
        save_learning_curve(history_df, history_dir / f"fold{fold_idx}_curve.jpg")
        history_df.to_csv(history_path, index=False)

        # Evaluate the best model in the fold
        best_model = model_class.load_model(
            f"{weight_dir}/fold{fold_idx}_best.pt", cfg.model.config
        ).to(device)
        res = trainer.evaluate(
            best_model, test_loader, [SimpleLogger(), RichProgressBar()]
        )
        fold_results[fold_idx] = res["metrics"]
        del train_loader, val_loader, test_loader, prepared, trainer, model, best_model

    # Show cross-validation results
    metric_keys = fold_results[0].keys()
    average_metrics = {}
    for key in metric_keys:
        values = [fold_results[i][key] for i in fold_results]
        average = sum(values) / len(values)
        average_metrics[key] = average

    logger.info("===== Cross Validation Results =====")
    for fold_idx, metrics in fold_results.items():
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"Fold {fold_idx}: {metrics_str}")

    average_metric_str = ", ".join(f"{k}={v:.4f}" for k, v in average_metrics.items())
    logger.info(f"Average Metrics: {average_metric_str}")
    logger.info("Training complete.")
    return fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Configuration file path",
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
    args = parser.parse_args()

    cfg = DotDict(yaml.safe_load(args.config.read_text()))
    main(cfg, args)

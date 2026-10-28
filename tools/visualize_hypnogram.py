import sys
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
import torch
import yaml
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from src.models.model_selector import ModelSelector
from src.training.utils import DotDict
from src.datasets.loader.cohort_data import load_prediction_record, validate_config
from src.datasets.loader.standard_sleep_data import read_record_metadata

# Use a clean style and serif fonts for publication
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Constants
EPOCH_PER_SEC = 30
EPOCH_PER_MIN = EPOCH_PER_SEC / 60  # Each epoch is 0.5 minute
STAGE_BG_COLORS = ["salmon", "khaki", "lightgreen", "cornflowerblue", "mediumpurple"]
STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"]
PARTICIPANT_SEX_LABELS = {0: "Male", 1: "Female", 2: "Unknown"}


def plot_hypnogram(ax, stages, title=""):
    """
    Plots a hypnogram on the given axis with an academic style.
    """
    # Plot a gray line connecting the epochs
    ax.plot(
        stages, marker="o", linestyle="-", color="gray", markersize=6, linewidth=1.5
    )

    # Plot each point with its corresponding color
    for i, stage in enumerate(stages):
        ax.plot(
            i,
            stage,
            marker="o",
            color=STAGE_BG_COLORS[stage] if 0 <= stage < 5 else "gray",
            markersize=8,
        )

    # Set x-axis ticks (convert epoch indices to minutes)
    n_ticks = min(20, len(stages))
    tick_indices = np.linspace(0, len(stages) - 1, n_ticks, dtype=int)
    tick_labels = [f"{i * EPOCH_PER_MIN:.1f}" for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Time [min]")

    # Set y-axis ticks and labels
    ax.set_yticks([-1, 0, 1, 2, 3, 4])
    ax.set_yticklabels(["Unknown", *STAGE_LABELS])

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.invert_yaxis()  # Invert y-axis for typical hypnogram style

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_hypnogram_with_errors(
    ax, pred, true, title="", description="", show_legend=False
):
    """
    Plots the predicted hypnogram and highlights epochs in pink where the
    prediction does not match the ground truth.
    Optionally attaches a legend mapping colors to sleep stages and the error highlight.
    A description (if provided) is appended to the title.
    """
    # Highlight errors in pink; each epoch highlighted from (i - 0.5) to (i + 0.5)
    for i, (t, p) in enumerate(zip(true, pred)):
        if t != p:
            ax.axvspan(i - 0.5, i + 0.5, color="pink", alpha=0.3)

    # Plot the prediction hypnogram on top of the highlights
    plot_hypnogram(ax, pred, title=title)

    if show_legend:
        # Create legend handles for each stage
        stage_handles = [
            mpatches.Patch(color=STAGE_BG_COLORS[i], label=STAGE_LABELS[i])
            for i in range(len(STAGE_LABELS))
        ]
        # Create a patch for the error highlight
        error_patch = mpatches.Patch(facecolor="pink", alpha=0.7, label="Error")
        # Combine the handles and attach the legend
        handles = stage_handles + [error_patch]
        ax.legend(
            handles=handles, loc="lower left", frameon=True, fontsize=10, fancybox=True
        )


def plot_stacked_confidences(ax, confidences, title):
    """
    Plots a stacked area chart representing the model's confidence over time for each stage.
    """
    confidences = np.array(confidences)  # shape = (num_epochs, num_classes)
    x = np.arange(confidences.shape[0])  # epoch indices

    ax.stackplot(
        x, confidences.T, labels=STAGE_LABELS, colors=STAGE_BG_COLORS, alpha=0.8
    )

    n_ticks = min(20, len(x))
    tick_indices = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    tick_labels = [f"{i * EPOCH_PER_MIN:.1f}" for i in tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    # y 軸を反転させる
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_comparision_hypnogram(true, preds, save_path, description=""):
    # Create a figure with 2 subplots (one for each hypnogram)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(18, 8), sharex=True)
    fig.suptitle("Hypnogram Comparison", fontsize=16)
    fig.text(0.5, 0.92, description, ha="center", va="center", fontsize=12)

    # Plot ground truth hypnogram
    plot_hypnogram(axs[0], true, title="Ground Truth")
    # Plot model prediction hypnogram with error highlights; toggle legend here
    plot_hypnogram_with_errors(
        axs[1], preds, true, title="Model Prediction (pink = error)"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)


def plot_ground_truth_only_hypnogram(true, save_path, description=""):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 4), sharex=True)
    fig.suptitle("Hypnogram", fontsize=16)
    fig.text(0.5, 0.89, description, ha="center", va="center", fontsize=12)

    plot_hypnogram(axs, true, title="Ground Truth")
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(save_path)


def main(args):
    if bool(args.config) != bool(args.model):
        raise ValueError("Specify both --config and --model for prediction")
    cfg = DotDict(yaml.safe_load(args.config.read_text())) if args.config else None
    dataset_config = dict(cfg.dataset) if cfg else {}
    if cfg:
        validate_config(dataset_config)
    metadata = read_record_metadata(
        args.npz_file,
        **({"channels": dataset_config["channel_set"]} if cfg else {}),
    )
    stages = metadata.stages
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"results/hypnograms/{metadata.record_id}.jpg")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    description = f"Cohort: {metadata.cohort_id} | Record: {metadata.record_id}"

    if cfg:
        model_class = ModelSelector.select(cfg.model.arch)
        model = model_class.load_model(args.model, cfg.model.config)
        model.eval()
        stats = None
        if dataset_config.get("normalization") == "train-wise":
            stats_path = args.normalization_stats or args.model.with_name(
                args.model.stem.rsplit("_", 1)[0] + "_normalization.pt"
            )
            stats = torch.load(stats_path, map_location="cpu")
        dataset, stages, target_indices = load_prediction_record(
            args.npz_file, dataset_config, normalization_stats=stats
        )
        # 窓の開始間隔を横軸の時間に反映する。
        global EPOCH_PER_MIN
        EPOCH_PER_MIN = dataset_config.get("stride_sec", 30.0) / 60.0
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for signal, _ in loader:
                preds.extend(model(signal).argmax(dim=1).numpy().tolist())
        preds = np.asarray(preds)
        true = stages[target_indices]
        valid = (true >= 0) & (true < 5)
        if not valid.any():
            raise ValueError("No valid labeled epochs for evaluation")
        accuracy = accuracy_score(true[valid], preds[valid])
        macro_f1 = f1_score(true[valid], preds[valid], average="macro")
        kappa = cohen_kappa_score(true[valid], preds[valid])
        timeline = np.full(len(stages), -1, dtype=int)
        timeline[target_indices] = preds
        description += (
            f" | Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}"
            f" | Cohen Kappa: {kappa:.3f}"
        )
        plot_comparision_hypnogram(stages, timeline, output_path, description)
    else:
        plot_ground_truth_only_hypnogram(stages, output_path, description)
    plt.close("all")
    logger.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hypnogram comparison")
    parser.add_argument(
        "-f",
        "--npz-file",
        required=True,
        help="Path to the NPZ file containing signals and labels",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=None,
        help="Path to the model checkpoint used for inference",
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
        "-o",
        "--output",
        default=None,
        help=(
            "Path to save the hypnogram plot."
            "Default path is `results/hypnograms/{record_id}.jpg`"
        ),
    )

    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=None,
        help="Saved fold normalization statistics for train-wise inference",
    )
    args = parser.parse_args()
    main(args)

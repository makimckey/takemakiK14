import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def seed_everything(seed: int = 42):
    """
    Fixes random seeds for reproducibility across numpy, random, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_devices(devices: str) -> str:
    """
    Parses the devices argument to ensure it is in the correct format.
    """
    if isinstance(devices, int):
        return devices

    if isinstance(devices, str):
        if devices == "auto":
            return devices

        # Try parsing as a single int
        try:
            return int(devices)
        except ValueError:
            pass

        # Try parsing as a comma-separated list of ints
        try:
            return [int(v.strip()) for v in devices.split(",")]
        except ValueError:
            pass

    raise ValueError("Devices must be 'auto', an int, or a list of ints.")


def setup_device(accelerator: str, devices: str) -> tuple[torch.device, list]:
    """
    Sets up computation device(s) for PyTorch.

    Selects and returns a torch.device object (or list of torch.device objects)
    and device indices based on the specified accelerator type and device selection.

    Args:
        accelerator (ACCELERATOR_TYPE): The accelerator to use ("auto", "cpu", "gpu", or "mps").
        devices (DEVICE_TYPE): Device selection ("auto", int, or list of ints).

    Returns:
        Tuple[Union[torch.device, List[torch.device]], List[int]]: Selected torch.device(s) and device indices.

    Raises:
        AssertionError: If the requested accelerator is not available.
        ValueError: If devices is not "auto", int, or list of ints.

    Examples:
        >>> setup_device("gpu", "auto")
        (device(type='cuda'), [0, 1])
        >>> setup_device("gpu", [0, 2])
        (device(type='cuda:0,2'), [0, 2])
        >>> setup_device("gpu", 1)
        (device(type='cuda:1'), [1])
        >>> setup_device("mps", "auto")
        (device(type='mps'), [])
        >>> setup_device("cpu", 0)
        (device(type='cpu'), [])
    """
    parsed_devices = _parse_devices(devices)

    if accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    # Try to use GPU if available
    if accelerator == "gpu":
        assert torch.cuda.is_available(), "GPU is not available."

        if parsed_devices == "auto":
            device_ids = list(range(torch.cuda.device_count()))
            device = torch.device("cuda")
        elif isinstance(parsed_devices, list):
            device_ids = parsed_devices
            device = torch.device("cuda")
        elif isinstance(parsed_devices, int):
            device_ids = [parsed_devices]
            device = torch.device(f"cuda:{parsed_devices}")
        else:
            raise ValueError("Devices must be a list or an integer.")

        logger.info(f"Using GPU device(s): {device}")
        return device, device_ids

    # Try to use MPS if available
    if accelerator == "mps":
        assert torch.backends.mps.is_available(), "MPS is not available."
        device = torch.device("mps")
        logger.info(f"Using MPS device: {device}")
        return device, []

    # Fallback to CPU
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device, []


class DotDict(dict):
    """
    A dictionary subclass that allows attribute-style access to keys.

    This class extends the built-in `dict` so that dictionary keys can be accessed,
    set, and deleted using dot notation (attribute access), in addition to standard
    item access (`dict[key]`).

    Features:
        - Access keys as attributes: `d.key` instead of `d['key']`.
        - Supports assignment via attributes: `d.key = value`.
        - Supports deletion via attributes: `del d.key`.
        - Recursively wraps nested dictionaries into `DotDict`.
        - Recursively processes lists of dictionaries, converting each element to a `DotDict`.

    Example:
    >>> d = DotDict({"a": 1, "b": {"c": 2}, "d": [{"e": 3}]})
    >>> d.a
    1
    >>> d.b.c
    2
    >>> d.d[0].e
    3
    >>> d.new_attr = 10
    >>> d["new_attr"]
    10
    >>> del d.a
    >>> "a" in d
    False
    """

    def __getattr__(self, key):
        val = self.get(key)
        return self._wrap(val)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        else:
            return value


def save_learning_curve(history_df, save_path):
    # Set figure style
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

    axs[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train Loss",
        linestyle="-",
    )
    axs[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Val Loss",
        linestyle="--",
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc="upper right")

    axs[1].plot(
        history_df["epoch"],
        history_df["train_acc"],
        label="Train Acc",
        linestyle="-",
    )
    axs[1].plot(
        history_df["epoch"],
        history_df["val_acc"],
        label="Val Acc",
        linestyle="--",
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc="lower right")

    plt.suptitle("Training & Validation Curve")
    plt.tight_layout()
    plt.savefig(save_path, format="jpeg", dpi=200)
    plt.close()

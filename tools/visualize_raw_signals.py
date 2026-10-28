"""Sleep-Data-Hub標準形式の任意の30秒エポックを描画する。"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.loader.standard_sleep_data import DEFAULT_CHANNELS, load_sleep_record

STAGE_LABELS = {0: "Wake", 1: "REM", 2: "N1", 3: "N2", 4: "N3", 5: "N4", -1: "Unknown"}


def main(args):
    record = load_sleep_record(args.npz_file, channels=args.channels)
    epoch = random.randrange(len(record.signals)) if args.epoch is None else args.epoch
    if not 0 <= epoch < len(record.signals):
        raise ValueError(f"epoch must be between 0 and {len(record.signals) - 1}")
    signal = record.signals[epoch]
    time_sec = np.arange(len(signal)) / record.sampling_rate
    fig, axes = plt.subplots(
        len(record.channel_names),
        1,
        sharex=True,
        figsize=(12, 2 * len(record.channel_names)),
        squeeze=False,
    )
    for channel, name in enumerate(record.channel_names):
        ax = axes[channel, 0]
        ax.plot(time_sec, signal[:, channel])
        ax.set_ylabel(name)
        ax.grid(True)
    axes[-1, 0].set_xlabel("Time [s]")
    if not args.disable_title:
        stage = STAGE_LABELS.get(int(record.stages[epoch]), "Unknown")
        fig.suptitle(f"Epoch {epoch} — {stage} — {record.sampling_rate:g} Hz")
    output = args.output or Path(
        f"results/raw_signals/{record.metadata.record_id}_epoch-{epoch}.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--npz-file", required=True)
    parser.add_argument("-n", "--epoch", type=int, default=None)
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--disable-title", action="store_true")
    main(parser.parse_args())

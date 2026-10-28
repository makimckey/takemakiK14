from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.loader.standard_sleep_data import load_sleep_record

import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-e", "--epoch", type=int, default=128)
    parser.add_argument("--channels", nargs="+", default=None)
    args = parser.parse_args()

    # データの読み込み
    record = load_sleep_record(
        args.file, **({"channels": args.channels} if args.channels else {})
    )
    if not 0 <= args.epoch < len(record.signals):
        parser.error(f"epoch must be between 0 and {len(record.signals) - 1}")
    signals = record.signals[args.epoch]
    sampling_rate = record.sampling_rate
    channel_names = record.channel_names
    stage = record.stages[args.epoch]

    # TODO: tmp_x, tmp_y を削除し，
    #       任意のエポックを可視化するPythonプログラムを作成してください．
    tmp_x = np.linspace(0, 2 * np.pi, 500)
    tmp_y = np.sin(tmp_x)

    # プロット
    plt.plot(tmp_x, tmp_y, label="sin(x)")
    plt.title("Sign Wave (Ignoring NPZ Data)")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/test.jpg")


if __name__ == "__main__":
    main()

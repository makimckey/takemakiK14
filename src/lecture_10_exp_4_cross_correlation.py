"""
第10・11回 実装課題 4

EEG_C3-RとEEG_C4-Rの相互相関を計算するプログラムを作成してください．
また，相互相関が高い例，低い例についてプロットしてください．
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.loader.standard_sleep_data import load_sleep_record

import argparse
import numpy as np
import matplotlib.pyplot as plt


def cross_corr(a, b):
    """相互相関を計算する関数. `a` と `b` には信号が渡される"""
    # TODO: implement
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--file", required=True)
    p.add_argument("--channels", nargs=2, default=["EEG_C3-R", "EEG_C4-R"])
    a = p.parse_args()

    record = load_sleep_record(a.file, channels=a.channels)
    sig = record.signals
    c3_signal, c4_signal = sig[:, :, 0], sig[:, :, 1]
    assert c3_signal.shape == c4_signal.shape

    corrs = [np.max(cross_corr(c3, c4)) for c3, c4 in zip(c3_signal, c4_signal)]
    high_i = np.argmax(corrs)
    low_i = np.argmin(corrs)

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    for title, idx in [("High correlation", high_i), ("Low correlation", low_i)]:
        plt.figure(figsize=(8, 3))
        plt.plot(c3_signal[idx], label=a.channels[0])
        plt.plot(c4_signal[idx], label=a.channels[1], alpha=0.7)
        plt.title(f"{title} (epoch {idx}) corr={corrs[idx]:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/plots/lecture_10_exp_4_{title.replace(' ', '-')}.jpg")


if __name__ == "__main__":
    main()

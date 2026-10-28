"""
第10・11回 実装課題 3

ある信号に対して，

- 周波数ベースリサンプリング
- 線形リサンプリング

を実行するプログラムを作成してください．
また，生のEOG_LOC-R信号を100Hzサンプリングに変換し，
２種類のリサンプリング後信号を並べてプロットするプログラムを作成してください．
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.loader.standard_sleep_data import load_sleep_record

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.interpolate import interp1d


def freq_based_resampling(sig, old_f, new_f):
    """周波数ベースリサンプリングを行う関数"""
    # TODO: implement
    pass


def linear_resampling(sig, old_f, new_f):
    """線形ベースリサンプリングを行う関数"""
    # TODO: implement
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--file", required=True)
    p.add_argument("-e", "--epoch", type=int, default=128)
    p.add_argument("--channel", default="EOG_LOC-R")
    a = p.parse_args()

    record = load_sleep_record(a.file, channels=[a.channel])
    if not 0 <= a.epoch < len(record.signals):
        p.error(f"epoch must be between 0 and {len(record.signals) - 1}")
    loc_mave = record.signals[a.epoch, :, 0]

    fr = freq_based_resampling(loc_mave, record.sampling_rate, 100)
    lr = linear_resampling(loc_mave, record.sampling_rate, 100)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(fr)
    axes[0].set_title("Freq-based Resampling (100Hz)")
    axes[1].plot(lr)
    axes[1].set_title("Linear Resampling (100Hz)")
    plt.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/lecture_10_exp_3_resampling.jpg")


if __name__ == "__main__":
    main()

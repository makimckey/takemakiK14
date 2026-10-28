"""
第10・11回 実装課題 1

ある信号に対し，フーリエ変換を実施しスペクトラムを得るプログラムを作成してください．
また，任意のエポックの生のEEG_C3-R信号と，そのスペクトラムを並べてプロットするプログラムを作成してください．
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.loader.standard_sleep_data import load_sleep_record

import argparse
import numpy as np
import matplotlib.pyplot as plt


def calc_spectrum(signal, fs):
    """スペクトラムを計算する関数. 周波数とそのスペクトラムを返す"""
    N = len(signal)

    # TODO: implement
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-e", "--epoch", type=int, default=128)
    parser.add_argument("--channel", default="EEG_C3-R")
    args = parser.parse_args()

    record = load_sleep_record(args.file, channels=[args.channel])
    if not 0 <= args.epoch < len(record.signals):
        parser.error(f"epoch must be between 0 and {len(record.signals) - 1}")
    c3_mave = record.signals[args.epoch, :, 0]
    freq, spectrum = calc_spectrum(c3_mave, record.sampling_rate)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(c3_mave)
    plt.title(f"{args.channel} signal (epoch={args.epoch})")
    plt.xlabel("Time [sample]")
    plt.ylabel("Amplitude")

    plt.subplot(1, 2, 2)
    plt.plot(freq, spectrum)
    plt.title(f"Spectrum (epoch={args.epoch})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/test.jpg")


if __name__ == "__main__":
    main()

"""
第10・11回 実装課題 2

ある睡眠エポックに対して，

- 脳波には0.35-30Hzバンドパス
- 眼電位には0.35-30Hzバンドパス
- 顎筋電位には10-99.5Hzバンドパスと50Hzノッチ

をかけるプログラムを作成してください．
また，1.のプログラムを用いて，フィルタリング前後でスペクトラムが変化していることを確認してください．
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.loader.standard_sleep_data import load_sleep_record

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, lowcut, highcut, fs):
    """信号にバンドパスをかける関数"""
    # TODO implement
    pass


def notch_filter(signal, freq, fs):
    """信号にノッチをかける関数"""
    # TODO implement
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-e", "--epoch", type=int, required=True)
    parser.add_argument(
        "--channels", nargs=3, default=["EEG_C3-R", "EOG_LOC-R", "EMG_L-EMG_R"]
    )
    args = parser.parse_args()

    record = load_sleep_record(args.file, channels=args.channels)
    if not 0 <= args.epoch < len(record.signals):
        parser.error(f"epoch must be between 0 and {len(record.signals) - 1}")
    epoch_data = record.signals[args.epoch]
    fs = record.sampling_rate

    # ---- フィルタ処理 ----
    # 脳波 (EEG_C3-R): 0.35–30 Hz bandpass
    eeg = bandpass_filter(epoch_data[:, 0], 0.35, 30, fs)

    # 眼電位 (EOG): 0.35–30 Hz bandpass
    eog = bandpass_filter(epoch_data[:, 1], 0.35, 30, fs)

    # 顎筋電位 (EMG): 10–99.5 Hz bandpass + 50 Hz notch
    emg = bandpass_filter(epoch_data[:, 2], 10, min(99.5, fs / 2 - 0.5), fs)
    emg = notch_filter(emg, 50, fs)

    # ---- スペクトル表示 ----
    N = len(eeg)
    freq = np.fft.rfftfreq(N, d=1 / fs)

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 2, 1)
    plt.plot(eeg)
    plt.title("EEG (0.35–30 Hz)")
    plt.subplot(3, 2, 2)
    plt.plot(freq, np.abs(np.fft.rfft(eeg)))
    plt.title("EEG Spectrum")

    plt.subplot(3, 2, 3)
    plt.plot(eog)
    plt.title("EOG (0.35–30 Hz)")
    plt.subplot(3, 2, 4)
    plt.plot(freq, np.abs(np.fft.rfft(eog)))
    plt.title("EOG Spectrum")

    plt.subplot(3, 2, 5)
    plt.plot(emg)
    plt.title("EMG (10–99.5 Hz + Notch 50 Hz)")
    plt.subplot(3, 2, 6)
    plt.plot(freq, np.abs(np.fft.rfft(emg)))
    plt.title("EMG Spectrum")

    plt.tight_layout()
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/plots/filtered_signals.jpg")
    plt.show()


if __name__ == "__main__":
    main()

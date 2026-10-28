from typing import Tuple

import numpy as np


class ExcludeLabels:
    """(N,T,C) の信号と (N,) のラベルから指定したエポックを除く。"""

    def __init__(self, exclude_classes, sampling_rate=200, epoch_per_sec=30):
        self.exclude_classes = exclude_classes
        # 既存設定との互換性のため引数を残す。エポック化後は不要。
        self.sampling_rate = sampling_rate
        self.epoch_per_sec = epoch_per_sec

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Expected aligned signal (N,T,C) and labels (N,)")
        keep = ~np.isin(y, self.exclude_classes)
        return x[keep], y[keep]

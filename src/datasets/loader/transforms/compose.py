from typing import List, Callable

import numpy as np


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, signal: np.ndarray, label: np.ndarray):
        for transform in self.transforms:
            signal, label = transform(signal, label)
        return signal, label

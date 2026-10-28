from typing import Tuple
import numpy as np


class SelectSignals:
    def __init__(self, selected_signal_indice: np.ndarray):
        # Check if selected_signal_indice is integer array
        for idx in selected_signal_indice:
            if isinstance(idx, (int, np.integer)):
                continue
            raise ValueError("All elements in selected_signal_indice must be integers.")

        self.selected_indice = np.asarray(selected_signal_indice)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects specific channels from the input signal array based on provided indices.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (N, T, C)
              - N: number of epochs
              - T: number of temporal samples per epoch
              - C: number of channels
        y : np.ndarray
            Labels associated with each epoch, returned unchanged.

        Returns
        -------
        x_selected : np.ndarray
            Shape (N, T, K), where K = number of selected channels
        y : np.ndarray
            Unchanged labels.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (N, T, C), but got shape {x.shape}")

        N, T, C = x.shape
        if (self.selected_indice < 0).any() or (self.selected_indice >= C).any():
            raise IndexError("Selected channel indices are out of range.")

        x_selected = x[:, :, self.selected_indice]
        return x_selected, y
